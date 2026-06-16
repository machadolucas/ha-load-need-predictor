"""Coordinator: holds per-load model state and the published prediction.

This is an *on-demand* ``DataUpdateCoordinator`` — there is no polling interval.
The daily jobs (``jobs.py``, from M3) drive it explicitly: predict + push in the
afternoon, capture + log in the evening. ``_async_update_data`` recomputes the
current forecast from the stored model + a live feature snapshot, so the sensors
always reflect the latest model even after a reload.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .const import DOMAIN, EVAL_WINDOW_DAYS, SUBENTRY_TYPE_LOAD
from .models import LoadConfig, load_config_from_data
from .occupancy import count_people_home, guests_active
from .persistence import PredictorStore, model_from_dict, model_to_dict
from .predictor import (
    ModelState,
    build_features,
    default_model_state,
    predict_kwh,
    predict_minutes,
    rolling_mae,
)

_LOGGER = logging.getLogger(__name__)

type LoadNeedPredictorConfigEntry = ConfigEntry[LoadNeedPredictorCoordinator]


@dataclass
class LoadResult:
    """What the sensors publish for one load."""

    predicted_minutes: int | None = None
    predicted_kwh: float | None = None
    last_delivered_kwh: float | None = None
    prediction_error_minutes: float | None = None
    rolling_mae_minutes: float | None = None
    sample_count: int = 0
    last_push_ok: bool | None = None


class LoadNeedPredictorCoordinator(DataUpdateCoordinator[dict[str, LoadResult]]):
    """Owns the model state per load and recomputes the published forecast."""

    config_entry: LoadNeedPredictorConfigEntry

    def __init__(self, hass: HomeAssistant, entry: LoadNeedPredictorConfigEntry) -> None:
        # update_interval=None → on-demand only; the jobs trigger refreshes.
        super().__init__(hass, _LOGGER, config_entry=entry, name=DOMAIN, update_interval=None)
        self._store = PredictorStore(hass, entry.entry_id)
        self.models: dict[str, ModelState] = {}
        self.training: dict[str, list[dict]] = {}
        self.eval_errors: dict[str, list[float]] = {}
        self._results: dict[str, LoadResult] = {}

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def async_load_runtime(self) -> None:
        """Load persisted model state + training rows from the Store."""
        data = await self._store.async_load()
        for subentry_id, payload in data.items():
            self.models[subentry_id] = model_from_dict(payload.get("model"))
            self.training[subentry_id] = list(payload.get("training", []))
            self.eval_errors[subentry_id] = list(payload.get("eval", []))

    def _runtime_snapshot(self) -> dict:
        """Serialise everything persistable (called by the debounced save)."""
        return {
            subentry_id: {
                "model": model_to_dict(self.models.get(subentry_id, default_model_state())),
                "training": self.training.get(subentry_id, []),
                "eval": self.eval_errors.get(subentry_id, []),
            }
            for subentry_id in self.load_configs()
        }

    def async_persist(self) -> None:
        """Schedule a debounced save of the model + training state."""
        self._store.async_schedule_save(self._runtime_snapshot)

    # ── config access ─────────────────────────────────────────────────────────

    def load_configs(self) -> dict[str, LoadConfig]:
        """Per-load configuration, keyed by subentry id."""
        out: dict[str, LoadConfig] = {}
        for subentry_id, subentry in self.config_entry.subentries.items():
            if subentry.subentry_type != SUBENTRY_TYPE_LOAD:
                continue
            out[subentry_id] = load_config_from_data(subentry.data)
        return out

    def model_for(self, subentry_id: str) -> ModelState:
        """The model for a load, seeded on first use."""
        return self.models.setdefault(subentry_id, default_model_state())

    # ── feature snapshot ───────────────────────────────────────────────────────

    def _state_float(self, entity_id: str | None) -> float | None:
        """Current numeric state of an entity, or None if missing/unparseable."""
        if not entity_id:
            return None
        state = self.hass.states.get(entity_id)
        if state is None or state.state in ("unknown", "unavailable"):
            return None
        try:
            return float(state.state)
        except (TypeError, ValueError):
            return None

    def build_snapshot(self, cfg: LoadConfig) -> dict:
        """Assemble the feature snapshot for the coming cycle from live state.

        v1 uses the current occupancy + guests; the temps are read for logging
        only. (Actual delivered energy is filled later from statistics — M3.)
        """
        now = dt_util.now()
        return {
            "people_home": count_people_home(self.hass, cfg.person_entities),
            "guests": guests_active(self.hass, cfg.guests_calendar_entity),
            "weekend": now.weekday() >= 5,
            "supply_temp": self._state_float(cfg.supply_temp_entity),
            "outdoor_temp": self._state_float(cfg.outdoor_temp_entity),
            "water_total_delta": None,
        }

    # ── daily jobs (driven by jobs.py) ─────────────────────────────────────────

    async def async_predict_and_push(self) -> None:
        """Recompute the forecast and push the target to the scheduler.

        M2: recompute only. The push to ``number.<load>_target`` is wired in M3.
        """
        await self.async_request_refresh()

    async def async_capture_and_log(self) -> None:
        """Capture the day's actual delivery, log it, and calibrate.

        M2: no-op. The statistics read, training-row append, gain update and
        eval refresh are wired in M3.
        """

    # ── prediction ──────────────────────────────────────────────────────────────

    async def _async_update_data(self) -> dict[str, LoadResult]:
        """Recompute the published forecast for every load."""
        results: dict[str, LoadResult] = {}
        for subentry_id, cfg in self.load_configs().items():
            state = self.model_for(subentry_id)
            features = build_features(self.build_snapshot(cfg))
            kwh = predict_kwh(state, features)
            minutes = predict_minutes(
                state,
                features,
                rated_power_kw=cfg.rated_power_kw,
                min_minutes=cfg.min_minutes,
                max_minutes=cfg.max_minutes,
            )
            errors = self.eval_errors.get(subentry_id, [])[-EVAL_WINDOW_DAYS:]
            prev = self._results.get(subentry_id, LoadResult())
            results[subentry_id] = LoadResult(
                predicted_minutes=minutes,
                predicted_kwh=round(kwh, 3),
                last_delivered_kwh=prev.last_delivered_kwh,
                prediction_error_minutes=prev.prediction_error_minutes,
                rolling_mae_minutes=round(rolling_mae(errors), 1) if errors else None,
                sample_count=state.sample_count,
                last_push_ok=prev.last_push_ok,
            )
        self._results = results
        return results
