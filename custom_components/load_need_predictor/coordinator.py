"""Coordinator: holds per-load model state and runs the daily loop.

This is an *on-demand* ``DataUpdateCoordinator`` — there is no polling interval.
``jobs.py`` drives it: predict + push in the afternoon, capture + log in the
evening. ``_async_update_data`` recomputes the current forecast from the stored
model + a live feature snapshot, so the sensors always reflect the latest model
even after a reload.

Prediction/actual alignment (v1): one training row per local calendar day. The
predict job writes the row's prediction; the capture job fills in that same
day's actual delivered energy and calibrates. Same-day alignment keeps the loop
simple — the online gain only needs a *consistent* ratio, and an occupancy-gated
daily model is insensitive to the exact overnight offset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .actuation import async_push_target
from .const import (
    DOMAIN,
    EVAL_WINDOW_DAYS,
    MAX_TRAINING_ROWS,
    MIN_REFIT_SAMPLES,
    SUBENTRY_TYPE_LOAD,
)
from .models import LoadConfig, load_config_from_data
from .occupancy import async_count_residents_home, async_guest_equivalents
from .persistence import PredictorStore, model_from_dict, model_to_dict
from .predictor import (
    SEED_E_BASE,
    SEED_E_DRAW_PER_PERSON,
    FeatureVector,
    ModelState,
    apply_observation,
    blend_param,
    build_features,
    default_model_state,
    is_valid_delivery,
    kwh_to_minutes,
    predict_kwh,
    predict_minutes,
    refit_occupancy_params,
    rolling_mae,
)
from .runtime import LoadNeedPredictorConfigEntry
from .statistics_source import async_daily_delivered_kwh

_LOGGER = logging.getLogger(__name__)


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
    """Owns the model state per load and runs predict/capture."""

    config_entry: LoadNeedPredictorConfigEntry

    def __init__(self, hass: HomeAssistant, entry: LoadNeedPredictorConfigEntry) -> None:
        # update_interval=None → on-demand only; the jobs trigger refreshes.
        super().__init__(hass, _LOGGER, config_entry=entry, name=DOMAIN, update_interval=None)
        self._store = PredictorStore(hass, entry.entry_id)
        self.models: dict[str, ModelState] = {}
        self.training: dict[str, list[dict]] = {}
        self.eval_errors: dict[str, list[float]] = {}
        self._push_ok: dict[str, bool] = {}

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

    async def async_build_snapshot(self, cfg: LoadConfig) -> dict:
        """Assemble the feature snapshot for the coming cycle.

        Occupancy is duration-based (residents home for most of the trailing day;
        guests weighted by visit length — see ``occupancy``), not a point-in-time
        snapshot. Temps are read for logging only (the model ignores them).
        """
        now = dt_util.now()
        return {
            "people_home": await async_count_residents_home(self.hass, cfg.person_entities),
            "guests": await async_guest_equivalents(self.hass, cfg.guests_calendar_entity),
            "weekend": now.weekday() >= 5,
            "supply_temp": self._state_float(cfg.supply_temp_entity),
            "outdoor_temp": self._state_float(cfg.outdoor_temp_entity),
            "water_total_delta": None,
        }

    # ── daily jobs (driven by jobs.py) ─────────────────────────────────────────

    async def async_predict_and_push(self, only: str | None = None) -> None:
        """Forecast each load, push the target, and record the prediction row.

        ``only`` restricts the work to a single subentry (used by the per-load
        "Predict now" button); the daily job passes nothing to do them all.
        """
        today = dt_util.now().date().isoformat()
        for subentry_id, cfg in self.load_configs().items():
            if only is not None and subentry_id != only:
                continue
            state = self.model_for(subentry_id)
            features = build_features(await self.async_build_snapshot(cfg))
            minutes = predict_minutes(
                state,
                features,
                rated_power_kw=cfg.rated_power_kw,
                min_minutes=cfg.min_minutes,
                max_minutes=cfg.max_minutes,
            )
            self._push_ok[subentry_id] = await async_push_target(
                self.hass, cfg.target_number_entity, minutes
            )
            self._upsert_prediction_row(subentry_id, today, features, state, minutes)
        self.async_persist()
        await self.async_refresh()

    async def async_capture_and_log(self) -> None:
        """Read each load's actual delivery, complete its row, and calibrate."""
        day_start = dt_util.start_of_local_day()
        today = day_start.date().isoformat()
        for subentry_id, cfg in self.load_configs().items():
            if not cfg.delivered_energy_entity:
                continue
            actual_kwh = await async_daily_delivered_kwh(
                self.hass, cfg.delivered_energy_entity, day_start
            )
            self._record_actual(subentry_id, today, cfg, actual_kwh)
        self.async_persist()
        await self.async_refresh()

    # ── training-row management ────────────────────────────────────────────────

    def _upsert_prediction_row(
        self, subentry_id: str, date: str, features: FeatureVector, state: ModelState, minutes: int
    ) -> None:
        """Write/replace today's row with the prediction, keeping any actuals."""
        rows = self.training.setdefault(subentry_id, [])
        row = {
            "date": date,
            "people_home": features.people_home,
            "guests": features.guests,
            "weekend": features.weekend,
            "supply_temp": features.supply_temp,
            "outdoor_temp": features.outdoor_temp,
            "water_total_delta": features.water_total_delta,
            "predicted_kwh": round(predict_kwh(state, features), 3),
            "predicted_minutes": minutes,
            "gain": round(state.gain, 4),
            "model_version": state.version,
            "actual_kwh": None,
            "actual_minutes": None,
            "abs_error_minutes": None,
            "data_quality": None,
        }
        existing = self._row_for_date(rows, date)
        if existing is not None:
            # Preserve actuals if the capture job already ran today.
            for key in ("actual_kwh", "actual_minutes", "abs_error_minutes", "data_quality"):
                row[key] = existing.get(key)
            rows[rows.index(existing)] = row
        else:
            rows.append(row)
        del rows[:-MAX_TRAINING_ROWS]  # cap the history

    def _record_actual(
        self, subentry_id: str, date: str, cfg: LoadConfig, actual_kwh: float | None
    ) -> None:
        """Fill today's row with the actual delivery and calibrate the model."""
        rows = self.training.setdefault(subentry_id, [])
        row = self._row_for_date(rows, date)
        if row is None:  # capture without a prior prediction (e.g. fresh install)
            row = {"date": date, "predicted_kwh": None, "predicted_minutes": None}
            rows.append(row)
            del rows[:-MAX_TRAINING_ROWS]

        valid = is_valid_delivery(actual_kwh)
        row["actual_kwh"] = round(actual_kwh, 3) if actual_kwh is not None else None
        row["data_quality"] = valid
        if actual_kwh is None:
            return

        actual_minutes = round(kwh_to_minutes(actual_kwh, cfg.rated_power_kw))
        row["actual_minutes"] = actual_minutes
        predicted_minutes = row.get("predicted_minutes")
        if predicted_minutes is not None:
            error = abs(predicted_minutes - actual_minutes)
            row["abs_error_minutes"] = error
            if valid:  # only learn from plausible days
                errors = self.eval_errors.setdefault(subentry_id, [])
                errors.append(error)
                del errors[:-MAX_TRAINING_ROWS]

        predicted_kwh = row.get("predicted_kwh")
        if valid and predicted_kwh:
            self.models[subentry_id] = apply_observation(
                self.model_for(subentry_id), predicted_kwh, actual_kwh
            )
            self._maybe_refit(subentry_id)

    def _maybe_refit(self, subentry_id: str) -> None:
        """Blend in an empirical fit of E_base / E_draw once enough data exists.

        Fits ``actual_kwh ~ people_home`` over the clean, occupancy-labelled rows
        and blends the result toward the seeds by sample count (so the structure
        shifts from prior to learned gradually). Needs occupancy variation —
        otherwise ``refit_occupancy_params`` returns None and the seeds stand.
        The online gain keeps handling any residual drift on top.
        """
        rows = [
            r
            for r in self.training.get(subentry_id, [])
            if r.get("data_quality")
            and r.get("actual_kwh") is not None
            and r.get("people_home") is not None
        ]
        if len(rows) < MIN_REFIT_SAMPLES:
            return
        fit = refit_occupancy_params([(r["people_home"], r["actual_kwh"]) for r in rows])
        if fit is None:
            return
        e_base_emp, e_draw_emp = fit
        n = len(rows)
        state = self.model_for(subentry_id)
        self.models[subentry_id] = replace(
            state,
            e_base=blend_param(SEED_E_BASE, e_base_emp, n),
            e_draw_per_person=blend_param(SEED_E_DRAW_PER_PERSON, e_draw_emp, n),
        )

    @staticmethod
    def _row_for_date(rows: list[dict], date: str) -> dict | None:
        for row in rows:
            if row.get("date") == date:
                return row
        return None

    def _last_completed(self, subentry_id: str) -> dict | None:
        """Most recent training row that has an actual delivery."""
        for row in reversed(self.training.get(subentry_id, [])):
            if row.get("actual_kwh") is not None:
                return row
        return None

    # ── prediction / published results ──────────────────────────────────────────

    async def _build_results(self) -> dict[str, LoadResult]:
        """Compute the published forecast + metrics for every load."""
        results: dict[str, LoadResult] = {}
        for subentry_id, cfg in self.load_configs().items():
            state = self.model_for(subentry_id)
            features = build_features(await self.async_build_snapshot(cfg))
            kwh = predict_kwh(state, features)
            minutes = predict_minutes(
                state,
                features,
                rated_power_kw=cfg.rated_power_kw,
                min_minutes=cfg.min_minutes,
                max_minutes=cfg.max_minutes,
            )
            errors = self.eval_errors.get(subentry_id, [])[-EVAL_WINDOW_DAYS:]
            completed = self._last_completed(subentry_id)
            results[subentry_id] = LoadResult(
                predicted_minutes=minutes,
                predicted_kwh=round(kwh, 3),
                last_delivered_kwh=completed.get("actual_kwh") if completed else None,
                prediction_error_minutes=completed.get("abs_error_minutes") if completed else None,
                rolling_mae_minutes=round(rolling_mae(errors), 1) if errors else None,
                sample_count=state.sample_count,
                last_push_ok=self._push_ok.get(subentry_id),
            )
        return results

    async def _async_update_data(self) -> dict[str, LoadResult]:
        return await self._build_results()
