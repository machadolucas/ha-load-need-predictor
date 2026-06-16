"""Coordinator for the beyond-horizon price forecast.

Once a day it (re)fits the price model from long-term statistics, reads the wind
+ temperature forecasts, predicts a daily price for the next few days and expands
that into hourly slots in the Load Scheduler's price-attribute shape. A second
daily pass reconciles each past forecast against the realised price so the
evaluation sensors show how well it's tracking.

This is deliberately separate from the load-need coordinator: the two
capabilities share nothing but the hub's schedule, and keeping them apart keeps
each one's state and tests simple. It uses its own ``.forecast`` Store file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .const import DOMAIN, EVAL_WINDOW_DAYS, MAX_TRAINING_ROWS, SUBENTRY_TYPE_PRICE_FORECAST
from .forecast_source import (
    async_daily_price_mean,
    async_daily_temp_forecast,
    async_fit_rows,
    async_wind_series_gw,
    daily_wind_means_gw,
)
from .models import PriceForecastConfig, price_forecast_config_from_data
from .persistence import PredictorStore
from .price_model import FittedModel, fit, mean_abs_error, predict_price
from .runtime import LoadNeedPredictorConfigEntry

_LOGGER = logging.getLogger(__name__)

_DAY_MS = 86_400_000


@dataclass
class ForecastResult:
    """What the price-forecast sensors publish for one subentry."""

    status: str = "no_data"
    slots: list[dict] = field(default_factory=list)  # the scheduler-shaped data_today
    days: list[dict] = field(default_factory=list)  # per-day summary for the UI
    mean_buy: float | None = None
    model_samples: int | None = None
    fit_mae: float | None = None
    forecast_mae: float | None = None
    forecast_samples: int = 0
    last_predicted: float | None = None
    last_actual: float | None = None
    last_error: float | None = None


class PriceForecastCoordinator(DataUpdateCoordinator[dict[str, ForecastResult]]):
    """Fits the price model, builds the forecast, and tracks its accuracy."""

    config_entry: LoadNeedPredictorConfigEntry

    def __init__(self, hass: HomeAssistant, entry: LoadNeedPredictorConfigEntry) -> None:
        super().__init__(
            hass, _LOGGER, config_entry=entry, name=f"{DOMAIN}_forecast", update_interval=None
        )
        self._store = PredictorStore(hass, entry.entry_id, ".forecast")
        self.models: dict[str, FittedModel | None] = {}
        self.log: dict[str, list[dict]] = {}
        self.eval_errors: dict[str, list[float]] = {}
        self.slots: dict[str, list[dict]] = {}
        self.days: dict[str, list[dict]] = {}
        self.fit_mae: dict[str, float | None] = {}

    # ── lifecycle ──────────────────────────────────────────────────────────────

    async def async_load_runtime(self) -> None:
        data = await self._store.async_load()
        for subentry_id, payload in data.items():
            self.models[subentry_id] = FittedModel.from_dict(payload.get("model"))
            self.log[subentry_id] = list(payload.get("log", []))
            self.eval_errors[subentry_id] = list(payload.get("eval", []))

    def _runtime_snapshot(self) -> dict:
        return {
            subentry_id: {
                "model": (m.to_dict() if (m := self.models.get(subentry_id)) else None),
                "log": self.log.get(subentry_id, []),
                "eval": self.eval_errors.get(subentry_id, []),
            }
            for subentry_id in self.forecast_configs()
        }

    def async_persist(self) -> None:
        self._store.async_schedule_save(self._runtime_snapshot)

    def forecast_configs(self) -> dict[str, PriceForecastConfig]:
        out: dict[str, PriceForecastConfig] = {}
        for subentry_id, subentry in self.config_entry.subentries.items():
            if subentry.subentry_type != SUBENTRY_TYPE_PRICE_FORECAST:
                continue
            out[subentry_id] = price_forecast_config_from_data(subentry.data)
        return out

    @property
    def has_loads(self) -> bool:
        """True if any price-forecast subentry is configured."""
        return bool(self.forecast_configs())

    # ── daily jobs ─────────────────────────────────────────────────────────────

    async def async_build_forecast(self) -> None:
        """Refit from LTS, then publish the beyond-horizon slots for each load."""
        today_start = dt_util.start_of_local_day()
        for subentry_id, cfg in self.forecast_configs().items():
            await self._build_one(subentry_id, cfg, today_start)
        self.async_persist()
        await self.async_refresh()

    async def _build_one(
        self, subentry_id: str, cfg: PriceForecastConfig, today_start: datetime
    ) -> None:
        # 1) Refit from history. A successful fit replaces the stored model; an
        #    insufficient one leaves the previous model (or the seed) in place.
        rows: list[tuple[float, float, float]] = []
        if cfg.price_entity and cfg.temp_history_entity and cfg.wind_entity:
            rows = await async_fit_rows(
                self.hass, cfg.price_entity, cfg.temp_history_entity, cfg.wind_entity, cfg.fit_days
            )
        fitted = fit(rows)
        if fitted is not None:
            self.models[subentry_id] = fitted
        model = self.models.get(subentry_id)
        self.fit_mae[subentry_id] = mean_abs_error(model, rows) if rows else None

        # 2) Future inputs.
        wind_daily = (
            daily_wind_means_gw(await async_wind_series_gw(self.hass, cfg.wind_entity))
            if cfg.wind_entity
            else {}
        )
        temp_daily = (
            await async_daily_temp_forecast(self.hass, cfg.weather_entity)
            if cfg.weather_entity
            else {}
        )

        # 3) Predict each future day → flat hourly slots (the margin absorbs the
        #    intraday shape, per the design).
        slots: list[dict] = []
        days: list[dict] = []
        for offset in range(1, cfg.forecast_days + 1):
            day_start = today_start + timedelta(days=offset)
            day = day_start.date()
            temp = temp_daily.get(day)
            wind = wind_daily.get(day)
            if temp is None or wind is None:
                continue
            price = round(predict_price(model, temp, wind), 5)
            days.append(
                {
                    "date": day.isoformat(),
                    "temp": round(temp, 1),
                    "wind_gw": round(wind, 2),
                    "buy": price,
                }
            )
            for hour in range(24):
                start = day_start + timedelta(hours=hour)
                slots.append(
                    {
                        "start": start.isoformat(),
                        "end": (start + timedelta(hours=1)).isoformat(),
                        "buy": price,
                    }
                )
            self._upsert_log(subentry_id, day.isoformat(), int(day_start.timestamp() * 1000), price)
        self.slots[subentry_id] = slots
        self.days[subentry_id] = days

    async def async_evaluate(self) -> None:
        """Reconcile each past forecast against the realised daily price."""
        now_ms = dt_util.utcnow().timestamp() * 1000
        for subentry_id, cfg in self.forecast_configs().items():
            if not cfg.price_entity:
                continue
            for entry in self.log.get(subentry_id, []):
                if entry.get("actual") is not None:
                    continue
                bucket_ms = entry.get("bucket_ms")
                if bucket_ms is None or bucket_ms + _DAY_MS > now_ms:
                    continue  # day not fully realised yet
                day_start = datetime.fromtimestamp(bucket_ms / 1000, tz=UTC)
                actual = await async_daily_price_mean(self.hass, cfg.price_entity, day_start)
                if actual is None:
                    continue
                entry["actual"] = round(actual, 5)
                error = abs(entry["predicted"] - actual)
                entry["abs_error"] = round(error, 5)
                errors = self.eval_errors.setdefault(subentry_id, [])
                errors.append(error)
                del errors[:-MAX_TRAINING_ROWS]
        self.async_persist()
        await self.async_refresh()

    # ── log helpers ────────────────────────────────────────────────────────────

    def _upsert_log(
        self, subentry_id: str, date_iso: str, bucket_ms: int, predicted: float
    ) -> None:
        log = self.log.setdefault(subentry_id, [])
        for entry in log:
            if entry.get("date") == date_iso:
                entry["predicted"] = predicted
                entry["bucket_ms"] = bucket_ms
                return
        log.append(
            {
                "date": date_iso,
                "bucket_ms": bucket_ms,
                "predicted": predicted,
                "actual": None,
                "abs_error": None,
            }
        )
        del log[:-MAX_TRAINING_ROWS]

    def _last_completed(self, subentry_id: str) -> dict | None:
        for entry in reversed(self.log.get(subentry_id, [])):
            if entry.get("actual") is not None:
                return entry
        return None

    # ── published results ────────────────────────────────────────────────────

    def _build_results(self) -> dict[str, ForecastResult]:
        results: dict[str, ForecastResult] = {}
        for subentry_id in self.forecast_configs():
            slots = self.slots.get(subentry_id, [])
            errors = self.eval_errors.get(subentry_id, [])
            recent = errors[-EVAL_WINDOW_DAYS:]
            model = self.models.get(subentry_id)
            completed = self._last_completed(subentry_id)
            buys = [s["buy"] for s in slots]
            fit_mae = self.fit_mae.get(subentry_id)
            results[subentry_id] = ForecastResult(
                status="ok" if slots else "no_data",
                slots=slots,
                days=self.days.get(subentry_id, []),
                mean_buy=round(sum(buys) / len(buys), 5) if buys else None,
                model_samples=model.n if model else None,
                fit_mae=round(fit_mae, 5) if fit_mae is not None else None,
                forecast_mae=round(sum(abs(e) for e in recent) / len(recent), 5)
                if recent
                else None,
                forecast_samples=len(errors),
                last_predicted=completed.get("predicted") if completed else None,
                last_actual=completed.get("actual") if completed else None,
                last_error=completed.get("abs_error") if completed else None,
            )
        return results

    async def _async_update_data(self) -> dict[str, ForecastResult]:
        return self._build_results()
