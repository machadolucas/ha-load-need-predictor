"""Coordinator for the beyond-horizon price forecast.

Two sources, one published series:

- **Wattcast** (primary): a free hosted spot-price forecast (≈ 7 days, 15-min,
  p10/p90 band). Fetched at most hourly on a cheap 5-minute due-check tick,
  **cached** (persisted, so it survives restarts) and kept until a newer fetch
  succeeds; failures back off (5 → 15 → 30 → 60 min, honouring ``Retry-After``)
  and, after a few hours, raise a repair issue. Its spot €/MWh is converted to
  the user's all-in €/kWh by a mapping learned from (settled spot, real buy)
  pairs.
- **Local** (fallback): the daily ridge on wind + temperature (``price_model``)
  shaped by a learned intraday profile — fills slots beyond Wattcast's coverage
  and everything before a first successful fetch.

Slots start at the first slot without a real price, so the Load Scheduler (which
ignores overlap with its real prices anyway) and the dashboard see only genuine
forecast. Every unsettled day's per-hour forecast is snapshotted per source on
each rebuild — the stored value is therefore the last forecast made *before*
the real prices existed — and the capture job scores each source against the
realised hourly prices. The primary source is whichever scores best over the
recent window (default Wattcast).

Deliberately separate from the load-need coordinator: the two capabilities share
nothing but the hub's schedule. It uses its own ``.forecast`` Store file.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    EVAL_WINDOW_DAYS,
    FORECAST_TICK_MINUTES,
    HOURLY_LOG_ROWS,
    MAX_TRAINING_ROWS,
    RETAIL_PAIRS_DAYS,
    SHAPE_FIT_DAYS,
    SUBENTRY_TYPE_PRICE_FORECAST,
    WATTCAST_BACKOFF_MIN,
    WATTCAST_FETCH_MINUTE,
    WATTCAST_ISSUE_AFTER_H,
    WATTCAST_STALE_AFTER_H,
)
from .forecast_source import (
    WattcastFetch,
    async_daily_price_mean,
    async_daily_temp_forecast,
    async_fetch_wattcast,
    async_fit_rows,
    async_hourly_price_rows,
    async_wind_series_gw,
    current_pair,
    daily_wind_means_gw,
    real_price_slots,
    retail_pairs,
)
from .models import PriceForecastConfig, price_forecast_config_from_data
from .persistence import PredictorStore
from .price_model import FittedModel, describe, fit, mean_abs_error, predict_price
from .runtime import LoadNeedPredictorConfigEntry
from .spot_forecast import (
    RetailMapping,
    WattcastSeries,
    build_slots,
    daily_mean,
    fit_intraday_shape,
    fit_retail_mapping,
    hourly_mae,
    hourly_vectors,
    seed_mapping,
    select_primary,
)

_LOGGER = logging.getLogger(__name__)

_DAY_MS = 86_400_000
# A day's per-hour snapshot is only (re)written while at least this many of its
# hours are still forecast — so a day that just became partially real (the CET
# delivery day leaks 00:00–01:00 into the previous list) keeps its full
# pre-publication snapshot instead of being overwritten by a sliver.
_MIN_SNAPSHOT_HOURS = 20
_BUTTON_MIN_REFETCH = timedelta(minutes=15)


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
    hourly_mae: float | None = None
    forecast_samples: int = 0
    last_predicted: float | None = None
    last_actual: float | None = None
    last_error: float | None = None
    fitted: bool = False  # True once a real fit replaced the seed formula
    coefficients: dict | None = None  # regression terms, for the card's rationale
    source: str | None = None  # primary source in force
    known_until: str | None = None  # end of real prices (forecast starts here)
    stale: bool = False
    wattcast_made_at: str | None = None
    cache_age_h: float | None = None
    fetch_error: str | None = None
    retail_mapping: dict | None = None
    mae_by_source: dict = field(default_factory=dict)


@dataclass
class _FetchState:
    """Per-subentry polling bookkeeping (``failing_since``/``last_error`` persist)."""

    next_fetch: datetime | None = None
    failures: int = 0
    failing_since: datetime | None = None
    last_error: str | None = None


def _floor_slot(when: datetime) -> datetime:
    return when.replace(minute=when.minute - when.minute % 15, second=0, microsecond=0)


def _next_issue_fetch(after: datetime) -> datetime:
    """The first HH:32 strictly after ``after`` (just past Wattcast's :25 re-issue)."""
    candidate = after.replace(minute=WATTCAST_FETCH_MINUTE, second=0, microsecond=0)
    return candidate if candidate > after else candidate + timedelta(hours=1)


def _iso(when: datetime | None) -> str | None:
    return when.isoformat() if when is not None else None


def _parse_iso(value) -> datetime | None:
    if not value:
        return None
    parsed = dt_util.parse_datetime(str(value))
    return dt_util.as_utc(parsed) if parsed is not None else None


class PriceForecastCoordinator(DataUpdateCoordinator[dict[str, ForecastResult]]):
    """Fetches/caches Wattcast, fits the fallback, builds + scores the forecast."""

    config_entry: LoadNeedPredictorConfigEntry

    def __init__(self, hass: HomeAssistant, entry: LoadNeedPredictorConfigEntry) -> None:
        super().__init__(
            hass, _LOGGER, config_entry=entry, name=f"{DOMAIN}_forecast", update_interval=None
        )
        self._store = PredictorStore(hass, entry.entry_id, ".forecast")
        self._lock = asyncio.Lock()
        self._unsub_tick = None
        # Local fallback model.
        self.models: dict[str, FittedModel | None] = {}
        self.fit_mae: dict[str, float | None] = {}
        self.shape: dict[str, dict[str, float]] = {}
        self.local_daily: dict[str, dict[date, float]] = {}
        self.local_meta: dict[str, dict[date, dict]] = {}
        self.wind_clim: dict[str, float | None] = {}
        # Wattcast cache + mapping.
        self.wattcast: dict[str, WattcastSeries | None] = {}
        self.fetched_at: dict[str, datetime | None] = {}
        self.fetch: dict[str, _FetchState] = {}
        self.pairs: dict[str, list[list]] = {}
        self.mapping: dict[str, RetailMapping] = {}
        # Published series + evaluation.
        self.slots: dict[str, list[dict]] = {}
        self.days: dict[str, list[dict]] = {}
        self.primary: dict[str, str] = {}
        self.known_until: dict[str, datetime | None] = {}
        self.log: dict[str, list[dict]] = {}
        self.eval_errors: dict[str, list[float]] = {}

    # ── lifecycle ──────────────────────────────────────────────────────────────

    async def async_load_runtime(self) -> None:
        data = await self._store.async_load()
        for subentry_id, payload in data.items():
            self.models[subentry_id] = FittedModel.from_dict(payload.get("model"))
            self.log[subentry_id] = list(payload.get("log", []))
            self.eval_errors[subentry_id] = list(payload.get("eval", []))
            self.shape[subentry_id] = dict(payload.get("shape") or {})
            self.wind_clim[subentry_id] = payload.get("wind_clim")
            self.pairs[subentry_id] = list(payload.get("pairs", []))
            mapping = RetailMapping.from_dict(payload.get("mapping"))
            if mapping is not None:
                self.mapping[subentry_id] = mapping
            cache = payload.get("wattcast") or {}
            self.wattcast[subentry_id] = WattcastSeries.from_dict(cache.get("series"))
            self.fetched_at[subentry_id] = _parse_iso(cache.get("fetched_at"))
            fetch = payload.get("fetch") or {}
            self.fetch[subentry_id] = _FetchState(
                failing_since=_parse_iso(fetch.get("failing_since")),
                last_error=fetch.get("last_error"),
            )

    def _runtime_snapshot(self) -> dict:
        out: dict = {}
        for subentry_id in self.forecast_configs():
            model = self.models.get(subentry_id)
            series = self.wattcast.get(subentry_id)
            mapping = self.mapping.get(subentry_id)
            fetch = self.fetch.get(subentry_id) or _FetchState()
            out[subentry_id] = {
                "model": model.to_dict() if model else None,
                "log": self.log.get(subentry_id, []),
                "eval": self.eval_errors.get(subentry_id, []),
                "shape": self.shape.get(subentry_id, {}),
                "wind_clim": self.wind_clim.get(subentry_id),
                "pairs": self.pairs.get(subentry_id, []),
                "mapping": mapping.to_dict() if mapping else None,
                # The cache is only ever replaced by a newer successful fetch.
                "wattcast": {
                    "series": series.to_dict() if series else None,
                    "fetched_at": _iso(self.fetched_at.get(subentry_id)),
                },
                "fetch": {
                    "failing_since": _iso(fetch.failing_since),
                    "last_error": fetch.last_error,
                },
            }
        return out

    def async_persist(self) -> None:
        self._store.async_schedule_save(self._runtime_snapshot)

    async def async_flush(self) -> None:
        """Write now (on unload) so a reload can't read a stale cache."""
        if self.has_loads:
            await self._store.async_save_now(self._runtime_snapshot())

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

    @callback
    def async_start(self) -> None:
        """Start the due-check tick (its own interval: see the tank tracker's why)."""
        if self._unsub_tick is not None or not self.has_loads:
            return
        self._unsub_tick = async_track_time_interval(
            self.hass, self._handle_tick, timedelta(minutes=FORECAST_TICK_MINUTES)
        )

    @callback
    def async_shutdown_ticker(self) -> None:
        if self._unsub_tick is not None:
            self._unsub_tick()
            self._unsub_tick = None

    async def _handle_tick(self, _now) -> None:
        try:
            await self.async_tick()
        except Exception:  # never let one bad tick kill the interval
            _LOGGER.exception("Price-forecast tick failed")

    # ── the tick: fetch if due, rebuild if anything moved ─────────────────────

    async def async_tick(self, *, force_rebuild: bool = False) -> None:
        async with self._lock:
            now = dt_util.utcnow()
            memo: dict[str, WattcastFetch] = {}
            changed = False
            for subentry_id, cfg in self.forecast_configs().items():
                fetched = False
                if not cfg.use_wattcast:
                    ir.async_delete_issue(self.hass, DOMAIN, self._issue_id(subentry_id))
                elif self._fetch_due(subentry_id, now):
                    outcome = memo.get(cfg.wattcast_zone)
                    if outcome is None:
                        outcome = await async_fetch_wattcast(self.hass, cfg.wattcast_zone)
                        memo[cfg.wattcast_zone] = outcome
                    if outcome.series is not None:
                        self._on_fetch_success(subentry_id, cfg, outcome.series, now)
                        fetched = True
                    else:
                        self._on_fetch_failure(subentry_id, cfg, outcome, now)
                    changed = True
                real_end = self._real_end(subentry_id, cfg)
                if fetched:
                    # Hourly is plenty for the weather/wind inputs too.
                    await self._refresh_local(subentry_id, cfg)
                if (
                    fetched
                    or force_rebuild
                    or real_end != self.known_until.get(subentry_id)
                    or subentry_id not in self.slots
                ):
                    self._rebuild(subentry_id, cfg, now)
                    changed = True
            if changed:
                self.async_persist()
                await self.async_refresh()

    def _fetch_due(self, subentry_id: str, now: datetime) -> bool:
        state = self.fetch.setdefault(subentry_id, _FetchState())
        if state.next_fetch is None:
            # First tick after (re)start: a cache from the current issue window
            # means no request until the next issue.
            fetched_at = self.fetched_at.get(subentry_id)
            state.next_fetch = _next_issue_fetch(fetched_at) if fetched_at else now
        return now >= state.next_fetch

    def _on_fetch_success(
        self, subentry_id: str, cfg: PriceForecastConfig, series: WattcastSeries, now: datetime
    ) -> None:
        state = self.fetch.setdefault(subentry_id, _FetchState())
        if state.failures:
            _LOGGER.info("Wattcast forecast reachable again after %d failures", state.failures)
        self.wattcast[subentry_id] = series
        self.fetched_at[subentry_id] = now
        state.failures = 0
        state.failing_since = None
        state.last_error = None
        state.next_fetch = _next_issue_fetch(now)
        ir.async_delete_issue(self.hass, DOMAIN, self._issue_id(subentry_id))
        self._update_mapping(subentry_id, cfg, series, now)

    def _on_fetch_failure(
        self, subentry_id: str, cfg: PriceForecastConfig, outcome: WattcastFetch, now: datetime
    ) -> None:
        state = self.fetch.setdefault(subentry_id, _FetchState())
        state.failures += 1
        state.failing_since = state.failing_since or now
        state.last_error = outcome.error
        step = WATTCAST_BACKOFF_MIN[min(state.failures - 1, len(WATTCAST_BACKOFF_MIN) - 1)]
        delay = timedelta(minutes=step)
        if outcome.retry_after_s:
            delay = max(delay, timedelta(seconds=outcome.retry_after_s))
        state.next_fetch = now + delay
        log = _LOGGER.warning if state.failures == 1 else _LOGGER.debug
        log(
            "Wattcast forecast fetch failed (%s); keeping the cached forecast, retry in %s",
            outcome.error,
            delay,
        )
        if now - state.failing_since >= timedelta(hours=WATTCAST_ISSUE_AFTER_H):
            last = self.fetched_at.get(subentry_id)
            ir.async_create_issue(
                self.hass,
                DOMAIN,
                self._issue_id(subentry_id),
                is_fixable=False,
                severity=ir.IssueSeverity.WARNING,
                translation_key="wattcast_unreachable",
                translation_placeholders={
                    "zone": cfg.wattcast_zone,
                    "hours": f"{WATTCAST_ISSUE_AFTER_H:g}",
                    "last_success": dt_util.as_local(last).strftime("%Y-%m-%d %H:%M")
                    if last
                    else "never",
                    "error": outcome.error or "unknown",
                },
            )

    def _issue_id(self, subentry_id: str) -> str:
        return f"wattcast_unreachable_{subentry_id}"

    def _update_mapping(
        self, subentry_id: str, cfg: PriceForecastConfig, series: WattcastSeries, now: datetime
    ) -> None:
        """Fold new (spot, buy) pairs into the rolling buffer and refit."""
        new: list[tuple[int, float, float]] = []
        if cfg.price_series_entity:
            new = retail_pairs(
                real_price_slots(self.hass.states.get(cfg.price_series_entity)), series
            )
        if not new and cfg.price_entity:
            pair = current_pair(self.hass.states.get(cfg.price_entity), series, now)
            if pair is not None:
                new = [pair]
        buffer = {int(row[0]): row for row in self.pairs.get(subentry_id, [])}
        for ms, spot, buy in new:
            buffer[ms] = [ms, spot, buy]
        cutoff = (now - timedelta(days=RETAIL_PAIRS_DAYS)).timestamp() * 1000
        rows = sorted((row for row in buffer.values() if row[0] >= cutoff), key=lambda r: r[0])
        self.pairs[subentry_id] = rows
        self.mapping[subentry_id] = fit_retail_mapping(
            [
                (dt_util.as_local(datetime.fromtimestamp(ms / 1000, tz=UTC)), spot, buy)
                for ms, spot, buy in rows
            ],
            cfg.vat,
        )

    def _real_end(self, subentry_id: str, cfg: PriceForecastConfig) -> datetime | None:
        """Where real prices end: the series entity if set, else Wattcast's known."""
        if cfg.price_series_entity:
            real = real_price_slots(self.hass.states.get(cfg.price_series_entity))
            if real:
                return max(end for _start, end, _buy in real)
        series = self.wattcast.get(subentry_id) if cfg.use_wattcast else None
        return series.known_end if series else None

    # ── daily jobs ─────────────────────────────────────────────────────────────

    async def async_build_forecast(self, only: str | None = None, *, fetch: bool = False) -> None:
        """Refit the local fallback (ridge + intraday shape), then rebuild.

        ``only`` restricts the work to one subentry (the "Update forecast now"
        button, which also passes ``fetch=True`` to pull a fresh Wattcast
        forecast — unless the cache is under 15 minutes old).
        """
        for subentry_id, cfg in self.forecast_configs().items():
            if only is not None and subentry_id != only:
                continue
            await self._refit_local(subentry_id, cfg)
            await self._refresh_local(subentry_id, cfg)
            series = self.wattcast.get(subentry_id) if cfg.use_wattcast else None
            if series is not None:
                # Refit the retail mapping from the cached settled spot too, so a
                # reconfigure/restart (e.g. a newly set price series entity)
                # takes effect now rather than at the next hourly fetch.
                self._update_mapping(subentry_id, cfg, series, dt_util.utcnow())
            if fetch and cfg.use_wattcast:
                # A manual refresh may pull early, but never through a failure
                # backoff / Retry-After, and not more than every 15 minutes.
                state = self.fetch.setdefault(subentry_id, _FetchState())
                fetched_at = self.fetched_at.get(subentry_id)
                now = dt_util.utcnow()
                fresh = fetched_at is not None and now - fetched_at < _BUTTON_MIN_REFETCH
                if not state.failures and not fresh:
                    state.next_fetch = now
        await self.async_tick(force_rebuild=True)

    async def _refit_local(self, subentry_id: str, cfg: PriceForecastConfig) -> None:
        # 1) Daily ridge. A successful fit replaces the stored model; an
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
        if rows:
            # Climatological wind for days past the wind feed's ~3.5-day reach.
            self.wind_clim[subentry_id] = sum(r[1] for r in rows) / len(rows)
        # 2) Intraday shape from the realised hourly prices.
        if cfg.price_entity:
            start = dt_util.start_of_local_day() - timedelta(days=SHAPE_FIT_DAYS)
            hourly = await async_hourly_price_rows(self.hass, cfg.price_entity, start)
            shape = fit_intraday_shape(hourly)
            if shape:
                self.shape[subentry_id] = shape

    async def _refresh_local(self, subentry_id: str, cfg: PriceForecastConfig) -> None:
        """Daily local-fallback prices for every day of the horizon."""
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
        model = self.models.get(subentry_id)
        clim = self.wind_clim.get(subentry_id)
        if clim is None and model is not None:
            clim = model.means[1]
        today = dt_util.start_of_local_day().date()
        daily: dict[date, float] = {}
        meta: dict[date, dict] = {}
        for offset in range(cfg.forecast_days + 1):
            day = today + timedelta(days=offset)
            temp = temp_daily.get(day)
            wind = wind_daily.get(day)
            wind_src = "forecast"
            if wind is None and clim is not None:
                wind, wind_src = clim, "climatology"
            if temp is None or wind is None:
                continue
            daily[day] = round(predict_price(model, temp, wind), 5)
            meta[day] = {"temp": round(temp, 1), "wind_gw": round(wind, 2), "wind_src": wind_src}
        self.local_daily[subentry_id] = daily
        self.local_meta[subentry_id] = meta

    # ── build ──────────────────────────────────────────────────────────────────

    def _rebuild(self, subentry_id: str, cfg: PriceForecastConfig, now: datetime) -> None:
        tz = dt_util.get_default_time_zone()
        real_end = self._real_end(subentry_id, cfg)
        self.known_until[subentry_id] = real_end
        start = _floor_slot(now)
        if real_end is not None and real_end > start:
            start = real_end
        end = dt_util.start_of_local_day() + timedelta(days=cfg.forecast_days + 1)
        series = self.wattcast.get(subentry_id) if cfg.use_wattcast else None
        if series is not None and (series.coverage_end is None or series.coverage_end <= start):
            series = None  # nothing left in the cache that's still ahead of us
        mapping = self.mapping.get(subentry_id) or seed_mapping(cfg.vat)
        local_daily = self.local_daily.get(subentry_id, {})
        shape = self.shape.get(subentry_id) or None

        def _build(source: str) -> list[dict]:
            if source == "local":
                wc, variant = None, "wattcast"
            else:
                wc, variant = series, source
            return build_slots(
                start=start,
                end=end,
                tz=tz,
                wattcast=wc,
                variant=variant,
                mapping=mapping,
                local_daily=local_daily if source == "local" else {},
                shape=shape,
            )

        candidates: dict[str, list[dict]] = {}
        if series is not None:
            candidates["wattcast"] = _build("wattcast")
            candidates["wattcast_raw"] = _build("wattcast_raw")
        if local_daily:
            candidates["local"] = _build("local")
        available = [src for src, slots in candidates.items() if slots]
        primary = select_primary(self._scores(subentry_id), available=available)
        self.primary[subentry_id] = primary

        # Published: the primary everywhere it reaches, the local fallback beyond.
        slots = build_slots(
            start=start,
            end=end,
            tz=tz,
            wattcast=series if primary != "local" else None,
            variant=primary if primary != "local" else "wattcast",
            mapping=mapping,
            local_daily=local_daily,
            shape=shape,
        )
        self.slots[subentry_id] = slots
        self.days[subentry_id] = self._day_summaries(subentry_id, slots, tz)
        self._snapshot(subentry_id, candidates, primary, tz)

    def _day_summaries(self, subentry_id: str, slots: list[dict], tz) -> list[dict]:
        per_day: dict[str, list[dict]] = {}
        for slot in slots:
            day = dt_util.parse_datetime(slot["start"]).astimezone(tz).date().isoformat()
            per_day.setdefault(day, []).append(slot)
        meta = self.local_meta.get(subentry_id, {})
        out: list[dict] = []
        for day, items in per_day.items():
            buys = [s["buy"] for s in items]
            srcs = [s["src"] for s in items]
            summary = {
                "date": day,
                "mean": round(sum(buys) / len(buys), 5),
                "min": min(buys),
                "max": max(buys),
                "src": max(set(srcs), key=srcs.count),
                "slots": len(items),
            }
            summary.update(meta.get(date.fromisoformat(day), {}))
            out.append(summary)
        return out

    def _snapshot(
        self, subentry_id: str, candidates: dict[str, list[dict]], primary: str, tz
    ) -> None:
        """Upsert each still-forecast day's per-hour prediction, per source."""
        vectors = {src: hourly_vectors(slots, tz) for src, slots in candidates.items()}
        days = sorted({d for vecs in vectors.values() for d in vecs})
        for day in days:
            hourly: dict[str, list] = {}
            for src, vecs in vectors.items():
                vec = vecs.get(day)
                if vec and sum(v is not None for v in vec) >= _MIN_SNAPSHOT_HOURS:
                    hourly[src] = vec
            if not hourly:
                continue
            entry = self._log_entry(subentry_id, day, tz)
            entry.setdefault("hourly", {}).update(hourly)
            entry["daily"] = {
                src: round(m, 5)
                for src, vec in entry["hourly"].items()
                if (m := daily_mean(vec)) is not None
            }
            main = primary if primary in entry["daily"] else next(iter(entry["daily"]))
            entry["src"] = main
            entry["predicted"] = entry["daily"][main]
        self._trim_log(subentry_id)

    def _log_entry(self, subentry_id: str, day_iso: str, tz) -> dict:
        log = self.log.setdefault(subentry_id, [])
        for entry in log:
            if entry.get("date") == day_iso:
                return entry
        start = datetime.combine(date.fromisoformat(day_iso), datetime.min.time(), tzinfo=tz)
        entry = {
            "date": day_iso,
            "bucket_ms": int(start.timestamp() * 1000),
            "predicted": None,
            "actual": None,
            "abs_error": None,
        }
        log.append(entry)
        log.sort(key=lambda e: e.get("date", ""))
        return entry

    def _trim_log(self, subentry_id: str) -> None:
        log = self.log.get(subentry_id, [])
        del log[:-MAX_TRAINING_ROWS]
        # Per-hour vectors are only needed until scored + for a recent window.
        for entry in log[:-HOURLY_LOG_ROWS]:
            entry.pop("hourly", None)

    # ── evaluation ─────────────────────────────────────────────────────────────

    async def async_evaluate(self) -> None:
        """Score each past forecast (per source) against the realised prices."""
        now_ms = dt_util.utcnow().timestamp() * 1000
        tz = dt_util.get_default_time_zone()
        async with self._lock:
            for subentry_id, cfg in self.forecast_configs().items():
                if not cfg.price_entity:
                    continue
                for entry in self.log.get(subentry_id, []):
                    if entry.get("actual") is not None or entry.get("predicted") is None:
                        continue
                    bucket_ms = entry.get("bucket_ms")
                    if bucket_ms is None or bucket_ms + _DAY_MS > now_ms:
                        continue  # day not fully realised yet
                    day_start = datetime.fromtimestamp(bucket_ms / 1000, tz=UTC)
                    # Local midnight to local midnight (wall-clock +1 day), so a
                    # 25-hour DST day isn't cut short by an hour.
                    day_end = dt_util.as_local(day_start) + timedelta(days=1)
                    rows = await async_hourly_price_rows(
                        self.hass, cfg.price_entity, day_start, day_end
                    )
                    actual_vec = hourly_vectors(
                        [{"start": when.isoformat(), "buy": value} for when, value in rows], tz
                    ).get(entry["date"])
                    actual = daily_mean(actual_vec) if actual_vec else None
                    if actual is None:
                        actual = await async_daily_price_mean(
                            self.hass, cfg.price_entity, day_start
                        )
                    if actual is None:
                        continue
                    entry["actual"] = round(actual, 5)
                    error = abs(entry["predicted"] - actual)
                    entry["abs_error"] = round(error, 5)
                    if actual_vec:
                        entry["scores"] = {
                            src: round(mae, 5)
                            for src, vec in (entry.get("hourly") or {}).items()
                            if (mae := hourly_mae(vec, actual_vec)) is not None
                        }
                    entry["daily_err"] = {
                        src: round(abs(pred - actual), 5)
                        for src, pred in (entry.get("daily") or {}).items()
                    }
                    errors = self.eval_errors.setdefault(subentry_id, [])
                    errors.append(error)
                    del errors[:-MAX_TRAINING_ROWS]
            self.async_persist()
        await self.async_refresh()

    def _scores(self, subentry_id: str) -> dict[str, list[float]]:
        """Chronological per-day hourly MAEs per source (for primary selection)."""
        out: dict[str, list[float]] = {}
        for entry in self.log.get(subentry_id, []):
            for src, mae in (entry.get("scores") or {}).items():
                out.setdefault(src, []).append(mae)
        return out

    def _mae_by_source(self, subentry_id: str) -> dict:
        scored = [e for e in self.log.get(subentry_id, []) if e.get("actual") is not None]
        recent = scored[-EVAL_WINDOW_DAYS:]
        out: dict[str, dict] = {}
        sources = {src for e in recent for src in (e.get("daily_err") or {})}
        for src in sorted(sources):
            hourly = [e["scores"][src] for e in recent if src in (e.get("scores") or {})]
            daily = [e["daily_err"][src] for e in recent if src in (e.get("daily_err") or {})]
            out[src] = {
                "hourly_mae": round(sum(hourly) / len(hourly), 5) if hourly else None,
                "daily_mae": round(sum(daily) / len(daily), 5) if daily else None,
                "n": len(daily),
            }
        return out

    def _last_completed(self, subentry_id: str) -> dict | None:
        for entry in reversed(self.log.get(subentry_id, [])):
            if entry.get("actual") is not None:
                return entry
        return None

    # ── published results ────────────────────────────────────────────────────

    def _build_results(self) -> dict[str, ForecastResult]:
        results: dict[str, ForecastResult] = {}
        now = dt_util.utcnow()
        configs = self.forecast_configs()
        for subentry_id in configs:
            slots = self.slots.get(subentry_id, [])
            errors = self.eval_errors.get(subentry_id, [])
            recent = errors[-EVAL_WINDOW_DAYS:]
            model = self.models.get(subentry_id)
            completed = self._last_completed(subentry_id)
            buys = [s["buy"] for s in slots]
            fit_mae = self.fit_mae.get(subentry_id)
            cfg = configs[subentry_id]
            series = self.wattcast.get(subentry_id) if cfg.use_wattcast else None
            fetched_at = self.fetched_at.get(subentry_id) if cfg.use_wattcast else None
            age_h = (now - fetched_at).total_seconds() / 3600 if fetched_at else None
            # Stale if we haven't fetched for a while, or the server keeps
            # serving an old issue (an hourly product > 3 h old is suspect).
            issue_age_h = (
                (now - series.made_at).total_seconds() / 3600 if series and series.made_at else None
            )
            stale = (age_h is not None and age_h > WATTCAST_STALE_AFTER_H) or (
                issue_age_h is not None and issue_age_h > WATTCAST_STALE_AFTER_H + 1
            )
            mapping = self.mapping.get(subentry_id)
            scored = [
                e["scores"][e["src"]]
                for e in self.log.get(subentry_id, [])
                if e.get("src") in (e.get("scores") or {})
            ][-EVAL_WINDOW_DAYS:]
            fetch = (self.fetch.get(subentry_id) if cfg.use_wattcast else None) or _FetchState()
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
                hourly_mae=round(sum(scored) / len(scored), 5) if scored else None,
                forecast_samples=len(errors),
                last_predicted=completed.get("predicted") if completed else None,
                last_actual=completed.get("actual") if completed else None,
                last_error=completed.get("abs_error") if completed else None,
                fitted=model is not None,
                coefficients=describe(model),
                source=self.primary.get(subentry_id),
                known_until=_iso(self.known_until.get(subentry_id)),
                stale=stale,
                wattcast_made_at=_iso(series.made_at) if series else None,
                cache_age_h=round(age_h, 2) if age_h is not None else None,
                fetch_error=fetch.last_error,
                retail_mapping=(
                    {
                        "slope_pos": round(mapping.slope_pos, 4),
                        "slope_neg": round(mapping.slope_neg, 4),
                        "base": round(mapping.base, 5),
                        "n": mapping.n,
                        "mae": round(mapping.mae, 5) if mapping.mae is not None else None,
                    }
                    if mapping
                    else None
                ),
                mae_by_source=self._mae_by_source(subentry_id),
            )
        return results

    async def _async_update_data(self) -> dict[str, ForecastResult]:
        return self._build_results()
