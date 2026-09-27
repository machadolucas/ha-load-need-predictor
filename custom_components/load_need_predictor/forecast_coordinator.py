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
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    EVAL_GIVE_UP_DAYS,
    EVAL_MIN_ACTUAL_HOURS,
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

# A day's per-hour snapshot is only (re)written while at least this many of its
# hours are still forecast — so a day that just became partially real (the CET
# delivery day leaks 00:00–01:00 into the previous list) keeps its full
# pre-publication snapshot instead of being overwritten by a sliver.
_MIN_SNAPSHOT_HOURS = 20
_BUTTON_MIN_REFETCH = timedelta(minutes=15)
# A persisted ``next_fetch`` further out than this is not a real backoff (clock
# jump, corrupt file) — ignore it rather than go silent. Generous on purpose: a
# server-directed ``Retry-After`` of days must survive a reload.
_MAX_RESTORED_BACKOFF = timedelta(days=7)
_ISSUE_PREFIX = "wattcast_unreachable_"
# Source tag of slots priced from Wattcast's *settled* spot (not a forecast):
# never snapshotted/scored as a forecast source.
_SRC_KNOWN = "wattcast_known"


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
    """Per-subentry polling bookkeeping (all of it persists, so a reload keeps a
    failure backoff / ``Retry-After`` and the button still sees the failures)."""

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


def _extends(new: WattcastSeries, old: WattcastSeries) -> bool:
    """True if ``new`` reaches strictly further (forecast or settled) than ``old``."""
    floor = datetime.min.replace(tzinfo=UTC)
    return (new.coverage_end or floor) > (old.coverage_end or floor) or (new.known_end or floor) > (
        old.known_end or floor
    )


def _fingerprint(cfg: PriceForecastConfig) -> dict[str, str | None]:
    """What makes the cached series / pairs / scores belong to *this* market.

    The Wattcast cache is a zone's spot series; the mapping pairs join that
    spot to a contract's buy price (``price_entity`` / ``price_series_entity``);
    the scores compare a zone's forecast with that contract's realised price.
    Keyed only by subentry id, a reconfigure (FI → EE) would otherwise restore
    the old market's prices and learning.
    """
    return {
        "zone": cfg.wattcast_zone,
        "price_entity": cfg.price_entity,
        "price_series_entity": cfg.price_series_entity,
    }


@callback
def async_delete_stale_wattcast_issues(
    hass: HomeAssistant, *, exclude_entry_id: str | None = None
) -> None:
    """Delete ``wattcast_unreachable_<sid>`` issues whose subentry is gone.

    An issue is otherwise only cleared by a success or by turning Wattcast
    off, so removing the subentry (or the whole hub) left it behind. Pass
    ``exclude_entry_id`` from ``async_remove_entry`` to also drop that hub's.
    """
    keep = {
        subentry_id
        for entry in hass.config_entries.async_entries(DOMAIN)
        if entry.entry_id != exclude_entry_id
        for subentry_id, subentry in entry.subentries.items()
        if subentry.subentry_type == SUBENTRY_TYPE_PRICE_FORECAST
    }
    registry = ir.async_get(hass)
    for domain, issue_id in list(registry.issues):
        if (
            domain == DOMAIN
            and issue_id.startswith(_ISSUE_PREFIX)
            and issue_id.removeprefix(_ISSUE_PREFIX) not in keep
        ):
            ir.async_delete_issue(hass, DOMAIN, issue_id)


class PriceForecastCoordinator(DataUpdateCoordinator[dict[str, ForecastResult]]):
    """Fetches/caches Wattcast, fits the fallback, builds + scores the forecast."""

    config_entry: LoadNeedPredictorConfigEntry

    def __init__(self, hass: HomeAssistant, entry: LoadNeedPredictorConfigEntry) -> None:
        super().__init__(
            hass, _LOGGER, config_entry=entry, name=f"{DOMAIN}_forecast", update_interval=None
        )
        self._store = PredictorStore(hass, entry.entry_id, ".forecast")
        self._lock = asyncio.Lock()
        self._closing = False  # set by async_flush: no further saves
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
        self._built_end: dict[str, datetime] = {}  # horizon end of the last rebuild
        self._fingerprints: dict[str, dict] = {}  # config the loaded state belongs to
        self.log: dict[str, list[dict]] = {}
        self.eval_errors: dict[str, list[float]] = {}

    # ── lifecycle ──────────────────────────────────────────────────────────────

    async def async_load_runtime(self) -> None:
        data = await self._store.async_load()
        configs = self.forecast_configs()
        now = dt_util.utcnow()
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
            next_fetch = _parse_iso(fetch.get("next_fetch"))
            if next_fetch is not None and next_fetch - now > _MAX_RESTORED_BACKOFF:
                next_fetch = None
            try:
                failures = max(0, int(fetch.get("failures") or 0))
            except (TypeError, ValueError):
                failures = 0
            self.fetch[subentry_id] = _FetchState(
                # None (a pre-fix payload) → derived from fetched_at on the first tick.
                next_fetch=next_fetch,
                failures=failures,
                failing_since=_parse_iso(fetch.get("failing_since")),
                last_error=fetch.get("last_error"),
            )
            stored = payload.get("fingerprint")
            if subentry_id in configs and isinstance(stored, Mapping):
                # A payload without a fingerprint predates it: adopt, don't wipe.
                current = _fingerprint(configs[subentry_id])
                self._invalidate(
                    subentry_id, {key for key, value in current.items() if stored.get(key) != value}
                )
        # The config this state now belongs to. Captured here, not re-read at
        # save time: a reconfigure updates the subentry *before* the unload
        # flush, which would otherwise stamp the old state with the new config.
        self._fingerprints = {sid: _fingerprint(cfg) for sid, cfg in configs.items()}
        async_delete_stale_wattcast_issues(self.hass)

    def _invalidate(self, subentry_id: str, changed: set[str]) -> None:
        """Drop the state that belonged to the previous market/contract."""
        if not changed:
            return
        _LOGGER.info(
            "Price forecast %s: %s changed; discarding the state learned for the old value",
            subentry_id,
            ", ".join(sorted(changed)),
        )
        # Every key feeds the (spot, buy) pairs: a new zone's spot, a new
        # contract's buy, or a new series's slots.
        self.pairs[subentry_id] = []
        self.mapping.pop(subentry_id, None)
        if changed & {"zone", "price_entity"}:
            # Snapshots/scores pit a zone's forecast against a contract's prices.
            self.log[subentry_id] = []
            self.eval_errors[subentry_id] = []
        if "zone" in changed:
            # The cache is another zone's series: fetch now, not at the next issue.
            self.wattcast[subentry_id] = None
            self.fetched_at[subentry_id] = None
            self.fetch[subentry_id] = _FetchState()
            ir.async_delete_issue(self.hass, DOMAIN, self._issue_id(subentry_id))
        if "price_entity" in changed:
            # Fit on the old contract's history; the next build refits from the new one.
            self.models[subentry_id] = None
            self.shape[subentry_id] = {}

    def _runtime_snapshot(self) -> dict:
        out: dict = {}
        for subentry_id, cfg in self.forecast_configs().items():
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
                    "next_fetch": _iso(fetch.next_fetch),
                    "failures": fetch.failures,
                    "failing_since": _iso(fetch.failing_since),
                    "last_error": fetch.last_error,
                },
                "fingerprint": self._fingerprints.get(subentry_id) or _fingerprint(cfg),
            }
        return out

    def async_persist(self) -> None:
        if self._closing:
            return
        self._store.async_schedule_save(self._runtime_snapshot)

    async def async_flush(self) -> None:
        """Drain in-flight work, stop further writes, and save now (on unload).

        A tick already awaiting the Wattcast HTTP call survives the timer's
        unsubscribe; without the lock + closing flag it could finish after the
        reload and overwrite the new coordinator's cache/backoff/fingerprint.
        """
        async with self._lock:
            self._closing = True
            if self.has_loads:
                await self._store.async_save_now(self._runtime_snapshot())

    def async_reopen(self) -> None:
        """Undo :meth:`async_flush`'s closing flag (the unload failed)."""
        self._closing = False

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
                    if outcome.series is None:
                        self._on_fetch_failure(subentry_id, cfg, outcome, now)
                    else:
                        verdict, reason = self._judge(subentry_id, outcome.series, now)
                        if verdict == "degraded":
                            self._on_fetch_degraded(subentry_id, cfg, reason, now)
                        elif verdict == "unchanged":
                            self._on_fetch_unchanged(subentry_id, now)
                        else:
                            self._on_fetch_success(subentry_id, cfg, outcome.series, now)
                            fetched = True
                    changed = True
                real_end = self._real_end(subentry_id, cfg)
                if fetched:
                    # Hourly is plenty for the weather/wind inputs too.
                    await self._refresh_local(subentry_id, cfg)
                if (
                    fetched
                    or force_rebuild
                    or real_end != self.known_until.get(subentry_id)
                    or self._expired(subentry_id, cfg, now)
                ):
                    self._rebuild(subentry_id, cfg, now)
                    changed = True
            if changed:
                self.async_persist()
                await self.async_refresh()

    def _expired(self, subentry_id: str, cfg: PriceForecastConfig, now: datetime) -> bool:
        """Whether the published series has aged, independent of any fetch.

        During an outage (fetches failing, real-price boundary unchanged)
        nothing else triggers a rebuild, so without this, past slots — and in
        the end a past-only series — would stay published as "ok". Cheap: at
        most one rebuild per 15-min slot boundary plus one per local midnight.
        """
        slots = self.slots.get(subentry_id)
        if slots is None:
            return True
        if slots and (end := _parse_iso(slots[0].get("end"))) is not None and end <= now:
            return True  # the first published slot is over
        return self._horizon_end(cfg) != self._built_end.get(subentry_id)  # new local day

    def _horizon_end(self, cfg: PriceForecastConfig) -> datetime:
        return dt_util.start_of_local_day() + timedelta(days=cfg.forecast_days + 1)

    def _judge(
        self, subentry_id: str, series: WattcastSeries, now: datetime
    ) -> tuple[str, str | None]:
        """Whether a parsed HTTP 200 may replace the cache.

        ``("accept", None)`` — a newer issue (or nothing cached); ``("unchanged",
        None)`` — the cached issue again, adding nothing (a button press or a
        late re-issue): keep the cache and its bookkeeping, just wait for the
        next issue; ``("degraded", why)`` — it would make things worse.

        Issue order is enforced whatever the cache's coverage: an *older* issue
        (a lagging edge cache) never replaces a newer one, even an expired one.
        A newer issue without forecast still ahead never replaces a usable cache
        (with no usable cache, even a known-only response is worth taking: its
        settled spot feeds the mapping and the ``wattcast_known`` slots).
        """
        cached = self.wattcast.get(subentry_id)
        if cached is None:
            return "accept", None
        if series.made_at and cached.made_at:
            if series.made_at < cached.made_at:
                return "degraded", "response older than the cached forecast"
            if series.made_at == cached.made_at and not _extends(series, cached):
                return "unchanged", None
        usable = cached.coverage_end is not None and cached.coverage_end > now
        if usable and (series.coverage_end is None or series.coverage_end <= now):
            return "degraded", "response without forecast"
        return "accept", None

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
        self._maybe_raise_issue(subentry_id, cfg, outcome.error, now)

    def _on_fetch_unchanged(self, subentry_id: str, now: datetime) -> None:
        """The cached issue confirmed: not a replacement, not a failure.

        ``fetched_at`` moves (we did reach the server, and it rate-limits the
        button); the series, mapping and failure bookkeeping stay as they are.
        A server stuck on one issue still shows as stale via the issue age.
        """
        self.fetched_at[subentry_id] = now
        self.fetch.setdefault(subentry_id, _FetchState()).next_fetch = _next_issue_fetch(now)

    def _on_fetch_degraded(
        self, subentry_id: str, cfg: PriceForecastConfig, reason: str, now: datetime
    ) -> None:
        """A 200 that would downgrade the cache: keep the cache, count it as failing.

        Not a transport failure, so no retry ladder: the server re-issues
        hourly, and a sooner retry would only get the same issue again — the
        next request waits for the next issue, exactly like a success (no extra
        requests). It still keeps the failure clock running, so a server that
        stays degraded for hours raises the repair issue.
        """
        state = self.fetch.setdefault(subentry_id, _FetchState())
        if state.last_error != reason:
            _LOGGER.warning("Wattcast %s; keeping the cached forecast", reason)
        state.failing_since = state.failing_since or now
        state.last_error = reason
        state.next_fetch = _next_issue_fetch(now)
        self._maybe_raise_issue(subentry_id, cfg, reason, now)

    def _maybe_raise_issue(
        self, subentry_id: str, cfg: PriceForecastConfig, error: str | None, now: datetime
    ) -> None:
        state = self.fetch.setdefault(subentry_id, _FetchState())
        if state.failing_since and now - state.failing_since >= timedelta(
            hours=WATTCAST_ISSUE_AFTER_H
        ):
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
                    "error": error or "unknown",
                },
            )

    def _issue_id(self, subentry_id: str) -> str:
        return f"{_ISSUE_PREFIX}{subentry_id}"

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
            try:
                await self._refit_local(subentry_id, cfg)
            except Exception:
                # Never let a bad fit skip the rebuild below (or other subentries).
                _LOGGER.exception("Refitting the local price model failed; keeping the old fit")
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
                # ``last_error`` also covers a degraded 200, whose deadline is the
                # next issue — pressing again would only re-fetch the same one.
                if not state.failures and state.last_error is None and not fresh:
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
        try:
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
        except Exception:
            # Keep the previous local forecast; the tick/rebuild must still run.
            _LOGGER.exception("Reading the local price-forecast inputs failed")
            return
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
        end = self._horizon_end(cfg)
        self._built_end[subentry_id] = end
        series = self.wattcast.get(subentry_id) if cfg.use_wattcast else None
        if series is not None and not any(
            edge is not None and edge > start for edge in (series.coverage_end, series.known_end)
        ):
            # Nothing left in the cache that's still ahead of us — neither
            # forecast nor settled spot (a known-only series still prices the
            # hours a lagging real-price entity hasn't published).
            series = None
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
        # Settled-spot slots are near-truth, not forecast: never snapshot/score them.
        candidates = {
            src: [s for s in slots if s.get("src") != _SRC_KNOWN]
            for src, slots in candidates.items()
        }
        available = [src for src, slots in candidates.items() if slots]
        primary = select_primary(self._scores(subentry_id), available=available)
        self.primary[subentry_id] = primary

        # Published, per slot: Wattcast's settled spot where it runs ahead of the
        # real-price entity, then the primary where it reaches, the other
        # forecast source where it doesn't (a local primary with weather for
        # only a few days must not drop the Wattcast days beyond).
        slots = build_slots(
            start=start,
            end=end,
            tz=tz,
            wattcast=series,
            variant=primary if primary != "local" else "wattcast",
            mapping=mapping,
            local_daily=local_daily,
            shape=shape,
            prefer_local=primary == "local",
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
        now = dt_util.utcnow()
        tz = dt_util.get_default_time_zone()
        async with self._lock:
            for subentry_id, cfg in self.forecast_configs().items():
                if not cfg.price_entity:
                    continue
                for entry in self.log.get(subentry_id, []):
                    if (
                        entry.get("actual") is not None
                        or entry.get("predicted") is None
                        or entry.get("unscorable")
                    ):
                        continue
                    bucket_ms = entry.get("bucket_ms")
                    if bucket_ms is None:
                        continue
                    day_start = datetime.fromtimestamp(bucket_ms / 1000, tz=UTC)
                    # Local midnight to local midnight (wall-clock +1 day), so a
                    # 25-hour DST day is neither cut short nor — at the 23:55
                    # capture on the fall-back day — finalised an hour early.
                    day_end = dt_util.as_utc(dt_util.as_local(day_start) + timedelta(days=1))
                    if day_end > now:
                        continue  # day not fully realised yet
                    rows = await async_hourly_price_rows(
                        self.hass, cfg.price_entity, day_start, day_end
                    )
                    actual_vec = hourly_vectors(
                        [{"start": when.isoformat(), "buy": value} for when, value in rows], tz
                    ).get(entry["date"])
                    hours = sum(v is not None for v in actual_vec or ())
                    actual = None
                    if hours >= EVAL_MIN_ACTUAL_HOURS:
                        actual = daily_mean(actual_vec)
                    elif not hours:
                        # No hourly LTS at all → the daily mean (exact day only).
                        actual = await async_daily_price_mean(
                            self.hass, cfg.price_entity, day_start
                        )
                    else:
                        actual_vec = None  # a partial day: retry, never score it
                    if actual is None:
                        # Retry nightly (a recorder may backfill), but not forever.
                        if now - day_end >= timedelta(days=EVAL_GIVE_UP_DAYS):
                            entry["unscorable"] = True
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
