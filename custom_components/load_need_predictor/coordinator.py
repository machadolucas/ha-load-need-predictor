"""Coordinator: holds per-load model state and runs the daily loop.

This is an *on-demand* ``DataUpdateCoordinator`` — there is no polling interval.
``jobs.py`` drives it: predict + push in the afternoon, capture + log in the
evening. The published runtime is *what was last pushed* (cached per load at
push time, rationale included), so a capture's gain update can't make the sensor
show a hypothetical target the scheduler never received; only a load with no
push yet (fresh start/reload) gets a live snapshot. Metrics are always fresh.

Prediction/actual alignment (v1): one training row per local calendar day. The
predict job writes the row's prediction; the capture job fills in the actual
delivered over the trailing 24 h ending at capture time and calibrates. The
online gain only needs a *consistent* ratio, and an occupancy-gated daily model
is insensitive to the exact overnight offset.

Predict and capture are serialised by one lock: each reads model state, awaits
the recorder/history/push, then writes it back — interleaved, the later writer
would silently drop the other's update (e.g. a button press losing a capture's
gain step).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import date as date_cls
from datetime import datetime, timedelta

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .actuation import async_push_target
from .const import (
    CLEAN_CYCLE_TOL_MINUTES,
    DOMAIN,
    EVAL_WINDOW_DAYS,
    MAX_TRAINING_ROWS,
    MIN_REFIT_SAMPLES,
    SUBENTRY_TYPE_LOAD,
)
from .models import LoadConfig, load_config_from_data
from .occupancy import async_count_residents_home, async_guest_equivalents
from .persistence import (
    PredictorStore,
    model_from_dict,
    model_to_dict,
    tank_from_dict,
    tank_to_dict,
)
from .predictor import (
    SEED_E_BASE,
    SEED_E_DRAW_PER_PERSON,
    FeatureVector,
    ModelState,
    apply_observation,
    blend_param,
    build_features,
    clamp_minutes,
    close_cycle,
    default_model_state,
    energy_balance_demand,
    explain_load,
    is_valid_delivery,
    kwh_to_minutes,
    open_cycle,
    predict_kwh,
    refit_occupancy_params,
    rolling_mae,
)
from .runtime import LoadNeedPredictorConfigEntry
from .statistics_source import (
    async_commanded_minutes,
    async_daily_delivered_kwh,
    async_statistic_change,
    capture_window,
)
from .tank_model import TankState, deficit_minutes_from_kwh

_LOGGER = logging.getLogger(__name__)

# Row keys the capture job owns; a later re-predict the same day keeps them.
_CAPTURE_KEYS = (
    "actual_kwh",
    "actual_minutes",
    "abs_error_minutes",
    "data_quality",
    "clean_cycle",
    "demand_kwh",
    "tank_deficit_end_kwh",
    "tank_deficit_end_at",
    "capture_window_start",
    "capture_window_end",
)

# The energy balance pairs yesterday's tank snapshot with today's delivery
# window; they must describe the same span. A capture-time change, a lock wait
# or recorder lag larger than this falls back to the clean-cycle gate.
ENERGY_BALANCE_TOL = timedelta(minutes=15)


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
    # Carried backlog (minutes) folded into the pushed target; None when deficit
    # carryover is disabled (no controlled switch configured).
    deficit_minutes: float | None = None
    # The prediction broken into its terms, for the dashboard card's rationale.
    rationale: dict | None = None


@dataclass(frozen=True)
class PublishedTarget:
    """The target-side half of a :class:`LoadResult`, frozen at push time.

    Cached per load when a predict pushes, so the sensor shows exactly what the
    scheduler holds (and a rationale built from the same features/deficit),
    rather than re-deriving it on every refresh with whatever gain/occupancy
    is current.
    """

    minutes: int
    kwh: float
    deficit_minutes: float | None
    rationale: dict
    # Where/when it was pushed — a restored cache is only trusted while the
    # load still targets the same entity.
    target_entity: str | None = None
    date: str = ""

    def to_dict(self) -> dict:
        return {
            "minutes": self.minutes,
            "kwh": self.kwh,
            "deficit_minutes": self.deficit_minutes,
            "rationale": self.rationale,
            "target_entity": self.target_entity,
            "date": self.date,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> PublishedTarget | None:
        """Rebuild a persisted cache entry; ``None`` if absent or malformed."""
        if not isinstance(data, dict):
            return None
        try:
            deficit = data.get("deficit_minutes")
            return cls(
                minutes=int(data["minutes"]),
                kwh=float(data["kwh"]),
                deficit_minutes=None if deficit is None else float(deficit),
                rationale=dict(data.get("rationale") or {}),
                target_entity=data.get("target_entity"),
                date=str(data.get("date", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None


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
        # Tank state-of-charge, mutated by the tank tracker but owned here so the
        # debounced snapshot doesn't drop it (see tank_tracker for the rationale).
        self.tanks: dict[str, TankState] = {}
        # What each load last pushed (see PublishedTarget); in-memory only.
        self._published: dict[str, PublishedTarget] = {}
        # Serialises predict/capture (see the module docstring).
        self._lock = asyncio.Lock()
        # Set by async_flush on unload: this coordinator is being replaced, so
        # any late predict/capture/persist must not write behind the new one.
        self._closing = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def async_load_runtime(self) -> None:
        """Load persisted model state + training rows from the Store."""
        data = await self._store.async_load()
        for subentry_id, payload in data.items():
            self.models[subentry_id] = model_from_dict(payload.get("model"))
            self.training[subentry_id] = list(payload.get("training", []))
            self.eval_errors[subentry_id] = list(payload.get("eval", []))
            if (tank := payload.get("tank")) and (restored := tank_from_dict(tank)) is not None:
                self.tanks[subentry_id] = restored
        # The last *successfully pushed* target survives a restart/reload, so
        # the sensor keeps showing what the scheduler holds rather than an
        # unpushed recomputation — unless the load now targets another entity.
        configs = self.load_configs()
        for subentry_id, payload in data.items():
            published = PublishedTarget.from_dict(payload.get("published"))
            cfg = configs.get(subentry_id)
            if (
                published is not None
                and cfg is not None
                and published.target_entity == cfg.target_number_entity
            ):
                self._published[subentry_id] = published

    def _runtime_snapshot(self) -> dict:
        """Serialise everything persistable (called by the debounced save)."""
        return {
            subentry_id: {
                "model": model_to_dict(self.models.get(subentry_id, default_model_state())),
                "training": self.training.get(subentry_id, []),
                "eval": self.eval_errors.get(subentry_id, []),
                # Additive: only present for loads with tank tracking enabled.
                **(
                    {"tank": tank_to_dict(self.tanks[subentry_id])}
                    if subentry_id in self.tanks
                    else {}
                ),
                # Additive (defaults-tolerant): the last successful push.
                **(
                    {"published": self._published[subentry_id].to_dict()}
                    if subentry_id in self._published
                    else {}
                ),
            }
            for subentry_id in self.load_configs()
        }

    def async_persist(self) -> None:
        """Schedule a debounced save of the model + training state.

        A no-op once closing: the final snapshot is already written, and a
        delayed write from this (replaced) coordinator could land after the
        reloaded one has read — and later saved — newer state.
        """
        if self._closing:
            return
        self._store.async_schedule_save(self._runtime_snapshot)

    async def async_flush(self) -> None:
        """Drain in-flight work, stop further writes, and save now (on unload).

        Every subentry add/edit reloads the entry; without this a reload inside
        the save debounce would lose the latest anchors/learning/boost state,
        and the old delayed write could land after the new coordinator loaded.
        Taking the lock waits out a predict/capture already awaiting the
        recorder, so its result is in the final snapshot rather than written
        after it; ``Store.async_save`` also cancels any pending delayed write.
        """
        async with self._lock:
            self._closing = True
            await self._store.async_save_now(self._runtime_snapshot())

    def async_reopen(self) -> None:
        """Undo :meth:`async_flush`'s closing flag (the unload failed)."""
        self._closing = False

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

    def _live_tank(self, subentry_id: str, cfg: LoadConfig) -> TankState | None:
        """The load's tank state when it may drive control, else ``None``.

        Only a *tracked* (see ``LoadConfig.tank_tracking_enabled``) and
        *calibrated* tank counts: after the detector is cleared the restored
        state would otherwise stay frozen and override the backlog forever.
        """
        if not cfg.tank_tracking_enabled:
            return None
        tank = self.tanks.get(subentry_id)
        return tank if tank is not None and tank.calibrated else None

    def _tank_deficit_minutes(self, subentry_id: str, cfg: LoadConfig) -> float | None:
        """A live tank's control-ledger deficit as runtime minutes (capped), or None.

        Reads the raw *clamped control* ledger on purpose: the sensor shows the
        saturation-curve display value (which creeps toward "full" while the
        element runs), but control wants the plain conservative kWh-below-
        setpoint number.
        """
        tank = self._live_tank(subentry_id, cfg)
        if tank is None:
            return None
        return min(
            max(deficit_minutes_from_kwh(tank.deficit_kwh, cfg.rated_power_kw), 0.0),
            cfg.deficit_cap_minutes,
        )

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
            # Today-so-far `change` in the meter's own unit (m³ here) — no energy
            # conversion. Log-only context: the v1 model ignores it (see
            # CLAUDE.md's data findings).
            "water_total_delta": (
                await async_statistic_change(
                    self.hass, cfg.water_total_entity, dt_util.start_of_local_day(), now
                )
                if cfg.water_total_entity
                else None
            ),
        }

    # ── daily jobs (driven by jobs.py) ─────────────────────────────────────────

    async def async_predict_and_push(self, only: str | None = None) -> bool:
        """Forecast each load, push the target, and record the prediction row.

        ``only`` restricts the work to a single subentry (used by the per-load
        "Predict now" button and the tank's low-charge boost); the daily job
        passes nothing to do them all.

        When a controlled switch is configured this also runs deficit carryover:
        the first predict of a local day closes the previous cycle (how much
        actually ran vs. what was asked) and rolls the unmet remainder into a
        backlog, then pushes today's occupancy need *plus* that backlog. A later
        predict the same day only *re-plans* the open cycle (see
        :meth:`_predict_one`). Without a switch it is the plain daily predictor.

        Returns True iff every push it attempted succeeded (vacuously True when
        no load has a target to push to) — the boost disarms only on success.
        One load failing never skips the others or the persist.
        """
        all_ok = True
        async with self._lock:
            if self._closing:  # being unloaded/replaced — push nothing, write nothing
                return False
            now = dt_util.now()
            for subentry_id, cfg in self.load_configs().items():
                if only is not None and subentry_id != only:
                    continue
                try:
                    ok = await self._predict_one(subentry_id, cfg, now)
                except Exception:  # noqa: BLE001 - one broken load must not stop the rest
                    _LOGGER.exception("Predict failed for load %s", subentry_id)
                    ok = False
                all_ok = all_ok and ok
            self.async_persist()
        # Outside the lock: the refresh only reads, and the cache makes it cheap.
        await self.async_refresh()
        return all_ok

    def _plan(
        self,
        cfg: LoadConfig,
        state: ModelState,
        features: FeatureVector,
        deficit: float,
    ) -> PublishedTarget:
        """Need + backlog → the target + its rationale, from one set of inputs.

        Both come from the same state/features/deficit, so the card's
        ``target_minutes`` is always the value actually pushed.
        """
        kwh = predict_kwh(state, features)
        need_raw = kwh_to_minutes(kwh, cfg.rated_power_kw)
        _, pushed = open_cycle(need_raw, deficit, cfg.min_minutes, cfg.max_minutes)
        rationale = explain_load(
            state,
            features,
            rated_power_kw=cfg.rated_power_kw,
            min_minutes=cfg.min_minutes,
            max_minutes=cfg.max_minutes,
            deficit_minutes=deficit,
        )
        return PublishedTarget(
            minutes=pushed,
            kwh=round(kwh, 3),
            deficit_minutes=round(deficit, 1) if cfg.controlled_switch_entity else None,
            rationale=rationale,
        )

    @staticmethod
    def _same_local_day(start_iso: str, now: datetime) -> bool:
        """Whether the open cycle began on ``now``'s local calendar day."""
        start = dt_util.parse_datetime(start_iso) if start_iso else None
        return start is not None and dt_util.as_local(start).date() == dt_util.as_local(now).date()

    async def _predict_one(self, subentry_id: str, cfg: LoadConfig, now: datetime) -> bool:
        """Predict + push one load; True unless an attempted push failed."""
        today = dt_util.as_local(now).date().isoformat()
        state = self.model_for(subentry_id)
        features = build_features(await self.async_build_snapshot(cfg))
        need_raw = kwh_to_minutes(predict_kwh(state, features), cfg.rated_power_kw)
        need_minutes = clamp_minutes(need_raw, cfg.min_minutes, cfg.max_minutes)

        # Commanded-minutes bookkeeping. Cycles are *daily*: only the first
        # predict of a local day closes the previous cycle. A mid-day re-predict
        # (the button, a low-charge boost) re-plans the open one instead —
        # closing it after a few hours would find little on-time, roll ~the whole
        # ask into the backlog and push it again on top of a fresh need. This is
        # the backlog fallback — it keeps running/persisting even when a
        # calibrated tank supersedes it for the pushed target.
        bookkept_deficit = 0.0
        replan = False
        if cfg.controlled_switch_entity:
            bookkept_deficit = state.deficit_minutes
            replan = self._same_local_day(state.cycle_start_iso, now)
            if not replan:
                commanded = await self._commanded_since(cfg, state.cycle_start_iso, now)
                if commanded is not None and state.pending_owed_minutes > 0:
                    bookkept_deficit = close_cycle(
                        state.pending_owed_minutes, commanded, cfg.deficit_cap_minutes
                    )

        # SoC feedback #1: a live calibrated tank's measured deficit is a better
        # backlog than the commanded-minutes bookkeeping — it self-heals on manual
        # heating, early thermostat trips and skips alike. Over-ask is physically
        # safe (the thermostat just trips).
        deficit = bookkept_deficit
        deficit_source = "commanded" if cfg.controlled_switch_entity else None
        if (tank_deficit := self._tank_deficit_minutes(subentry_id, cfg)) is not None:
            deficit = tank_deficit
            deficit_source = "tank"

        # ``plan.minutes`` reflects the effective (possibly tank-overridden) deficit.
        plan = self._plan(cfg, state, features, deficit)

        if cfg.controlled_switch_entity:
            # Persist the open cycle from the *bookkept* deficit (not the tank
            # override) so the commanded-minutes backlog stays a valid fallback
            # and the clean-cycle gain gate reads an untouched deficit. A re-plan
            # keeps the cycle's start so the day's on-time still counts toward it.
            bookkept_pending, _ = open_cycle(
                need_raw, bookkept_deficit, cfg.min_minutes, cfg.max_minutes
            )
            self.models[subentry_id] = state = replace(
                state,
                deficit_minutes=bookkept_deficit,
                pending_owed_minutes=bookkept_pending,
                cycle_start_iso=state.cycle_start_iso if replan else now.isoformat(),
            )

        ok = await async_push_target(self.hass, cfg.target_number_entity, plan.minutes)
        self._push_ok[subentry_id] = ok
        # The cache is "what the scheduler holds": a failed push leaves the old
        # value in force (``last_push_ok`` shows the failure). A publish-only
        # load has no scheduler, so its computed plan *is* what it publishes.
        if ok or not cfg.target_number_entity:
            self._published[subentry_id] = replace(
                plan, target_entity=cfg.target_number_entity, date=today
            )
        self._upsert_prediction_row(
            subentry_id,
            today,
            features,
            state,
            need_minutes,
            plan.minutes,
            deficit,
            deficit_source,
        )
        # A publish-only load (no target configured) attempted nothing to fail.
        return ok or not cfg.target_number_entity

    async def _commanded_since(self, cfg: LoadConfig, start_iso: str, end) -> float | None:
        """Minutes the load's switch was on since ``start_iso`` (None if unknown).

        ``None`` (no open cycle yet, unparseable timestamp, or the recorder
        unavailable) tells the caller to leave the backlog untouched rather than
        treat a missing reading as "nothing ran".
        """
        start = dt_util.parse_datetime(start_iso) if start_iso else None
        if start is None or start >= end:
            return None
        return await async_commanded_minutes(self.hass, cfg.controlled_switch_entity, start, end)

    async def async_capture_and_log(self) -> None:
        """Read each load's actual delivery, complete its row, and calibrate.

        The window is ``capture_window(now)`` — the day ending at the last
        already-compiled 5-minute bucket, tiling exactly with the previous
        capture (not the calendar day: at a 23:55 capture the day's last hour
        isn't compiled into hourly stats yet, and the next capture reads a
        different day, so a calendar-day read never counted 23:00–24:00). The
        row stays keyed by today's date.
        """
        async with self._lock:
            if self._closing:
                return
            now = dt_util.utcnow()
            start, end = capture_window(now)
            today = dt_util.as_local(now).date().isoformat()
            for subentry_id, cfg in self.load_configs().items():
                if not cfg.delivered_energy_entity:
                    continue
                try:
                    actual_kwh = await async_daily_delivered_kwh(
                        self.hass, cfg.delivered_energy_entity, start, end
                    )
                    # Switch on-time over the same window tells us whether the
                    # scheduler actually ran the ask — so the gain only learns
                    # from days the meter reflects true demand, not a price-driven
                    # skip/defer.
                    commanded_today = None
                    if cfg.controlled_switch_entity:
                        commanded_today = await async_commanded_minutes(
                            self.hass, cfg.controlled_switch_entity, start, end
                        )
                    self._record_actual(
                        subentry_id,
                        today,
                        cfg,
                        actual_kwh,
                        commanded_today,
                        window=(start, end),
                        snapshot_at=dt_util.utcnow(),
                    )
                except Exception:  # noqa: BLE001 - one broken load must not stop the rest
                    _LOGGER.exception("Capture failed for load %s", subentry_id)
            self.async_persist()
        await self.async_refresh()

    # ── training-row management ────────────────────────────────────────────────

    def _upsert_prediction_row(
        self,
        subentry_id: str,
        date: str,
        features: FeatureVector,
        state: ModelState,
        need_minutes: int,
        pushed_minutes: int,
        deficit_minutes: float,
        deficit_source: str | None = None,
    ) -> None:
        """Write/replace today's row with the prediction, keeping any actuals.

        ``predicted_minutes`` stays the occupancy *need* (so the error/eval/gain
        signals measure demand, unaffected by carryover); ``pushed_minutes`` is
        what was actually sent to the scheduler (need + ``deficit_minutes``).
        ``deficit_source`` records where the backlog came from — ``"tank"`` (a
        calibrated tank's measured deficit), ``"commanded"`` (the switch on-time
        bookkeeping), or ``None`` (carryover disabled).
        """
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
            "predicted_minutes": need_minutes,
            "pushed_minutes": pushed_minutes,
            "deficit_minutes": round(deficit_minutes, 1),
            "deficit_source": deficit_source,
            "gain": round(state.gain, 4),
            "model_version": state.version,
            "actual_kwh": None,
            "actual_minutes": None,
            "abs_error_minutes": None,
            "data_quality": None,
        }
        existing = self._row_for_date(rows, date)
        if existing is not None and existing.get("actual_kwh") is not None:
            # Already captured: the observation was judged against *this* row's
            # prediction (gain, features, ask). A later re-predict the same date
            # (still pushed + published) must not rewrite it, or a refit would
            # divide that demand by a gain it was never predicted with.
            return
        if existing is not None:
            # Preserve actuals if the capture job already ran today — incl. the
            # tank snapshot tomorrow's energy balance starts from.
            for key in _CAPTURE_KEYS:
                if key in existing:
                    row[key] = existing[key]
            rows[rows.index(existing)] = row
        else:
            rows.append(row)
        del rows[:-MAX_TRAINING_ROWS]  # cap the history

    def _record_actual(
        self,
        subentry_id: str,
        date: str,
        cfg: LoadConfig,
        actual_kwh: float | None,
        commanded_today: float | None = None,
        *,
        window: tuple[datetime, datetime] | None = None,
        snapshot_at: datetime | None = None,
    ) -> None:
        """Fill today's row with the actual delivery and calibrate the model.

        ``window`` is the delivery window read; ``snapshot_at`` is when the tank
        ledger was sampled. Both are stored so the next capture can check that
        its energy balance pairs spans that actually meet.
        """
        rows = self.training.setdefault(subentry_id, [])
        row = self._row_for_date(rows, date)
        if row is None:  # capture without a prior prediction (e.g. fresh install)
            row = {"date": date, "predicted_kwh": None, "predicted_minutes": None}
            rows.append(row)
            del rows[:-MAX_TRAINING_ROWS]

        # Energy-balance snapshot: the live tank's control ledger at capture
        # time. Taken whatever the meter did, so tomorrow's balance has its start.
        tank = self._live_tank(subentry_id, cfg)
        tank_end = tank.deficit_kwh if tank is not None else None
        if window is not None:
            row["capture_window_start"] = window[0].isoformat()
            row["capture_window_end"] = window[1].isoformat()
        if tank_end is not None:
            row["tank_deficit_end_kwh"] = round(tank_end, 3)
            row["tank_deficit_end_at"] = (snapshot_at or dt_util.utcnow()).isoformat()
        else:
            row.pop("tank_deficit_end_kwh", None)
            row.pop("tank_deficit_end_at", None)

        valid = is_valid_delivery(actual_kwh)
        row["actual_kwh"] = round(actual_kwh, 3) if actual_kwh is not None else None
        row["data_quality"] = valid
        if actual_kwh is None:
            return

        actual_minutes = round(kwh_to_minutes(actual_kwh, cfg.rated_power_kw))
        row["actual_minutes"] = actual_minutes

        # With a tank snapshot at both ends of the 24 h window, demand is known
        # from the energy balance (delivered + Δdeficit), so skips/refills no
        # longer bias it and the day is a fair sample whatever the scheduler did.
        # Otherwise fall back to the clean-cycle gate: only a cycle that ran
        # roughly the full ask with no backlog in play has meter ≈ demand.
        learn_kwh = actual_kwh
        prev = self._row_for_date(rows, _previous_date(date))
        prev_end = prev.get("tank_deficit_end_kwh") if prev is not None else None
        if (
            tank_end is not None
            and prev_end is not None
            and _balance_spans_match(prev, window, snapshot_at)
        ):
            learn_kwh = energy_balance_demand(actual_kwh, tank_end, float(prev_end))
            row["demand_kwh"] = round(learn_kwh, 3)
            clean = True
        else:
            row.pop("demand_kwh", None)
            clean = self._cycle_is_clean(subentry_id, cfg, row, commanded_today)
        row["clean_cycle"] = clean

        predicted_minutes = row.get("predicted_minutes")
        if predicted_minutes is not None:
            error = abs(predicted_minutes - round(kwh_to_minutes(learn_kwh, cfg.rated_power_kw)))
            row["abs_error_minutes"] = error
            if valid and clean:  # only learn from plausible, supply-clean days
                errors = self.eval_errors.setdefault(subentry_id, [])
                errors.append(error)
                del errors[:-MAX_TRAINING_ROWS]

        predicted_kwh = row.get("predicted_kwh")
        if valid and clean and predicted_kwh:
            # The row's gain is the one the prediction was made with — the
            # learner needs it to converge on the true ratio (see update_gain).
            self.models[subentry_id] = apply_observation(
                self.model_for(subentry_id),
                predicted_kwh,
                learn_kwh,
                predicted_gain=row.get("gain"),
            )
            self._maybe_refit(subentry_id)

    def _cycle_is_clean(
        self, subentry_id: str, cfg: LoadConfig, row: dict, commanded_today: float | None
    ) -> bool:
        """Whether today's delivery is a trustworthy demand sample for the gain.

        Without carryover (no switch) every plausible day is fair game — the
        original behaviour. With it, a day is *not* clean while a backlog is being
        worked off (the meter under-reads demand then) or when the scheduler ran
        materially less than asked (a price/solar skip/defer). A missing recorder
        reading leaves learning enabled rather than starving the model.
        """
        if not cfg.controlled_switch_entity:
            return True
        if self.model_for(subentry_id).deficit_minutes >= CLEAN_CYCLE_TOL_MINUTES:
            return False
        # The row's deficit is the *effective* backlog folded into the push — with
        # a calibrated tank it can be nonzero while the bookkept value above is 0.
        # A tank-refill day delivers backlog + demand, so it must not teach either.
        if (row.get("deficit_minutes") or 0) >= CLEAN_CYCLE_TOL_MINUTES:
            return False
        if commanded_today is None:
            return True
        asked = row.get("pushed_minutes")
        if asked is None:
            asked = row.get("predicted_minutes") or 0
        return commanded_today >= asked - CLEAN_CYCLE_TOL_MINUTES

    def _maybe_refit(self, subentry_id: str) -> None:
        """Blend in an empirical fit of E_base / E_draw once enough data exists.

        Fits the de-gained, de-guested, de-factored demand on ``people_home``
        (see ``refit_occupancy_params`` — it must invert the same equation
        ``predict_kwh`` uses, or the fit is biased and the gain double-applies)
        over the valid, *clean*, occupancy-labelled rows (a legacy row without
        ``clean_cycle`` counts as clean), and blends toward the seeds by sample
        count. The target is the energy-balance ``demand_kwh`` when the row has
        one. Needs a zero-person and a multi-person day — otherwise the fit
        returns None and the seeds stand. The online gain keeps handling
        residual drift on top.
        """
        rows = [
            r
            for r in self.training.get(subentry_id, [])
            if r.get("data_quality")
            and r.get("clean_cycle", True)
            and _learning_kwh(r) is not None
            and r.get("people_home") is not None
        ]
        if len(rows) < MIN_REFIT_SAMPLES:
            return
        state = self.model_for(subentry_id)
        fit = refit_occupancy_params(
            [
                (
                    r["people_home"],
                    _learning_kwh(r),
                    r.get("guests") or 0.0,
                    r.get("gain") or state.gain,
                )
                for r in rows
            ],
            guest_bonus=state.guest_bonus,
            empty_house_factor=state.empty_house_factor,
        )
        if fit is None:
            return
        e_base_emp, e_draw_emp = fit
        n = len(rows)
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
        """Compute the published forecast + metrics for every load.

        A load that has pushed publishes its cached :class:`PublishedTarget`
        (what the scheduler actually holds); only a load with none yet builds a
        live snapshot. One load failing (e.g. a recorder hiccup) keeps its
        previous result instead of failing the whole coordinator — which would
        mark every load's sensors unavailable until the next job.
        """
        previous = self.data or {}
        results: dict[str, LoadResult] = {}
        for subentry_id, cfg in self.load_configs().items():
            try:
                results[subentry_id] = await self._result_for(subentry_id, cfg)
            except Exception:  # noqa: BLE001 - one broken load must not blank the rest
                _LOGGER.exception("Building the result failed for load %s", subentry_id)
                if subentry_id in previous:
                    results[subentry_id] = previous[subentry_id]
        return results

    async def _result_for(self, subentry_id: str, cfg: LoadConfig) -> LoadResult:
        """One load's published result: cached/live target + always-fresh metrics."""
        state = self.model_for(subentry_id)
        plan = self._published.get(subentry_id)
        if plan is None:
            # Fresh start: show what a push *would* send now — occupancy need
            # plus the standing backlog (0 when carryover is off), a live
            # calibrated tank's measured deficit superseding the bookkept one.
            features = build_features(await self.async_build_snapshot(cfg))
            deficit = state.deficit_minutes if cfg.controlled_switch_entity else 0.0
            if (tank_deficit := self._tank_deficit_minutes(subentry_id, cfg)) is not None:
                deficit = tank_deficit
            plan = self._plan(cfg, state, features, deficit)
        errors = self.eval_errors.get(subentry_id, [])[-EVAL_WINDOW_DAYS:]
        completed = self._last_completed(subentry_id)
        return LoadResult(
            predicted_minutes=plan.minutes,
            predicted_kwh=plan.kwh,
            last_delivered_kwh=completed.get("actual_kwh") if completed else None,
            prediction_error_minutes=completed.get("abs_error_minutes") if completed else None,
            rolling_mae_minutes=round(rolling_mae(errors), 1) if errors else None,
            sample_count=state.sample_count,
            last_push_ok=self._push_ok.get(subentry_id),
            deficit_minutes=plan.deficit_minutes,
            rationale=plan.rationale,
        )

    async def _async_update_data(self) -> dict[str, LoadResult]:
        return await self._build_results()


def _previous_date(date: str) -> str:
    """The ISO date one calendar day before ``date`` (the previous capture's row)."""
    return (date_cls.fromisoformat(date) - timedelta(days=1)).isoformat()


def _learning_kwh(row: dict) -> float | None:
    """A row's demand sample: the energy-balance demand if known, else the meter."""
    demand = row.get("demand_kwh")
    return demand if demand is not None else row.get("actual_kwh")


def _balance_spans_match(
    prev: dict, window: tuple[datetime, datetime] | None, snapshot_at: datetime | None
) -> bool:
    """Whether Δdeficit (prev snapshot → now) covers the delivery window.

    A same-date row is no proof of *when* its snapshot was taken: a changed
    capture time, a lock wait or recorder lag would pair a delivery window with
    a different ΔSoC span and admit fictitious demand as clean. Both ends must
    agree within :data:`ENERGY_BALANCE_TOL`; a legacy row without a timestamp
    never matches (→ the clean-cycle gate).
    """
    if window is None or snapshot_at is None:
        return False
    prev_at = dt_util.parse_datetime(prev.get("tank_deficit_end_at") or "")
    if prev_at is None:
        return False
    start, end = window
    return (
        abs(prev_at - start) <= ENERGY_BALANCE_TOL and abs(snapshot_at - end) <= ENERGY_BALANCE_TOL
    )
