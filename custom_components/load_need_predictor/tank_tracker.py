"""HA-side tank charge tracker: a self-ticking coordinator.

The pure energy balance lives in :mod:`tank_model`; this is the thin Home
Assistant shell that feeds it. Once per :data:`TANK_TICK_SECONDS` it reads each
tank-load's counters/meter/binary states straight from ``hass.states`` (never the
recorder — the tank math is purely instantaneous), unit-normalises both
counters (energy → kWh, water → L, via HA's own converters; an unknown unit reads
as unavailable, never guessed), re-baselines a counter whose configured entity
changed, computes sustained-state durations from ``last_changed``, calls
:func:`tank_model.apply_tick`, and publishes a :class:`TankResult`.

**Why a self-driven tick instead of ``update_interval``.** A polling
``DataUpdateCoordinator`` only polls while an entity is listening, so a disabled
tank-charge sensor would silently freeze the integration and its parameter
learning. We register our own ``async_track_time_interval`` (only when a load has
opted in) so the balance advances regardless of who's watching — mirroring how
``PredictorJobs`` owns its own time listeners.

**Tank state ownership.** The :class:`tank_model.TankState` lives in
``LoadNeedPredictorCoordinator.tanks`` (not here), because that coordinator's
``_runtime_snapshot`` rebuilds the whole per-subentry Store dict on every save —
a key owned elsewhere would be silently dropped. This tracker mutates that dict
and asks the load coordinator to persist.

**Two deficits, two audiences.** The model keeps a clamped *control* ledger and
a saturation-curve *display* deficit (see :mod:`tank_model`). What we publish as
``deficit_kwh`` (and therefore the SoC %, litres and showers) is the shown one:
it reaches exactly 100 % only on a real anchor *transition* and then drifts back
down as the tank relaxes past the thermostat trip — a latched tank is **not**
held at 100 %. The raw control ledger rides along as ``deficit_raw_kwh``, which
is what the SoC→prediction feedback in the load coordinator asks heating for.
Because ``anchored`` is now the transition tick alone, it is still exactly the
"learned something, persist now" signal.

**SoC → prediction feedback #2 (low-charge boost).** After each tick, if a
calibrated tank's SoC has fallen below the load's boost threshold, the tracker
re-runs the load's predict/push (which, via feedback #1 in the coordinator, folds
the measured deficit into a bigger target) so the scheduler books more heating
before the tank empties. Hysteresis + a rate limit live in the pure model; the
push runs as its own task (it can outlast a tick; while it is in flight that
load's boost isn't re-evaluated), the disarm is applied onto the *latest* tank
state only once the push actually succeeded (a dead scheduler target must not
consume the trigger), and a failed push is retried at most every
:data:`BOOST_RETRY_INTERVAL` after it completed, so it can't re-plan every tick.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from homeassistant.const import UnitOfEnergy, UnitOfVolume
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import BaseUnitConverter, EnergyConverter, VolumeConverter

from . import occupancy
from .const import DOMAIN, TANK_TICK_SECONDS
from .coordinator import LoadNeedPredictorCoordinator
from .models import LoadConfig
from .runtime import LoadNeedPredictorConfigEntry
from .tank_model import (
    TankParams,
    TickInputs,
    apply_tick,
    capacity_kwh,
    initial_state,
    liters_at_temp,
    profile_of,
    rebind_sources,
    should_boost,
    showers_left,
)
from .tank_model import soc as model_soc

_LOGGER = logging.getLogger(__name__)

TICK_INTERVAL = timedelta(seconds=TANK_TICK_SECONDS)
# Persist at least this often even without an anchor/learn/boost, so a long quiet
# stretch of drift still survives a restart within ~15 minutes.
PERSIST_EVERY_TICKS = 15

# A failed boost push (scheduler target unavailable) keeps the trigger armed but
# is retried no more often than this, so a dead target can't re-plan every tick.
BOOST_RETRY_INTERVAL = timedelta(minutes=15)

# Spellings of "litres" seen on template/legacy meters that HA's VolumeConverter
# (which only knows ``L``) would otherwise reject; accepted as before.
_LITER_ALIASES = frozenset({"l", "liter", "liters", "Liter", "litre", "litres"})

_UNKNOWN_STATES = ("unknown", "unavailable")


@dataclass
class TankResult:
    """What the tank charge sensor publishes for one load.

    Two deficits are published on purpose: ``deficit_kwh`` is the *shown* one
    (the saturation-curve display value the SoC % is derived from), while
    ``deficit_raw_kwh`` is the clamped control ledger the SoC→prediction feedback
    actually asks heating for. ``uncertainty_kwh`` is the σ that curve uses, so
    the card can say how much of the reading is model slack.
    """

    soc_pct: float
    deficit_kwh: float
    deficit_raw_kwh: float
    uncertainty_kwh: float
    hysteresis_kwh: float
    capacity_kwh: float
    hot_fraction: float
    hot_fraction_profile: list[float]
    standby_w: float
    calibrated: bool
    latched: bool
    last_full: str | None
    draw_source: str
    liters_40c: float
    showers_left: float


class TankTracker(DataUpdateCoordinator[dict[str, TankResult]]):
    """Ticks the pure tank model on its own timer and publishes each load's SoC."""

    config_entry: LoadNeedPredictorConfigEntry

    def __init__(
        self,
        hass: HomeAssistant,
        entry: LoadNeedPredictorConfigEntry,
        load: LoadNeedPredictorCoordinator,
    ) -> None:
        # update_interval=None → no polling; our own time interval drives ticks.
        super().__init__(
            hass, _LOGGER, config_entry=entry, name=f"{DOMAIN}_tank", update_interval=None
        )
        self._load = load
        self._unsub: callable | None = None
        self._ticks = 0
        # Per-load "don't retry the boost push before" after a failed push
        # (in memory: a restart may retry at once, which is harmless).
        self._boost_retry_at: dict[str, datetime] = {}
        # Loads whose boost push is running right now (see ``_async_boost``).
        self._boost_in_flight: set[str] = set()
        # Their tasks, so unload can drain them before the final flush (a boost
        # finishing after it would lose its cooldown and re-fire on reload).
        self._boost_tasks: set[asyncio.Task] = set()
        # (entity_id, unit) pairs already warned about, so a misconfigured unit
        # logs once instead of every minute.
        self._warned_units: set[tuple[str, str | None]] = set()

    # ── config access ──────────────────────────────────────────────────────────

    def tank_configs(self) -> dict[str, LoadConfig]:
        """Loads that opted into tank tracking (``LoadConfig.tank_tracking_enabled``)."""
        return {
            sid: cfg for sid, cfg in self._load.load_configs().items() if cfg.tank_tracking_enabled
        }

    @property
    def has_tanks(self) -> bool:
        """True when at least one load has tank tracking enabled."""
        return bool(self.tank_configs())

    # ── lifecycle ──────────────────────────────────────────────────────────────

    @callback
    def async_start(self) -> None:
        """Register the periodic tick — only when a load opted in (idempotent)."""
        if self._unsub is not None or not self.has_tanks:
            return
        self._unsub = async_track_time_interval(self.hass, self._handle_tick, TICK_INTERVAL)

    @callback
    def async_shutdown_ticker(self) -> None:
        """Cancel the tick (idempotent; safe on unload)."""
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    async def _handle_tick(self, now: datetime) -> None:
        """Timer callback → run one tick."""
        await self.async_tick()

    # ── state reads (instantaneous — no recorder) ───────────────────────────────

    def _counter(
        self,
        entity_id: str | None,
        converter: type[BaseUnitConverter],
        target_unit: str,
        aliases: dict[str, str] | None = None,
    ) -> float | None:
        """A cumulative counter normalised to ``target_unit``, or ``None``.

        The model's baselines are in kWh / L, so a Wh counter stepping 500 must
        read as 0.5 kWh, not 500 — the unit comes from ``unit_of_measurement`` via
        HA's own converter. A missing or unsupported unit is treated as
        *unavailable* (logged once), never guessed: a wrong guess is a bogus
        delta the balance can't take back, whereas "unavailable" just waits.
        """
        if not entity_id:
            return None
        state = self.hass.states.get(entity_id)
        if state is None or state.state in _UNKNOWN_STATES:
            return None
        try:
            value = float(state.state)
        except (TypeError, ValueError):
            return None
        unit = state.attributes.get("unit_of_measurement")
        unit = (aliases or {}).get(unit, unit)
        if unit not in converter.VALID_UNITS:
            if (entity_id, unit) not in self._warned_units:
                self._warned_units.add((entity_id, unit))
                _LOGGER.warning(
                    "Tank counter %s has unit %r, which can't be converted to %s; "
                    "ignoring its readings until the unit is fixed",
                    entity_id,
                    unit,
                    target_unit,
                )
            return None
        return converter.convert(value, unit, target_unit)

    def _energy_kwh(self, entity_id: str | None) -> float | None:
        """Delivered-energy counter in kWh (Wh/MWh/J/… converted), or ``None``."""
        return self._counter(entity_id, EnergyConverter, UnitOfEnergy.KILO_WATT_HOUR)

    def _water_liters(self, entity_id: str | None) -> float | None:
        """Cold-meter reading in litres (m³/gal/ft³/mL/… converted), or ``None``."""
        return self._counter(
            entity_id,
            VolumeConverter,
            UnitOfVolume.LITERS,
            aliases=dict.fromkeys(_LITER_ALIASES, UnitOfVolume.LITERS),
        )

    def _changed_iso(self, entity_id: str | None) -> str | None:
        """When the entity's state last changed (ISO), or ``None`` if unknown."""
        state = self.hass.states.get(entity_id) if entity_id else None
        return state.last_changed.isoformat() if state is not None else None

    def _bool_and_duration(
        self, entity_id: str | None, now: datetime
    ) -> tuple[bool | None, float | None]:
        """Tri-state on/off + seconds held → ``(is_on|None, held_s|None)``.

        ``"on"`` → True, ``"off"`` → False; anything else (unknown/unavailable/
        missing) → ``None`` so the model can fail closed (a missing sensor never
        anchors). The duration is seconds since ``last_changed`` — which the
        ``unavailable`` blips both entities show reset, so it self-debounces.
        """
        state = self.hass.states.get(entity_id) if entity_id else None
        if state is None:
            return None, None
        if state.state == "on":
            is_on: bool | None = True
        elif state.state == "off":
            is_on = False
        else:
            return None, None
        return is_on, (now - state.last_changed).total_seconds()

    # ── the tick ─────────────────────────────────────────────────────────────────

    async def async_tick(self) -> None:
        """Advance every tank-load by one tick and publish the results.

        Each load's work is isolated in ``try/except`` — one broken entity or load
        must never kill the loop or raise (the integration degrades, never breaks).
        """
        self._ticks += 1
        now = dt_util.utcnow()
        now_iso = now.isoformat()
        results: dict[str, TankResult] = dict(self.data or {})
        persist_needed = self._ticks % PERSIST_EVERY_TICKS == 0

        for sid, cfg in self.tank_configs().items():
            try:
                result, changed = await self._tick_one(sid, cfg, now, now_iso)
            except Exception:  # noqa: BLE001 - a broken load must not stop the others
                _LOGGER.exception("Tank tick failed for load %s", sid)
                continue
            results[sid] = result
            persist_needed = persist_needed or changed

        if persist_needed:
            self._load.async_persist()
        self.async_set_updated_data(results)

    async def _tick_one(
        self, sid: str, cfg: LoadConfig, now: datetime, now_iso: str
    ) -> tuple[TankResult, bool]:
        """Run one load's tick; returns ``(result, save_now)``.

        ``save_now`` is True when the tick took an anchor *transition* (which is
        also the only time the model learns) — the event worth persisting
        promptly (a successful boost persists from its own task). Ordinary
        latched ticks are just drift and ride the periodic save.
        """
        params = TankParams(cfg.tank_volume_l, cfg.tank_setpoint_c, cfg.tank_cold_in_c)
        capacity = capacity_kwh(cfg.tank_volume_l, cfg.tank_setpoint_c, cfg.tank_cold_in_c)
        state = self._load.tanks.get(sid) or initial_state(capacity)
        # A baseline belongs to one entity: a reconfigured counter re-baselines.
        state = rebind_sources(
            state, cfg.delivered_energy_entity or "", cfg.water_total_entity or ""
        )

        # Elapsed since the last tick — restart reconciliation is just this same
        # arithmetic over the (possibly long) gap since the persisted timestamp.
        last = dt_util.parse_datetime(state.last_tick_iso) if state.last_tick_iso else None
        elapsed_s = (now - last).total_seconds() if last is not None else float(TANK_TICK_SECONDS)

        contactor_on, contactor_on_for_s = self._bool_and_duration(
            cfg.controlled_switch_entity, now
        )
        heating_on, heating_off_for_s = self._bool_and_duration(cfg.heating_active_entity, now)

        model = self._load.model_for(sid)
        water_l = self._water_liters(cfg.water_total_entity)
        inputs = TickInputs(
            now_iso=now_iso,
            elapsed_s=elapsed_s,
            energy_counter_kwh=self._energy_kwh(cfg.delivered_energy_entity),
            water_counter_l=water_l,
            contactor_on=contactor_on,
            heating_on=heating_on,
            contactor_on_for_s=contactor_on_for_s,
            heating_off_for_s=heating_off_for_s,
            # Instantaneous count (the fallback estimate needs no history here).
            people_home=occupancy.count_people_home(self.hass, cfg.person_entities),
            e_base=model.e_base,
            e_draw_per_person=model.e_draw_per_person,
            empty_house_factor=model.empty_house_factor,
            # Display-only smoothing between the energy counter's coarse steps.
            rated_power_kw=cfg.rated_power_kw,
            # The daypart hot-fraction buckets are wall-clock buckets (evenings
            # are showers), so the model needs the *local* hour — ``now`` is UTC.
            local_hour=dt_util.as_local(now).hour,
            # The meter's own step time, so a slow meter's step is rated over
            # its real interval rather than the tick it happened to land in.
            water_changed_iso=(
                self._changed_iso(cfg.water_total_entity) if water_l is not None else None
            ),
        )

        tick = apply_tick(state, params, inputs)
        self._load.tanks[sid] = tick.state

        if cfg.tank_boost_soc_pct is not None and sid not in self._boost_in_flight:
            # The boost is a *control* decision, so it reads the same clamped raw
            # ledger the coordinator sizes the ask from — not the display curve.
            # While a boost push is in flight nothing is evaluated: another fire
            # would only queue a duplicate predict behind the coordinator lock.
            fire, boosted_state = should_boost(
                tick.state,
                model_soc(tick.state.deficit_kwh, tick.capacity_kwh),
                cfg.tank_boost_soc_pct,
                now_iso,
            )
            retry_at = self._boost_retry_at.get(sid)
            if fire and retry_at is not None and now < retry_at:
                # A recent push failed: don't re-plan before the retry time, and
                # commit nothing — ``boosted_state`` is the disarmed one, while
                # the tick's own state is still armed for the retry.
                pass
            elif not fire:
                # Store any re-arm so hysteresis persists across ticks.
                self._load.tanks[sid] = boosted_state
            else:
                _LOGGER.info(
                    "Tank charge for %s fell to %.0f%% (< %.0f%%); requesting an early re-plan",
                    sid,
                    tick.soc * 100.0,
                    cfg.tank_boost_soc_pct,
                )
                # The push (predict → recorder → scheduler) can outlast a tick, so
                # it runs as its own task: the tick loop keeps integrating, and
                # nothing is committed until the push reports back.
                self._boost_in_flight.add(sid)
                task = self.config_entry.async_create_task(
                    self.hass,
                    self._async_boost(sid, boosted_state.last_boost_iso),
                    name=f"{DOMAIN}_tank_boost_{sid}",
                )
                self._boost_tasks.add(task)
                task.add_done_callback(self._boost_tasks.discard)

        # Litres/showers are a *display* figure, so they follow the shown deficit
        # (the same number the SoC % is derived from) — not the control ledger.
        available_kwh = tick.capacity_kwh - tick.deficit_shown_kwh
        liters_40c = liters_at_temp(available_kwh, cfg.tank_cold_in_c)
        result = TankResult(
            soc_pct=round(tick.soc * 100.0, 1),
            deficit_kwh=tick.deficit_shown_kwh,
            deficit_raw_kwh=tick.state.deficit_kwh,
            uncertainty_kwh=tick.uncertainty_kwh,
            hysteresis_kwh=tick.state.hysteresis_kwh,
            capacity_kwh=tick.capacity_kwh,
            # The profile mean, plus the four per-daypart fractions behind it.
            hot_fraction=tick.state.hot_fraction,
            hot_fraction_profile=list(profile_of(tick.state)),
            standby_w=tick.state.standby_w,
            calibrated=tick.state.calibrated,
            latched=tick.latched,
            # The most recent 100 % anchor (incl. one that just fired this tick).
            last_full=tick.state.last_anchor_iso or None,
            draw_source=tick.draw_source,
            liters_40c=liters_40c,
            showers_left=showers_left(liters_40c),
        )
        return result, tick.anchored

    async def async_drain(self, timeout: float = 30.0) -> None:
        """Wait (bounded) for in-flight boost pushes — call after stopping the tick."""
        if self._boost_tasks:
            await asyncio.wait(set(self._boost_tasks), timeout=timeout)

    async def _async_boost(self, sid: str, fired_iso: str) -> None:
        """Run one boost re-plan and commit its bookkeeping only on success.

        Feedback #1 folds the measured deficit into the push. On success only
        the boost fields (disarm + the rate-limit stamp) are applied — onto the
        *latest* tank state, because ticks kept running during the await and may
        have anchored, learned or moved the counters. On failure nothing is
        committed (the trigger stays armed) and the retry waits
        :data:`BOOST_RETRY_INTERVAL` from *completion*, so a push that hung for
        a while doesn't retry the moment it gives up.
        """
        try:
            try:
                pushed = await self._load.async_predict_and_push(only=sid)
            except Exception:  # noqa: BLE001 - a failed push must not lose the trigger
                _LOGGER.exception("Boost re-plan failed for load %s", sid)
                pushed = False
            # ``None`` (an older signature) counts as success: it can't report
            # failure, and never disarming would re-plan every tick.
            if pushed is False:
                self._boost_retry_at[sid] = dt_util.utcnow() + BOOST_RETRY_INTERVAL
                _LOGGER.warning(
                    "Boost push for %s failed; will retry in %s", sid, BOOST_RETRY_INTERVAL
                )
                return
            self._boost_retry_at.pop(sid, None)
            latest = self._load.tanks.get(sid)
            if latest is not None:
                self._load.tanks[sid] = replace(latest, boost_armed=False, last_boost_iso=fired_iso)
                self._load.async_persist()
        finally:
            self._boost_in_flight.discard(sid)
