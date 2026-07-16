"""HA-side tank charge tracker: a self-ticking coordinator.

The pure energy balance lives in :mod:`tank_model`; this is the thin Home
Assistant shell that feeds it. Once per :data:`TANK_TICK_SECONDS` it reads each
tank-load's counters/meter/binary states straight from ``hass.states`` (never the
recorder — the tank math is purely instantaneous), unit-normalises the water
meter, computes sustained-state durations from ``last_changed``, calls
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

**SoC → prediction feedback #2 (low-charge boost).** After each tick, if a
calibrated tank's SoC has fallen below the load's boost threshold, the tracker
re-runs the load's predict/push (which, via feedback #1 in the coordinator, folds
the measured deficit into a bigger target) so the scheduler books more heating
before the tank empties. Hysteresis + a rate limit live in the pure model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

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
    should_boost,
    showers_left,
)

_LOGGER = logging.getLogger(__name__)

TICK_INTERVAL = timedelta(seconds=TANK_TICK_SECONDS)
# Persist at least this often even without an anchor/learn/boost, so a long quiet
# stretch of drift still survives a restart within ~15 minutes.
PERSIST_EVERY_TICKS = 15

# Water-meter unit strings already in litres; anything else (the OCR meter reports
# m³) is scaled up. Only litres and m³ are expected in practice.
_LITER_UNITS = frozenset({"L", "l", "liter", "liters", "Liter"})
_LITERS_PER_M3 = 1000.0

_UNKNOWN_STATES = ("unknown", "unavailable")


@dataclass
class TankResult:
    """What the tank charge sensor publishes for one load."""

    soc_pct: float
    deficit_kwh: float
    capacity_kwh: float
    hot_fraction: float
    standby_w: float
    calibrated: bool
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

    # ── config access ──────────────────────────────────────────────────────────

    def tank_configs(self) -> dict[str, LoadConfig]:
        """Loads that opted into tank tracking (``heating_active_entity`` set)."""
        return {
            sid: cfg for sid, cfg in self._load.load_configs().items() if cfg.heating_active_entity
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

    def _float_state(self, entity_id: str | None) -> float | None:
        """Current numeric state of an entity, or ``None`` if missing/unparseable."""
        if not entity_id:
            return None
        state = self.hass.states.get(entity_id)
        if state is None or state.state in _UNKNOWN_STATES:
            return None
        try:
            return float(state.state)
        except (TypeError, ValueError):
            return None

    def _water_liters(self, entity_id: str | None) -> float | None:
        """Cold-meter reading normalised to litres (m³ → ×1000), or ``None``."""
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
        if unit in _LITER_UNITS:
            return value
        # Assume m³ (the OCR water meter's native unit) → litres.
        return value * _LITERS_PER_M3

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

        ``save_now`` is True when the tick anchored (which is also the only time
        the model learns) or fired a boost — the events worth persisting promptly.
        """
        params = TankParams(cfg.tank_volume_l, cfg.tank_setpoint_c, cfg.tank_cold_in_c)
        capacity = capacity_kwh(cfg.tank_volume_l, cfg.tank_setpoint_c, cfg.tank_cold_in_c)
        state = self._load.tanks.get(sid) or initial_state(capacity)

        # Elapsed since the last tick — restart reconciliation is just this same
        # arithmetic over the (possibly long) gap since the persisted timestamp.
        last = dt_util.parse_datetime(state.last_tick_iso) if state.last_tick_iso else None
        elapsed_s = (now - last).total_seconds() if last is not None else float(TANK_TICK_SECONDS)

        contactor_on, contactor_on_for_s = self._bool_and_duration(
            cfg.controlled_switch_entity, now
        )
        heating_on, heating_off_for_s = self._bool_and_duration(cfg.heating_active_entity, now)

        model = self._load.model_for(sid)
        inputs = TickInputs(
            now_iso=now_iso,
            elapsed_s=elapsed_s,
            energy_counter_kwh=self._float_state(cfg.delivered_energy_entity),
            water_counter_l=self._water_liters(cfg.water_total_entity),
            contactor_on=contactor_on,
            heating_on=heating_on,
            contactor_on_for_s=contactor_on_for_s,
            heating_off_for_s=heating_off_for_s,
            # Instantaneous count (the fallback estimate needs no history here).
            people_home=occupancy.count_people_home(self.hass, cfg.person_entities),
            e_base=model.e_base,
            e_draw_per_person=model.e_draw_per_person,
            empty_house_factor=model.empty_house_factor,
        )

        tick = apply_tick(state, params, inputs)
        self._load.tanks[sid] = tick.state

        boosted = False
        if cfg.tank_boost_soc_pct is not None:
            fire, boosted_state = should_boost(
                tick.state, tick.soc, cfg.tank_boost_soc_pct, now_iso
            )
            # Always store the re-armed/disarmed state so hysteresis + the rate
            # limit persist across ticks, whether or not this one fires.
            self._load.tanks[sid] = boosted_state
            if fire:
                boosted = True
                _LOGGER.info(
                    "Tank charge for %s fell to %.0f%% (< %.0f%%); requesting an early re-plan",
                    sid,
                    tick.soc * 100.0,
                    cfg.tank_boost_soc_pct,
                )
                # Re-runs predict/push; feedback #1 folds the measured deficit in.
                await self._load.async_predict_and_push(only=sid)

        available_kwh = tick.capacity_kwh - tick.state.deficit_kwh
        liters_40c = liters_at_temp(available_kwh, cfg.tank_cold_in_c)
        result = TankResult(
            soc_pct=round(tick.soc * 100.0, 1),
            deficit_kwh=tick.state.deficit_kwh,
            capacity_kwh=tick.capacity_kwh,
            hot_fraction=tick.state.hot_fraction,
            standby_w=tick.state.standby_w,
            calibrated=tick.state.calibrated,
            # The most recent 100 % anchor (incl. one that just fired this tick).
            last_full=tick.state.last_anchor_iso or None,
            draw_source=tick.draw_source,
            liters_40c=liters_40c,
            showers_left=showers_left(liters_40c),
        )
        return result, (tick.anchored or boosted)
