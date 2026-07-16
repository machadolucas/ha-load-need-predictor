"""The tank charge tracker: energy-balance ticks, anchoring, fallback, boost, and
the SoC → prediction feedback.

Setup follows ``test_deficit.py``: a load subentry with the tank fields, a mocked
``number.set_value`` for the scheduler push, and the ``freezer`` fixture for
deterministic time (``hass.states.async_set`` stamps ``last_changed`` at the
frozen time, so state durations are controllable). The tracker's own timer drives
the first case; the rest call ``async_tick()`` directly.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
    async_mock_service,
)

from custom_components.load_need_predictor.const import DOMAIN, SUBENTRY_TYPE_LOAD
from custom_components.load_need_predictor.tank_model import (
    KWH_PER_LITER_KELVIN,
    capacity_kwh,
    initial_state,
)

_STATS = "custom_components.load_need_predictor.coordinator.async_daily_delivered_kwh"
_CMD = "custom_components.load_need_predictor.coordinator.async_commanded_minutes"

_ENERGY = "sensor.lvv_energy"
_SWITCH = "switch.lvv"
_LED = "binary_sensor.led"
_WATER = "sensor.water"
_TARGET = "number.lvv_target"

_TANK_LOAD = {
    "name": "LVV",
    "target_number_entity": _TARGET,
    "delivered_energy_entity": _ENERGY,
    "person_entities": ["person.a"],
    "rated_power_kw": 3.0,
    "controlled_switch_entity": _SWITCH,
    "heating_active_entity": _LED,
    "water_total_entity": _WATER,
    "tank_volume_l": 300,
    "tank_setpoint_c": 75,
    "tank_cold_in_c": 12,
    "tank_boost_soc_pct": 20,
}

_PLAIN_LOAD = {
    "name": "Other",
    "target_number_entity": "number.other_target",
    "delivered_energy_entity": "sensor.other_energy",
    "person_entities": ["person.a"],
    "rated_power_kw": 3.0,
}

_CAPACITY = capacity_kwh(300, 75, 12)  # ≈ 21.98 kWh (ΔT 63 K)


async def _setup(hass: HomeAssistant, loads: list[dict] | None = None) -> MockConfigEntry:
    hass.states.async_set("person.a", "home")
    hass.states.async_set(_TARGET, "0")
    subs = loads if loads is not None else [_TANK_LOAD]
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"name": "Predictor", "predict_time": "14:00:00", "capture_time": "23:55:00"},
        subentries_data=[
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_LOAD, title=d["name"], unique_id=None, data=d
            )
            for d in subs
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


def _tank_sid(entry: MockConfigEntry) -> str:
    """Subentry id of the (first) load that opted into tank tracking."""
    return next(s for s, sub in entry.subentries.items() if sub.data.get("heating_active_entity"))


def _sensor_id(hass: HomeAssistant, sid: str, key: str) -> str | None:
    return er.async_get(hass).async_get_entity_id("sensor", DOMAIN, f"{sid}_{key}")


def _is_number(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


# 1 ─ the tracker's own timer drives a tick ───────────────────────────────────


async def test_timer_registers_and_drives_tick(hass: HomeAssistant, freezer) -> None:
    entry = await _setup(hass)
    tracker = entry.runtime_data.tank
    sid = _tank_sid(entry)

    # The backgrounded initial tick already seeded the sensor.
    eid = _sensor_id(hass, sid, "tank_soc")
    assert eid is not None
    assert _is_number(hass.states.get(eid).state)

    ticks_before = tracker._ticks
    freezer.tick(timedelta(seconds=61))
    async_fire_time_changed(hass)
    await hass.async_block_till_done()

    assert tracker._ticks > ticks_before  # the interval fired a tick
    assert _is_number(hass.states.get(eid).state)


# 2 ─ delivered energy raises SoC ──────────────────────────────────────────────


async def test_energy_in_raises_soc(hass: HomeAssistant, freezer) -> None:
    entry = await _setup(hass)
    tracker, coord = entry.runtime_data.tank, entry.runtime_data.load
    sid = _tank_sid(entry)

    hass.states.async_set(_ENERGY, "100.0")
    hass.states.async_set(_WATER, "500.0", {"unit_of_measurement": "m³"})
    coord.tanks[sid] = replace(
        initial_state(_CAPACITY),
        last_tick_iso=dt_util.utcnow().isoformat(),
        energy_baseline_kwh=100.0,
    )

    await tracker.async_tick()  # settle baselines (no delta yet)
    soc0 = tracker.data[sid].soc_pct

    hass.states.async_set(_ENERGY, "101.5")  # +1.5 kWh delivered
    freezer.tick(timedelta(seconds=60))
    await tracker.async_tick()
    r = tracker.data[sid]

    assert r.soc_pct > soc0
    assert (r.soc_pct - soc0) == pytest.approx(1.5 / _CAPACITY * 100.0, abs=1.0)
    # SoC and its reported deficit/capacity agree.
    assert r.capacity_kwh == pytest.approx(_CAPACITY, abs=0.01)
    assert r.soc_pct == pytest.approx((1.0 - r.deficit_kwh / r.capacity_kwh) * 100.0, abs=0.1)


# 3 ─ the 100 % anchor requires sustained states ──────────────────────────────


async def test_anchor_pins_full_after_sustained_idle(hass: HomeAssistant, freezer) -> None:
    entry = await _setup(hass)
    tracker, coord = entry.runtime_data.tank, entry.runtime_data.load
    sid = _tank_sid(entry)
    coord.tanks[sid] = replace(initial_state(_CAPACITY), last_tick_iso=dt_util.utcnow().isoformat())

    # Commanded on + element idle, both freshly set → durations below thresholds.
    hass.states.async_set(_SWITCH, "on")
    hass.states.async_set(_LED, "off")
    await tracker.async_tick()
    assert tracker.data[sid].soc_pct != 100.0
    assert tracker.data[sid].calibrated is False

    # Sustain both past the 120 s / 60 s thresholds → anchor fires.
    freezer.tick(timedelta(seconds=200))
    await tracker.async_tick()
    r = tracker.data[sid]
    assert r.soc_pct == 100.0
    assert r.calibrated is True
    assert r.last_full is not None


async def test_unavailable_heating_never_anchors(hass: HomeAssistant, freezer) -> None:
    entry = await _setup(hass)
    tracker, coord = entry.runtime_data.tank, entry.runtime_data.load
    sid = _tank_sid(entry)
    coord.tanks[sid] = replace(initial_state(_CAPACITY), last_tick_iso=dt_util.utcnow().isoformat())

    hass.states.async_set(_SWITCH, "on")
    hass.states.async_set(_LED, "unavailable")  # tri-state None → fails closed
    freezer.tick(timedelta(seconds=300))
    await tracker.async_tick()

    r = tracker.data[sid]
    assert r.soc_pct != 100.0
    assert r.calibrated is False


# 4 ─ a stale meter falls back to the occupancy estimate ──────────────────────


async def test_stale_meter_uses_fallback(hass: HomeAssistant, freezer) -> None:
    entry = await _setup(hass)
    tracker, coord = entry.runtime_data.tank, entry.runtime_data.load
    sid = _tank_sid(entry)

    hass.states.async_set(_WATER, "500.0", {"unit_of_measurement": "m³"})
    coord.tanks[sid] = replace(initial_state(_CAPACITY), last_tick_iso=dt_util.utcnow().isoformat())
    await tracker.async_tick()  # establishes the water baseline at T0

    hass.states.async_set(_WATER, "unavailable")
    freezer.tick(timedelta(seconds=1000))  # > WATER_STALE_AFTER_S (900)
    await tracker.async_tick()

    assert tracker.data[sid].draw_source == "fallback"


# 5 ─ the m³ water meter is normalised to litres ──────────────────────────────


async def test_m3_meter_normalised_to_liters(hass: HomeAssistant, freezer) -> None:
    entry = await _setup(hass)
    tracker, coord = entry.runtime_data.tank, entry.runtime_data.load
    sid = _tank_sid(entry)

    hass.states.async_set(_ENERGY, "100.0")
    hass.states.async_set(_WATER, "412.35", {"unit_of_measurement": "m³"})
    coord.tanks[sid] = replace(
        initial_state(_CAPACITY),
        last_tick_iso=dt_util.utcnow().isoformat(),
        energy_baseline_kwh=100.0,  # no energy delivered → energy_in stays 0
    )
    await tracker.async_tick()  # water baseline = 412 350 L
    deficit0 = tracker.data[sid].deficit_kwh

    hass.states.async_set(_WATER, "412.36", {"unit_of_measurement": "m³"})  # +0.010 m³ = 10 L
    # A 2-minute span keeps the 8 L/min hot-flow cap above the 10 L draw.
    freezer.tick(timedelta(seconds=120))
    await tracker.async_tick()
    deficit1 = tracker.data[sid].deficit_kwh

    expected_draw = 10.0 * 0.25 * 63.0 * KWH_PER_LITER_KELVIN  # ≈ 0.183 kWh
    assert (deficit1 - deficit0) == pytest.approx(expected_draw, abs=0.01)
    assert tracker.data[sid].draw_source == "meter"


# 6 ─ opt-in: the sensor only exists for a tank-configured load ───────────────


async def test_opt_in_and_shared_device(hass: HomeAssistant) -> None:
    entry = await _setup(hass, loads=[_TANK_LOAD, _PLAIN_LOAD])
    ent_reg = er.async_get(hass)
    tank_sid = _tank_sid(entry)
    plain_sid = next(
        s for s, sub in entry.subentries.items() if not sub.data.get("heating_active_entity")
    )

    # The plain load gets no tank sensor.
    assert _sensor_id(hass, plain_sid, "tank_soc") is None

    # The tank load's tank sensor shares the device of its runtime sensor.
    tank_eid = _sensor_id(hass, tank_sid, "tank_soc")
    runtime_eid = _sensor_id(hass, tank_sid, "predicted_runtime")
    assert tank_eid is not None and runtime_eid is not None
    assert ent_reg.async_get(tank_eid).device_id == ent_reg.async_get(runtime_eid).device_id


# 7 ─ a hub with no tank-configured load registers no timer ───────────────────


async def test_no_tank_hub_registers_no_timer(hass: HomeAssistant) -> None:
    entry = await _setup(hass, loads=[_PLAIN_LOAD])
    tracker = entry.runtime_data.tank
    assert tracker.has_tanks is False
    assert tracker._unsub is None


# 8 ─ a low calibrated charge boosts the scheduler push ───────────────────────


async def test_low_charge_boost_pushes_once(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    tracker, coord = entry.runtime_data.tank, entry.runtime_data.load
    sid = _tank_sid(entry)
    calls = async_mock_service(hass, "number", "set_value")

    # Both booleans off → no anchor to reset the seeded deficit.
    hass.states.async_set(_SWITCH, "off")
    hass.states.async_set(_LED, "off")
    # Calibrated tank, ~10 % SoC (deficit 0.9 × capacity) → below the 20 % default.
    coord.tanks[sid] = replace(
        initial_state(_CAPACITY), deficit_kwh=0.9 * _CAPACITY, calibrated=True
    )

    await tracker.async_tick()
    assert len(calls) == 1  # one predict/push fired by the boost
    # The push folds in the measured tank deficit → far above the ~105 min need.
    assert calls[-1].data["value"] > 105
    assert coord.tanks[sid].boost_armed is False  # disarmed by the fire

    # A second tick right away: still low, but disarmed + rate-limited → no re-push.
    await tracker.async_tick()
    assert len(calls) == 1


# 9 ─ SoC → prediction feedback #1 (calibrated tank overrides the backlog) ─────
#    (the two feedback cases live here, not duplicated in test_deficit.py)


async def test_predict_uses_calibrated_tank_deficit(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    coord = entry.runtime_data.load
    sid = _tank_sid(entry)
    calls = async_mock_service(hass, "number", "set_value")

    # deficit 3.0 kWh at 3 kW → 60 min; added to the ~104 min need → 165 (15-step).
    coord.tanks[sid] = replace(initial_state(_CAPACITY), deficit_kwh=3.0, calibrated=True)
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        await coord.async_predict_and_push()

    assert calls[-1].data["value"] == 165
    row = coord.training[sid][-1]
    assert row["deficit_source"] == "tank"


async def test_predict_uses_commanded_when_tank_uncalibrated(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    coord = entry.runtime_data.load
    sid = _tank_sid(entry)
    calls = async_mock_service(hass, "number", "set_value")

    # An uncalibrated tank is ignored → the commanded-minutes bookkeeping (0 here)
    # leaves the plain need on the wire.
    coord.tanks[sid] = replace(initial_state(_CAPACITY), deficit_kwh=3.0, calibrated=False)
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        await coord.async_predict_and_push()

    assert calls[-1].data["value"] == 105
    row = coord.training[sid][-1]
    assert row["deficit_source"] == "commanded"


async def test_gain_not_learned_on_tank_refill_day(hass: HomeAssistant) -> None:
    """A tank-driven backlog day delivers refill + demand — it must not teach.

    The bookkept deficit stays 0 when the tank overrides, so without the row-level
    gate the day would look clean and the extra refill energy would inflate the
    gain.
    """
    entry = await _setup(hass)
    coord = entry.runtime_data.load
    sid = _tank_sid(entry)
    async_mock_service(hass, "number", "set_value")

    # Calibrated tank 3 kWh low → 60 min folded into the push (≥ the 30 min gate).
    coord.tanks[sid] = replace(initial_state(_CAPACITY), deficit_kwh=3.0, calibrated=True)
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        await coord.async_predict_and_push()
    assert coord.training[sid][-1]["deficit_source"] == "tank"
    assert coord.models[sid].deficit_minutes == 0  # bookkept fallback untouched

    # Full delivery of the boosted ask: demand (~5.2 kWh) + refill (3 kWh).
    before = coord.models[sid].gain
    with (
        patch(_STATS, new=AsyncMock(return_value=8.2)),
        patch(_CMD, new=AsyncMock(return_value=165.0)),
    ):
        await coord.async_capture_and_log()

    row = coord.training[sid][-1]
    assert row["clean_cycle"] is False
    assert coord.models[sid].sample_count == 0
    assert coord.models[sid].gain == before
