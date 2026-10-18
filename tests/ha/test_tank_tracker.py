"""The tank charge tracker: energy-balance ticks, anchoring, fallback, boost, and
the SoC → prediction feedback.

Setup follows ``test_deficit.py``: a load subentry with the tank fields, a mocked
``number.set_value`` for the scheduler push, and the ``freezer`` fixture for
deterministic time (``hass.states.async_set`` stamps ``last_changed`` at the
frozen time, so state durations are controllable). The tracker's own timer drives
the first case; the rest call ``async_tick()`` directly.
"""

from __future__ import annotations

import asyncio
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
_KWH = {"unit_of_measurement": "kWh"}

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

    hass.states.async_set(_ENERGY, "100.0", _KWH)
    hass.states.async_set(_WATER, "500.0", {"unit_of_measurement": "m³"})
    coord.tanks[sid] = replace(
        initial_state(_CAPACITY),
        last_tick_iso=dt_util.utcnow().isoformat(),
        energy_baseline_kwh=100.0,
    )

    await tracker.async_tick()  # settle baselines (no delta yet)
    soc0 = tracker.data[sid].soc_pct

    hass.states.async_set(_ENERGY, "101.5", _KWH)  # +1.5 kWh delivered
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

    # Sustain both past the 120 s / 60 s thresholds → the anchor *transition*
    # fires and pins the tank exactly full.
    freezer.tick(timedelta(seconds=200))
    await tracker.async_tick()
    r = tracker.data[sid]
    assert r.soc_pct == 100.0
    assert r.calibrated is True
    assert r.latched is True
    assert r.last_full is not None
    assert r.deficit_kwh == pytest.approx(0.0, abs=1e-9)

    # A following latched idle tick is *not* held at 100: the tank keeps losing
    # energy (standby + post-trip mixing) until the element re-engages.
    freezer.tick(timedelta(seconds=1800))
    await tracker.async_tick()
    after = tracker.data[sid]
    assert after.latched is True
    assert after.soc_pct < 100.0
    assert after.soc_pct > 90.0  # a slow drift, not a collapse
    assert after.deficit_kwh > 0.0

    # The v2 diagnostics ride along on every result.
    assert len(after.hot_fraction_profile) == 4
    assert after.deficit_raw_kwh >= 0.0
    assert after.uncertainty_kwh >= 0.05
    assert after.hysteresis_kwh > 0.0


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

    hass.states.async_set(_ENERGY, "100.0", _KWH)
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
    await hass.async_block_till_done()  # the boost push runs as its own task
    assert len(calls) == 1  # one predict/push fired by the boost
    # The push folds in the measured tank deficit → far above the ~105 min need.
    assert calls[-1].data["value"] > 105
    assert coord.tanks[sid].boost_armed is False  # disarmed by the fire

    # A second tick right away: still low, but disarmed + rate-limited → no re-push.
    await tracker.async_tick()
    await hass.async_block_till_done()
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


# 10 ─ counter units are converted, never guessed ─────────────────────────────


async def test_wh_energy_counter_normalised_to_kwh(hass: HomeAssistant, freezer) -> None:
    entry = await _setup(hass)
    tracker, coord = entry.runtime_data.tank, entry.runtime_data.load
    sid = _tank_sid(entry)

    hass.states.async_set(_ENERGY, "100000", {"unit_of_measurement": "Wh"})
    coord.tanks[sid] = replace(initial_state(_CAPACITY), last_tick_iso=dt_util.utcnow().isoformat())
    await tracker.async_tick()
    assert coord.tanks[sid].energy_baseline_kwh == pytest.approx(100.0)

    # +500 Wh is 0.5 kWh — not 500 kWh.
    hass.states.async_set(_ENERGY, "100500", {"unit_of_measurement": "Wh"})
    before = coord.tanks[sid].deficit_kwh
    freezer.tick(timedelta(seconds=60))
    await tracker.async_tick()
    assert coord.tanks[sid].energy_baseline_kwh == pytest.approx(100.5)
    assert before - coord.tanks[sid].deficit_kwh == pytest.approx(0.5, abs=0.05)


async def test_water_units_converted_liters_and_gallons(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    tracker = entry.runtime_data.tank
    hass.states.async_set(_WATER, "412.35", {"unit_of_measurement": "m³"})
    assert tracker._water_liters(_WATER) == pytest.approx(412_350.0)
    hass.states.async_set(_WATER, "1234", {"unit_of_measurement": "L"})
    assert tracker._water_liters(_WATER) == pytest.approx(1234.0)
    hass.states.async_set(_WATER, "1234", {"unit_of_measurement": "liters"})  # legacy alias
    assert tracker._water_liters(_WATER) == pytest.approx(1234.0)
    hass.states.async_set(_WATER, "10", {"unit_of_measurement": "gal"})
    assert tracker._water_liters(_WATER) == pytest.approx(37.854, abs=0.01)


async def test_unknown_unit_reads_unavailable_and_logs_once(
    hass: HomeAssistant, freezer, caplog
) -> None:
    entry = await _setup(hass)
    tracker, coord = entry.runtime_data.tank, entry.runtime_data.load
    sid = _tank_sid(entry)

    hass.states.async_set(_ENERGY, "100.0", {"unit_of_measurement": "bananas"})
    hass.states.async_set(_WATER, "500.0")  # no unit at all → not guessed as m³
    coord.tanks[sid] = replace(initial_state(_CAPACITY), last_tick_iso=dt_util.utcnow().isoformat())
    caplog.clear()
    await tracker.async_tick()
    freezer.tick(timedelta(seconds=60))
    await tracker.async_tick()

    assert coord.tanks[sid].energy_baseline_kwh is None
    assert coord.tanks[sid].water_baseline_l is None
    warnings = [r for r in caplog.records if "can't be converted" in r.getMessage()]
    assert len(warnings) == 2  # once per entity, not once per tick


# 11 ─ a reconfigured counter re-baselines instead of dumping its difference ───


async def test_changed_energy_entity_rebaselines(hass: HomeAssistant, freezer) -> None:
    entry = await _setup(hass)
    tracker, coord = entry.runtime_data.tank, entry.runtime_data.load
    sid = _tank_sid(entry)

    # The stored baseline came from an older counter at 100 kWh; the configured
    # one (_ENERGY) reads 8115 kWh.
    hass.states.async_set(_ENERGY, "8115.0", _KWH)
    coord.tanks[sid] = replace(
        initial_state(_CAPACITY),
        last_tick_iso=dt_util.utcnow().isoformat(),
        energy_baseline_kwh=100.0,
        energy_source="sensor.old_lvv_energy",
    )
    before = coord.tanks[sid].deficit_kwh
    await tracker.async_tick()
    after = coord.tanks[sid]
    assert after.energy_source == _ENERGY
    assert after.energy_baseline_kwh == pytest.approx(8115.0)
    assert after.deficit_kwh == pytest.approx(before, abs=0.1)  # not pinned full
    assert after.cycle_clean is False


async def test_legacy_state_adopts_configured_source_without_loss(
    hass: HomeAssistant, freezer
) -> None:
    entry = await _setup(hass)
    tracker, coord = entry.runtime_data.tank, entry.runtime_data.load
    sid = _tank_sid(entry)

    hass.states.async_set(_ENERGY, "101.0", _KWH)
    coord.tanks[sid] = replace(
        initial_state(_CAPACITY),
        last_tick_iso=dt_util.utcnow().isoformat(),
        energy_baseline_kwh=100.0,  # saved before sources were recorded
    )
    before = coord.tanks[sid].deficit_kwh
    await tracker.async_tick()
    after = coord.tanks[sid]
    assert after.energy_source == _ENERGY
    assert before - after.deficit_kwh == pytest.approx(1.0, abs=0.05)  # delta credited


# 12 ─ a failed boost push keeps the trigger, with bounded retries ─────────────


def _low_calibrated(hass: HomeAssistant, coord, sid: str) -> None:
    hass.states.async_set(_SWITCH, "off")
    hass.states.async_set(_LED, "off")
    coord.tanks[sid] = replace(
        initial_state(_CAPACITY), deficit_kwh=0.9 * _CAPACITY, calibrated=True
    )


async def test_failed_boost_push_stays_armed_and_backs_off(hass: HomeAssistant, freezer) -> None:
    entry = await _setup(hass)
    tracker, coord = entry.runtime_data.tank, entry.runtime_data.load
    sid = _tank_sid(entry)
    _low_calibrated(hass, coord, sid)

    push = AsyncMock(return_value=False)  # scheduler target unavailable
    with patch.object(coord, "async_predict_and_push", new=push):
        await tracker.async_tick()
        assert push.await_count == 1
        assert coord.tanks[sid].boost_armed is True  # trigger not consumed
        assert coord.tanks[sid].last_boost_iso == ""

        # Still low a minute later: no re-plan storm while backing off.
        freezer.tick(timedelta(seconds=60))
        await tracker.async_tick()
        assert push.await_count == 1
        assert coord.tanks[sid].boost_armed is True

        # After the retry interval the push is attempted again — and succeeds.
        push.return_value = True
        freezer.tick(timedelta(minutes=15))
        await tracker.async_tick()
        assert push.await_count == 2
        assert coord.tanks[sid].boost_armed is False
        assert coord.tanks[sid].last_boost_iso != ""


async def test_boost_push_raising_counts_as_failure(hass: HomeAssistant, freezer) -> None:
    entry = await _setup(hass)
    tracker, coord = entry.runtime_data.tank, entry.runtime_data.load
    sid = _tank_sid(entry)
    _low_calibrated(hass, coord, sid)

    push = AsyncMock(side_effect=RuntimeError("boom"))
    with patch.object(coord, "async_predict_and_push", new=push):
        await tracker.async_tick()
    assert push.await_count == 1
    assert coord.tanks[sid].boost_armed is True
    assert tracker.data[sid] is not None  # the tick still published


async def test_boost_push_returning_none_counts_as_success(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    tracker, coord = entry.runtime_data.tank, entry.runtime_data.load
    sid = _tank_sid(entry)
    _low_calibrated(hass, coord, sid)

    push = AsyncMock(return_value=None)  # an older signature that returns nothing
    with patch.object(coord, "async_predict_and_push", new=push):
        await tracker.async_tick()
        await tracker.async_tick()
    assert push.await_count == 1
    assert coord.tanks[sid].boost_armed is False


def _gated_push(result):
    """An ``async_predict_and_push`` stand-in that blocks until released."""
    gate = asyncio.Event()

    async def _push(only=None):
        await gate.wait()
        return result

    return gate, AsyncMock(side_effect=_push)


async def test_in_flight_boost_is_not_refired_and_commits_onto_latest_state(
    hass: HomeAssistant, freezer
) -> None:
    """A push slower than a tick: no duplicate predict, and no stale overwrite.

    The second tick must neither queue another predict nor lose its own updates
    when the first push finally succeeds — only the boost fields are applied, on
    top of whatever the tank state is by then.
    """
    entry = await _setup(hass)
    tracker, coord = entry.runtime_data.tank, entry.runtime_data.load
    sid = _tank_sid(entry)
    _low_calibrated(hass, coord, sid)
    hass.states.async_set(_ENERGY, "100.0", _KWH)
    coord.tanks[sid] = replace(coord.tanks[sid], energy_baseline_kwh=100.0)
    fired_at = dt_util.utcnow().isoformat()

    gate, push = _gated_push(True)
    with patch.object(coord, "async_predict_and_push", new=push):
        await tracker.async_tick()  # fires; the push is now blocked in flight
        assert push.await_count == 1
        assert coord.tanks[sid].boost_armed is True  # nothing committed yet

        hass.states.async_set(_ENERGY, "101.0", _KWH)  # the tank keeps integrating
        freezer.tick(timedelta(seconds=90))
        await tracker.async_tick()
        assert push.await_count == 1  # in flight → not re-evaluated
        assert coord.tanks[sid].energy_baseline_kwh == pytest.approx(101.0)

        gate.set()
        await hass.async_block_till_done()

    state = coord.tanks[sid]
    assert state.boost_armed is False
    assert state.last_boost_iso == fired_at
    assert state.energy_baseline_kwh == pytest.approx(101.0)  # tick 2 survived
    assert sid not in tracker._boost_in_flight


async def test_failed_boost_backoff_runs_from_completion(hass: HomeAssistant, freezer) -> None:
    """A push that hangs 20 min then fails must not retry the moment it gives up."""
    entry = await _setup(hass)
    tracker, coord = entry.runtime_data.tank, entry.runtime_data.load
    sid = _tank_sid(entry)
    _low_calibrated(hass, coord, sid)

    gate, push = _gated_push(False)
    with patch.object(coord, "async_predict_and_push", new=push):
        await tracker.async_tick()  # T0: fires, blocked
        freezer.tick(timedelta(minutes=20))
        gate.set()  # T0+20: fails
        await hass.async_block_till_done()
        assert coord.tanks[sid].boost_armed is True

        freezer.tick(timedelta(minutes=5))  # T0+25: past fire+15, not completion+15
        await tracker.async_tick()
        await hass.async_block_till_done()
        assert push.await_count == 1

        freezer.tick(timedelta(minutes=11))  # T0+36: completion+16 → retried
        await tracker.async_tick()
        await hass.async_block_till_done()
        assert push.await_count == 2


async def test_unload_drains_an_in_flight_boost_before_flushing(
    hass: HomeAssistant, freezer, hass_storage
) -> None:
    """A boost finishing mid-unload must still commit its cooldown to disk.

    Otherwise the reload restores the armed trigger and re-boosts at once,
    bypassing the rate limit.
    """
    entry = await _setup(hass)
    tracker, coord = entry.runtime_data.tank, entry.runtime_data.load
    sid = _tank_sid(entry)
    _low_calibrated(hass, coord, sid)

    gate, push = _gated_push(True)
    with patch.object(coord, "async_predict_and_push", new=push):
        await tracker.async_tick()  # fires; blocked in flight
        assert sid in tracker._boost_in_flight
        unload = hass.async_create_task(hass.config_entries.async_unload(entry.entry_id))
        await asyncio.sleep(0)
        assert not unload.done()  # waiting on the boost, not flushing yet
        gate.set()
        assert await unload

    saved = hass_storage[f"{DOMAIN}.{entry.entry_id}"]["data"]
    assert saved[sid]["tank"]["boost_armed"] is False  # the flush saw the cooldown


async def test_shadow_observations_never_change_control_and_feedback_survives_reload(
    hass: HomeAssistant, freezer
) -> None:
    """Real tracker/service path: a candidate correction cannot change control."""
    from homeassistant.exceptions import ServiceValidationError

    hass.states.async_set(_ENERGY, "100", _KWH)
    hass.states.async_set(_WATER, "1000", {"unit_of_measurement": "L"})
    hass.states.async_set(_SWITCH, "off")
    hass.states.async_set(_LED, "off")
    hass.states.async_set("binary_sensor.shower", "off")
    hass.states.async_set("sensor.humidity", "35", {"unit_of_measurement": "%"})
    base = {**_TANK_LOAD, "tank_boost_soc_pct": None}
    observed = {
        **base,
        "name": "Observed",
        "shower_motion_entity": "binary_sensor.shower",
        "shower_humidity_entity": "sensor.humidity",
    }
    entry = await _setup(hass, [base, observed])
    coord, tracker = entry.runtime_data.load, entry.runtime_data.tank
    plain_sid, observed_sid = list(entry.subentries)
    tracker.async_shutdown_ticker()  # This test explicitly drives each tick.
    for minute in range(1, 11):
        freezer.tick(timedelta(minutes=1))
        if 4 <= minute <= 6:
            hass.states.async_set(
                _WATER, str(1000 + (minute - 3) * 10), {"unit_of_measurement": "L"}
            )
            hass.states.async_set("binary_sensor.shower", "on")
            hass.states.async_set(
                "sensor.humidity", str(35 + (minute - 3) * 2), {"unit_of_measurement": "%"}
            )
        elif minute > 6:
            hass.states.async_set("binary_sensor.shower", "off")
        await hass.async_block_till_done()
        await tracker.async_tick()
        assert coord.tanks[plain_sid] == coord.tanks[observed_sid]
    events = coord.tank_observations[observed_sid]["events"]
    assert any(e["confidence"] == "confident" for e in events), [
        e["quality_reasons"] for e in events
    ]
    event = next(e for e in events if e["confidence"] == "confident")
    assert event["water_l"] == 30
    assert event["correction_kwh"] > 0
    control_before = coord.tanks[observed_sid]
    eid = _sensor_id(hass, observed_sid, "tank_soc")
    service_data = {"entity_id": eid, "event_id": event["id"], "comfort": "not_warm_enough"}
    with patch.object(coord, "async_predict_and_push", new=AsyncMock()) as push:
        await hass.services.async_call(DOMAIN, "report_shower_comfort", service_data, blocking=True)
        push.assert_not_called()
    assert coord.tanks[observed_sid] == control_before
    assert hass.states.get(eid).attributes["recent_showers"][0]["feedback"] == "not_warm_enough"
    assert hass.states.get(eid).attributes["hot_water_availability"] == "unknown"
    assert "showers_left" not in hass.states.get(eid).attributes
    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN,
            "report_shower_comfort",
            {**service_data, "event_id": "wrong-load-or-expired"},
            blocking=True,
        )
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    restored = entry.runtime_data.load.tank_observations[observed_sid]
    assert (
        next(e for e in restored["events"] if e["id"] == event["id"])["feedback"]
        == "not_warm_enough"
    )


async def test_short_motion_pulse_between_ticks_is_observed(hass: HomeAssistant, freezer) -> None:
    load = {**_TANK_LOAD, "shower_motion_entity": "binary_sensor.shower"}
    hass.states.async_set("binary_sensor.shower", "off")
    entry = await _setup(hass, [load])
    tracker = entry.runtime_data.tank
    hass.states.async_set("binary_sensor.shower", "on")
    await hass.async_block_till_done()
    freezer.tick(timedelta(seconds=10))
    hass.states.async_set("binary_sensor.shower", "off")
    await hass.async_block_till_done()
    assert "binary_sensor.shower" in tracker._motion_seen
    await tracker.async_tick()
    assert not tracker._motion_seen
    tracker.async_shutdown_ticker()
    assert tracker._unsub_motion is None


async def test_observation_failure_does_not_break_tank(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    tracker = entry.runtime_data.tank
    with patch(
        "custom_components.load_need_predictor.tank_tracker.advance_observation",
        side_effect=ValueError("bad diagnostic state"),
    ):
        await tracker.async_tick()
    result = tracker.data[_tank_sid(entry)]
    assert result.soc_pct is not None
    assert "Shower observations unavailable" in result.quality_warnings


def test_appliance_unknown_is_not_idle() -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    from custom_components.load_need_predictor.tank_tracker import TankTracker

    tracker = object.__new__(TankTracker)
    tracker.hass = SimpleNamespace(states=Mock())
    tracker.hass.states.get.return_value = SimpleNamespace(state="mystery")
    assert tracker._appliance_activity(("sensor.washer",)) is None
    tracker.hass.states.get.return_value = SimpleNamespace(state="running")
    assert tracker._appliance_activity(("sensor.washer",)) is True
    tracker.hass.states.get.return_value = SimpleNamespace(state="idle")
    assert tracker._appliance_activity(("sensor.washer",)) is False
    assert tracker._appliance_activity(()) is False


async def test_appliance_context_records_short_pulses_and_does_not_change_control(
    hass: HomeAssistant, freezer
) -> None:
    hass.states.async_set(_ENERGY, "100", _KWH)
    hass.states.async_set(_WATER, "1000", {"unit_of_measurement": "L"})
    hass.states.async_set(_SWITCH, "off")
    hass.states.async_set(_LED, "off")
    hass.states.async_set("sensor.washer", "power_off")
    hass.states.async_set("binary_sensor.bathroom", "off")
    base = {**_TANK_LOAD, "tank_boost_soc_pct": None}
    observed = {
        **base,
        "name": "Observed",
        "cold_appliance_entities": ["sensor.washer"],
        "appliance_household_activity_entities": ["binary_sensor.bathroom"],
    }
    entry = await _setup(hass, [base, observed])
    tracker, coord = entry.runtime_data.tank, entry.runtime_data.load
    plain_sid, observed_sid = list(entry.subentries)
    hass.states.async_set("sensor.washer", "detecting")
    hass.states.async_set("binary_sensor.bathroom", "on")
    await hass.async_block_till_done()
    freezer.tick(timedelta(seconds=10))
    hass.states.async_set("binary_sensor.bathroom", "off")
    await hass.async_block_till_done()
    cfg = tracker.tank_configs()[observed_sid]
    assert tracker._household_activity(cfg, dt_util.utcnow()) is True
    assert "sensor.washer" in tracker._appliance_changed
    for minute in range(1, 5):
        freezer.tick(timedelta(seconds=20))
        hass.states.async_set(_WATER, str(1000 + 4 * minute), {"unit_of_measurement": "L"})
        await tracker.async_tick()
        assert coord.tanks[plain_sid] == coord.tanks[observed_sid]
    # Dropping offline mid-run is a lost connection: unknown, not idle — and it
    # outlives the tick that saw it.
    hass.states.async_set("sensor.washer", "unavailable")
    await hass.async_block_till_done()
    await tracker.async_tick()
    await tracker.async_tick()
    assert tracker._appliance_activity(("sensor.washer",)) is None
    assert not any("sensor.washer" in item for item in tracker.data[observed_sid].quality_warnings)
    # Once it reports idle, a later drop offline is a power-off.
    hass.states.async_set("sensor.washer", "power_off")
    await hass.async_block_till_done()
    assert tracker._appliance_activity(("sensor.washer",)) is False
    hass.states.async_set("sensor.washer", "unavailable")
    await hass.async_block_till_done()
    await tracker.async_tick()
    assert tracker._appliance_activity(("sensor.washer",)) is False
    # A restored state means the integration isn't loaded: unknown and warned.
    hass.states.async_set("sensor.washer", "unavailable", {"restored": True})
    await hass.async_block_till_done()
    await tracker.async_tick()
    assert tracker._appliance_activity(("sensor.washer",)) is None
    assert any("sensor.washer" in item for item in tracker.data[observed_sid].quality_warnings)


async def test_appliance_offline_since_startup_is_idle(hass: HomeAssistant) -> None:
    """An LG washer left off is ``unavailable`` from HA start until powered on."""
    from custom_components.load_need_predictor.tank_tracker import TankTracker

    tracker = object.__new__(TankTracker)
    tracker.hass = hass
    tracker._appliance_lost_mid_run = set()
    hass.states.async_set("sensor.washer", "unavailable")
    assert tracker._appliance_activity(("sensor.washer",)) is False


async def test_bad_context_units_never_corroborate_shower(hass: HomeAssistant) -> None:
    from custom_components.load_need_predictor.tank_tracker import TankTracker

    tracker = object.__new__(TankTracker)
    tracker.hass = hass
    hass.states.async_set("sensor.humidity", "50", {"unit_of_measurement": "g/m³"})
    hass.states.async_set("sensor.supply", "12", {"unit_of_measurement": "bananas"})
    assert tracker._number("sensor.humidity", humidity=True) is None
    assert tracker._number("sensor.supply", temperature=True) is None
    hass.states.async_set("sensor.humidity", "50", {"unit_of_measurement": "%"})
    assert tracker._number("sensor.humidity", humidity=True) == 50
