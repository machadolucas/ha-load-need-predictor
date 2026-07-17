"""Unit tests for the pure tank state-of-charge model (no Home Assistant).

Loaded standalone via ``importlib`` so these run without importing the HA-bound
package. Mirrors the loader in ``test_predictor.py``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "custom_components"
    / "load_need_predictor"
    / "tank_model.py"
)
_spec = importlib.util.spec_from_file_location("lnp_tank_model", _PATH)
tank = importlib.util.module_from_spec(_spec)
sys.modules["lnp_tank_model"] = tank
_spec.loader.exec_module(tank)

TankParams = tank.TankParams
TankState = tank.TankState
TickInputs = tank.TickInputs

PARAMS = TankParams(volume_l=300.0, setpoint_c=75.0, cold_in_c=12.0)
CAP = tank.capacity_kwh(300.0, 75.0, 12.0)
K = tank.KWH_PER_LITER_KELVIN


def _state(deficit_kwh: float = 10.0, **over) -> TankState:
    return TankState(deficit_kwh=deficit_kwh, **over)


def _inputs(**over) -> TickInputs:
    base = dict(
        now_iso="2026-07-16T10:00:00+03:00",
        elapsed_s=60.0,
        energy_counter_kwh=None,
        water_counter_l=None,
        contactor_on=None,
        heating_on=None,
        contactor_on_for_s=None,
        heating_off_for_s=None,
        people_home=1,
        e_base=3.0,
        e_draw_per_person=2.2,
        empty_house_factor=0.4,
    )
    base.update(over)
    return TickInputs(**base)


# ── capacity_kwh ──────────────────────────────────────────────────────────────


def test_capacity_matches_heat_time_crosscheck():
    # 300 L, 75 °C setpoint, 12 °C inlet → ~22 kWh (the user's 7–8 h × 3 kW check).
    assert tank.capacity_kwh(300.0, 75.0, 12.0) == pytest.approx(22.0, abs=0.05)
    assert tank.capacity_kwh(300.0, 75.0, 12.0) == pytest.approx(300 * (4.186 / 3600) * 63)


def test_capacity_lower_setpoint():
    # 75 / 22 → ΔT 53 K ≈ 18.49 kWh.
    assert tank.capacity_kwh(300.0, 75.0, 22.0) == pytest.approx(18.49, abs=0.02)


def test_capacity_delta_t_floored():
    # Inverted config (cold ≥ setpoint) → ΔT floored at 1 K, never zero/negative.
    assert tank.capacity_kwh(300.0, 20.0, 25.0) == pytest.approx(300 * K * 1.0)


# ── soc ───────────────────────────────────────────────────────────────────────


def test_soc_clamps_and_midpoint():
    assert tank.soc(0.0, 22.0) == 1.0
    assert tank.soc(22.0, 22.0) == 0.0
    assert tank.soc(11.0, 22.0) == pytest.approx(0.5)
    assert tank.soc(30.0, 22.0) == 0.0  # over-empty clamps to 0
    assert tank.soc(-5.0, 22.0) == 1.0  # over-full clamps to 1


def test_soc_nonpositive_capacity_guard():
    assert tank.soc(5.0, 0.0) == 0.0
    assert tank.soc(5.0, -1.0) == 0.0


# ── initial_state ─────────────────────────────────────────────────────────────


def test_initial_state_half_full_uncalibrated():
    st = tank.initial_state(CAP)
    assert st.deficit_kwh == pytest.approx(0.5 * CAP)
    assert tank.soc(st.deficit_kwh, CAP) == pytest.approx(0.5)
    assert st.calibrated is False
    assert st.hot_fraction == tank.SEED_HOT_FRACTION
    assert st.standby_w == tank.SEED_STANDBY_W
    assert st.boost_armed is True
    assert st.version == "v1"


# ── counter_delta ─────────────────────────────────────────────────────────────


def test_counter_delta_normal():
    assert tank.counter_delta(100.0, 150.0) == (50.0, 150.0)


def test_counter_delta_none_reading_keeps_baseline():
    assert tank.counter_delta(100.0, None) == (0.0, 100.0)


def test_counter_delta_none_baseline_adopts_reading():
    assert tank.counter_delta(None, 100.0) == (0.0, 100.0)


def test_counter_delta_reset_rebaselines():
    # A decrease (counter reset / meter swap) → no delta, re-baseline to reading.
    assert tank.counter_delta(150.0, 100.0) == (0.0, 100.0)


def test_counter_delta_gap_recovery_is_lossless():
    # A big cumulative jump across a restart gap is the honest delta.
    assert tank.counter_delta(1000.0, 1042.0) == (42.0, 1042.0)


# ── water_delta ───────────────────────────────────────────────────────────────


def test_water_delta_normal():
    assert tank.water_delta(1000.0, 1010.0, 10.0) == (10.0, 1010.0)


def test_water_delta_negative_is_misread():
    assert tank.water_delta(1000.0, 990.0, 10.0) == (None, 990.0)


def test_water_delta_too_fast_over_short_span():
    # 100 L in 2 min ⇒ 50 L/min > 30 L/min cap → misread, re-baseline.
    assert tank.water_delta(1000.0, 1100.0, 2.0) == (None, 1100.0)


def test_water_delta_same_delta_over_long_restart_span_passes():
    # The identical 100 L delta accumulated over an 8 h gap is plausible → passes.
    assert tank.water_delta(1000.0, 1100.0, 480.0) == (100.0, 1100.0)


def test_water_delta_none_reading_keeps_baseline():
    assert tank.water_delta(1000.0, None, 10.0) == (None, 1000.0)


def test_water_delta_none_baseline_adopts_reading():
    assert tank.water_delta(None, 1000.0, 10.0) == (None, 1000.0)


# ── hot_attributable_liters ───────────────────────────────────────────────────


def test_hot_attributable_shower_rate_passes():
    # 40 L over 10 min = 4 L/min ≤ 8 L/min cap → untouched.
    assert tank.hot_attributable_liters(40.0, 10.0) == 40.0


def test_hot_attributable_garden_hose_rate_caps():
    # 200 L over 10 min → capped at 8 × 10 = 80 L (excess is cold-only).
    assert tank.hot_attributable_liters(200.0, 10.0) == 80.0


def test_hot_attributable_span_floored():
    # Zero span still allows a 1-minute window of hot flow.
    assert tank.hot_attributable_liters(5.0, 0.0) == 5.0
    assert tank.hot_attributable_liters(20.0, 0.0) == 8.0


# ── draw_kwh_from_liters ──────────────────────────────────────────────────────


def test_draw_kwh_from_liters():
    # 100 L × 0.25 hot × 63 K × c.
    assert tank.draw_kwh_from_liters(100.0, 0.25, 75.0, 12.0) == pytest.approx(
        100.0 * 0.25 * 63.0 * K
    )


def test_draw_kwh_from_liters_delta_t_floored():
    assert tank.draw_kwh_from_liters(100.0, 0.25, 20.0, 25.0) == pytest.approx(
        100.0 * 0.25 * 1.0 * K
    )


# ── fallback_draw_kwh ─────────────────────────────────────────────────────────


def test_fallback_draw_people_none_assumes_one():
    # people None → 1 present: 1×(3.0 + 2.2) − 1.68 standby = 3.52 kWh/day.
    assert tank.fallback_draw_kwh(3.0, 2.2, 0.4, None, 70.0, 1440.0) == pytest.approx(3.52)


def test_fallback_draw_empty_house_uses_factor():
    # Nobody home → base scaled by 0.4 (10 high enough to survive the standby sub).
    daily = 0.4 * 10.0 - 70.0 * 24.0 / 1000.0
    assert tank.fallback_draw_kwh(10.0, 2.2, 0.4, 0, 70.0, 1440.0) == pytest.approx(daily)


def test_fallback_draw_floored_at_zero():
    # Empty house at the seeds → 0.4×3.0 − 1.68 < 0 → floored to 0.
    assert tank.fallback_draw_kwh(3.0, 2.2, 0.4, 0, 70.0, 1440.0) == 0.0


def test_fallback_draw_prorated_to_elapsed():
    full = tank.fallback_draw_kwh(3.0, 2.2, 0.4, 1, 70.0, 1440.0)
    hour = tank.fallback_draw_kwh(3.0, 2.2, 0.4, 1, 70.0, 60.0)
    assert hour == pytest.approx(full / 24.0)


# ── standby_kwh ───────────────────────────────────────────────────────────────


def test_standby_kwh():
    assert tank.standby_kwh(70.0, 1440.0) == pytest.approx(1.68)
    assert tank.standby_kwh(70.0, 60.0) == pytest.approx(0.07)


# ── should_anchor (truth table) ───────────────────────────────────────────────


def test_should_anchor_all_conditions_met():
    assert tank.should_anchor(True, False, 120.0, 60.0) is True


def test_should_anchor_contactor_duration_boundary():
    assert tank.should_anchor(True, False, 119.0, 60.0) is False
    assert tank.should_anchor(True, False, 120.0, 60.0) is True


def test_should_anchor_heating_off_duration_boundary():
    assert tank.should_anchor(True, False, 120.0, 59.0) is False
    assert tank.should_anchor(True, False, 120.0, 60.0) is True


def test_should_anchor_contactor_off():
    assert tank.should_anchor(False, False, 120.0, 60.0) is False


def test_should_anchor_element_still_heating():
    assert tank.should_anchor(True, True, 120.0, 60.0) is False


def test_should_anchor_none_states_fail_closed():
    assert tank.should_anchor(None, False, 120.0, 60.0) is False
    assert tank.should_anchor(True, None, 120.0, 60.0) is False


def test_should_anchor_none_durations_fail_closed():
    assert tank.should_anchor(True, False, None, 60.0) is False
    assert tank.should_anchor(True, False, 120.0, None) is False


# ── apply_tick: integrating deficit ───────────────────────────────────────────


def test_apply_tick_energy_in_shrinks_deficit():
    st = _state(
        deficit_kwh=10.0,
        energy_baseline_kwh=100.0,
        water_baseline_l=500.0,
        water_baseline_iso="2026-07-16T09:00:00+03:00",
        calibrated=True,
    )
    # 60 min, 2 kWh in, zero water draw → deficit − 2 + standby(0.07).
    inp = _inputs(elapsed_s=3600.0, energy_counter_kwh=102.0, water_counter_l=500.0)
    res = tank.apply_tick(st, PARAMS, inp)
    assert res.draw_source == "meter"
    assert res.energy_in_kwh == pytest.approx(2.0)
    assert res.draw_kwh == 0.0
    assert res.standby_kwh == pytest.approx(0.07)
    assert res.state.deficit_kwh == pytest.approx(10.0 + 0.07 - 2.0)


def test_apply_tick_draw_and_standby_grow_deficit():
    st = _state(
        deficit_kwh=5.0,
        energy_baseline_kwh=100.0,
        water_baseline_l=500.0,
        water_baseline_iso="2026-07-16T09:00:00+03:00",
        calibrated=True,
    )
    # 60 L over 60 min (span from 09:00), no energy in.
    inp = _inputs(elapsed_s=3600.0, energy_counter_kwh=100.0, water_counter_l=560.0)
    res = tank.apply_tick(st, PARAMS, inp)
    draw = tank.draw_kwh_from_liters(60.0, 0.25, 75.0, 12.0)
    assert res.draw_source == "meter"
    assert res.draw_kwh == pytest.approx(draw)
    assert res.state.deficit_kwh == pytest.approx(5.0 + draw + 0.07)


def test_apply_tick_deficit_clamped_to_zero():
    # No water baseline on the state + no reading → fallback, but energy_in dominates.
    st = _state(deficit_kwh=1.0, energy_baseline_kwh=100.0, calibrated=True, standby_w=0.0)
    inp = _inputs(elapsed_s=3600.0, energy_counter_kwh=105.0)  # 5 kWh in ≫ deficit
    res = tank.apply_tick(st, PARAMS, inp)
    assert res.state.deficit_kwh == 0.0
    assert res.soc == 1.0


def test_apply_tick_deficit_clamped_to_capacity():
    st = _state(
        deficit_kwh=CAP - 0.1,
        energy_baseline_kwh=100.0,
        water_baseline_l=1000.0,
        water_baseline_iso="2026-07-16T09:00:00+03:00",
        calibrated=True,
    )
    # A huge draw would overflow the deficit → clamped at capacity (SoC 0).
    inp = _inputs(elapsed_s=3600.0, energy_counter_kwh=100.0, water_counter_l=2000.0)
    res = tank.apply_tick(st, PARAMS, inp)
    assert res.state.deficit_kwh == pytest.approx(CAP)
    assert res.soc == 0.0


# ── apply_tick: anchor transition + latch ─────────────────────────────────────


def test_apply_tick_first_anchor_resets_calibrates_latches_no_learn():
    st = _state(deficit_kwh=10.0, energy_baseline_kwh=100.0, calibrated=False)
    inp = _inputs(
        energy_counter_kwh=100.0,
        contactor_on=True,
        heating_on=False,
        contactor_on_for_s=120.0,
        heating_off_for_s=60.0,
    )
    res = tank.apply_tick(st, PARAMS, inp)
    assert res.anchored is True
    assert res.state.deficit_kwh == 0.0
    assert res.state.calibrated is True
    assert res.state.anchor_latched is True
    assert res.soc == 1.0
    # First anchor never learns — params stay at the seeds.
    assert res.state.hot_fraction == tank.SEED_HOT_FRACTION
    assert res.state.standby_w == tank.SEED_STANDBY_W


def test_apply_tick_consecutive_anchor_holds_no_relearn_then_latch_clears():
    # Enter the anchor.
    st = _state(deficit_kwh=8.0, energy_baseline_kwh=100.0, calibrated=False)
    first = tank.apply_tick(
        st,
        PARAMS,
        _inputs(
            energy_counter_kwh=100.0,
            contactor_on=True,
            heating_on=False,
            contactor_on_for_s=120.0,
            heating_off_for_s=60.0,
        ),
    )
    # A second still-anchored tick holds full and must not re-learn.
    second = tank.apply_tick(
        first.state,
        PARAMS,
        _inputs(
            now_iso="2026-07-16T10:05:00+03:00",
            energy_counter_kwh=101.0,
            contactor_on=True,
            heating_on=False,
            contactor_on_for_s=420.0,
            heating_off_for_s=360.0,
        ),
    )
    assert second.anchored is True
    assert second.state.deficit_kwh == 0.0
    assert second.state.hot_fraction == first.state.hot_fraction
    assert second.state.standby_w == first.state.standby_w
    # Heating resumes (element on) → not an anchor → latch clears.
    third = tank.apply_tick(
        second.state,
        PARAMS,
        _inputs(
            now_iso="2026-07-16T10:10:00+03:00",
            energy_counter_kwh=102.0,
            contactor_on=True,
            heating_on=True,
            contactor_on_for_s=720.0,
            heating_off_for_s=0.0,
        ),
    )
    assert third.anchored is False
    assert third.state.anchor_latched is False


def test_apply_tick_second_anchor_learns_hot_fraction_on_clean_cycle():
    # A calibrated, clean cycle: 100 metered L, energy chosen so the implied hot
    # fraction is 0.5 over a 10 h cycle → EWMA 0.8×0.25 + 0.2×0.5 = 0.3.
    cycle_energy = 0.5 * (100.0 * 63.0 * K) + 70.0 * 10.0 / 1000.0
    st = _state(
        deficit_kwh=6.0,
        calibrated=True,
        cycle_clean=True,
        cycle_liters=100.0,
        cycle_energy_in_kwh=cycle_energy,
        cycle_start_iso="2026-07-16T00:00:00+03:00",
        energy_baseline_kwh=200.0,
        water_baseline_l=1000.0,
        water_baseline_iso="2026-07-16T09:50:00+03:00",
        standby_w=70.0,
        hot_fraction=0.25,
    )
    inp = _inputs(
        now_iso="2026-07-16T10:00:00+03:00",
        elapsed_s=600.0,
        energy_counter_kwh=200.0,  # 0 delta this tick
        water_counter_l=1000.0,  # 0 delta this tick
        contactor_on=True,
        heating_on=False,
        contactor_on_for_s=200.0,
        heating_off_for_s=120.0,
    )
    res = tank.apply_tick(st, PARAMS, inp)
    assert res.anchored is True
    assert res.state.deficit_kwh == 0.0
    assert res.state.hot_fraction == pytest.approx(0.3, abs=1e-4)


# ── learn_from_cycle (EWMA + clamps, both branches) ──────────────────────────


def _learn(**over) -> TankState:
    base = dict(
        deficit_kwh=0.0,
        calibrated=True,
        cycle_clean=True,
        hot_fraction=0.25,
        standby_w=70.0,
    )
    base.update(over)
    return TankState(**base)


def test_learn_hot_fraction_ewma():
    # implied 0.5 over 10 h with 100 L → 0.8×0.25 + 0.2×0.5 = 0.3.
    energy = 0.5 * (100.0 * 63.0 * K) + 70.0 * 10.0 / 1000.0
    st = _learn(cycle_liters=100.0, cycle_energy_in_kwh=energy)
    out = tank.learn_from_cycle(st, PARAMS, cycle_hours=10.0)
    assert out.hot_fraction == pytest.approx(0.3, abs=1e-6)
    assert out.standby_w == 70.0  # untouched


def test_learn_hot_fraction_clamped_high():
    # An absurd implied fraction still can't push past HOT_FRACTION_MAX.
    st = _learn(cycle_liters=100.0, cycle_energy_in_kwh=500.0)
    out = tank.learn_from_cycle(st, PARAMS, cycle_hours=10.0)
    assert out.hot_fraction == tank.HOT_FRACTION_MAX


def test_learn_hot_fraction_skips_nonpositive_numerator():
    # Delivered energy below standby → numerator ≤ 0 → no update.
    st = _learn(cycle_liters=100.0, cycle_energy_in_kwh=0.1)
    out = tank.learn_from_cycle(st, PARAMS, cycle_hours=10.0)
    assert out.hot_fraction == 0.25


def test_learn_standby_ewma():
    # < 10 L and ≥ 12 h → standby branch. implied_w chosen to be 100 W.
    draw_est = tank.draw_kwh_from_liters(5.0, 0.25, 75.0, 12.0)
    energy = 100.0 * 24.0 / 1000.0 + draw_est  # implied_w = 100
    st = _learn(cycle_liters=5.0, cycle_energy_in_kwh=energy)
    out = tank.learn_from_cycle(st, PARAMS, cycle_hours=24.0)
    # 0.8×70 + 0.2×100 = 76.
    assert out.standby_w == pytest.approx(76.0, abs=1e-6)
    assert out.hot_fraction == 0.25  # untouched


def test_learn_standby_clamped_high():
    st = _learn(cycle_liters=2.0, cycle_energy_in_kwh=50.0)
    out = tank.learn_from_cycle(st, PARAMS, cycle_hours=24.0)
    assert out.standby_w == tank.STANDBY_W_MAX


def test_learn_midsize_cycle_learns_nothing():
    # 10–50 L matches neither regime.
    st = _learn(cycle_liters=30.0, cycle_energy_in_kwh=3.0)
    out = tank.learn_from_cycle(st, PARAMS, cycle_hours=24.0)
    assert out.hot_fraction == 0.25
    assert out.standby_w == 70.0
    assert out is st  # no change → same object


def test_learn_dirty_cycle_learns_nothing():
    st = _learn(cycle_liters=100.0, cycle_energy_in_kwh=5.0, cycle_clean=False)
    out = tank.learn_from_cycle(st, PARAMS, cycle_hours=10.0)
    assert out is st


def test_learn_uncalibrated_learns_nothing():
    st = _learn(cycle_liters=100.0, cycle_energy_in_kwh=5.0, calibrated=False)
    out = tank.learn_from_cycle(st, PARAMS, cycle_hours=10.0)
    assert out is st


def test_learn_zero_hours_learns_nothing():
    st = _learn(cycle_liters=100.0, cycle_energy_in_kwh=5.0)
    out = tank.learn_from_cycle(st, PARAMS, cycle_hours=0.0)
    assert out is st


# ── apply_tick: pending fallback reconciliation ──────────────────────────────


def test_pending_fallback_reconciled_into_next_metered_draw():
    st = _state(
        deficit_kwh=5.0,
        energy_baseline_kwh=0.0,
        water_baseline_l=1000.0,
        water_baseline_iso="2026-07-16T00:00:00+03:00",
        calibrated=True,
        standby_w=0.0,
    )
    # Tick A: meter stale (>900 s) → fallback charged, pending accumulates.
    a = tank.apply_tick(
        st,
        PARAMS,
        _inputs(
            now_iso="2026-07-16T00:16:40+03:00",  # +1000 s
            elapsed_s=1000.0,
            energy_counter_kwh=0.0,
            water_counter_l=None,
        ),
    )
    fb = tank.fallback_draw_kwh(3.0, 2.2, 0.4, 1, 0.0, 1000.0 / 60.0)
    assert a.draw_source == "fallback"
    assert a.state.pending_fallback_kwh == pytest.approx(fb)
    assert a.state.water_baseline_iso == "2026-07-16T00:00:00+03:00"  # not re-baselined
    # Tick B: a valid metered read returns → its draw is netted against pending.
    b = tank.apply_tick(
        a.state,
        PARAMS,
        _inputs(
            now_iso="2026-07-16T00:33:20+03:00",  # +2000 s from baseline
            elapsed_s=1000.0,
            energy_counter_kwh=0.0,
            water_counter_l=1005.0,  # 5 L over 2000 s span → valid
        ),
    )
    raw = tank.draw_kwh_from_liters(5.0, 0.25, 75.0, 12.0)
    assert b.draw_source == "meter"
    assert b.draw_kwh == pytest.approx(max(0.0, raw - fb))
    assert b.state.pending_fallback_kwh == 0.0


def test_pending_fallback_floors_metered_draw_at_zero():
    # Big accumulated pending vs a tiny metered delta → draw floors to 0, resets.
    st = _state(
        deficit_kwh=5.0,
        energy_baseline_kwh=0.0,
        water_baseline_l=1000.0,
        water_baseline_iso="2026-07-16T00:00:00+03:00",
        calibrated=True,
        standby_w=0.0,
        pending_fallback_kwh=5.0,  # pre-loaded, far bigger than any 1 L draw
    )
    b = tank.apply_tick(
        st,
        PARAMS,
        _inputs(
            now_iso="2026-07-16T00:16:40+03:00",
            elapsed_s=1000.0,
            energy_counter_kwh=0.0,
            water_counter_l=1001.0,  # 1 L
        ),
    )
    assert b.draw_source == "meter"
    assert b.draw_kwh == 0.0
    assert b.state.pending_fallback_kwh == 0.0


def test_misread_return_resets_pending_without_applying():
    st = _state(
        deficit_kwh=5.0,
        energy_baseline_kwh=0.0,
        water_baseline_l=1000.0,
        water_baseline_iso="2026-07-16T00:00:00+03:00",
        calibrated=True,
        standby_w=0.0,
        pending_fallback_kwh=0.5,
    )
    # Reading dropped below the baseline → misread → fallback this tick, pending
    # dropped (not applied to reduce anything).
    res = tank.apply_tick(
        st,
        PARAMS,
        _inputs(
            now_iso="2026-07-16T00:16:40+03:00",
            elapsed_s=1000.0,
            energy_counter_kwh=0.0,
            water_counter_l=990.0,
        ),
    )
    fb = tank.fallback_draw_kwh(3.0, 2.2, 0.4, 1, 0.0, 1000.0 / 60.0)
    assert res.draw_source == "fallback"
    assert res.draw_kwh == pytest.approx(fb)  # full fallback, not netted
    assert res.state.pending_fallback_kwh == 0.0
    assert res.state.water_baseline_l == 990.0  # re-baselined


def test_short_meter_blip_draws_nothing():
    st = _state(
        deficit_kwh=5.0,
        energy_baseline_kwh=0.0,
        water_baseline_l=1000.0,
        water_baseline_iso="2026-07-16T09:59:30+03:00",  # 30 s ago
        calibrated=True,
        standby_w=0.0,
    )
    res = tank.apply_tick(
        st,
        PARAMS,
        _inputs(now_iso="2026-07-16T10:00:00+03:00", elapsed_s=30.0, water_counter_l=None),
    )
    assert res.draw_source == "none"
    assert res.draw_kwh == 0.0
    assert res.state.pending_fallback_kwh == 0.0


# ── apply_tick: restart reconciliation ────────────────────────────────────────


def test_restart_reconciliation_single_tick_over_8h_gap():
    # One tick spans an 8 h restart gap: cumulative deltas + 8 h standby fall out
    # of the ordinary arithmetic; nothing special-cased.
    st = _state(
        deficit_kwh=4.0,
        energy_baseline_kwh=1000.0,
        water_baseline_l=2000.0,
        water_baseline_iso="2026-07-16T02:00:00+03:00",
        last_tick_iso="2026-07-16T02:00:00+03:00",
        calibrated=True,
        standby_w=70.0,
        hot_fraction=0.25,
    )
    inp = _inputs(
        now_iso="2026-07-16T10:00:00+03:00",
        elapsed_s=8 * 3600.0,
        energy_counter_kwh=1002.0,  # 2 kWh in over the gap
        water_counter_l=2100.0,  # 100 L over the gap (all hot: under both caps)
    )
    res = tank.apply_tick(st, PARAMS, inp)
    draw = tank.draw_kwh_from_liters(100.0, 0.25, 75.0, 12.0)
    standby = tank.standby_kwh(70.0, 8 * 60.0)
    assert res.energy_in_kwh == pytest.approx(2.0)
    assert res.draw_source == "meter"
    assert res.draw_kwh == pytest.approx(draw)
    assert res.standby_kwh == pytest.approx(standby)
    assert res.state.deficit_kwh == pytest.approx(4.0 + draw + standby - 2.0)
    assert res.state.energy_baseline_kwh == 1002.0
    assert res.state.water_baseline_l == 2100.0


# ── should_boost (truth table) ────────────────────────────────────────────────


def test_boost_uncalibrated_never_fires():
    st = _state(calibrated=False, boost_armed=True)
    fire, _ = tank.should_boost(
        st, soc_value=0.10, threshold_pct=20.0, now_iso="2026-07-16T10:00:00+03:00"
    )
    assert fire is False


def test_boost_below_threshold_armed_fires_once():
    st = _state(calibrated=True, boost_armed=True, last_boost_iso="")
    fire, out = tank.should_boost(
        st, soc_value=0.10, threshold_pct=20.0, now_iso="2026-07-16T10:00:00+03:00"
    )
    assert fire is True
    assert out.boost_armed is False
    assert out.last_boost_iso == "2026-07-16T10:00:00+03:00"


def test_boost_rate_limited_within_interval():
    st = _state(calibrated=True, boost_armed=True, last_boost_iso="2026-07-16T07:00:00+03:00")
    # Only 3 h since the last boost (< 6 h) → suppressed.
    fire, out = tank.should_boost(
        st, soc_value=0.10, threshold_pct=20.0, now_iso="2026-07-16T10:00:00+03:00"
    )
    assert fire is False
    assert out.boost_armed is True  # still armed for later


def test_boost_fires_after_interval_elapsed():
    st = _state(calibrated=True, boost_armed=True, last_boost_iso="2026-07-16T03:00:00+03:00")
    # 7 h since the last boost (≥ 6 h) → fires.
    fire, _ = tank.should_boost(
        st, soc_value=0.10, threshold_pct=20.0, now_iso="2026-07-16T10:00:00+03:00"
    )
    assert fire is True


def test_boost_rearms_only_above_threshold_plus_margin():
    # Disarmed; SoC 30 % is below threshold+margin (35 %) → stays disarmed.
    st = _state(calibrated=True, boost_armed=False)
    fire, out = tank.should_boost(
        st, soc_value=0.30, threshold_pct=20.0, now_iso="2026-07-16T10:00:00+03:00"
    )
    assert fire is False
    assert out.boost_armed is False
    # SoC 40 % ≥ 35 % → re-arm (but doesn't fire — it's above the threshold).
    fire, out = tank.should_boost(
        st, soc_value=0.40, threshold_pct=20.0, now_iso="2026-07-16T10:00:00+03:00"
    )
    assert fire is False
    assert out.boost_armed is True


def test_boost_disarmed_below_threshold_does_not_fire():
    st = _state(calibrated=True, boost_armed=False)
    fire, out = tank.should_boost(
        st, soc_value=0.10, threshold_pct=20.0, now_iso="2026-07-16T10:00:00+03:00"
    )
    assert fire is False
    assert out.boost_armed is False


# ── deficit_minutes_from_kwh ──────────────────────────────────────────────────


def test_deficit_minutes_from_kwh():
    assert tank.deficit_minutes_from_kwh(3.0, 3.0) == pytest.approx(60.0)
    assert tank.deficit_minutes_from_kwh(1.5, 3.0) == pytest.approx(30.0)


def test_deficit_minutes_rejects_nonpositive_power():
    with pytest.raises(ValueError):
        tank.deficit_minutes_from_kwh(5.0, 0.0)


# ── liters_at_temp / showers_left ─────────────────────────────────────────────


def test_liters_at_temp():
    # 5 kWh available at 40 °C mix, 12 °C inlet → ΔT 28 K.
    assert tank.liters_at_temp(5.0, 12.0) == pytest.approx(5.0 / (28.0 * K))


def test_liters_at_temp_delta_t_floored():
    assert tank.liters_at_temp(5.0, 45.0, 40.0) == pytest.approx(5.0 / (1.0 * K))


def test_liters_at_temp_negative_available_is_zero():
    assert tank.liters_at_temp(-1.0, 12.0) == 0.0


def test_showers_left():
    assert tank.showers_left(160.0) == pytest.approx(4.0)
    assert tank.showers_left(0.0) == 0.0
    assert tank.showers_left(160.0, per_shower=0.0) == 0.0


# ── apply_tick: heating-active deficit floor (the anchor's inverse) ───────────


def test_heating_active_floors_deficit():
    # Element actively drawing ≥ 60 s → the tank cannot read (near-)full even if
    # the balance says so: floor at HEATING_MIN_DEFICIT_KWH.
    st = _state(deficit_kwh=0.5, energy_baseline_kwh=100.0, calibrated=True, standby_w=0.0)
    inp = _inputs(
        elapsed_s=60.0,
        energy_counter_kwh=102.0,  # 2 kWh in would clamp the deficit to 0…
        contactor_on=True,
        heating_on=True,
        contactor_on_for_s=600.0,
        heating_off_for_s=600.0,  # …but it has been heating for 10 min
    )
    res = tank.apply_tick(st, PARAMS, inp)
    assert res.state.deficit_kwh == pytest.approx(tank.HEATING_MIN_DEFICIT_KWH)
    assert res.soc < 1.0


def test_heating_active_floor_needs_sustained_on():
    # A fresh heating flank (< 60 s) doesn't floor — detector blips can't inject it.
    st = _state(deficit_kwh=0.0, energy_baseline_kwh=100.0, calibrated=True, standby_w=0.0)
    inp = _inputs(
        elapsed_s=60.0,
        energy_counter_kwh=100.0,
        e_base=0.0,
        e_draw_per_person=0.0,
        heating_on=True,
        heating_off_for_s=30.0,
    )
    res = tank.apply_tick(st, PARAMS, inp)
    assert res.state.deficit_kwh == 0.0


def test_heating_floor_leaves_larger_deficit_alone():
    st = _state(deficit_kwh=8.0, energy_baseline_kwh=100.0, calibrated=True, standby_w=0.0)
    inp = _inputs(
        elapsed_s=60.0,
        energy_counter_kwh=100.0,
        e_base=0.0,
        e_draw_per_person=0.0,
        heating_on=True,
        heating_off_for_s=600.0,
    )
    res = tank.apply_tick(st, PARAMS, inp)
    assert res.state.deficit_kwh == pytest.approx(8.0)


def test_heating_floor_does_not_apply_when_idle_or_unknown():
    for heating_on, held in ((False, 600.0), (None, None)):
        st = _state(deficit_kwh=0.0, energy_baseline_kwh=100.0, calibrated=True, standby_w=0.0)
        inp = _inputs(
            elapsed_s=60.0,
            energy_counter_kwh=100.0,
            e_base=0.0,
            e_draw_per_person=0.0,
            heating_on=heating_on,
            heating_off_for_s=held,
        )
        res = tank.apply_tick(st, PARAMS, inp)
        assert res.state.deficit_kwh == 0.0


def test_anchor_still_wins_over_heating_floor():
    # A genuine commanded-on-but-idle anchor resets to full; the floor only
    # applies while the element is actually drawing.
    st = _state(
        deficit_kwh=tank.HEATING_MIN_DEFICIT_KWH,
        energy_baseline_kwh=100.0,
        calibrated=True,
    )
    inp = _inputs(
        energy_counter_kwh=100.0,
        contactor_on=True,
        heating_on=False,
        contactor_on_for_s=600.0,
        heating_off_for_s=120.0,
    )
    res = tank.apply_tick(st, PARAMS, inp)
    assert res.anchored is True
    assert res.state.deficit_kwh == 0.0
    assert res.soc == 1.0


# ── HA-free contract ──────────────────────────────────────────────────────────


def test_no_homeassistant_import():
    src = _PATH.read_text()
    assert "homeassistant" not in src
