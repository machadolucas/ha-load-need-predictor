"""Unit tests for the pure tank state-of-charge model (no Home Assistant).

Loaded standalone via ``importlib`` so these run without importing the HA-bound
package. Mirrors the loader in ``test_predictor.py``.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
from datetime import datetime, timedelta

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

# All tick timestamps hang off one base instant so multi-tick sequences read as
# "+n minutes" instead of hand-written ISO strings.
_T0 = datetime.fromisoformat("2026-07-16T10:00:00+03:00")


def _iso(minutes: float = 0.0) -> str:
    return (_T0 + timedelta(minutes=minutes)).isoformat()


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
        rated_power_kw=None,
        local_hour=None,
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
    assert st.version == "v2"


def test_initial_state_v2_seeds():
    st = tank.initial_state(CAP)
    # No profile yet ⇒ profile_of() falls back to the flat seeded hot fraction.
    assert st.hot_fraction_profile == ()
    assert st.hysteresis_kwh == tank.SEED_HYSTERESIS_KWH
    assert st.residual_ratio == tank.SEED_RESIDUAL_RATIO
    # None (not 0.0) means "the learning ledger starts at the deficit".
    assert st.cycle_unclamped_kwh is None
    assert st.cycle_relax_kwh == 0.0
    assert st.cycle_gross_kwh == 0.0
    assert st.cycle_hot_liters_by_bucket == (0.0,) * tank.N_DAYPARTS
    assert st.led_kwh_since_counter == 0.0


# ── counter_delta ─────────────────────────────────────────────────────────────


def test_counter_delta_normal():
    assert tank.counter_delta(100.0, 150.0) == (50.0, 150.0)


def test_counter_delta_none_reading_keeps_baseline():
    assert tank.counter_delta(100.0, None) == (0.0, 100.0)


def test_counter_delta_none_baseline_adopts_reading():
    assert tank.counter_delta(None, 100.0) == (0.0, 100.0)


def test_counter_delta_reset_rebaselines():
    # A *large* decrease (counter reset / meter swap) → no delta, re-baseline.
    assert tank.counter_delta(150.0, 100.0) == (0.0, 100.0)


def test_counter_delta_small_rollback_keeps_baseline():
    # Restore-after-restart: powercalc re-published an older value 0.5 kWh back.
    # Below the tolerance ⇒ keep the baseline, so nothing is lost or doubled.
    assert tank.counter_delta(7961.55, 7961.05) == (0.0, 7961.55)
    # Catching back up to the old baseline still yields no delta…
    assert tank.counter_delta(7961.55, 7961.55) == (0.0, 7961.55)
    # …and only genuine progress past it counts.
    assert tank.counter_delta(7961.55, 7962.05) == (pytest.approx(0.5), 7962.05)


def test_counter_delta_rollback_tolerance_boundary():
    # Exactly at the tolerance is a reset (strict <), just below it a rollback.
    assert tank.counter_delta(100.0, 99.0) == (0.0, 99.0)
    assert tank.counter_delta(100.0, 99.01) == (0.0, 100.0)
    # …and the tolerance is overridable (the water meter uses litres).
    assert tank.counter_delta(100.0, 95.0, rollback_tol=10.0) == (0.0, 100.0)


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


# ── daypart_bucket / profile_of ───────────────────────────────────────────────


def test_daypart_bucket_boundaries():
    # DAYPART_BOUNDS = night [0,6), morning [6,11), day [11,17), evening [17,24).
    assert [tank.daypart_bucket(h) for h in (0, 5)] == [0, 0]
    assert [tank.daypart_bucket(h) for h in (6, 10)] == [1, 1]
    assert [tank.daypart_bucket(h) for h in (11, 16)] == [2, 2]
    assert [tank.daypart_bucket(h) for h in (17, 23)] == [3, 3]


def test_daypart_bucket_wraps_out_of_range_hours():
    assert tank.daypart_bucket(24) == 0  # 00:00
    assert tank.daypart_bucket(30) == 1  # 06:00
    assert tank.daypart_bucket(-1) == 3  # 23:00


def test_profile_of_empty_profile_is_flat_hot_fraction():
    st = _state(hot_fraction=0.31)
    assert tank.profile_of(st) == (0.31,) * tank.N_DAYPARTS


def test_profile_of_returns_stored_profile():
    profile = (0.2, 0.15, 0.18, 0.3)
    assert tank.profile_of(_state(hot_fraction_profile=profile)) == profile


def test_profile_of_wrong_length_falls_back_to_flat():
    st = _state(hot_fraction=0.22, hot_fraction_profile=(0.4, 0.4))
    assert tank.profile_of(st) == (0.22,) * tank.N_DAYPARTS


# ── post_trip_relax_kwh ───────────────────────────────────────────────────────


def test_post_trip_relax_first_hour():
    # τ = 45 min: over the first hour the tank has given up 1 − e^(−60/45) of the
    # hysteresis band.
    assert tank.post_trip_relax_kwh(0.8, 0.0, 1.0) == pytest.approx(
        0.8 * (1.0 - math.exp(-60.0 / 45.0))
    )


def test_post_trip_relax_sums_to_hysteresis_over_long_interval():
    assert tank.post_trip_relax_kwh(0.8, 0.0, 24.0) == pytest.approx(0.8, abs=1e-6)


def test_post_trip_relax_is_additive_across_ticks():
    # Summing per-tick increments must equal one long interval (telescoping).
    total = sum(tank.post_trip_relax_kwh(0.8, t / 60.0, (t + 1) / 60.0) for t in range(60))
    assert total == pytest.approx(tank.post_trip_relax_kwh(0.8, 0.0, 1.0))


def test_post_trip_relax_zero_for_nonpositive_interval():
    assert tank.post_trip_relax_kwh(0.8, 1.0, 1.0) == 0.0
    assert tank.post_trip_relax_kwh(0.8, 2.0, 1.0) == 0.0


def test_post_trip_relax_negative_start_floored():
    assert tank.post_trip_relax_kwh(0.8, -1.0, 1.0) == pytest.approx(
        tank.post_trip_relax_kwh(0.8, 0.0, 1.0)
    )


# ── display_deficit_kwh / uncertainty_kwh (the saturation curve) ─────────────


def test_display_deficit_identity_above_sigma():
    assert tank.display_deficit_kwh(1.0, 0.5) == 1.0
    assert tank.display_deficit_kwh(0.5, 0.5) == 0.5


def test_display_deficit_continuous_and_slope_one_at_sigma():
    sigma = 0.5
    h = 1e-7
    # Value: the two branches meet at u = σ.
    assert tank.display_deficit_kwh(sigma - h, sigma) == pytest.approx(sigma, abs=1e-6)
    # Slope: the curved branch arrives at the kink with derivative 1 (C¹).
    left = (tank.display_deficit_kwh(sigma, sigma) - tank.display_deficit_kwh(sigma - h, sigma)) / h
    assert left == pytest.approx(1.0, abs=1e-4)


def test_display_deficit_positive_and_decreasing_past_full():
    sigma = 0.5
    values = [tank.display_deficit_kwh(u, sigma) for u in (0.0, -1.0, -10.0, -1000.0)]
    assert all(v > 0.0 for v in values)
    assert values == sorted(values, reverse=True)
    # 1/|u| tail: σ² / (2σ − u).
    assert tank.display_deficit_kwh(-1.0, 0.5) == pytest.approx(0.25 / 2.0)


def test_display_deficit_sigma_floored():
    # σ below SIGMA_MIN_KWH is lifted to it, so the curve never divides by ~0.
    assert tank.display_deficit_kwh(-1.0, 0.0) == pytest.approx(
        tank.display_deficit_kwh(-1.0, tank.SIGMA_MIN_KWH)
    )
    assert tank.display_deficit_kwh(-1.0, 0.0) == pytest.approx(0.05**2 / (0.1 + 1.0))


def test_uncertainty_kwh_scales_with_flow_and_floors():
    assert tank.uncertainty_kwh(0.15, 10.0) == pytest.approx(1.5)
    assert tank.uncertainty_kwh(0.15, 0.1) == tank.SIGMA_MIN_KWH  # 0.015 → floored
    assert tank.uncertainty_kwh(0.15, 0.0) == tank.SIGMA_MIN_KWH
    assert tank.uncertainty_kwh(0.5, -3.0) == tank.SIGMA_MIN_KWH  # negative gross guarded


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


# ── latch_holds (truth table) ─────────────────────────────────────────────────


def test_latch_holds_requires_an_existing_latch():
    assert tank.latch_holds(False, True, False) is False
    assert tank.latch_holds(False, None, None) is False


def test_latch_holds_while_commanded_on_and_idle():
    assert tank.latch_holds(True, True, False) is True


def test_latch_holds_through_unknown_states():
    # Unlike should_anchor, the latch fails *open*: an unavailable blip must not
    # drop it (it only dedupes the transition).
    assert tank.latch_holds(True, None, None) is True
    assert tank.latch_holds(True, None, False) is True
    assert tank.latch_holds(True, True, None) is True


def test_latch_releases_on_definite_heating_or_contactor_off():
    assert tank.latch_holds(True, True, True) is False
    assert tank.latch_holds(True, False, False) is False
    assert tank.latch_holds(True, None, True) is False
    assert tank.latch_holds(True, False, None) is False


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
    # The learning ledger starts from the deficit when it was still None.
    assert res.state.cycle_unclamped_kwh == pytest.approx(10.0 + 0.07 - 2.0)


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
    # Control ledger clamps at "full"…
    assert res.state.deficit_kwh == 0.0
    # …while the learning ledger keeps the overshoot (it *is* the information).
    assert res.state.cycle_unclamped_kwh < 0.0
    # The display saturates toward 100 % but only an anchor shows exactly full.
    assert res.soc < 1.0
    assert res.soc > 0.999
    assert res.anchored is False


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
    assert res.state.cycle_unclamped_kwh == pytest.approx(CAP)  # unclamped ledger too
    assert res.deficit_shown_kwh == pytest.approx(CAP)
    assert res.soc == 0.0


def test_apply_tick_draw_uses_the_daypart_hot_fraction():
    profile = (0.8, 0.1, 0.1, 0.1)
    st = _state(
        deficit_kwh=5.0,
        energy_baseline_kwh=100.0,
        water_baseline_l=1000.0,
        water_baseline_iso="2026-07-16T09:00:00+03:00",
        calibrated=True,
        standby_w=0.0,
        hot_fraction_profile=profile,
    )
    inp = _inputs(
        elapsed_s=3600.0,
        energy_counter_kwh=100.0,
        water_counter_l=1060.0,  # 60 L over the 60 min span
        local_hour=3,  # night bucket → 0.8
    )
    res = tank.apply_tick(st, PARAMS, inp)
    assert res.draw_kwh == pytest.approx(tank.draw_kwh_from_liters(60.0, 0.8, 75.0, 12.0))
    # …and the litres land in that bucket so the learner attributes them there.
    assert res.state.cycle_hot_liters_by_bucket == pytest.approx((60.0, 0.0, 0.0, 0.0))


def test_apply_tick_daypart_defaults_to_now_iso_hour():
    profile = (0.1, 0.1, 0.1, 0.8)
    st = _state(
        deficit_kwh=5.0,
        energy_baseline_kwh=100.0,
        water_baseline_l=1000.0,
        water_baseline_iso="2026-07-16T19:00:00+03:00",
        calibrated=True,
        standby_w=0.0,
        hot_fraction_profile=profile,
    )
    inp = _inputs(
        now_iso="2026-07-16T20:00:00+03:00",  # evening bucket → 0.8
        elapsed_s=3600.0,
        energy_counter_kwh=100.0,
        water_counter_l=1060.0,
    )
    res = tank.apply_tick(st, PARAMS, inp)
    assert res.draw_kwh == pytest.approx(tank.draw_kwh_from_liters(60.0, 0.8, 75.0, 12.0))
    assert res.state.cycle_hot_liters_by_bucket == pytest.approx((0.0, 0.0, 0.0, 60.0))


# ── apply_tick: LED smoothing between counter steps ──────────────────────────


def test_led_smoothing_grows_while_heating_and_lifts_soc():
    st = _state(
        deficit_kwh=0.5,
        cycle_unclamped_kwh=0.5,
        energy_baseline_kwh=100.0,
        water_baseline_l=500.0,
        water_baseline_iso=_iso(-60),
        calibrated=True,
        standby_w=0.0,
    )
    socs = []
    for tick in range(1, 6):
        res = tank.apply_tick(
            st,
            PARAMS,
            _inputs(
                now_iso=_iso(tick),
                energy_counter_kwh=100.0,  # counter stalled between its 0.5 kWh steps
                water_counter_l=500.0,
                heating_on=True,
                heating_off_for_s=600.0,
                rated_power_kw=3.0,
            ),
        )
        st = res.state
        socs.append(res.soc)
        # 3 kW × 60 s = 0.05 kWh per tick.
        assert st.led_kwh_since_counter == pytest.approx(0.05 * tick)
    assert socs == sorted(socs)
    assert socs[0] < socs[-1]
    # The authoritative ledger is untouched by the display smoothing.
    assert st.cycle_unclamped_kwh == pytest.approx(0.5)


def test_led_smoothing_resets_when_the_counter_steps():
    st = _state(
        deficit_kwh=0.5,
        cycle_unclamped_kwh=0.5,
        energy_baseline_kwh=100.0,
        water_baseline_l=500.0,
        water_baseline_iso=_iso(-60),
        calibrated=True,
        standby_w=0.0,
        led_kwh_since_counter=0.4,
    )
    res = tank.apply_tick(
        st,
        PARAMS,
        _inputs(
            now_iso=_iso(1),
            energy_counter_kwh=100.5,  # the counter finally moves
            water_counter_l=500.0,
            heating_on=True,
            heating_off_for_s=600.0,
            rated_power_kw=3.0,
        ),
    )
    assert res.energy_in_kwh == pytest.approx(0.5)
    assert res.state.led_kwh_since_counter == 0.0


def test_led_smoothing_capped():
    st = _state(
        deficit_kwh=5.0,
        cycle_unclamped_kwh=5.0,
        energy_baseline_kwh=100.0,
        water_baseline_l=500.0,
        water_baseline_iso=_iso(-60),
        calibrated=True,
        standby_w=0.0,
    )
    for tick in range(1, 31):
        st = tank.apply_tick(
            st,
            PARAMS,
            _inputs(
                now_iso=_iso(tick),
                energy_counter_kwh=100.0,
                water_counter_l=500.0,
                heating_on=True,
                heating_off_for_s=600.0,
                rated_power_kw=3.0,
            ),
        ).state
    # 30 × 0.05 = 1.5 kWh of on-time, but a stalled counter can't run it away.
    assert st.led_kwh_since_counter == tank.LED_SMOOTHING_CAP_KWH


def test_led_smoothing_needs_heating_and_a_rated_power():
    for heating_on, rated in ((False, 3.0), (None, 3.0), (True, None)):
        st = _state(
            deficit_kwh=0.5,
            cycle_unclamped_kwh=0.5,
            energy_baseline_kwh=100.0,
            water_baseline_l=500.0,
            water_baseline_iso=_iso(-60),
            calibrated=True,
            standby_w=0.0,
        )
        res = tank.apply_tick(
            st,
            PARAMS,
            _inputs(
                now_iso=_iso(1),
                energy_counter_kwh=100.0,
                water_counter_l=500.0,
                heating_on=heating_on,
                heating_off_for_s=600.0,
                rated_power_kw=rated,
            ),
        )
        assert res.state.led_kwh_since_counter == 0.0


# ── apply_tick: never 100 % while heating (soft saturation) ──────────────────


def test_heating_never_reads_full_but_creeps_toward_it():
    # The balance runs past "full" while the element is still on: the % must creep
    # up monotonically and never reach exactly 100 without an anchor.
    st = _state(
        deficit_kwh=0.3,
        cycle_unclamped_kwh=0.3,
        energy_baseline_kwh=100.0,
        water_baseline_l=500.0,
        water_baseline_iso=_iso(-60),
        calibrated=True,
        standby_w=0.0,
    )
    counter = 100.0
    socs = []
    for tick in range(1, 11):
        counter += 0.05  # 3 kW × 60 s
        res = tank.apply_tick(
            st,
            PARAMS,
            _inputs(
                now_iso=_iso(tick),
                energy_counter_kwh=counter,
                water_counter_l=500.0,
                contactor_on=True,
                heating_on=True,
                contactor_on_for_s=1800.0,
                heating_off_for_s=1800.0,
                rated_power_kw=3.0,
            ),
        )
        st = res.state
        socs.append(res.soc)
        assert res.anchored is False
        assert res.soc < 1.0
        assert res.deficit_shown_kwh > 0.0
    assert socs == sorted(socs)  # monotone non-decreasing
    assert socs[-1] > 0.99
    # The control ledger has long since clamped to "full" — the display has not.
    assert st.deficit_kwh == 0.0
    assert st.cycle_unclamped_kwh < 0.0


def test_no_soc_drop_when_the_element_restarts_after_a_trip():
    anchored = tank.apply_tick(
        _state(
            deficit_kwh=6.0,
            energy_baseline_kwh=100.0,
            water_baseline_l=500.0,
            water_baseline_iso=_iso(-10),
            calibrated=True,
            standby_w=0.0,
        ),
        PARAMS,
        _inputs(
            now_iso=_iso(0),
            energy_counter_kwh=100.0,
            water_counter_l=500.0,
            contactor_on=True,
            heating_on=False,
            contactor_on_for_s=600.0,
            heating_off_for_s=300.0,
            rated_power_kw=3.0,
        ),
    )
    assert anchored.soc == 1.0
    # The element re-engages. Whether the energy counter has stepped yet or the LED
    # smoothing is standing in, the % must not lurch downward.
    for counter in (100.0, 100.5):
        res = tank.apply_tick(
            anchored.state,
            PARAMS,
            _inputs(
                now_iso=_iso(1),
                energy_counter_kwh=counter,
                water_counter_l=500.0,
                contactor_on=True,
                heating_on=True,
                contactor_on_for_s=660.0,
                heating_off_for_s=1.0,
                rated_power_kw=3.0,
            ),
        )
        assert res.anchored is False
        assert res.latched is False  # definite "heating" releases the latch
        assert anchored.soc - res.soc < 0.005  # < 0.5 percentage points


def test_soc_full_only_on_the_anchor_tick_then_relaxes_while_latched():
    st = _state(
        deficit_kwh=6.0,
        energy_baseline_kwh=100.0,
        water_baseline_l=500.0,
        water_baseline_iso=_iso(-10),
        calibrated=True,
        standby_w=0.0,
    )
    first = tank.apply_tick(
        st,
        PARAMS,
        _inputs(
            now_iso=_iso(0),
            energy_counter_kwh=100.0,
            water_counter_l=500.0,
            contactor_on=True,
            heating_on=False,
            contactor_on_for_s=600.0,
            heating_off_for_s=300.0,
        ),
    )
    assert first.anchored is True
    assert first.soc == 1.0
    assert first.deficit_shown_kwh == 0.0
    assert first.uncertainty_kwh == tank.SIGMA_MIN_KWH
    socs = [first.soc]
    state = first.state
    for tick in range(1, 4):
        res = tank.apply_tick(
            state,
            PARAMS,
            _inputs(
                now_iso=_iso(tick),
                energy_counter_kwh=100.0,
                water_counter_l=500.0,
                contactor_on=True,
                heating_on=False,
                contactor_on_for_s=600.0 + tick * 60.0,
                heating_off_for_s=300.0 + tick * 60.0,
            ),
        )
        assert res.anchored is False  # the latch dedupes the transition
        assert res.latched is True
        state = res.state
        socs.append(res.soc)
    assert socs[1] < 1.0  # post-trip relaxation pulls it off 100 immediately
    assert socs == sorted(socs, reverse=True)  # …and keeps pulling
    # The relaxation the ticks accrued is exactly the closed-form increment.
    assert state.cycle_relax_kwh == pytest.approx(
        tank.post_trip_relax_kwh(tank.SEED_HYSTERESIS_KWH, 0.0, 3.0 / 60.0)
    )
    assert state.deficit_kwh == pytest.approx(state.cycle_relax_kwh)


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
    assert res.latched is True
    assert res.state.deficit_kwh == 0.0
    assert res.state.cycle_unclamped_kwh == 0.0
    assert res.state.cycle_relax_kwh == 0.0
    assert res.state.calibrated is True
    assert res.state.anchor_latched is True
    assert res.soc == 1.0
    assert res.deficit_shown_kwh == 0.0
    # First anchor never learns — params stay at the seeds.
    assert res.state.hot_fraction == tank.SEED_HOT_FRACTION
    assert res.state.standby_w == tank.SEED_STANDBY_W
    assert res.state.hysteresis_kwh == tank.SEED_HYSTERESIS_KWH
    assert res.state.residual_ratio == tank.SEED_RESIDUAL_RATIO
    # A fresh cycle opens at the anchor.
    assert res.state.last_anchor_iso == inp.now_iso
    assert res.state.cycle_start_iso == inp.now_iso
    assert res.state.cycle_liters == 0.0
    assert res.state.cycle_energy_in_kwh == 0.0
    assert res.state.cycle_gross_kwh == 0.0
    assert res.state.cycle_hot_liters_by_bucket == (0.0,) * tank.N_DAYPARTS


def test_apply_tick_consecutive_anchor_holds_no_relearn_then_latch_clears():
    # Enter the anchor.
    st = _state(
        deficit_kwh=8.0,
        energy_baseline_kwh=100.0,
        water_baseline_l=500.0,
        water_baseline_iso=_iso(-10),
        calibrated=False,
    )
    first = tank.apply_tick(
        st,
        PARAMS,
        _inputs(
            now_iso=_iso(0),
            energy_counter_kwh=100.0,
            water_counter_l=500.0,
            contactor_on=True,
            heating_on=False,
            contactor_on_for_s=120.0,
            heating_off_for_s=60.0,
        ),
    )
    assert first.anchored is True
    # A second still-anchored tick is *not* a new transition (the latch dedupes),
    # must not re-learn, and keeps integrating: standby + relaxation accrue so the
    # tank drifts off 100 instead of being pinned there.
    second = tank.apply_tick(
        first.state,
        PARAMS,
        _inputs(
            now_iso=_iso(5),
            elapsed_s=300.0,
            energy_counter_kwh=100.0,
            water_counter_l=500.0,
            contactor_on=True,
            heating_on=False,
            contactor_on_for_s=420.0,
            heating_off_for_s=360.0,
        ),
    )
    assert second.anchored is False
    assert second.latched is True
    assert second.state.deficit_kwh > 0.0
    assert second.state.hot_fraction == first.state.hot_fraction
    assert second.state.standby_w == first.state.standby_w
    assert second.state.hysteresis_kwh == first.state.hysteresis_kwh
    # The cycle is *not* restarted on every latched tick.
    assert second.state.cycle_start_iso == first.state.cycle_start_iso
    assert second.state.last_anchor_iso == first.state.last_anchor_iso
    # Heating resumes (element on) → not an anchor → latch clears.
    third = tank.apply_tick(
        second.state,
        PARAMS,
        _inputs(
            now_iso=_iso(10),
            elapsed_s=300.0,
            energy_counter_kwh=102.0,
            water_counter_l=500.0,
            contactor_on=True,
            heating_on=True,
            contactor_on_for_s=720.0,
            heating_off_for_s=0.0,
        ),
    )
    assert third.anchored is False
    assert third.latched is False
    assert third.state.anchor_latched is False


def test_latch_survives_unknown_blip_without_a_second_anchor():
    st = _state(
        deficit_kwh=6.0,
        energy_baseline_kwh=100.0,
        water_baseline_l=500.0,
        water_baseline_iso=_iso(-10),
        calibrated=True,
    )
    first = tank.apply_tick(
        st,
        PARAMS,
        _inputs(
            now_iso=_iso(0),
            energy_counter_kwh=100.0,
            water_counter_l=500.0,
            contactor_on=True,
            heating_on=False,
            contactor_on_for_s=600.0,
            heating_off_for_s=300.0,
        ),
    )
    assert first.anchored is True
    # A 1-second `unavailable` blip on both entities.
    blip = tank.apply_tick(
        first.state,
        PARAMS,
        _inputs(
            now_iso=_iso(1.0 / 60.0),
            elapsed_s=1.0,
            energy_counter_kwh=100.0,
            water_counter_l=500.0,
            contactor_on=None,
            heating_on=None,
        ),
    )
    assert blip.anchored is False
    assert blip.latched is True
    assert blip.state.last_anchor_iso == first.state.last_anchor_iso
    # Back to commanded-on-but-idle, now with freshly-reset short durations: still
    # the same trip, so no second anchor and no second learn.
    back = tank.apply_tick(
        blip.state,
        PARAMS,
        _inputs(
            now_iso=_iso(1),
            energy_counter_kwh=100.0,
            water_counter_l=500.0,
            contactor_on=True,
            heating_on=False,
            contactor_on_for_s=1.0,
            heating_off_for_s=1.0,
        ),
    )
    assert back.anchored is False
    assert back.latched is True
    assert back.state.last_anchor_iso == first.state.last_anchor_iso


def test_latch_releases_on_contactor_off_then_a_new_anchor_fires():
    st = _state(
        deficit_kwh=6.0,
        energy_baseline_kwh=100.0,
        water_baseline_l=500.0,
        water_baseline_iso=_iso(-10),
        calibrated=True,
    )
    first = tank.apply_tick(
        st,
        PARAMS,
        _inputs(
            now_iso=_iso(0),
            energy_counter_kwh=100.0,
            water_counter_l=500.0,
            contactor_on=True,
            heating_on=False,
            contactor_on_for_s=600.0,
            heating_off_for_s=300.0,
        ),
    )
    # Slot ends: the contactor drops → the latch releases.
    released = tank.apply_tick(
        first.state,
        PARAMS,
        _inputs(
            now_iso=_iso(1),
            energy_counter_kwh=100.0,
            water_counter_l=500.0,
            contactor_on=False,
            heating_on=False,
            contactor_on_for_s=0.0,
            heating_off_for_s=360.0,
        ),
    )
    assert released.latched is False
    assert released.anchored is False
    # …so the *next* commanded-on-but-idle really is a new trip.
    again = tank.apply_tick(
        released.state,
        PARAMS,
        _inputs(
            now_iso=_iso(200),
            energy_counter_kwh=100.0,
            water_counter_l=500.0,
            contactor_on=True,
            heating_on=False,
            contactor_on_for_s=600.0,
            heating_off_for_s=300.0,
        ),
    )
    assert again.anchored is True
    assert again.latched is True
    assert again.state.last_anchor_iso == _iso(200)


def test_apply_tick_second_anchor_learns_hot_fraction_on_clean_cycle():
    # A calibrated, clean 10 h cycle whose litres all landed in the evening bucket
    # and which trips with a +1 kWh unclamped residual (the model over-estimated
    # the draw) → that bucket's hot fraction must come *down*.
    st = _state(
        deficit_kwh=6.0,
        cycle_unclamped_kwh=1.0,
        calibrated=True,
        cycle_clean=True,
        cycle_liters=100.0,
        cycle_hot_liters_by_bucket=(0.0, 0.0, 0.0, 100.0),
        cycle_gross_kwh=10.0,
        cycle_start_iso="2026-07-16T00:00:00+03:00",
        energy_baseline_kwh=200.0,
        water_baseline_l=1000.0,
        water_baseline_iso="2026-07-16T09:50:00+03:00",
        standby_w=0.0,
        hot_fraction=0.25,
    )
    inp = _inputs(
        now_iso=_iso(0),
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
    assert res.state.cycle_unclamped_kwh == 0.0
    # Normalised-LMS step on the single wet bucket, then the ridge pull.
    x = 100.0 * 63.0 * K
    stepped = 0.25 - tank.LEARN_BETA * 1.0 / x
    mean = (3 * 0.25 + stepped) / tank.N_DAYPARTS
    assert res.state.hot_fraction_profile[3] == pytest.approx(
        0.95 * stepped + 0.05 * mean, abs=1e-9
    )
    assert res.state.hot_fraction_profile[3] < 0.25
    assert res.state.hot_fraction == pytest.approx(mean, abs=1e-9)
    assert res.state.hot_fraction < 0.25
    # …and the same cycle taught the saturation curve its σ scale.
    assert res.state.residual_ratio == pytest.approx(0.8 * 0.15 + 0.2 * (1.0 / 10.0))


# ── learn_from_cycle: gates ───────────────────────────────────────────────────


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


def test_learn_uncalibrated_learns_nothing():
    st = _learn(cycle_liters=100.0, calibrated=False)
    assert tank.learn_from_cycle(st, PARAMS, 10.0, 1.0) is st


def test_learn_dirty_cycle_learns_nothing():
    st = _learn(cycle_liters=100.0, cycle_clean=False)
    assert tank.learn_from_cycle(st, PARAMS, 10.0, 1.0) is st


def test_learn_zero_hours_learns_nothing():
    st = _learn(cycle_liters=100.0)
    assert tank.learn_from_cycle(st, PARAMS, 0.0, 1.0) is st
    assert tank.learn_from_cycle(st, PARAMS, -1.0, 1.0) is st


# ── learn_from_cycle: short cycles teach hysteresis only ─────────────────────


def test_learn_hysteresis_from_short_dry_reheat():
    # An 18-minute, 0 L, 0.81 kWh re-heat: the energy beyond standby *is* the
    # hysteresis band. 0.8×0.8 + 0.2×(0.81 − 70×0.3/1000) = 0.7978.
    st = _learn(cycle_liters=0.0, cycle_energy_in_kwh=0.81)
    out = tank.learn_from_cycle(st, PARAMS, 0.3, 0.0)
    observed = 0.81 - 70.0 * 0.3 / 1000.0
    assert out.hysteresis_kwh == pytest.approx(0.8 * 0.8 + 0.2 * observed)
    assert out.hysteresis_kwh == pytest.approx(0.79780)
    # Nothing else moves on a short cycle.
    assert out.hot_fraction == 0.25
    assert out.hot_fraction_profile == ()
    assert out.standby_w == 70.0
    assert out.residual_ratio == tank.SEED_RESIDUAL_RATIO


def test_learn_hysteresis_needs_a_dry_cycle():
    # Water was drawn during the re-heat → the energy isn't purely the band.
    st = _learn(cycle_liters=20.0, cycle_energy_in_kwh=0.81)
    assert tank.learn_from_cycle(st, PARAMS, 0.3, 0.0) is st


def test_learn_hysteresis_needs_a_real_topup():
    # Below HYSTERESIS_MIN_ENERGY_KWH this was a blip, not a re-heat.
    st = _learn(cycle_liters=0.0, cycle_energy_in_kwh=0.1)
    assert tank.learn_from_cycle(st, PARAMS, 0.3, 0.0) is st


def test_learn_hysteresis_skips_nonpositive_observation():
    # Delivered energy below the cycle's own standby → no information.
    st = _learn(cycle_liters=0.0, cycle_energy_in_kwh=0.25)
    assert tank.learn_from_cycle(st, PARAMS, 3.9, 0.0) is st


def test_learn_hysteresis_clamped():
    st = _learn(cycle_liters=0.0, cycle_energy_in_kwh=50.0)
    out = tank.learn_from_cycle(st, PARAMS, 0.5, 0.0)
    assert out.hysteresis_kwh == tank.HYSTERESIS_MAX_KWH
    st = _learn(cycle_liters=0.0, cycle_energy_in_kwh=0.3, hysteresis_kwh=0.21)
    out = tank.learn_from_cycle(st, PARAMS, 0.3, 0.0)
    assert out.hysteresis_kwh >= tank.HYSTERESIS_MIN_KWH


def test_learn_long_cycle_does_not_touch_hysteresis():
    # A 20 h dry cycle is a standby observation, not a re-heat.
    st = _learn(cycle_liters=0.0, cycle_energy_in_kwh=0.81)
    out = tank.learn_from_cycle(st, PARAMS, 20.0, 0.0)
    assert out.hysteresis_kwh == tank.SEED_HYSTERESIS_KWH


# ── learn_from_cycle: the hot-fraction profile (normalised LMS + ridge) ──────


def test_learn_profile_lms_step_single_bucket():
    # 100 evening litres, +1 kWh residual ⇒ the model over-charged that bucket.
    st = _learn(cycle_liters=100.0, cycle_hot_liters_by_bucket=(0.0, 0.0, 0.0, 100.0))
    out = tank.learn_from_cycle(st, PARAMS, 10.0, 1.0)
    # x_b = L_b·ΔT·k; with one wet bucket the normalised step is β·residual/x.
    x = 100.0 * 63.0 * K
    stepped = 0.25 - tank.LEARN_BETA * 1.0 / x
    mean = (3 * 0.25 + stepped) / tank.N_DAYPARTS
    assert out.hot_fraction_profile[3] == pytest.approx(0.95 * stepped + 0.05 * mean, abs=1e-9)
    assert out.hot_fraction_profile[3] == pytest.approx(0.2368610, abs=1e-6)
    # The dry buckets only feel the ridge, not the step.
    for index in (0, 1, 2):
        assert out.hot_fraction_profile[index] == pytest.approx(0.95 * 0.25 + 0.05 * mean, abs=1e-9)
        assert out.hot_fraction_profile[index] == pytest.approx(0.2498294, abs=1e-6)
    # hot_fraction is kept as the profile mean, which the ridge preserves.
    assert out.hot_fraction == pytest.approx(mean, abs=1e-9)
    assert out.hot_fraction == pytest.approx(sum(out.hot_fraction_profile) / 4.0)


def test_learn_profile_step_direction_follows_residual_sign():
    st = _learn(cycle_liters=100.0, cycle_hot_liters_by_bucket=(0.0, 0.0, 0.0, 100.0))
    # Under-estimated the draw (negative residual = tank emptier than modelled)
    # → that bucket's hot fraction goes up.
    up = tank.learn_from_cycle(st, PARAMS, 10.0, -1.0)
    assert up.hot_fraction_profile[3] > 0.25
    assert up.hot_fraction > 0.25
    # A zero residual leaves the step alone (only the ridge acts).
    flat = tank.learn_from_cycle(st, PARAMS, 10.0, 0.0)
    assert flat.hot_fraction_profile == pytest.approx((0.25,) * 4)


def test_learn_profile_step_scales_with_bucket_litres():
    # Twice the litres ⇒ twice the LMS step (x_b ∝ L_b), and the ridge is a linear
    # contraction so the *difference* between buckets scales by (1 − ridge).
    st = _learn(cycle_liters=150.0, cycle_hot_liters_by_bucket=(0.0, 0.0, 50.0, 100.0))
    out = tank.learn_from_cycle(st, PARAMS, 10.0, 1.0)
    x_day = 50.0 * 63.0 * K
    x_eve = 100.0 * 63.0 * K
    norm = x_day**2 + x_eve**2
    step_day = tank.LEARN_BETA * 1.0 * x_day / norm
    step_eve = tank.LEARN_BETA * 1.0 * x_eve / norm
    assert step_eve == pytest.approx(2.0 * step_day)
    assert out.hot_fraction_profile[2] - out.hot_fraction_profile[3] == pytest.approx(
        (1.0 - tank.PROFILE_RIDGE) * (step_eve - step_day), abs=1e-9
    )
    assert out.hot_fraction_profile[3] < out.hot_fraction_profile[2] < 0.25


def test_learn_profile_ridge_pulls_every_bucket_toward_the_mean():
    # Residual 0 isolates the ridge: each bucket moves 5 % of its distance to the
    # profile mean (0.2 here), and the mean itself is preserved.
    st = _learn(
        cycle_liters=100.0,
        cycle_hot_liters_by_bucket=(0.0, 0.0, 0.0, 100.0),
        hot_fraction_profile=(0.1, 0.1, 0.1, 0.5),
    )
    out = tank.learn_from_cycle(st, PARAMS, 10.0, 0.0)
    assert out.hot_fraction_profile == pytest.approx((0.105, 0.105, 0.105, 0.485))
    assert out.hot_fraction == pytest.approx(0.2)


def test_learn_profile_clamped_both_ends():
    st = _learn(cycle_liters=100.0, cycle_hot_liters_by_bucket=(0.0, 0.0, 0.0, 100.0))
    low = tank.learn_from_cycle(st, PARAMS, 10.0, 1000.0)
    assert low.hot_fraction_profile[3] == tank.HOT_FRACTION_MIN
    high = tank.learn_from_cycle(st, PARAMS, 10.0, -1000.0)
    assert high.hot_fraction_profile[3] == tank.HOT_FRACTION_MAX
    for profile in (low.hot_fraction_profile, high.hot_fraction_profile):
        assert all(tank.HOT_FRACTION_MIN <= hf <= tank.HOT_FRACTION_MAX for hf in profile)


def test_learn_profile_needs_enough_metered_liters():
    # Below LEARN_MIN_LITERS the attribution is too thin to move the profile.
    st = _learn(cycle_liters=49.0, cycle_hot_liters_by_bucket=(0.0, 0.0, 0.0, 49.0))
    out = tank.learn_from_cycle(st, PARAMS, 10.0, 1.0)
    assert out.hot_fraction == 0.25
    assert out.hot_fraction_profile == pytest.approx((0.25,) * 4)


def test_learn_profile_skipped_when_no_bucket_carried_liters():
    # Litres were counted but never bucketed (e.g. pre-v2 state) → zero norm, no
    # step; the flat profile is simply materialised.
    st = _learn(cycle_liters=100.0, cycle_hot_liters_by_bucket=(0.0, 0.0, 0.0, 0.0))
    out = tank.learn_from_cycle(st, PARAMS, 10.0, 1.0)
    assert out.hot_fraction == 0.25
    assert out.hot_fraction_profile == pytest.approx((0.25,) * 4)


def test_learn_profile_tolerates_a_wrong_length_stored_profile():
    st = _learn(
        cycle_liters=100.0,
        cycle_hot_liters_by_bucket=(0.0, 100.0),  # bad shape → treated as zeros
    )
    out = tank.learn_from_cycle(st, PARAMS, 10.0, 1.0)
    assert out.hot_fraction_profile == pytest.approx((0.25,) * 4)


# ── learn_from_cycle: standby on long dry cycles ─────────────────────────────


def test_learn_standby_from_residual():
    # 24 h, < 10 L: standby absorbs the residual. 70 − 0.1×0.48/24×1000 = 68 W.
    st = _learn(cycle_liters=5.0)
    out = tank.learn_from_cycle(st, PARAMS, 24.0, 0.48)
    assert out.standby_w == pytest.approx(68.0)
    assert out.hot_fraction == 0.25  # untouched


def test_learn_standby_direction_follows_residual_sign():
    st = _learn(cycle_liters=5.0)
    out = tank.learn_from_cycle(st, PARAMS, 24.0, -0.48)
    assert out.standby_w == pytest.approx(72.0)


def test_learn_standby_clamped_both_ends():
    st = _learn(cycle_liters=5.0)
    assert tank.learn_from_cycle(st, PARAMS, 24.0, -100.0).standby_w == tank.STANDBY_W_MAX
    assert tank.learn_from_cycle(st, PARAMS, 24.0, 100.0).standby_w == tank.STANDBY_W_MIN


def test_learn_standby_needs_a_long_cycle():
    # ≥ 4 h (so not a re-heat) but < STANDBY_MIN_HOURS → standby doesn't dominate.
    st = _learn(cycle_liters=5.0)
    out = tank.learn_from_cycle(st, PARAMS, 6.0, 0.48)
    assert out.standby_w == 70.0


def test_learn_midsize_cycle_learns_no_structure():
    # 10–50 L matches neither the profile nor the standby regime.
    st = _learn(cycle_liters=30.0, cycle_hot_liters_by_bucket=(0.0, 0.0, 0.0, 30.0))
    out = tank.learn_from_cycle(st, PARAMS, 24.0, 1.0)
    assert out.hot_fraction == 0.25
    assert out.standby_w == 70.0
    assert out.hot_fraction_profile == pytest.approx((0.25,) * 4)


# ── learn_from_cycle: residual_ratio (the σ scale) ───────────────────────────


def test_learn_residual_ratio_ewma():
    # |residual| / gross = 2/10 = 0.2 → 0.8×0.15 + 0.2×0.2 = 0.16.
    st = _learn(cycle_liters=0.0, cycle_gross_kwh=10.0)
    out = tank.learn_from_cycle(st, PARAMS, 10.0, 2.0)
    assert out.residual_ratio == pytest.approx(0.16)
    # The sign doesn't matter — it's an error *scale*.
    out = tank.learn_from_cycle(st, PARAMS, 10.0, -2.0)
    assert out.residual_ratio == pytest.approx(0.16)


def test_learn_residual_ratio_observation_capped():
    # A wild residual over a tiny cycle can't blow σ up: the observation caps at
    # RESIDUAL_RATIO_MAX before the EWMA.
    st = _learn(cycle_liters=0.0, cycle_gross_kwh=1.0)
    out = tank.learn_from_cycle(st, PARAMS, 10.0, 100.0)
    assert out.residual_ratio == pytest.approx(0.8 * 0.15 + 0.2 * tank.RESIDUAL_RATIO_MAX)


def test_learn_residual_ratio_clamped_low():
    st = _learn(cycle_liters=0.0, cycle_gross_kwh=10.0, residual_ratio=0.05)
    out = tank.learn_from_cycle(st, PARAMS, 10.0, 0.0)
    assert out.residual_ratio == tank.RESIDUAL_RATIO_MIN


def test_learn_residual_ratio_needs_flow():
    # Almost nothing flowed → the residual says nothing about attribution error.
    st = _learn(cycle_liters=0.0, cycle_gross_kwh=0.4)
    out = tank.learn_from_cycle(st, PARAMS, 10.0, 2.0)
    assert out.residual_ratio == tank.SEED_RESIDUAL_RATIO


def test_learn_residual_ratio_not_updated_by_short_cycles():
    st = _learn(cycle_liters=0.0, cycle_energy_in_kwh=0.81, cycle_gross_kwh=10.0)
    out = tank.learn_from_cycle(st, PARAMS, 0.3, 2.0)
    assert out.residual_ratio == tank.SEED_RESIDUAL_RATIO


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
    assert a.state.cycle_clean is False  # a fallback tick dirties the cycle
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
    assert res.state.cycle_clean is False


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
    assert res.state.cycle_clean is True  # a blip is not a data-quality problem


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


# ── HA-free contract ──────────────────────────────────────────────────────────


def test_no_homeassistant_import():
    src = _PATH.read_text()
    assert "homeassistant" not in src
