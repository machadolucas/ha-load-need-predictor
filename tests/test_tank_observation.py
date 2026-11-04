"""Shadow attribution is standalone and cannot change the tank's control state."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import pathlib
import sys
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "custom_components"
    / "load_need_predictor"
    / "tank_observation.py"
)
_spec = importlib.util.spec_from_file_location("lnp_tank_observation", _PATH)
observation = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = observation
_spec.loader.exec_module(observation)
_tank_spec = importlib.util.spec_from_file_location(
    "lnp_observation_tank", _PATH.with_name("tank_model.py")
)
tank = importlib.util.module_from_spec(_tank_spec)
sys.modules[_tank_spec.name] = tank
_tank_spec.loader.exec_module(tank)
BASE = datetime.fromisoformat("2026-10-01T12:00:00+03:00")


def iso(minute=0):
    return (BASE + timedelta(minutes=minute)).isoformat()


def tick(saved=None, minute=0, **overrides):
    values = {
        "now_iso": iso(minute),
        "source_signature": "bathroom-v1",
        "water_l": 100.0,
        "motion": False,
        "humidity": 45.0,
        "ventilation": False,
        "appliance_active": False,
    }
    values.update(overrides)
    return observation.advance_observation(saved, observation.ObservationInputs(**values))


def shower(*, motion=True, humidity=True, fan=False, appliance=False):
    state = None
    for minute in range(4):
        state = tick(state, minute, anchored=minute == 0)
    for minute in range(4, 8):
        state = tick(
            state,
            minute,
            water_l=100 + (minute - 3) * 5,
            motion=motion,
            humidity=50 if humidity else 45,
            ventilation=fan,
            appliance_active=appliance,
            baseline_draw_kwh=0.1,
        )
    for minute in (8, 9):
        state = tick(state, minute, water_l=120)
    return state


def test_confident_shower_uses_mixed_water_demand_not_tank_setpoint():
    state = shower()
    event = state["events"][0]
    expected = 20 * observation.KWH_PER_LITER_KELVIN * (40 - 12)
    assert event["confidence"] == "confident"
    assert event["baseline_draw_kwh"] == pytest.approx(0.4)
    assert event["replacement_kwh"] == pytest.approx(expected)
    assert state["cycle"]["correction_kwh"] == pytest.approx(expected - 0.4)
    json.dumps(state, allow_nan=False)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"humidity": False}, "no_environment_corroboration"),
        ({"appliance": True}, "appliance_overlap"),
        ({"appliance": None}, "source_unknown"),
    ],
)
def test_ambiguous_events_keep_entire_baseline(overrides, reason):
    state = shower(**overrides)
    event = state["events"][0]
    assert event["confidence"] == "ambiguous"
    assert reason in event["quality_reasons"]
    assert event["replacement_kwh"] == event["baseline_draw_kwh"]
    assert state["cycle"]["correction_kwh"] == 0


def test_fan_transition_corroborates_but_static_fan_does_not():
    assert shower(humidity=False, fan=True)["events"][0]["confidence"] == "confident"
    state = None
    for minute in range(4):
        state = tick(state, minute, ventilation=True)
    for minute in range(4, 8):
        state = tick(state, minute, water_l=100 + (minute - 3) * 5, ventilation=True, motion=True)
    for minute in (8, 9):
        state = tick(state, minute, water_l=120, ventilation=True)
    assert state["events"][0]["confidence"] == "ambiguous"


def test_correlated_humidity_and_fan_never_replace_motion_evidence():
    state = shower(motion=False, humidity=True, fan=True)
    assert state["events"] == []
    assert state["cycle"]["correction_kwh"] == 0
    assert observation.summarize_observation(state)["recent_showers"] == []


def test_inputs_are_copied_and_supply_temperature_is_log_only():
    before = shower()
    snapshot = copy.deepcopy(before)
    after = tick(before, 10, water_l=120, supply_temp_c=30)
    assert before == snapshot
    assert after["cycle"]["correction_kwh"] == before["cycle"]["correction_kwh"]
    assert after["latest"]["supply_temperature_role"] == "log_only"


def test_duplicate_tick_is_idempotent_even_during_an_active_event():
    state = tick(anchored=True)
    for minute in range(1, 4):
        state = tick(state, minute)
        assert tick(state, minute) == state
    state = tick(state, 4, water_l=105, motion=True, humidity=50, baseline_draw_kwh=0.1)
    repeated = tick(state, 4, water_l=105, motion=True, humidity=50, baseline_draw_kwh=0.1)
    assert repeated == state
    assert repeated is not state


def test_same_instant_changed_meter_is_not_safely_idempotent():
    state = tick(anchored=True)
    state = tick(state, 0, water_l=110)
    assert state["cycle"]["complete"] is False
    assert "observation_gap" in state["cycle"]["quality_reasons"]


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"water_l": 90}, "meter_reset"),
        ({"water_l": None}, "water_unavailable"),
        ({"water_l": 150}, "batched_or_overlapping_flow"),
        ({"source_signature": "changed"}, "sources_changed"),
        ({"water_source_quality": "misread", "water_l": 115}, "water_quality"),
    ],
)
def test_interruption_ends_event_without_claiming_unseen_water(overrides, reason):
    state = None
    for minute in range(4):
        state = tick(state, minute, anchored=minute == 0)
    for minute in (4, 5, 6):
        state = tick(state, minute, water_l=100 + (minute - 3) * 5, motion=True, humidity=50)
    state = tick(state, 7, **overrides)
    assert reason in state["events"][0]["quality_reasons"]
    assert state["events"][0]["correction_kwh"] == 0
    assert state["cycle"]["complete"] is False


def test_restart_gap_never_classifies_unobserved_interval():
    state = shower()
    state = tick(state, 30, water_l=180, motion=True, humidity=55)
    assert "active_event" not in state
    assert state["cycle"]["complete"] is False
    state = tick(state, 31, water_l=185, motion=True, humidity=58)
    assert state["active_event"]["complete"] is False


def test_fresh_restart_in_middle_of_shower_has_unobserved_start():
    state = tick(minute=0)
    for minute in range(1, 5):
        state = tick(state, minute, water_l=100 + minute * 5, motion=True, humidity=50)
    for minute in (5, 6):
        state = tick(state, minute, water_l=120)
    assert "start_unobserved" in state["events"][0]["quality_reasons"]


def test_anchor_residual_reconstructs_before_the_baseline_reset():
    state = shower()
    correction = state["cycle"]["correction_kwh"]
    state = tick(
        state,
        10,
        water_l=120,
        anchored=True,
        previous_cycle_unclamped_kwh=0.8,
        baseline_draw_kwh=0.2,
        standby_kwh=0.1,
        energy_in_kwh=0.3,
    )
    cycle = state["cycles"][0]
    assert cycle["baseline_residual_kwh"] == pytest.approx(0.8)
    assert cycle["shadow_residual_kwh"] == pytest.approx(0.8 + correction)
    assert cycle["complete"] and cycle["clean"]
    assert state["cycle"]["correction_kwh"] == 0


def test_anchor_mid_event_disqualifies_both_partial_cycles():
    state = tick(anchored=True)
    for minute in range(1, 4):
        state = tick(state, minute)
    state = tick(state, 4, water_l=105, motion=True, humidity=50)
    state = tick(state, 5, water_l=110, motion=True, humidity=50)
    state = tick(state, 6, water_l=115, motion=True, humidity=50, anchored=True)
    assert "anchor_mid_event" in state["events"][0]["quality_reasons"]
    assert state["cycles"][0]["complete"] is False
    assert state["cycle"]["complete"] is False
    state = tick(state, 7, water_l=115, anchored=True)
    assert state["cycle"]["complete"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"previous_cycle_clean": False},
        {"draw_source": "fallback"},
        {"energy_available": False},
    ],
)
def test_dirty_current_anchor_tick_cannot_appear_clean_after_reset(overrides):
    state = tick(anchored=True)
    state = tick(state, 1, anchored=True, **overrides)
    assert state["cycles"][0]["clean"] is False


def test_saturated_baseline_cycle_is_ineligible_and_residual_matches_control():
    state = tick(anchored=True)
    state = tick(
        state,
        1,
        anchored=True,
        tank_capacity_kwh=22,
        previous_cycle_unclamped_kwh=21.5,
        baseline_draw_kwh=2,
    )
    assert state["cycles"][0]["baseline_residual_kwh"] == 22
    assert state["cycles"][0]["complete"] is False


def test_cycle_diagnostics_keep_independent_accounting_totals():
    state = tick(anchored=True)
    state = tick(state, 1, water_l=104, baseline_draw_kwh=0.2, energy_in_kwh=0.5, standby_kwh=0.01)
    state = tick(state, 2, water_l=104, anchored=True, standby_kwh=0.02)
    cycle = state["cycles"][0]
    assert cycle["water_l"] == 4
    assert cycle["baseline_draw_kwh"] == 0.2
    assert cycle["energy_in_kwh"] == 0.5
    assert cycle["standby_kwh"] == pytest.approx(0.03)


def test_pipe_idle_unknown_until_first_confident_hot_draw():
    assert observation.summarize_observation(tick())["pipe_idle_minutes"] is None
    state = shower()
    state = tick(state, 10, water_l=120)
    summary = observation.summarize_observation(state)
    assert summary["pipe_idle_minutes"] == 3
    assert summary["comfort_risk"] == "unknown"
    assert summary["automatic_promotion"] is False


def test_feedback_attaches_exact_event_and_sparse_reports_remain_unknown():
    state = shower()
    event_id = state["events"][0]["id"]
    after = observation.record_feedback(
        state, event_id=event_id, report="not_warm_enough", now_iso=iso(10)
    )
    assert after["events"][0]["feedback"] == "not_warm_enough"
    assert state["events"][0]["feedback"] is None
    assert observation.summarize_observation(after)["comfort_risk"] == "unknown"
    with pytest.raises(ValueError):
        observation.record_feedback(state, event_id="wrong", report="comfortable", now_iso=iso(10))
    with pytest.raises(ValueError):
        observation.record_feedback(state, event_id=event_id, report="safe", now_iso=iso(10))


def test_feedback_before_observations_raises_meaningful_value_error():
    with pytest.raises(ValueError, match="not retained"):
        observation.record_feedback(None, event_id="missing", report="comfortable", now_iso=iso(1))


def test_ninety_day_retention_and_feedback_expiry():
    state = shower()
    event_id = state["events"][0]["id"]
    future = iso(91 * 24 * 60)
    state = observation.advance_observation(
        state,
        observation.ObservationInputs(
            now_iso=future,
            source_signature="bathroom-v1",
            water_l=120,
            motion=False,
            humidity=45,
            ventilation=False,
            appliance_active=False,
        ),
    )
    assert state["events"] == []
    assert observation.summarize_observation(state)["pipe_idle_minutes"] is None
    with pytest.raises(ValueError):
        observation.record_feedback(state, event_id=event_id, report="comfortable", now_iso=future)


def evaluation_state(baseline=1.0, candidate=0.5):
    return {
        "source_signature": "same",
        "cycles": [
            {
                "start": iso(i * 1440),
                "end": iso((i + 1) * 1440),
                "complete": True,
                "clean": True,
                "synchronized": True,
                "source_signature": "same",
                "confident_events": 1,
                "water_l": 100.0,
                "baseline_residual_kwh": baseline,
                "shadow_residual_kwh": candidate,
            }
            for i in range(12)
        ],
    }


def test_evaluator_requires_chronological_holdout_with_rmse_and_bias_gates():
    state = evaluation_state()
    # Early observations must not dominate the held-out comparison.
    for row in state["cycles"][:6]:
        row["shadow_residual_kwh"] = 99
    report = observation.evaluate_observations(state)
    assert report["status"] == "passes_shadow_comparison"
    assert report["candidate"]["rmse_kwh"] == 0.5
    assert report["heldout_start"] == iso(6 * 1440)
    assert report["automatic_promotion"] is False


def test_lower_rmse_cannot_hide_worse_absolute_bias():
    state = evaluation_state(candidate=0.5)
    for i, row in enumerate(state["cycles"]):
        row["baseline_residual_kwh"] = 1 if i % 2 else -1
    assert observation.evaluate_observations(state)["status"] == "fails_shadow_comparison"


def test_evaluator_ignores_old_setup_dirty_partial_and_nonfinite_cycles():
    state = evaluation_state()
    state["cycles"][0]["complete"] = False
    state["cycles"][1]["clean"] = False
    state["cycles"][2]["source_signature"] = "old"
    state["cycles"][3]["shadow_residual_kwh"] = float("nan")
    report = observation.evaluate_observations(state)
    assert report["status"] == "insufficient_evidence"
    assert report["eligible_cycles"] == 8


def test_evaluator_skips_post_trip_reheats_and_dry_cycles():
    """Re-heats and dry cycles have nothing to correct: they only dilute the error."""
    state = evaluation_state()
    reheat = state["cycles"][0]
    reheat["end"] = iso(3 * 60 + 59)  # cycle 0 starts at minute 0
    state["cycles"][1]["water_l"] = 0.0
    state["cycles"][2].pop("water_l")
    report = observation.evaluate_observations(state)
    assert report["eligible_cycles"] == 9
    assert report["status"] == "insufficient_evidence"
    assert report["appliance_exploration"]["eligible_cycles"] == 9


def test_evaluator_requires_recognized_events_not_just_anchor_count():
    state = evaluation_state()
    for row in state["cycles"]:
        row["confident_events"] = 0
    assert observation.evaluate_observations(state)["status"] == "insufficient_evidence"


def test_naive_timestamps_rejected():
    with pytest.raises(ValueError):
        tick(now_iso="2026-10-01T12:00:00")


@pytest.mark.parametrize("appliance_assumption", [None, False])
def test_real_51_liter_shower_trace_and_conditional_appliance_assumption(appliance_assumption):
    """Missing appliance history stays ambiguous; False is a conditional replay.

    The fixture contains observed water/motion/humidity/fan/energy only. It does
    not establish appliance idleness, tank anchors, or a true comfort outcome.
    """
    path = pathlib.Path(__file__).parent / "fixtures" / "shower_observation.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    state = None
    previous_water = float(rows[0]["water_l"])
    for row in rows:
        water = float(row["water_l"])
        baseline_draw = (
            min(8, max(0, water - previous_water)) * 0.25 * observation.KWH_PER_LITER_KELVIN * 63
        )
        state = observation.advance_observation(
            state,
            observation.ObservationInputs(
                now_iso=row["timestamp"],
                source_signature="sanitized-real-trace-v1",
                water_l=water,
                motion=row["motion"] == "on",
                humidity=float(row["humidity"]),
                ventilation=row["ventilation"] == "on",
                appliance_active=appliance_assumption,
                baseline_draw_kwh=baseline_draw,
            ),
        )
        previous_water = water
        assert row["appliance_active"] == ""
    assert len(state["events"]) == 1
    event = state["events"][0]
    assert event["water_l"] == 51
    assert event["motion_seen"] and event["corroborated"]
    if appliance_assumption is None:
        assert event["confidence"] == "ambiguous"
        assert "source_unknown" in event["quality_reasons"]
        assert event["correction_kwh"] == 0
    else:
        assert event["confidence"] == "confident"
        assert event["mixed_demand_kwh"] == pytest.approx(
            51 * observation.KWH_PER_LITER_KELVIN * 28
        )
    assert observation.evaluate_observations(state)["status"] == "insufficient_evidence"


def test_retention_can_expire_disabled_observer_without_new_ticks():
    state = shower()
    future = (BASE + timedelta(days=91)).isoformat()
    expired = observation.prune_observation(state, future)
    assert expired["events"] == []
    assert expired["cycles"] == []
    assert "water_l" not in expired
    assert state["events"]  # The old in-memory snapshot is not mutated.


def appliance_tick(saved=None, minute=0, *, running=False, **overrides):
    values = {
        "appliance_states": {"washer": running},
        "appliance_active": running,
        "household_activity": False,
    }
    values.update(overrides)
    return tick(saved, minute, **values)


def test_appliances_unconfigured_do_not_claim_observations():
    assert (
        observation.summarize_observation(tick())["appliance_profiles"]["status"]
        == "not_configured"
    )


def test_run_profiles_keep_gross_interior_flow_and_separate_exploratory_bound():
    state = appliance_tick(anchored=True)
    state = appliance_tick(state, 1, running=True, water_l=102, baseline_draw_kwh=0.1)
    state = appliance_tick(state, 2, running=True, water_l=110, baseline_draw_kwh=0.2)
    for minute in (3, 4):
        state = appliance_tick(state, minute, running=True, water_l=110)
    state = appliance_tick(state, 5, running=True, water_l=114, baseline_draw_kwh=0.1)
    state = appliance_tick(state, 6, water_l=114, anchored=True, previous_cycle_unclamped_kwh=1)
    store = state["appliance_observations"]
    run = store["runs"][0]
    assert run["complete"]
    assert run["gross_household_l"] == 14
    assert run["observed_interior_household_l"] == 12
    assert run["observed_interior_seconds"] == 240
    assert run["household_flow_episodes"] == 2
    assert run["exploratory_cold_upper_liters"] == 12
    assert run["exploratory_max_draw_reduction_kwh"] == pytest.approx(0.3)
    assert run["attribution"] == "gross_household_water_during_run_not_appliance_fill"
    cycle = state["cycles"][0]
    assert cycle["shadow_residual_kwh"] == cycle["baseline_residual_kwh"] == 1
    assert cycle["exploratory_min_residual_kwh"] == pytest.approx(0.7)
    profile = observation.summarize_observation(state)["appliance_profiles"]
    assert profile["accepted_cold_attribution_kwh"] == 0
    assert profile["cold_attribution_status"] == "unvalidated"
    assert profile["per_source"]["washer"]["mean_gross_household_l_per_metered_run"] == 14
    assert profile["exploratory_cold_upper_liters"] == 12
    json.dumps(state, allow_nan=False)


@pytest.mark.parametrize(
    "overrides",
    [
        {"household_activity": None},
        {"household_activity": True},
        {"motion": None},
        {"motion": True},
        {"appliance_states": {"washer": True, "dishwasher": None}},
        {"appliance_states": {"washer": True, "dishwasher": True}},
        {"appliance_boundaries_known": False},
        {"draw_source": "fallback"},
    ],
)
def test_ambiguous_intervals_never_contribute_to_cold_bound(overrides):
    states = overrides.get("appliance_states", {"washer": True})
    state = appliance_tick(appliance_states={source: False for source in states})
    state = appliance_tick(state, 1, running=True, appliance_states=states)
    state = appliance_tick(state, 2, running=True, water_l=108, baseline_draw_kwh=0.2, **overrides)
    profile = observation.summarize_observation(state)["appliance_profiles"]
    assert profile["exploratory_cold_upper_liters"] == 0
    assert profile["exploratory_max_draw_reduction_kwh"] == 0
    assert profile["accepted_cold_attribution_kwh"] == 0


def test_overlapping_appliance_profiles_do_not_double_count_exploratory_water():
    state = appliance_tick(appliance_states={"washer": False, "dishwasher": False})
    state = appliance_tick(
        state, 1, running=True, appliance_states={"washer": True, "dishwasher": True}
    )
    state = appliance_tick(
        state,
        2,
        running=True,
        water_l=110,
        baseline_draw_kwh=0.3,
        appliance_states={"washer": True, "dishwasher": True},
    )
    state = appliance_tick(
        state, 3, water_l=110, appliance_states={"washer": False, "dishwasher": False}
    )
    runs = state["appliance_observations"]["runs"]
    assert len(runs) == 2
    assert [run["gross_household_l"] for run in runs] == [10, 10]
    assert all("appliance_overlap" in run["quality_flags"] for run in runs)
    summary = observation.summarize_observation(state)["appliance_profiles"]
    assert summary["gross_liters_overlap_between_sources"] is True
    assert summary["exploratory_cold_upper_liters"] == 0


def test_no_metered_water_does_not_mean_demonstrated_zero_consumption():
    state = appliance_tick()
    state = appliance_tick(state, 1, running=True)
    state = appliance_tick(state, 2, running=True)
    state = appliance_tick(state, 3)
    run = state["appliance_observations"]["runs"][0]
    assert "no_metered_water" in run["quality_flags"]
    # Appliances always fill: a dry run is a missed meter, never a clean sample.
    assert run["complete"] is False
    assert run["interior_profile_usable"] is False
    profile = observation.summarize_observation(state)["appliance_profiles"]["per_source"]["washer"]
    assert profile["runs_with_metered_water"] == 0
    assert profile["mean_gross_household_l_per_metered_run"] is None
    assert profile["cold_volume_consistency_status"] == "not_established"


def test_short_restart_cannot_attribute_unseen_flow_or_claim_complete_run():
    state = appliance_tick()
    state = appliance_tick(state, 1, running=True)
    state = appliance_tick(
        state,
        2,
        running=True,
        water_l=110,
        observation_restarted=True,
        appliance_boundaries_known=False,
        baseline_draw_kwh=0.3,
    )
    assert state["appliance_observations"]["runs"][0]["complete"] is False
    assert "restart_or_gap" in state["appliance_observations"]["runs"][0]["quality_flags"]
    assert state["appliance_observations"]["active_runs"]["washer"]["gross_household_l"] == 0
    assert (
        observation.summarize_observation(state)["appliance_profiles"][
            "exploratory_cold_upper_liters"
        ]
        == 0
    )


def test_unknown_run_state_then_off_remains_incomplete():
    state = appliance_tick()
    state = appliance_tick(state, 1, running=True)
    state = appliance_tick(state, 2, running=None, water_l=108)
    state = appliance_tick(state, 3, water_l=108)
    run = state["appliance_observations"]["runs"][0]
    assert run["complete"] is False
    assert "appliance_state_unknown" in run["quality_flags"]
    assert run["gross_household_l"] == 8


def test_normal_uncertain_edge_keeps_usable_observed_run_interior():
    state = appliance_tick()
    state = appliance_tick(state, 1, running=True, appliance_boundaries_known=False)
    state = appliance_tick(state, 2, running=True, water_l=108, baseline_draw_kwh=0.2)
    state = appliance_tick(state, 3, water_l=108, appliance_boundaries_known=False)
    run = state["appliance_observations"]["runs"][0]
    assert not run["complete"]
    assert run["interior_profile_usable"]
    assert run["observed_interior_household_l"] == 8
    assert run["exploratory_cold_upper_liters"] == 8


def test_between_tick_appliance_pulse_ending_idle_invalidates_shower():
    state = None
    for minute in range(4):
        state = appliance_tick(state, minute, anchored=minute == 0)
    for minute in range(4, 8):
        state = appliance_tick(
            state,
            minute,
            water_l=100 + (minute - 3) * 5,
            motion=True,
            humidity=50,
            appliance_boundaries_known=minute != 5,
            baseline_draw_kwh=0.1,
        )
    for minute in (8, 9):
        state = appliance_tick(state, minute, water_l=120)
    event = state["events"][0]
    assert event["confidence"] == "ambiguous"
    assert "appliance_boundary_unobserved" in event["quality_reasons"]
    assert event["correction_kwh"] == 0


def test_appliance_counter_and_source_changes_invalidate_in_progress_run():
    state = appliance_tick()
    state = appliance_tick(state, 1, running=True)
    state = appliance_tick(state, 2, running=True, water_l=90)
    state = appliance_tick(state, 3, water_l=90)
    assert "meter_reset" in state["appliance_observations"]["runs"][0]["quality_flags"]
    assert not state["appliance_observations"]["runs"][0]["complete"]
    state = appliance_tick(state, 4, running=True, water_l=90)
    state = appliance_tick(state, 5, running=True, water_l=94, source_signature="new")
    assert "restart_or_gap" in state["appliance_observations"]["runs"][-1]["quality_flags"]
    assert state["appliance_observations"]["active_runs"]["washer"]["complete"] is False


def test_appliance_runs_and_bounds_expire_after_ninety_days():
    state = appliance_tick()
    state = appliance_tick(state, 1, running=True)
    state = appliance_tick(state, 2, running=True, water_l=108, baseline_draw_kwh=0.2)
    state = appliance_tick(state, 3, water_l=108)
    expired = observation.prune_observation(state, (BASE + timedelta(days=91)).isoformat())
    assert expired["appliance_observations"]["runs"] == []
    assert expired["appliance_observations"]["daily_bounds"] == []
    assert expired["appliance_observations"]["last_states"] == {}
    assert (
        observation.summarize_observation(expired)["appliance_profiles"][
            "exploratory_cold_upper_liters"
        ]
        == 0
    )


def test_legacy_partial_appliance_store_loads_without_raising():
    state = tick()
    state["appliance_observations"] = {"runs": []}
    state = appliance_tick(state, 1)
    assert state["appliance_observations"]["last_states"] == {"washer": False}


def test_improved_exploratory_error_never_passes_accepted_attribution_gate():
    state = evaluation_state(baseline=1, candidate=1)
    state["appliance_observations"] = {
        "last_states": {"washer": False},
        "daily_bounds": [],
        "runs": [],
    }
    for row in state["cycles"]:
        row["exploratory_max_draw_reduction_kwh"] = 1
        row["confident_events"] = 0
    report = observation.evaluate_observations(state)
    assert report["status"] == "insufficient_evidence"
    exploration = report["appliance_exploration"]
    assert exploration["status"] == "unvalidated_upper_bound"
    assert exploration["hypothetical_min_residual"]["rmse_kwh"] == 0
    assert exploration["accepted_cold_attribution_kwh"] == 0
    assert exploration["automatic_promotion"] is False


def test_refit_constants_match_live_model():
    assert observation.DAYPART_BOUNDS == tank.DAYPART_BOUNDS
    assert observation.REFIT_PRIOR_SCALES == (
        *((tank.PRIOR_HOT_FRACTION,) * 4),
        tank.PRIOR_STANDBY_W,
        tank.PRIOR_STANDBY_W_PER_K,
    )
    assert observation.REFIT_PARAM_BOUNDS == (
        *((tank.HOT_FRACTION_MIN, tank.HOT_FRACTION_MAX),) * 4,
        (tank.STANDBY_W_MIN, tank.STANDBY_W_MAX),
        (tank.STANDBY_W_PER_K_MIN, tank.STANDBY_W_PER_K_MAX),
    )
    assert observation.EVAL_MIN_CYCLE_S == tank.LEARN_MIN_CYCLE_HOURS * 3600
    for hour in range(24):
        inputs = observation.ObservationInputs(
            iso(), "same", 0, False, 0, False, False, local_hour=hour
        )
        assert observation._daypart(inputs) == tank.daypart_bucket(hour)


def test_anchor_features_include_final_tick_and_reproduce_multitick_physics():
    start = BASE.replace(hour=5, minute=58)
    params = tank.TankParams(300, 75, 12)
    e = tank.draw_kwh_from_liters(1, 1, 75, 12)
    live = replace(
        tank.initial_state(22),
        hot_fraction_profile=(0.2, 0.3, 0.4, 0.5),
        standby_w=90,
        standby_w_per_k=2,
    )
    saved, water, energy = None, 1000.0, 100.0
    last = 12 * 60 + 4  # 18:02, crossing every daypart boundary.
    for minute in range(last + 1):
        now = start + timedelta(minutes=minute)
        water += 2 if minute in (2, 4, 303, last) else 0
        energy += 0.001 if minute else 0
        previous = live
        anchored = minute in (0, last)
        model_inputs = tank.TickInputs(
            now_iso=now.isoformat(),
            elapsed_s=60,
            energy_counter_kwh=energy,
            water_counter_l=water,
            contactor_on=True,
            heating_on=not anchored,
            contactor_on_for_s=120,
            heating_off_for_s=60,
            local_hour=now.hour,
            ambient_c=10,
            people_home=1,
            e_base=3.0,
            e_draw_per_person=2.2,
            empty_house_factor=0.4,
        )
        result = tank.apply_tick(previous, params, model_inputs)
        live = result.state
        saved = observation.advance_observation(
            saved,
            observation.ObservationInputs(
                now_iso=now.isoformat(),
                source_signature="same",
                water_l=water,
                motion=False,
                humidity=40,
                ventilation=False,
                appliance_active=False,
                appliance_states={"washer": 1 <= minute <= 4 or 303 <= minute <= 305},
                household_activity=False,
                local_hour=now.hour,
                baseline_draw_kwh=result.draw_kwh,
                standby_kwh=result.standby_kwh,
                energy_in_kwh=result.energy_in_kwh,
                anchored=result.anchored,
                previous_cycle_unclamped_kwh=previous.cycle_unclamped_kwh or 0,
                previous_cycle_clean=previous.cycle_clean,
                previous_cycle_hot_liters_by_bucket=previous.cycle_hot_liters_by_bucket,
                previous_cycle_cold_kh=previous.cycle_cold_kh,
                previous_hot_fraction_profile=tank.profile_of(previous),
                previous_standby_w=previous.standby_w,
                previous_standby_w_per_k=previous.standby_w_per_k,
                kwh_per_hot_liter=e,
                tick_cold_kh=5 / 60,
            ),
        )
    row = saved["cycles"][0]
    features = row["features"]
    assert features["hot_liters_by_bucket"] == pytest.approx([0, 4, 2, 2])
    assert features["appliances"]["washer"] == {
        "run_starts_by_bucket": [1, 0, 1, 0],
        "liters_by_bucket": [0, 4, 2, 0],
    }
    assert features["cycle_hours"] == pytest.approx(last / 60)
    assert features["cold_kh"] == pytest.approx(5 * last / 60)
    assert features["cutoff_rebased"] is False
    p0 = observation._param_vector(features["params"])
    assert observation._refit_residual(row, p0, {}) == row["baseline_residual_kwh"]
    # This independent balance detects missing final-tick litres or cold hours;
    # 1e-12 covers only repeated floating-point integration over 724 ticks.
    expected = (
        e
        * sum(
            hf * liters for hf, liters in zip(p0[:4], features["hot_liters_by_bucket"], strict=True)
        )
        + p0[4] * features["cycle_hours"] / 1000
        + p0[5] * features["cold_kh"] / 1000
        - row["energy_in_kwh"]
    )
    assert expected == pytest.approx(row["baseline_residual_kwh"], abs=1e-12)
    assert features["params"]["standby_w"] != live.standby_w
    json.dumps(saved, allow_nan=False)


def refit_state(*, cold_heldout=True, count=48):
    """Known physics: TRAIN is cold; optional held-out regime makes it fail."""
    saved = {
        "mode": "shadow",
        "source_signature": "same",
        "cycles": [],
        "appliance_observations": {"runs": []},
    }
    profile = [0.25, 0.3, 0.35, 0.4]
    e = tank.draw_kwh_from_liters(1, 1, 75, 12)
    for index in range(count):
        start = BASE + timedelta(days=index)
        heldout = index >= count // 2
        n = 3 if heldout and not cold_heldout else 1 + index % 3
        bucket = index % 4
        counts = [n if b == bucket else 0 for b in range(4)]
        residual = e * profile[bucket] * 20 * n if cold_heldout or not heldout else 0.0
        saved["cycles"].append(
            {
                "start": start.isoformat(),
                "end": (start + timedelta(days=1)).isoformat(),
                "complete": True,
                "clean": True,
                "synchronized": True,
                "water_l": 600,
                "source_signature": "same",
                "baseline_residual_kwh": residual,
                "shadow_residual_kwh": residual,
                "features": {
                    "hot_liters_by_bucket": [
                        80 + (index * (b + 3) * 17 + b * 31) % 160 for b in range(4)
                    ],
                    "cycle_hours": 24.0,
                    "cold_kh": 24 * (index % 9),
                    "kwh_per_hot_liter": e,
                    "params": {
                        "hot_fraction_profile": profile[:],
                        "standby_w": 100.0,
                        "standby_w_per_k": 2.0,
                    },
                    "cutoff_rebased": False,
                    "appliances": {
                        "washer": {
                            "run_starts_by_bucket": counts,
                            "liters_by_bucket": [20 * c for c in counts],
                        }
                    },
                },
            }
        )
    for index in range(5):
        saved["appliance_observations"]["runs"].append(
            {
                "source": "washer",
                "source_signature": "same",
                "complete": True,
                "start": (BASE + timedelta(hours=index * 2)).isoformat(),
                "end": (BASE + timedelta(hours=index * 2 + 1)).isoformat(),
                "observed_interior_household_l": 20.0,
                "observed_interior_seconds": 3600.0,
                "gross_household_l": 20.0,
                "interior_profile_usable": True,
                "household_flow_episodes": 1,
                "quality_flags": ["state_transition_interval"],
            }
        )
    return saved


def test_refit_passes_when_appliance_water_is_cold_without_mutating_saved():
    saved = refit_state()
    before = copy.deepcopy(saved)
    report = observation.evaluate_appliance_refit(saved)
    assert saved == before
    assert report["status"] == "passes_shadow_comparison"
    assert report["reasons"] == []
    assert report["C2"]["rmse_kwh"] < report["B1"]["rmse_kwh"]
    assert abs(report["C2"]["bias_kwh"]) <= abs(report["B1"]["bias_kwh"]) + 1e-12
    assert report["n_heldout_with_consistent_runs"] == 24
    assert report["per_appliance"]["washer"] == {
        "n": 5,
        "median_l": 20,
        "mean_l": 20,
        "sd_l": 0,
        "cv": 0,
        "consistent": True,
    }
    assert report["automatic_promotion"] is False
    assert report["accepted_cold_attribution_kwh"] == 0
    assert report["interpretation"] == "shadow_refit_comparison_not_control"
    published = observation.summarize_observation(saved)
    assert published["evaluation"]["appliance_refit"] == report
    json.dumps(published, allow_nan=False)


def test_refit_fails_when_deduction_worsens_heldout_bias():
    report = observation.evaluate_appliance_refit(refit_state(cold_heldout=False))
    assert report["status"] == "fails_shadow_comparison"
    assert abs(report["C2"]["bias_kwh"]) > abs(report["B1"]["bias_kwh"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("flag", sorted(observation.REFIT_RUN_REJECTIONS))
def test_refit_rejects_every_unclean_run_flag(flag):
    saved = refit_state()
    bad_run = copy.deepcopy(saved["appliance_observations"]["runs"][0])
    bad_run.update(quality_flags=[flag], observed_interior_household_l=10000)
    saved["appliance_observations"]["runs"].append(bad_run)
    assert observation.evaluate_appliance_refit(saved)["per_appliance"]["washer"]["n"] == 5


def test_refit_run_statistics_do_not_leak_future_or_old_setup_data():
    saved = refit_state()
    run = saved["appliance_observations"]["runs"][0]
    split = saved["cycles"][24]["start"]
    for overrides in (
        {"start": split, "end": iso(40 * 24 * 60)},
        {"start": iso(), "end": iso(40 * 24 * 60)},
        {"source_signature": "old"},
        {"complete": False},
        {"observed_interior_household_l": float("nan")},
        {"observed_interior_household_l": 0},
    ):
        saved["appliance_observations"]["runs"].append(
            {**run, "observed_interior_household_l": 10000, **overrides}
        )
    assert observation.evaluate_appliance_refit(saved)["per_appliance"]["washer"]["n"] == 5


def test_refit_ignores_legacy_rebased_and_invalid_cycles():
    saved = refit_state()
    saved["cycles"][0].pop("features")
    saved["cycles"][1]["features"]["cutoff_rebased"] = True
    saved["cycles"][2]["features"]["kwh_per_hot_liter"] = float("nan")
    saved["cycles"][3]["features"]["hot_liters_by_bucket"] = []
    assert observation.evaluate_appliance_refit(saved)["eligible_cycles"] == 44
    legacy = evaluation_state()
    report = observation.evaluate_appliance_refit(legacy)
    assert report["eligible_cycles"] == 0
    assert report["reasons"] == [
        "too_few_cycles",
        "no_consistent_appliance",
        "too_few_heldout_cycles_with_runs",
    ]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "count, expected", [(11, "too_few_cycles"), (12, "too_few_heldout_cycles_with_runs")]
)
def test_refit_minimum_cycle_evidence(count, expected):
    report = observation.evaluate_appliance_refit(refit_state(count=count))
    assert report["status"] == "insufficient_evidence"
    assert expected in report["reasons"]


@pytest.mark.parametrize("volumes", [[20] * 4, [5, 10, 20, 30, 40]])
def test_refit_requires_five_consistent_volume_samples(volumes):
    saved = refit_state()
    runs = saved["appliance_observations"]["runs"][: len(volumes)]
    saved["appliance_observations"]["runs"] = runs
    for run, volume in zip(runs, volumes, strict=True):
        run["observed_interior_household_l"] = volume
    report = observation.evaluate_appliance_refit(saved)
    assert report["status"] == "insufficient_evidence"
    assert "no_consistent_appliance" in report["reasons"]
    assert report["per_appliance"]["washer"]["consistent"] is False
    assert report["B1"] == report["C2"]


def test_refit_cv_threshold_is_strict():
    saved = refit_state()
    sample = saved["appliance_observations"]["runs"][0]
    saved["appliance_observations"]["runs"] = [
        {**sample, "observed_interior_household_l": volume} for volume in [15, 25] * 4
    ]
    stats = observation.evaluate_appliance_refit(saved)["per_appliance"]["washer"]
    assert stats["cv"] == 0.25
    assert stats["consistent"] is False


def test_offline_evaluation_tool_loads_refit_standalone():
    path = _PATH.parents[2] / "tools" / "evaluate_tank_observations.py"
    spec = importlib.util.spec_from_file_location("lnp_refit_tool", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    saved = refit_state()
    stores = tool.find_observations({"load": {"observation": saved}})
    reports = {
        name: tool.load_model().evaluate_observations(state) for name, state in stores.items()
    }
    assert (
        reports["root.load.observation"]["appliance_refit"]["status"] == "passes_shadow_comparison"
    )
    json.dumps(reports, allow_nan=False)


def test_refit_ridge_solution_and_hard_bounds():
    row = refit_state()["cycles"][0]
    row["features"].update(hot_liters_by_bucket=[0] * 4, cycle_hours=1000, cold_kh=0)
    row["baseline_residual_kwh"] = 10
    fitted = observation._fit_refit([row], {})
    assert fitted[4] == pytest.approx(100 - 900 * 10 / 901)
    assert fitted[:4] == row["features"]["params"]["hot_fraction_profile"]
    for residual, expected in ((1e6, 20), (-1e6, 250)):
        row["baseline_residual_kwh"] = residual
        assert observation._fit_refit([row], {})[4] == expected


def test_cutoff_flag_is_sticky_but_clears_for_next_observation_cycle():
    state = appliance_tick(anchored=True)
    state = appliance_tick(state, 1, previous_cutoff_rebased=True)
    assert state["cycle"]["features"]["cutoff_rebased"] is True
    state = appliance_tick(state, 2)
    state = appliance_tick(
        state,
        3,
        anchored=True,
        kwh_per_hot_liter=0.07,
        previous_cycle_hot_liters_by_bucket=(0, 0, 0, 0),
        previous_cycle_cold_kh=0,
        previous_hot_fraction_profile=(0.25,) * 4,
        previous_standby_w=100,
        previous_standby_w_per_k=0,
    )
    assert state["cycles"][0]["features"]["cutoff_rebased"] is True
    assert state["cycle"]["features"]["cutoff_rebased"] is False


def test_legacy_in_progress_cycle_is_not_given_partial_features():
    state = appliance_tick(anchored=True)
    state["cycle"].pop("features")
    state = appliance_tick(
        state,
        1,
        anchored=True,
        kwh_per_hot_liter=0.07,
        previous_cycle_hot_liters_by_bucket=(0, 0, 0, 0),
        previous_cycle_cold_kh=0,
        previous_hot_fraction_profile=(0.25,) * 4,
        previous_standby_w=100,
        previous_standby_w_per_k=0,
    )
    assert "features" not in state["cycles"][0]
