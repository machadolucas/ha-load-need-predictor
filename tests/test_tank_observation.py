"""Shadow attribution is standalone and cannot change the tank's control state."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import pathlib
import sys
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
