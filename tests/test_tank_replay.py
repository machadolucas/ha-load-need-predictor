"""Regression test: replay a real week of telemetry through the tank model.

Pure — ``tools/replay_tank.py`` is loaded standalone via ``importlib`` (like
``tests/test_predictor.py`` loads ``predictor``), and it in turn loads
``tank_model.py`` by path, so nothing here touches Home Assistant.

Why this test exists on top of ``test_tank_model.py``: the unit tests pin single
ticks and single functions, but the estimator is a *continuous integrator with
self-correcting anchors*, and its real failure modes are shapes over hours —
the clamp swallowing delivered energy and paying it back as a 13-point jump, the
display frozen at one value for an afternoon, a 100 % reading while the element
is visibly drawing. Only a replayed week can see those, so the week is a
fixture and the assertions below are the acceptance criteria the v2 rework had
to meet (see the module docstring of ``tank_model.py``). Everything is a
one-sided bound, not an exact expected value: the point is to fail on a
regression in *behaviour*, not to freeze arithmetic that a legitimate model
improvement would shift.

Runs ~10k ticks in well under a second.
"""

from __future__ import annotations

import importlib.util
import pathlib
import statistics
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PATH = _ROOT / "tools" / "replay_tank.py"
_spec = importlib.util.spec_from_file_location("lnp_replay_tank", _PATH)
replay_tank = importlib.util.module_from_spec(_spec)
sys.modules["lnp_replay_tank"] = replay_tank
_spec.loader.exec_module(replay_tank)

FIXTURE = _ROOT / "tests" / "fixtures" / "tank_week_2026-08-28.csv"


@pytest.fixture(scope="module")
def week():
    """The week replayed once, learning ON, from the params the live sensor ran."""
    return replay_tank.replay(
        FIXTURE,
        hot_fraction=replay_tank.LIVE_HOT_FRACTION,
        standby_w=replay_tank.LIVE_STANDBY_W,
        learn=True,
    )


def test_replay_covers_the_whole_week(week):
    """Sanity: the fixture really is a week of minutes with the element running."""
    assert week["ticks"] > 9_000
    assert week["heating_minutes"] > 500


def test_no_non_anchor_soc_jumps(week):
    """SoC may only step discontinuously at an anchor (where physics re-zeroes it).

    Any other ≥3-point step in one minute is the estimator contradicting itself —
    the v0.8 clamp did exactly this, discarding delivered energy and then paying
    it back in 13–19-point lurches.
    """
    assert week["jumps"] == [], f"non-anchor SoC jumps: {week['jumps']}"


def test_trip_residuals_are_small_and_unbiased(week):
    """At a thermostat trip the unclamped ledger *is* the model's cycle error.

    A tank that holds ~22 kWh should close a real cycle within ~1 kWh, and the
    bias must stay near zero — a systematic bias is a mis-attributed draw
    (hot fraction / standby), which the learner would then chase.
    """
    assert week["residual_rms"] < 1.5, week["residuals"]
    assert abs(week["residual_bias"]) < 0.6, week["residuals"]


def test_the_week_anchors_often_enough_to_learn(week):
    """The anchor is the only ground truth: it has to fire, and often.

    ``long_cycle_anchors`` are the ones closing a ≥4 h cycle — the only ones that
    teach the hot-fraction profile / standby — so both counts are asserted.
    """
    assert week["long_cycle_anchors"] >= 5
    assert week["anchors"] >= 10


def test_anchors_read_exactly_full(week):
    """An anchor is ground truth "the tank is full" ⇒ the display must say 100.0.

    This is the one place the saturation curve is bypassed; if an anchor ever
    showed 99.x the card would never reach full.
    """
    assert all(value == pytest.approx(100.0) for value in week["anchor_socs"]), week["anchor_socs"]


def test_soc_declines_after_a_trip_without_falling_off_a_cliff(week):
    """After the trip the element idles while mixing pulls the tank back down.

    Half an hour later the display should have drifted a couple of points below
    full: still ≥94 (a bigger drop would be the old post-anchor collapse) and
    ≤99.5 (flat at 100 would be the pre-v2 behaviour that hid the re-heat).
    """
    median_30 = week["post_trip_median"][30]
    assert median_30 is not None, "no idle+latched samples ~30 min after a trip"
    assert 94.0 <= median_30 <= 99.5, week["post_trip_soc"][30]


def test_no_frozen_soc_plateau_while_heating(week):
    """While the element draws, the displayed % must keep moving.

    v0.8 pinned it at 95.4 for hours (the heating-active deficit floor), which
    made the card look broken. A run of ≥20 identical minutes at a high SoC is
    that pathology returning.
    """
    assert week["max_plateau_run"] < 20, f"identical-SoC run of {week['max_plateau_run']} ticks"


def test_never_full_while_the_element_is_still_drawing(week):
    """The element drawing proves the tank is below setpoint — so SoC < 100.

    Only a genuine anchor may read exactly full; the saturation curve's heavy
    tail is what keeps the approach asymptotic instead of clipping.
    """
    assert week["heating_soc_max"] < 100.0


def test_median_post_trip_samples_are_consistent(week):
    """The 5/30/60-minute post-trip samples must decline monotonically.

    The relaxation is a single exponential toward ``hysteresis_kwh``, so a later
    sample can never read higher than an earlier one; if it does, energy is
    being double-counted somewhere in the relax ledger.
    """
    medians = [week["post_trip_median"][minute] for minute in (5, 30, 60)]
    present = [value for value in medians if value is not None]
    assert len(present) >= 2
    assert present == sorted(present, reverse=True), medians
    # And the decline is real, not rounding noise.
    assert present[0] - present[-1] > 1.0


def test_learning_stays_inside_the_guardrails(week):
    """A week of ±1 kWh cycle noise must not walk the learnable params away.

    The v2 learner is deliberately slow; if a week moves the hot fraction by
    more than a few percent it is chasing noise (the failure the fast learner
    showed in the replay).
    """
    state = week["final_state"]
    profile = state.hot_fraction_profile or (state.hot_fraction,) * 4
    assert all(0.15 <= value <= 0.40 for value in profile), profile
    assert abs(statistics.fmean(profile) - replay_tank.LIVE_HOT_FRACTION) < 0.05
    assert 60.0 <= state.standby_w <= 170.0
    assert 0.2 <= state.hysteresis_kwh <= 1.5
