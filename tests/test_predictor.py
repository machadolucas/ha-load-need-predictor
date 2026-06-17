"""Unit tests for the pure prediction model (no Home Assistant).

Loaded standalone via ``importlib`` so these run without importing the HA-bound
package. Mirrors the ``test_baseline.py`` pattern in ha-load-scheduler.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "custom_components"
    / "load_need_predictor"
    / "predictor.py"
)
_spec = importlib.util.spec_from_file_location("lnp_predictor", _PATH)
predictor = importlib.util.module_from_spec(_spec)
sys.modules["lnp_predictor"] = predictor
_spec.loader.exec_module(predictor)

FeatureVector = predictor.FeatureVector
ModelState = predictor.ModelState


# ── default_model_state ──────────────────────────────────────────────────────


def test_default_model_state_uses_seeds():
    s = predictor.default_model_state()
    assert s.e_base == predictor.SEED_E_BASE
    assert s.e_draw_per_person == predictor.SEED_E_DRAW_PER_PERSON
    assert s.guest_bonus == predictor.SEED_GUEST_BONUS
    assert s.gain == 1.0
    assert s.sample_count == 0
    assert s.version == "v1"


# ── build_features ───────────────────────────────────────────────────────────


def test_build_features_maps_and_coerces():
    fv = predictor.build_features(
        {
            "people_home": 2,
            "guests": 0.5,
            "weekend": True,
            "supply_temp": "12.5",
            "outdoor_temp": 9,
            "inside_temp": None,
            "water_total_delta": "0.3",
        }
    )
    assert fv.people_home == 2
    assert fv.guests == 0.5  # guest-equivalent weight (no longer a 0/1 flag)
    assert fv.weekend is True
    assert fv.supply_temp == 12.5
    assert fv.outdoor_temp == 9.0
    assert fv.inside_temp is None
    assert fv.water_total_delta == 0.3


def test_build_features_missing_occupancy_assumes_one_person():
    # Conservative: unknown occupancy must not under-serve the tank.
    fv = predictor.build_features({})
    assert fv.people_home == 1
    assert fv.guests == 0.0
    assert fv.supply_temp is None


def test_build_features_unparseable_context_is_none():
    fv = predictor.build_features({"people_home": 0, "supply_temp": "n/a"})
    assert fv.people_home == 0
    assert fv.supply_temp is None


def test_build_features_negative_people_clamped_to_zero():
    fv = predictor.build_features({"people_home": -3})
    assert fv.people_home == 0


# ── predict_kwh ──────────────────────────────────────────────────────────────


def test_predict_kwh_two_people_matches_mean():
    s = predictor.default_model_state()
    kwh = predictor.predict_kwh(s, FeatureVector(people_home=2))
    assert kwh == pytest.approx(3.0 + 2 * 2.2)  # 7.4


def test_predict_kwh_empty_house_scales_base():
    s = predictor.default_model_state()
    kwh = predictor.predict_kwh(s, FeatureVector(people_home=0))
    assert kwh == pytest.approx(0.4 * 3.0)  # 1.2


def test_predict_kwh_guests_scale_with_weight():
    # guests is a guest-equivalent weight: the bonus scales with it.
    s = predictor.default_model_state()
    base = predictor.predict_kwh(s, FeatureVector(people_home=2, guests=0.0))
    short = predictor.predict_kwh(s, FeatureVector(people_home=2, guests=0.5))
    long = predictor.predict_kwh(s, FeatureVector(people_home=2, guests=2.0))
    assert short - base == pytest.approx(0.5 * 2.5)  # short visit ≈ +1.25 kWh
    assert long - base == pytest.approx(2.0 * 2.5)  # long visit ≈ +5.0 kWh


def test_predict_kwh_negative_guests_ignored():
    s = predictor.default_model_state()
    assert predictor.predict_kwh(
        s, FeatureVector(people_home=1, guests=-1.0)
    ) == predictor.predict_kwh(s, FeatureVector(people_home=1, guests=0.0))


def test_predict_kwh_monotonic_in_people():
    s = predictor.default_model_state()
    vals = [predictor.predict_kwh(s, FeatureVector(people_home=p)) for p in range(0, 5)]
    # Each additional person adds positive draw → strictly increasing.
    assert all(b > a for a, b in zip(vals, vals[1:], strict=False))


def test_predict_kwh_gain_scales_linearly():
    base = predictor.default_model_state()
    hot = predictor.ModelState(gain=1.2)
    fv = FeatureVector(people_home=2)
    assert predictor.predict_kwh(hot, fv) == pytest.approx(predictor.predict_kwh(base, fv) * 1.2)


def test_predict_kwh_temperature_ignored_in_v1():
    s = predictor.default_model_state()
    cold = FeatureVector(people_home=2, supply_temp=3.0, outdoor_temp=-5.0)
    warm = FeatureVector(people_home=2, supply_temp=18.0, outdoor_temp=25.0)
    assert predictor.predict_kwh(s, cold) == predictor.predict_kwh(s, warm)


# ── kwh_to_minutes ───────────────────────────────────────────────────────────


def test_kwh_to_minutes():
    assert predictor.kwh_to_minutes(3.0, 3.0) == pytest.approx(60.0)
    assert predictor.kwh_to_minutes(7.4, 3.0) == pytest.approx(148.0)


def test_kwh_to_minutes_rejects_nonpositive_power():
    with pytest.raises(ValueError):
        predictor.kwh_to_minutes(5.0, 0.0)


# ── clamp_minutes ────────────────────────────────────────────────────────────


def test_clamp_minutes_rounds_to_step():
    assert predictor.clamp_minutes(148, 40, 240) == 150
    assert predictor.clamp_minutes(151, 40, 240) == 150
    assert predictor.clamp_minutes(158, 40, 240) == 165


def test_clamp_minutes_respects_inward_bounds():
    # min 40 → smallest 15-step ≥ 40 is 45; max 240 → 240.
    assert predictor.clamp_minutes(10, 40, 240) == 45
    assert predictor.clamp_minutes(1000, 40, 240) == 240
    assert predictor.clamp_minutes(0, 40, 240) == 45


def test_clamp_minutes_degenerate_config():
    # min and max in the same 15-step bucket: still returns a valid value.
    assert predictor.clamp_minutes(100, 50, 55) == 60


# ── predict_minutes (end-to-end) ─────────────────────────────────────────────


def test_predict_minutes_end_to_end():
    s = predictor.default_model_state()
    fv = FeatureVector(people_home=2)
    # 7.4 kWh / 3 kW * 60 = 148 min → round to 150, within [45, 240].
    assert (
        predictor.predict_minutes(s, fv, rated_power_kw=3.0, min_minutes=40, max_minutes=240) == 150
    )


def test_predict_minutes_empty_house_hits_safety_floor():
    s = predictor.default_model_state()
    fv = FeatureVector(people_home=0)
    # 1.2 kWh / 3 kW * 60 = 24 min → below the 40-min floor → clamped to 45.
    assert (
        predictor.predict_minutes(s, fv, rated_power_kw=3.0, min_minutes=40, max_minutes=240) == 45
    )


# ── is_valid_delivery ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwh", "valid"),
    [
        (None, False),
        (0.0, False),
        (0.1, False),
        (0.2, True),
        (7.0, True),
        (18.0, True),
        (18.1, False),
    ],
)
def test_is_valid_delivery(kwh, valid):
    assert predictor.is_valid_delivery(kwh) is valid


# ── update_gain / apply_observation ──────────────────────────────────────────


def test_update_gain_perfect_prediction_keeps_gain():
    s = predictor.ModelState(gain=1.0)
    out = predictor.update_gain(s, predicted_kwh=7.0, actual_kwh=7.0)
    assert out.gain == pytest.approx(1.0)


def test_update_gain_moves_toward_ratio():
    s = predictor.ModelState(gain=1.0)
    # actual double predicted → ratio 2.0, gain = 0.85*1 + 0.15*2 = 1.15.
    out = predictor.update_gain(s, predicted_kwh=5.0, actual_kwh=10.0)
    assert out.gain == pytest.approx(1.15)


def test_update_gain_ratio_is_clamped():
    s = predictor.ModelState(gain=1.0)
    huge = predictor.update_gain(s, predicted_kwh=1.0, actual_kwh=100.0)
    capped = predictor.update_gain(s, predicted_kwh=1.0, actual_kwh=2.0)  # ratio 2.0
    assert huge.gain == pytest.approx(capped.gain)  # ratio clamped at 2.0


def test_update_gain_never_exceeds_bounds():
    s = predictor.ModelState(gain=1.0)
    for _ in range(100):  # relentless over-delivery
        s = predictor.update_gain(s, predicted_kwh=1.0, actual_kwh=100.0)
    assert s.gain <= predictor.GAIN_MAX
    s = predictor.ModelState(gain=1.0)
    for _ in range(100):  # relentless under-delivery
        s = predictor.update_gain(s, predicted_kwh=10.0, actual_kwh=0.1)
    assert s.gain >= predictor.GAIN_MIN


def test_update_gain_noop_for_tiny_prediction():
    s = predictor.ModelState(gain=1.3)
    assert predictor.update_gain(s, predicted_kwh=0.1, actual_kwh=9.0).gain == 1.3


def test_apply_observation_updates_gain_and_count():
    s = predictor.default_model_state()
    out = predictor.apply_observation(s, predicted_kwh=5.0, actual_kwh=10.0)
    assert out.sample_count == 1
    assert out.gain == pytest.approx(1.15)


# ── blend_param ──────────────────────────────────────────────────────────────


def test_blend_param_no_data_is_prior():
    assert predictor.blend_param(3.0, 9.0, n=0) == 3.0


def test_blend_param_equal_weight_at_n_prior():
    assert predictor.blend_param(3.0, 9.0, n=predictor.N_PRIOR) == pytest.approx(6.0)


def test_blend_param_large_n_approaches_empirical():
    assert predictor.blend_param(3.0, 9.0, n=1000) == pytest.approx(9.0, abs=0.1)


# ── refit_occupancy_params ───────────────────────────────────────────────────


def test_refit_recovers_known_line():
    # actual = 3.0 + 2.2 * people, exactly.
    rows = [(p, 3.0 + 2.2 * p) for p in (0, 0, 1, 2, 2)]
    e_base, e_draw = predictor.refit_occupancy_params(rows)
    assert e_base == pytest.approx(3.0)
    assert e_draw == pytest.approx(2.2)


def test_refit_needs_variation():
    assert predictor.refit_occupancy_params([(2, 7.0), (2, 8.0)]) is None


def test_refit_needs_two_rows():
    assert predictor.refit_occupancy_params([(2, 7.0)]) is None


def test_refit_floors_negative_at_zero():
    # A perverse downward fit must not yield negative parameters.
    rows = [(0, 10.0), (1, 5.0), (2, 0.0)]
    e_base, e_draw = predictor.refit_occupancy_params(rows)
    assert e_base >= 0.0
    assert e_draw == 0.0


# ── rolling_mae ──────────────────────────────────────────────────────────────


def test_rolling_mae_basic():
    assert predictor.rolling_mae([1.0, -3.0, 2.0]) == pytest.approx(2.0)


def test_rolling_mae_empty_is_zero():
    assert predictor.rolling_mae([]) == 0.0


def test_rolling_mae_ignores_none():
    assert predictor.rolling_mae([None, 4.0, None, 2.0]) == pytest.approx(3.0)


# ── explain_load ───────────────────────────────────────────────────────────--


def _explain(state, features, **kw):
    defaults = {"rated_power_kw": 3.0, "min_minutes": 40, "max_minutes": 240}
    defaults.update(kw)
    return predictor.explain_load(state, features, **defaults)


def test_explain_load_matches_predict():
    # The rationale must never drift from the real prediction.
    state = predictor.default_model_state()
    fv = FeatureVector(people_home=2, guests=0.5)
    info = _explain(state, fv)
    assert info["predicted_kwh"] == predictor.predict_kwh(state, fv)
    assert info["predicted_minutes"] == predictor.predict_minutes(
        state, fv, rated_power_kw=3.0, min_minutes=40, max_minutes=240
    )


def test_explain_load_terms_add_up():
    state = predictor.default_model_state()  # E_base 3.0, draw 2.2, guest_bonus 2.5
    info = _explain(state, FeatureVector(people_home=2, guests=1.0))
    assert info["occupancy_factor"] == 1.0
    assert info["base_kwh"] == pytest.approx(7.4)  # 3.0 + 2*2.2
    assert info["occupied_kwh"] == pytest.approx(7.4)
    assert info["guest_kwh"] == pytest.approx(2.5)  # guest_bonus * 1.0
    assert info["pre_gain_kwh"] == pytest.approx(9.9)
    assert info["predicted_kwh"] == pytest.approx(9.9)  # gain 1.0


def test_explain_load_empty_house_uses_factor():
    state = predictor.default_model_state()
    info = _explain(state, FeatureVector(people_home=0))
    assert info["occupancy_factor"] == predictor.SEED_EMPTY_HOUSE_FACTOR
    assert info["occupied_kwh"] == pytest.approx(
        predictor.SEED_EMPTY_HOUSE_FACTOR * predictor.SEED_E_BASE
    )


def test_explain_load_clamped_flag():
    state = predictor.default_model_state()
    # Empty house → ~1.2 kWh ≈ 24 min, below the floor → bound applied.
    low = _explain(state, FeatureVector(people_home=0))
    assert low["clamped"] is True
    assert low["predicted_minutes"] == 45  # 40-min floor pulled up to the 15-min step
    # A normal 2-person day lands inside the band → no bound applied.
    mid = _explain(state, FeatureVector(people_home=2))
    assert mid["clamped"] is False
    assert mid["predicted_minutes"] == 150


def test_no_homeassistant_import():
    # Guard the pure contract: the model must never pull Home Assistant.
    src = _PATH.read_text()
    assert "homeassistant" not in src
    assert not math.isnan(predictor.predict_kwh(predictor.default_model_state(), FeatureVector(1)))
