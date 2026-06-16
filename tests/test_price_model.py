"""Unit tests for the pure price model (no Home Assistant).

Loaded standalone via ``importlib`` like ``test_predictor.py``.
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
    / "price_model.py"
)
_spec = importlib.util.spec_from_file_location("lnp_price_model", _PATH)
pm = importlib.util.module_from_spec(_spec)
sys.modules["lnp_price_model"] = pm
_spec.loader.exec_module(pm)


def _true(temp: float, wind: float) -> float:
    """A known price surface (base lifted so nothing clamps to 0 on the grid)."""
    ch = max(0.0, -temp)
    return 0.12 - 0.001 * temp - 0.008 * wind + 0.004 * ch - 0.0008 * wind * ch


def _grid(l2_prices=_true) -> list[tuple[float, float, float]]:
    rows = []
    for temp in range(-20, 21, 2):
        for wind in (1.0, 3.0, 5.0):
            rows.append((float(temp), wind, l2_prices(temp, wind)))
    return rows


# ── features ─────────────────────────────────────────────────────────────────


def test_cold_hinge():
    assert pm.cold_hinge(10) == 0.0  # warm → no cold term
    assert pm.cold_hinge(0) == 0.0
    assert pm.cold_hinge(-15) == 15.0


def test_build_features_interaction():
    f = pm.build_features(-10.0, 2.0)
    assert f == [-10.0, 2.0, 10.0, 20.0]  # temp, wind, cold_hinge, wind*cold_hinge
    assert pm.build_features(5.0, 2.0) == [5.0, 2.0, 0.0, 0.0]


# ── seed formula ─────────────────────────────────────────────────────────────


def test_seed_cold_dearer_than_warm():
    assert pm.seed_predict(-15, 2) > pm.seed_predict(15, 2)


def test_seed_more_wind_cheaper():
    assert pm.seed_predict(-15, 5) < pm.seed_predict(-15, 1)


def test_seed_clamped_non_negative():
    # A very warm, very windy day drives the raw formula below zero.
    assert pm.seed_predict(40, 50) == 0.0


# ── fit ──────────────────────────────────────────────────────────────────────


def test_fit_too_few_rows_returns_none():
    rows = [(0.0, 2.0, 0.1)] * (pm.MIN_FIT_ROWS - 1)
    assert pm.fit(rows) is None


def test_fit_recovers_known_surface_without_ridge():
    model = pm.fit(_grid(), l2=0.0)  # pure OLS → exact on noiseless linear data
    assert model is not None
    for temp, wind in ((-18.0, 1.0), (-6.0, 3.0), (8.0, 5.0), (18.0, 1.0)):
        assert model.predict(temp, wind) == pytest.approx(_true(temp, wind), abs=1e-6)


def test_fit_with_ridge_is_close():
    model = pm.fit(_grid())  # default L2 → slight shrinkage, still close
    assert pm.mean_abs_error(model, _grid()) < 0.01


def test_fitted_model_directional():
    model = pm.fit(_grid())
    assert model.predict(-15, 2) > model.predict(15, 2)  # colder dearer
    assert model.predict(-15, 5) < model.predict(-15, 1)  # more wind cheaper


def test_predict_clamped_non_negative():
    # Fit on data whose surface goes negative when warm+windy; predict clamps.
    rows = [(float(t), w, _true(t, w) - 0.15) for t in range(-20, 21, 2) for w in (1.0, 5.0)]
    model = pm.fit(rows, l2=0.0)
    assert model.predict(20, 5) >= 0.0


# ── serialization ────────────────────────────────────────────────────────────


def test_to_from_dict_round_trip():
    model = pm.fit(_grid())
    restored = pm.FittedModel.from_dict(model.to_dict())
    assert restored == model


def test_from_dict_none_and_garbage():
    assert pm.FittedModel.from_dict(None) is None
    assert pm.FittedModel.from_dict({}) is None
    assert (
        pm.FittedModel.from_dict({"betas": ["x"], "means": [], "stds": [], "intercept": 0, "n": 1})
        is None
    )


# ── predict_price dispatch + MAE ─────────────────────────────────────────────


def test_predict_price_falls_back_to_seed_when_unfitted():
    assert pm.predict_price(None, -15, 2) == pm.seed_predict(-15, 2)


def test_predict_price_uses_model_when_fitted():
    model = pm.fit(_grid())
    assert pm.predict_price(model, -15, 2) == model.predict(-15, 2)


def test_mean_abs_error_empty():
    assert pm.mean_abs_error(None, []) is None


def test_no_homeassistant_import():
    assert "homeassistant" not in _PATH.read_text()
