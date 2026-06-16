"""Pure beyond-horizon electricity-price model.

**No Home Assistant imports** — this is the testable heart of the price-forecast
capability, loaded standalone (via ``importlib``) by the pure unit tests.

We forecast *price* (opportunity), not demand: hot-water demand is essentially
unpredictable day to day, but Finnish day-after-tomorrow price is far more
forecastable from weather. On ~356 days of the author's data:

- more wind ⇒ lower price (r ≈ −0.23 overall, −0.49 when below −5 °C),
- colder ⇒ higher price (r ≈ −0.45 overall, weak when warm),
- jointly R² ≈ 0.37, stronger in the cold; mean ~11.4 c/kWh on cold days vs
  ~4.2 c/kWh on warm ones.

So the model is a small, explainable linear regression with a **cold-weather
interaction**: the price rises as temperature drops below freezing, and wind's
price-suppressing effect strengthens in the cold.

    features = [temp, wind, cold_hinge, wind × cold_hinge]   # cold_hinge = max(0, −temp)
    price ≈ intercept + Σ βᵢ · zᵢ                            # zᵢ = standardised feature

Fit by ridge-regularised least squares on standardised features (so the very
different scales — °C, GW, degree-hinge — don't destabilise the solve). A
hand-seeded formula covers the cold start before enough history exists.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Temperature (°C) below which the "cold" regime kicks in (the hinge knee).
COLD_THRESHOLD = 0.0

# Minimum clean daily rows before we trust a fitted model over the seed formula.
MIN_FIT_ROWS = 20

# Ridge penalty in standardised feature space (features have unit variance, and
# n is ~hundreds, so a small constant only nudges an ill-conditioned solve).
DEFAULT_L2 = 1.0

FEATURE_COUNT = 4

# ── Seed formula (cold start) — rough €/kWh, replaced by the fit ──────────────
SEED_BASE = 0.06
SEED_TEMP = -0.001  # per °C (warmer ⇒ slightly cheaper)
SEED_WIND = -0.008  # per GW (more wind ⇒ cheaper)
SEED_COLD = 0.004  # per °C below freezing (colder ⇒ dearer)
SEED_WIND_COLD = -0.0008  # wind suppresses price more in the cold


def cold_hinge(temp: float) -> float:
    """Degrees below the cold threshold (0 when warm)."""
    return max(0.0, COLD_THRESHOLD - temp)


def build_features(temp: float, wind: float) -> list[float]:
    """The four model features for one day's mean temperature + wind."""
    ch = cold_hinge(temp)
    return [temp, wind, ch, wind * ch]


def seed_predict(temp: float, wind: float) -> float:
    """Hand-seeded price (€/kWh) used until a fit exists. Clamped non-negative."""
    ch = cold_hinge(temp)
    price = (
        SEED_BASE
        + SEED_TEMP * temp
        + SEED_WIND * wind
        + SEED_COLD * ch
        + SEED_WIND_COLD * wind * ch
    )
    return max(0.0, price)


@dataclass(frozen=True)
class FittedModel:
    """A standardised ridge fit. ``predict`` reverses the standardisation."""

    means: tuple[float, ...]
    stds: tuple[float, ...]
    betas: tuple[float, ...]
    intercept: float
    n: int

    def predict(self, temp: float, wind: float) -> float:
        raw = build_features(temp, wind)
        price = self.intercept
        for value, mean, std, beta in zip(raw, self.means, self.stds, self.betas, strict=True):
            price += beta * ((value - mean) / std)
        return max(0.0, price)

    def to_dict(self) -> dict:
        return {
            "means": list(self.means),
            "stds": list(self.stds),
            "betas": list(self.betas),
            "intercept": self.intercept,
            "n": self.n,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> FittedModel | None:
        if not data or "betas" not in data:
            return None
        try:
            return cls(
                means=tuple(float(x) for x in data["means"]),
                stds=tuple(float(x) for x in data["stds"]),
                betas=tuple(float(x) for x in data["betas"]),
                intercept=float(data["intercept"]),
                n=int(data["n"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Solve ``matrix · x = rhs`` by Gaussian elimination with partial pivoting."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None  # singular
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col] / pivot_val
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def fit(rows: Sequence[tuple[float, float, float]], l2: float = DEFAULT_L2) -> FittedModel | None:
    """Ridge-fit price on [temp, wind, cold_hinge, wind·cold_hinge].

    ``rows`` is a sequence of ``(temp, wind, price)``. Returns ``None`` (so the
    caller falls back to the seed) when there are too few rows or the system is
    singular (e.g. no variation in the features).
    """
    n = len(rows)
    if n < MIN_FIT_ROWS:
        return None

    feats = [build_features(t, w) for t, w, _ in rows]
    prices = [p for _, _, p in rows]

    means = [sum(col) / n for col in zip(*feats, strict=True)]
    stds = []
    for j, mean in enumerate(means):
        var = sum((feats[i][j] - mean) ** 2 for i in range(n)) / n
        stds.append(math.sqrt(var) or 1.0)  # a constant feature → std 1 (no scaling)

    z = [[(feats[i][j] - means[j]) / stds[j] for j in range(FEATURE_COUNT)] for i in range(n)]
    intercept = sum(prices) / n
    yc = [p - intercept for p in prices]

    # Normal equations with ridge: (ZᵀZ + λI) β = Zᵀ yc
    ztz = [
        [
            sum(z[i][a] * z[i][b] for i in range(n)) + (l2 if a == b else 0.0)
            for b in range(FEATURE_COUNT)
        ]
        for a in range(FEATURE_COUNT)
    ]
    zty = [sum(z[i][a] * yc[i] for i in range(n)) for a in range(FEATURE_COUNT)]
    betas = _solve(ztz, zty)
    if betas is None:
        return None

    return FittedModel(
        means=tuple(means), stds=tuple(stds), betas=tuple(betas), intercept=intercept, n=n
    )


def predict_price(model: FittedModel | None, temp: float, wind: float) -> float:
    """Predicted price (€/kWh): the fitted model, or the seed when unfitted."""
    if model is None:
        return seed_predict(temp, wind)
    return model.predict(temp, wind)


def mean_abs_error(
    model: FittedModel | None, rows: Sequence[tuple[float, float, float]]
) -> float | None:
    """In-sample MAE of the model (or seed) over ``(temp, wind, price)`` rows."""
    if not rows:
        return None
    total = sum(abs(predict_price(model, t, w) - p) for t, w, p in rows)
    return total / len(rows)
