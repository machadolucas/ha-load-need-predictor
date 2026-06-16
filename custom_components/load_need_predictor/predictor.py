"""Pure prediction model for Load Need Predictor.

**This module imports nothing from Home Assistant.** It is the testable heart of
the integration and is loaded standalone (via ``importlib``) by the pure unit
tests, so keep it dependency-free.

Why the model looks the way it does (see CLAUDE.md for the full analysis): on
~3–4 months of long-term statistics, daily hot-water energy turned out to be
dominated by stochastic *draw*, with temperature/season and total-water carrying
no usable day-ahead signal — a flat constant beat every weather model. The one
feature with real leverage, occupancy, has no statistics history. So the model
is a calibrated constant gated by occupancy:

    predicted_kWh = occupancy_factor × [E_base + E_draw_per_person × people_home]
                    + guest_bonus × guests
    predicted_kWh ×= gain                      # online EWMA drift correction
    minutes = clamp(predicted_kWh / rated_kW × 60, min, max)

``E_base`` / ``E_draw_per_person`` are seeded from priors and refined from
self-logged data; ``gain`` tracks slow drift day to day. Temperature/water live
on the ``FeatureVector`` only so they are logged for a future v2 — the v1
``predict_kwh`` never reads them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

# ── Seed priors (cold start, day 1) ──────────────────────────────────────────
# Back-solved from the observed ~7.3 kWh/day mean: 2 people ≈ 3.0 + 2×2.2 = 7.4.
SEED_E_BASE = 3.0  # kWh/day: standby + minimal-occupancy floor (~p10 of observed)
SEED_E_DRAW_PER_PERSON = 2.2  # kWh per resident physically home that day
SEED_GUEST_BONUS = 2.5  # kWh added when guests are present (~1 extra person)
SEED_GAIN = 1.0  # neutral multiplicative calibration
SEED_EMPTY_HOUSE_FACTOR = 0.4  # scale base when nobody is home (still some standby)

# ── Online-gain tuning ───────────────────────────────────────────────────────
GAIN_BETA = 0.15  # EWMA weight on the newest day → ~6-day half-life
GAIN_MIN = 0.7  # anti-drift guardrail: the gain may correct ±50%, never run away
GAIN_MAX = 1.5
RATIO_MIN = 0.5  # reject single-day outliers before they hit the gain
RATIO_MAX = 2.0

# ── Prior ↔ empirical blend ──────────────────────────────────────────────────
# The prior is worth this many days of data; after ~N_PRIOR logged days the
# empirical estimate carries half the weight.
N_PRIOR = 10

# ── Data-quality gate (kWh/day) ──────────────────────────────────────────────
# Exclude meter resets / zero-read days from calibration (≈5 of 90 in the data).
ENERGY_VALID_MIN = 0.2
ENERGY_VALID_MAX = 18.0

# ── Output rounding ──────────────────────────────────────────────────────────
DEFAULT_STEP_MINUTES = 15  # the scheduler's target number accepts 15-min steps

MODEL_VERSION = "v1"


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` to the inclusive ``[lo, hi]`` range."""
    return max(lo, min(value, hi))


@dataclass(frozen=True)
class FeatureVector:
    """Inputs for one day's prediction.

    Only ``people_home`` and ``guests`` drive the v1 model; the remaining fields
    are recorded for evaluation and a future weather-aware model but are not read
    by :func:`predict_kwh`.
    """

    people_home: int
    guests: bool = False
    weekend: bool = False
    # Log-only context (None when the source sensor is missing/unavailable).
    supply_temp: float | None = None
    outdoor_temp: float | None = None
    inside_temp: float | None = None
    water_total_delta: float | None = None


@dataclass(frozen=True)
class ModelState:
    """The learnable + seeded parameters. Immutable; updates return a new state."""

    e_base: float = SEED_E_BASE
    e_draw_per_person: float = SEED_E_DRAW_PER_PERSON
    guest_bonus: float = SEED_GUEST_BONUS
    gain: float = SEED_GAIN
    empty_house_factor: float = SEED_EMPTY_HOUSE_FACTOR
    sample_count: int = 0  # valid daily outcomes folded in so far (confidence)
    version: str = MODEL_VERSION


def default_model_state() -> ModelState:
    """The seeded model used on day 1, before any data has been logged."""
    return ModelState()


def build_features(snapshot: dict) -> FeatureVector:
    """Assemble a :class:`FeatureVector` from a plain snapshot dict.

    The Home-Assistant layer (occupancy + statistics sources) produces the dict;
    this stays pure so it is unit-testable. Occupancy missing → assume one person
    present (conservative: never under-serve the tank). Context temps stay ``None``
    when absent.
    """
    people = snapshot.get("people_home")
    people = 1 if people is None else max(0, int(people))
    return FeatureVector(
        people_home=people,
        guests=bool(snapshot.get("guests", False)),
        weekend=bool(snapshot.get("weekend", False)),
        supply_temp=_opt_float(snapshot.get("supply_temp")),
        outdoor_temp=_opt_float(snapshot.get("outdoor_temp")),
        inside_temp=_opt_float(snapshot.get("inside_temp")),
        water_total_delta=_opt_float(snapshot.get("water_total_delta")),
    )


def _opt_float(value) -> float | None:
    """Coerce to float, or ``None`` if missing/unparseable (log-only fields)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def predict_kwh(state: ModelState, features: FeatureVector) -> float:
    """Predict the day's delivered energy (kWh) from occupancy + guests."""
    people = max(0, features.people_home)
    occupancy_factor = 1.0 if people > 0 else state.empty_house_factor
    base = state.e_base + state.e_draw_per_person * people
    kwh = occupancy_factor * base
    if features.guests:
        kwh += state.guest_bonus
    kwh *= state.gain
    return max(0.0, kwh)


def kwh_to_minutes(kwh: float, rated_power_kw: float) -> float:
    """Convert delivered energy to heater runtime minutes at the rated power."""
    if rated_power_kw <= 0:
        raise ValueError("rated_power_kw must be positive")
    return kwh / rated_power_kw * 60.0


def clamp_minutes(
    minutes: float,
    min_minutes: float,
    max_minutes: float,
    step: int = DEFAULT_STEP_MINUTES,
) -> int:
    """Round to the nearest ``step`` and clamp inside ``[min, max]``.

    The bounds are pulled *inward* to the nearest step multiple (ceil the low
    bound, floor the high bound) so the result is always a valid step value that
    still respects the safety floor and the cap.
    """
    lo = math.ceil(min_minutes / step) * step
    hi = math.floor(max_minutes / step) * step
    if hi < lo:  # degenerate config (min and max in the same step bucket)
        hi = lo
    val = round(minutes / step) * step
    return int(_clamp(val, lo, hi))


def predict_minutes(
    state: ModelState,
    features: FeatureVector,
    *,
    rated_power_kw: float,
    min_minutes: float,
    max_minutes: float,
    step: int = DEFAULT_STEP_MINUTES,
) -> int:
    """End-to-end: features → kWh → clamped runtime minutes for the scheduler."""
    kwh = predict_kwh(state, features)
    minutes = kwh_to_minutes(kwh, rated_power_kw)
    return clamp_minutes(minutes, min_minutes, max_minutes, step)


def is_valid_delivery(
    kwh: float | None,
    lo: float = ENERGY_VALID_MIN,
    hi: float = ENERGY_VALID_MAX,
) -> bool:
    """True when a measured daily delivery is plausible (not a meter reset)."""
    return kwh is not None and lo <= kwh <= hi


def update_gain(
    state: ModelState,
    predicted_kwh: float,
    actual_kwh: float,
    *,
    beta: float = GAIN_BETA,
) -> ModelState:
    """EWMA-update the calibration gain from one day's actual/predicted ratio.

    No-op for a near-zero prediction (ratio undefined). The ratio and the gain
    are both clamped so a single wild day can't destabilise the model.
    """
    if predicted_kwh <= 0.5:
        return state
    ratio = _clamp(actual_kwh / predicted_kwh, RATIO_MIN, RATIO_MAX)
    new_gain = _clamp((1.0 - beta) * state.gain + beta * ratio, GAIN_MIN, GAIN_MAX)
    return replace(state, gain=new_gain)


def apply_observation(state: ModelState, predicted_kwh: float, actual_kwh: float) -> ModelState:
    """Fold one valid day's outcome into the model: update gain + bump count.

    The caller must gate on :func:`is_valid_delivery` first.
    """
    state = update_gain(state, predicted_kwh, actual_kwh)
    return replace(state, sample_count=state.sample_count + 1)


def blend_param(prior: float, empirical: float, n: int, n_prior: int = N_PRIOR) -> float:
    """Sample-count weighted blend of a prior and an empirical estimate."""
    return (n_prior * prior + n * empirical) / (n_prior + n)


def refit_occupancy_params(
    rows: Sequence[tuple[float, float]],
) -> tuple[float, float] | None:
    """2-parameter OLS of ``actual_kwh ~ people_home`` over logged rows.

    ``rows`` is a sequence of ``(people_home, actual_kwh)``. Returns
    ``(e_base, e_draw_per_person)`` (intercept, slope), each floored at 0 since
    negatives are unphysical, or ``None`` when the slope isn't identifiable
    (fewer than 2 rows, or no variation in ``people_home``).
    """
    pts = [(float(p), float(k)) for p, k in rows]
    if len(pts) < 2:
        return None
    xs = [p for p, _ in pts]
    ys = [k for _, k in pts]
    if len(set(xs)) < 2:  # all observations at the same occupancy → slope undefined
        return None
    n = len(pts)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    if sxx == 0:
        return None
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    return (max(0.0, intercept), max(0.0, slope))


def rolling_mae(errors: Sequence[float | None]) -> float:
    """Mean absolute error over the given errors (None entries ignored)."""
    vals = [abs(e) for e in errors if e is not None]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
