"""Pure hot-water tank state-of-charge model for Load Need Predictor.

**This module imports nothing from Home Assistant.** Like ``predictor.py`` and
``price_model.py`` it is loaded standalone (via ``importlib``) by the pure unit
tests, so it must stay dependency-free. The two non-obvious imports are
``datetime`` (parsing fixed ISO timestamps passed in by the caller — no wall-clock
access) and ``math`` (the saturation curve), which keep the module just as
reproducible as the rest.

Why an energy balance (see the approved plan + CLAUDE.md): the water heater has
no "how full is the tank" sensor, and the daily *need* predictor already showed
that draw is stochastic. But three cumulative signals — delivered energy, the
cold-water meter, and the contactor/heating-element states — let us *integrate*
the tank's charge continuously and self-correct at natural checkpoints:

    deficit_kwh += draw + standing_loss + post_trip_relax − energy_in   # per tick

``deficit_kwh`` is the energy the tank is *below* "full at the thermostat trip";
capacity is ``E_cap = volume × c × ΔT`` with ``ΔT = setpoint − cold_inlet``
(cross-checked on the author's LVV: a fully depleted 300 L / 75 °C tank takes
~7–8 h at 3 kW ≈ 22 kWh, matching a ~12 °C inlet).

**Three ledgers.** ``deficit_kwh`` is clamped to ``[0, E_cap]`` and drives
*control* (the SoC→prediction feedback — over-asking is physically safe, the
thermostat just trips). ``cycle_unclamped_kwh`` runs the same arithmetic without
clamps and without the post-trip relaxation: it is the *learning* ledger, because
a week of replayed real data (v0.9.0) showed the true error at thermostat trips
is ~±1 kWh while the clamps had silently discarded delivered energy and turned
that into 13–19-point jumps — the clamp is a control convenience, not information.
``cycle_relax_kwh`` holds the post-trip relaxation not yet paid back by energy
in; the *display* is ``unclamped + relax`` through the saturation curve.

**The anchor trick.** Integration drifts, so we re-zero it whenever physics hands
us ground truth: the contactor is commanded *on* but the heating element has gone
*idle* (its internal thermostat tripped) ⇒ the tank is full ⇒ ``deficit = 0``.
On the author's LVV this happens roughly daily. Anchors fire on the *transition*
only and require *sustained* states, so sub-second ``unavailable`` blips never
trigger a false anchor; once latched, the latch survives blips and re-toggles and
releases only on a definite "element heating" or "contactor off".

**After the trip the tank is not held at 100 %.** The element idles while the
tank keeps losing energy — standing loss plus internal mixing that pulls the
thermostat back below its band — until it re-engages ~1 h later and delivers a
near-constant top-up (0.8 kWh on the author's LVV). That top-up is modelled as a
*post-trip relaxation* toward a learned ``hysteresis_kwh``, so the % drifts down
smoothly after a trip and the re-heat pays that balance back first. It is kept out
of the learning ledger: trip-to-trip energy conservation has no mixing loss in it,
and treating it as one biased the learner by +0.9 kWh in the replay.

**Soft saturation instead of a floor.** The clamped ledger can read "full" while
the element is still running (or "not full" while it trips). Rather than pinning
a hard floor, the displayed deficit passes through a C¹-smooth curve that is the
identity above the model's own uncertainty ``σ`` and a slow (1/|u|) approach to
zero below it — the % creeps toward 100 while heating and reaches it only at the
trip. ``σ`` scales with the flow the balance had to attribute since the last
anchor (× a learned ``residual_ratio``), so it is ~0 right at an anchor (no
post-anchor drop) and grows with unmetered draw.

**Learning, with guardrails.** Each qualifying anchor→anchor cycle closes with a
*residual* — the unclamped deficit at the trip, which is exactly the model's error
over that cycle. Short cycles (< 4 h, the post-trip re-heats) teach
``hysteresis_kwh`` only. Long cycles teach the daypart ``hot_fraction_profile``
(a normalised-LMS step on the residual, ridge-pulled toward the profile mean so
sparse buckets can't run away) when they metered enough water, or ``standby_w``
when they metered almost none. Everything is EWMA-slow and hard-clamped: the
replay showed a fast learner chases the ±1 kWh cycle noise and gets worse.

Every public function here is pure; ``apply_tick`` is the orchestrator the
HA-side ``tank_tracker`` calls once per tick, and ``dataclasses.replace`` is the
only way state changes (the dataclasses are frozen).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime

# ── Physics constants ─────────────────────────────────────────────────────────
# Water's volumetric heat capacity expressed in kWh per litre per kelvin:
# 4.186 kJ/(L·K) ÷ 3600 kJ/kWh. The single conversion factor used everywhere the
# model turns litres × ΔT into energy (and back).
KWH_PER_LITER_KELVIN = 4.186 / 3600.0
DEFAULT_COLD_IN_C = 12.0  # annual underground-transit inlet average at the house

# ── Learnable parameter seeds + clamps ────────────────────────────────────────
# hot_fraction: share of metered household water that flows through the tank.
# Seed 0.25 back-solved from ~5.6 kWh/day tank draw ≈ 74 L vs ~300 L/day total.
# It is learned per *daypart* (see DAYPART_BOUNDS): evenings are showers, days
# and nights include toilets/dishwasher/washing machine (cold only) and, in
# summer, the garden hose — a week of trips already separated evening ≈ 0.27
# from the rest of the day ≈ 0.20 on the author's house.
SEED_HOT_FRACTION = 0.25
HOT_FRACTION_MIN, HOT_FRACTION_MAX = 0.05, 0.9
# Daypart buckets by local hour: [start, end) — night, morning, day, evening.
DAYPART_BOUNDS = ((0, 6), (6, 11), (11, 17), (17, 24))
N_DAYPARTS = len(DAYPART_BOUNDS)
PROFILE_RIDGE = 0.05  # per-anchor pull of each bucket toward the profile mean
# standby_w: standing loss. Seed 70 W ≈ 1.68 kWh/day, matching the observed
# ~1.7 kWh gap between mean delivery and estimated draws.
SEED_STANDBY_W = 70.0
STANDBY_W_MIN, STANDBY_W_MAX = 20.0, 200.0
# hysteresis_kwh: the post-trip top-up the thermostat asks for once mixing has
# pulled it back below its band. Observed 0.81 kWh twice on the author's LVV.
SEED_HYSTERESIS_KWH = 0.8
HYSTERESIS_MIN_KWH, HYSTERESIS_MAX_KWH = 0.2, 3.0
POST_TRIP_TAU_MIN = 45.0  # relaxation time constant (re-engage seen ~45–60 min after trips)
# residual_ratio: typical |trip residual| per kWh of flow the balance attributed
# during the cycle — the model's own uncertainty scale for the saturation curve.
SEED_RESIDUAL_RATIO = 0.15
RESIDUAL_RATIO_MIN, RESIDUAL_RATIO_MAX = 0.05, 0.6
SIGMA_MIN_KWH = 0.05  # numerical floor for σ (never divide by ~0)

LEARN_BETA = 0.1  # per-anchor step (~daily anchors → adapts over 2–4 weeks)
LEARN_BETA_FAST = 0.2  # for hysteresis / residual_ratio (few, clean observations)
LEARN_MIN_LITERS = 50.0  # a cycle must meter ≥ this to move the hot-fraction profile
STANDBY_MAX_LITERS = 10.0  # near-zero-draw cycle → its energy is (almost) all standby
STANDBY_MIN_HOURS = 12.0  # …and long enough that standby dominates the balance
LEARN_MIN_CYCLE_HOURS = 4.0  # shorter cycles are post-trip re-heats: hysteresis only
HYSTERESIS_MIN_ENERGY_KWH = 0.2  # a re-heat must deliver this much to count as one
RESIDUAL_RATIO_MIN_GROSS_KWH = 0.5  # don't learn σ from cycles with ~no flow

# ── Water-meter attribution + guards ──────────────────────────────────────────
# Hot water only reaches taps + showers, whose combined flow tops out ~8 L/min;
# anything faster (garden hose, dishwasher/washing-machine fill) is cold-only and
# must not be charged to the tank. The cap is a *rate* over the span since the
# previous valid read (one tick normally, the whole gap after a restart), so a
# restart never clips a legitimately accumulated delta.
MAX_HOT_FLOW_LPM = 8.0
# Misread guard: even every tap open at once can't exceed ~30 L/min, so a larger
# implied rate is an OCR misread → drop the delta and re-baseline.
MAX_PLAUSIBLE_FLOW_LPM = 30.0
# A missing reading below this age is a blip: wait (draw 0), the cumulative meter
# catches up. Beyond it, fall back to the occupancy estimate.
WATER_STALE_AFTER_S = 900.0

# ── Energy-counter guards ─────────────────────────────────────────────────────
# A cumulative counter that steps *down* by less than this is a restore-after-
# restart rollback (powercalc re-published an older value on 2026-09-03), not a
# reset: keep the old baseline and let the counter catch up. Larger drops are
# genuine resets / meter swaps → re-baseline.
COUNTER_ROLLBACK_TOL_KWH = 1.0
# Display-only smoothing: the powercalc counter steps 0.5 kWh every 10 min, so
# between steps the element's on-time × rated power fills in (capped so a stalled
# counter can't run it away).
LED_SMOOTHING_CAP_KWH = 1.0

# ── Anchor thresholds ─────────────────────────────────────────────────────────
# Sustained-state requirements for the *transition*: the element must have been
# idle ≥ 60 s (a real thermostat trip, not a blip) while the contactor has been
# commanded on ≥ 120 s (long enough that "on but idle" means full, not just
# switching on).
ANCHOR_MIN_HEATING_OFF_S = 60.0
ANCHOR_MIN_CONTACTOR_ON_S = 120.0

# ── Cold start ────────────────────────────────────────────────────────────────
# No ground truth on day 1 → assume half-full; ``calibrated`` stays False until
# the first anchor so the SoC feedback never acts on this guess.
FIRST_INSTALL_DEFICIT_FRACTION = 0.5

# ── Low-charge boost (SoC → prediction feedback) ──────────────────────────────
# Hysteresis: after a boost, SoC must climb this many points above the threshold
# before another can arm — stops thrash around the trigger line.
BOOST_REARM_MARGIN_PCT = 15.0
BOOST_MIN_INTERVAL_H = 6.0  # rate limit: at most one boost re-plan per 6 h

# ── Human-readable helpers ────────────────────────────────────────────────────
# "Equivalent litres" and "showers left" are expressed at a comfortable mixed
# tap temperature, not the tank setpoint, so the card's number matches intuition.
MIX_TEMP_C = 40.0
SHOWER_LITERS_40C = 40.0

TANK_VERSION = "v2"

_ZERO_PROFILE = (0.0,) * N_DAYPARTS


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` to the inclusive ``[lo, hi]`` range."""
    return max(lo, min(value, hi))


def _parse_iso(value: str) -> datetime | None:
    """Parse a fixed ISO timestamp, or ``None`` if empty/unparseable.

    Pure string arithmetic — no clock is read — so the module stays reproducible.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _span_minutes(baseline_iso: str, now_iso: str, fallback_min: float) -> float:
    """Minutes elapsed since ``baseline_iso`` (for rate/attribution windows).

    Falls back to ``fallback_min`` (this tick's elapsed) when either timestamp is
    missing — e.g. no water baseline has ever been taken.
    """
    base = _parse_iso(baseline_iso)
    now = _parse_iso(now_iso)
    if base is None or now is None:
        return max(0.0, fallback_min)
    return max(0.0, (now - base).total_seconds() / 60.0)


def _hours_between(start_iso: str, end_iso: str) -> float:
    """Hours between two ISO timestamps; ``0.0`` when either is missing."""
    start = _parse_iso(start_iso)
    end = _parse_iso(end_iso)
    if start is None or end is None:
        return 0.0
    return max(0.0, (end - start).total_seconds() / 3600.0)


@dataclass(frozen=True)
class TankParams:
    """Fixed physical description of one tank (from the load's config)."""

    volume_l: float
    setpoint_c: float
    cold_in_c: float


@dataclass(frozen=True)
class TankState:
    """The evolving tank estimate + learnable params. Frozen; ``replace`` to update.

    ``deficit_kwh`` is the clamped (control) charge deficit; ``cycle_unclamped_kwh``
    is the relaxation-free, unclamped balance since the last anchor (the learning
    ledger); ``cycle_relax_kwh`` is the post-trip relaxation still outstanding (paid
    down first by any energy in) — display = unclamped + relax.
    The baselines (energy counter, water counter, and *when* the water baseline was
    taken) make the deltas cumulative and lossless across restarts. The ``cycle_*``
    fields accumulate the current anchor→anchor cycle so :func:`learn_from_cycle`
    can close the balance; ``pending_fallback_kwh`` is fallback draw charged since
    the last valid meter read, reconciled out of the next metered delta so an OCR
    dropout doesn't double-count. ``boost_armed`` / ``last_boost_iso`` back the
    low-charge boost hysteresis + rate limit. ``hot_fraction`` is kept as the mean
    of ``hot_fraction_profile`` (an empty profile means "all buckets = hot_fraction",
    which is how pre-v2 stored state migrates).
    """

    deficit_kwh: float
    hot_fraction: float = SEED_HOT_FRACTION
    standby_w: float = SEED_STANDBY_W
    calibrated: bool = False
    anchor_latched: bool = False
    energy_baseline_kwh: float | None = None
    water_baseline_l: float | None = None
    water_baseline_iso: str = ""
    pending_fallback_kwh: float = 0.0
    last_tick_iso: str = ""
    last_anchor_iso: str = ""
    last_boost_iso: str = ""
    boost_armed: bool = True
    cycle_start_iso: str = ""
    cycle_energy_in_kwh: float = 0.0
    cycle_liters: float = 0.0
    cycle_clean: bool = True
    # v2 additions (all defaulted so older persisted shapes still load).
    hot_fraction_profile: tuple[float, ...] = ()
    hysteresis_kwh: float = SEED_HYSTERESIS_KWH
    residual_ratio: float = SEED_RESIDUAL_RATIO
    cycle_unclamped_kwh: float | None = None  # None ⇒ same as deficit_kwh (pre-v2 / fresh)
    cycle_gross_kwh: float = 0.0
    cycle_relax_kwh: float = 0.0  # post-trip relaxation accrued and not yet paid by energy in
    cycle_hot_liters_by_bucket: tuple[float, ...] = _ZERO_PROFILE
    led_kwh_since_counter: float = 0.0
    version: str = TANK_VERSION


@dataclass(frozen=True)
class TickInputs:
    """Everything one tick needs, read by the tracker from ``hass.states``.

    Cumulative counters may be ``None`` (sensor missing/unavailable). The
    contactor/heating booleans are tri-state (``None`` when unknown — never
    treated as off, so a missing sensor can't spuriously anchor). The trailing
    ``e_*`` fields are the load model's current occupancy params, used only for
    the fallback draw estimate. ``rated_power_kw`` enables the display-only
    energy smoothing between counter steps; ``local_hour`` selects the daypart
    bucket (derived from ``now_iso``'s own offset when omitted).
    """

    now_iso: str
    elapsed_s: float
    energy_counter_kwh: float | None
    water_counter_l: float | None
    contactor_on: bool | None
    heating_on: bool | None
    # Seconds the entity has held its *current* state (from ``last_changed``),
    # None when the state itself is unknown. The names reflect the anchor's
    # reading (contactor on / heating off); when ``heating_on`` is True,
    # ``heating_off_for_s`` therefore holds the time spent heating.
    contactor_on_for_s: float | None
    heating_off_for_s: float | None
    people_home: int | None
    e_base: float
    e_draw_per_person: float
    empty_house_factor: float
    rated_power_kw: float | None = None
    local_hour: int | None = None


@dataclass(frozen=True)
class TickResult:
    """The new state plus per-tick diagnostics (published + logged).

    ``draw_source`` records how the draw was attributed this tick — ``"meter"``
    (a valid cold-meter delta), ``"fallback"`` (occupancy estimate; meter stale or
    misread), or ``"none"`` (short blip, waiting for the meter to catch up).
    ``anchored`` is True only on the anchor *transition* tick (the moment the tank
    is known full and the model learns); ``latched`` while the trip still holds.
    ``soc`` is derived from ``deficit_shown_kwh`` (the saturation-curve display
    deficit), not from the clamped control ``state.deficit_kwh``.
    """

    state: TankState
    soc: float
    capacity_kwh: float
    draw_source: str
    anchored: bool
    latched: bool
    energy_in_kwh: float
    draw_kwh: float
    standby_kwh: float
    deficit_shown_kwh: float
    uncertainty_kwh: float


def capacity_kwh(volume_l: float, setpoint_c: float, cold_in_c: float) -> float:
    """Energy to raise the whole tank from the cold inlet to the setpoint (kWh).

    ``ΔT`` is floored at 1 K so a mis-set / inverted config can never yield a
    zero-or-negative capacity (which would make SoC undefined).
    """
    delta_t = max(setpoint_c - cold_in_c, 1.0)
    return volume_l * KWH_PER_LITER_KELVIN * delta_t


def soc(deficit_kwh: float, capacity: float) -> float:
    """State of charge in ``[0, 1]``: full when the deficit is zero.

    Guards a non-positive capacity (degenerate config) by reporting empty.
    """
    if capacity <= 0:
        return 0.0
    return _clamp(1.0 - deficit_kwh / capacity, 0.0, 1.0)


def initial_state(capacity: float) -> TankState:
    """Cold-start state: half-full and uncalibrated until the first anchor."""
    return TankState(deficit_kwh=FIRST_INSTALL_DEFICIT_FRACTION * capacity, calibrated=False)


# ── Daypart hot-fraction profile ──────────────────────────────────────────────


def daypart_bucket(hour: int) -> int:
    """Index of the daypart bucket a local hour falls in (``DAYPART_BOUNDS``)."""
    hour = int(hour) % 24
    for index, (start, end) in enumerate(DAYPART_BOUNDS):
        if start <= hour < end:
            return index
    return N_DAYPARTS - 1


def profile_of(state: TankState) -> tuple[float, ...]:
    """The in-force per-daypart hot fractions (pre-v2 state ⇒ flat profile)."""
    if len(state.hot_fraction_profile) == N_DAYPARTS:
        return state.hot_fraction_profile
    return (state.hot_fraction,) * N_DAYPARTS


def _local_hour(inputs: TickInputs) -> int:
    """Local hour for bucketing: the explicit field, else ``now_iso``'s own clock."""
    if inputs.local_hour is not None:
        return int(inputs.local_hour) % 24
    now = _parse_iso(inputs.now_iso)
    return now.hour if now is not None else 12


# ── Counter / meter primitives ────────────────────────────────────────────────


def counter_delta(
    baseline: float | None,
    reading: float | None,
    rollback_tol: float = COUNTER_ROLLBACK_TOL_KWH,
) -> tuple[float, float | None]:
    """Consume a cumulative counter → ``(delta_since_baseline, new_baseline)``.

    A missing reading yields no delta and keeps the baseline (wait for the next
    read). A missing baseline yields no delta but adopts the reading. A *small*
    decrease (< ``rollback_tol``) is a restore-after-restart rollback: no delta,
    baseline kept, so counting resumes once the counter passes it again and no
    energy is lost or double-counted. A *large* decrease is a counter reset /
    meter swap → no delta but re-baseline, so a reset never charges a phantom
    delta.
    """
    if reading is None:
        return 0.0, baseline
    if baseline is None:
        return 0.0, reading
    if reading < baseline:
        if baseline - reading < rollback_tol:
            return 0.0, baseline
        return 0.0, reading
    return reading - baseline, reading


def water_delta(
    baseline: float | None,
    reading_l: float | None,
    span_min: float,
    max_flow_lpm: float = MAX_PLAUSIBLE_FLOW_LPM,
) -> tuple[float | None, float | None]:
    """Cold-meter delta with a rate misread-guard → ``(delta_or_None, new_baseline)``.

    ``None`` delta means "no usable metered draw this tick": either there is no
    reading (keep the baseline and wait) or no baseline yet (adopt the reading),
    or the implied flow is impossible — negative, or faster than ``max_flow_lpm``
    over ``span_min`` — in which case it is treated as a misread and we
    re-baseline. The span is the time since the previous valid read (the baseline
    stamp advances on every valid read, changed or not), so it is one tick in
    steady state and the whole gap after a restart: a legitimate multi-litre delta
    that accumulated while HA was down still passes, while the same delta over a
    single tick is correctly rejected.
    """
    if reading_l is None:
        return None, baseline
    if baseline is None:
        return None, reading_l
    delta = reading_l - baseline
    if delta < 0 or delta > max_flow_lpm * max(span_min, 1.0):
        return None, reading_l
    return delta, reading_l


def hot_attributable_liters(
    delta_liters: float, span_min: float, max_hot_flow_lpm: float = MAX_HOT_FLOW_LPM
) -> float:
    """Litres of the metered draw plausibly heated (taps/showers only).

    Water above the hot-flow rate cap over ``span_min`` is cold-only usage (garden
    hose, appliance fill) and is *not* charged to the tank. The span is the time
    since the previous valid meter read — normally one tick, so this is in effect
    an 8 L/tick cap, but the whole gap after a restart, so a delta that accumulated
    while HA was down is not clipped.
    """
    return min(max(0.0, delta_liters), max_hot_flow_lpm * max(span_min, 1.0))


def draw_kwh_from_liters(
    liters: float, hot_fraction: float, setpoint_c: float, cold_in_c: float
) -> float:
    """Energy the tank gave up delivering ``liters`` of (partly hot) draw (kWh)."""
    delta_t = max(setpoint_c - cold_in_c, 1.0)
    return liters * hot_fraction * delta_t * KWH_PER_LITER_KELVIN


def fallback_draw_kwh(
    e_base: float,
    e_draw_per_person: float,
    empty_house_factor: float,
    people_home: int | None,
    standby_w: float,
    elapsed_min: float,
) -> float:
    """Occupancy-based draw estimate for when the meter is unusable (kWh).

    Reuses the load model's daily-energy formula, subtracts standby (the daily
    figure is *delivered* energy, which includes standing loss the balance already
    counts separately), floors at zero, and prorates to the tick. ``people_home``
    ``None`` ⇒ assume 1, matching ``build_features`` — never under-serve.
    """
    people = 1 if people_home is None else max(0, people_home)
    occ_factor = 1.0 if people > 0 else empty_house_factor
    daily_kwh = occ_factor * (e_base + e_draw_per_person * people) - standby_w * 24.0 / 1000.0
    daily_kwh = max(0.0, daily_kwh)
    return daily_kwh / 1440.0 * elapsed_min


def standby_kwh(standby_w: float, elapsed_min: float) -> float:
    """Standing loss over the tick (kWh) from the watt rating."""
    return standby_w * elapsed_min / 60.0 / 1000.0


def post_trip_relax_kwh(
    hysteresis_kwh: float, hours_since_trip_start: float, hours_since_trip_end: float
) -> float:
    """Mixing loss accrued between two instants after a thermostat trip (kWh).

    The tank relaxes toward ``hysteresis_kwh`` below the trip point with time
    constant ``POST_TRIP_TAU_MIN``: this returns the increment over the interval,
    so summing over ticks totals ``hysteresis_kwh`` (99 % of it within 5 τ). Zero
    for a non-positive interval.
    """
    if hours_since_trip_end <= hours_since_trip_start:
        return 0.0
    tau_h = POST_TRIP_TAU_MIN / 60.0
    start = max(0.0, hours_since_trip_start)
    return hysteresis_kwh * (math.exp(-start / tau_h) - math.exp(-hours_since_trip_end / tau_h))


def display_deficit_kwh(unclamped_kwh: float, sigma_kwh: float) -> float:
    """Saturation curve turning the unclamped deficit into the displayed one.

    Identity above ``σ``; below it ``σ² / (2σ − u)`` — continuous *and* slope-
    continuous at ``u = σ``, decaying toward (never reaching) zero like ``1/|u|``
    as the balance goes past "full". The heavy tail is deliberate: while the
    element keeps running after the model already thinks the tank is full, the %
    creeps toward 100 slowly enough that a rounded display still reads 99, and
    only a genuine anchor shows exactly 100.
    """
    sigma = max(sigma_kwh, SIGMA_MIN_KWH)
    if unclamped_kwh >= sigma:
        return unclamped_kwh
    return sigma * sigma / (2.0 * sigma - unclamped_kwh)


def uncertainty_kwh(residual_ratio: float, cycle_gross_kwh: float) -> float:
    """The model's error scale ``σ`` for the current cycle (kWh, floored)."""
    return max(SIGMA_MIN_KWH, residual_ratio * max(0.0, cycle_gross_kwh))


def should_anchor(
    contactor_on: bool | None,
    heating_on: bool | None,
    contactor_on_for_s: float | None,
    heating_off_for_s: float | None,
    *,
    min_heating_off_s: float = ANCHOR_MIN_HEATING_OFF_S,
    min_contactor_on_s: float = ANCHOR_MIN_CONTACTOR_ON_S,
) -> bool:
    """True when "commanded on but element idle" holds — the tank is full.

    Requires the contactor sustained *on* and the element sustained *off*, both
    for their thresholds. Any ``None`` (unknown/unavailable sensor) fails closed,
    so a missing reading can never anchor. Uses identity checks so a ``None``
    boolean never masquerades as its truthy/falsy value.
    """
    if contactor_on is not True or heating_on is not False:
        return False
    if contactor_on_for_s is None or heating_off_for_s is None:
        return False
    return contactor_on_for_s >= min_contactor_on_s and heating_off_for_s >= min_heating_off_s


def latch_holds(latched: bool, contactor_on: bool | None, heating_on: bool | None) -> bool:
    """Whether an existing anchor latch survives this tick's readings.

    The latch only dedupes the transition, so it ignores durations and survives
    unknown/unavailable blips and the scheduler's off→on re-toggle at a slot
    boundary; it releases on a *definite* "element heating" or "contactor off".
    """
    if not latched:
        return False
    if heating_on is True or contactor_on is False:
        return False
    return True


def learn_from_cycle(
    state: TankState, params: TankParams, cycle_hours: float, residual_kwh: float
) -> TankState:
    """Refine the learnable params from a closed anchor→anchor cycle.

    ``residual_kwh`` is the unclamped deficit at the trip — positive means the
    model over-estimated what the cycle took out of the tank. Gated on a
    calibrated tank (the first cycle starts from the seeded guess), a clean cycle
    (no fallback/misread ticks) and a positive duration. Cycle regimes:

    - **short** (< ``LEARN_MIN_CYCLE_HOURS``): a post-trip re-heat. Only
      ``hysteresis_kwh`` learns, from the energy the element delivered beyond
      standby, and only when the cycle metered ~no water and delivered a real
      top-up.
    - **long, wet** (≥ ``LEARN_MIN_LITERS``): the ``hot_fraction_profile`` takes
      a normalised-LMS step on the residual across the buckets that carried
      litres, then every bucket is pulled ``PROFILE_RIDGE`` toward the mean.
    - **long, dry** (< ``STANDBY_MAX_LITERS`` and ≥ ``STANDBY_MIN_HOURS``):
      ``standby_w`` absorbs the residual.

    Long cycles also update ``residual_ratio`` (|residual| per kWh of attributed
    flow) — the saturation curve's σ. Every update is EWMA-slow and hard-clamped.
    """
    if not state.calibrated or not state.cycle_clean or cycle_hours <= 0:
        return state
    delta_t = max(params.setpoint_c - params.cold_in_c, 1.0)
    standby_over_cycle = state.standby_w * cycle_hours / 1000.0

    if cycle_hours < LEARN_MIN_CYCLE_HOURS:
        if (
            state.cycle_liters < STANDBY_MAX_LITERS
            and state.cycle_energy_in_kwh >= HYSTERESIS_MIN_ENERGY_KWH
        ):
            observed = state.cycle_energy_in_kwh - standby_over_cycle
            if observed > 0:
                new_hyst = _clamp(
                    (1.0 - LEARN_BETA_FAST) * state.hysteresis_kwh + LEARN_BETA_FAST * observed,
                    HYSTERESIS_MIN_KWH,
                    HYSTERESIS_MAX_KWH,
                )
                return replace(state, hysteresis_kwh=new_hyst)
        return state

    profile = profile_of(state)
    new_profile = profile
    new_standby_w = state.standby_w
    if state.cycle_liters >= LEARN_MIN_LITERS:
        buckets = state.cycle_hot_liters_by_bucket
        if len(buckets) != N_DAYPARTS:
            buckets = _ZERO_PROFILE
        features = [liters * delta_t * KWH_PER_LITER_KELVIN for liters in buckets]
        norm = sum(x * x for x in features)
        if norm > 0:
            stepped = [
                hf - LEARN_BETA * residual_kwh * x / norm
                for hf, x in zip(profile, features, strict=True)
            ]
            mean = sum(stepped) / N_DAYPARTS
            new_profile = tuple(
                _clamp(hf - PROFILE_RIDGE * (hf - mean), HOT_FRACTION_MIN, HOT_FRACTION_MAX)
                for hf in stepped
            )
    elif state.cycle_liters < STANDBY_MAX_LITERS and cycle_hours >= STANDBY_MIN_HOURS:
        new_standby_w = _clamp(
            state.standby_w - LEARN_BETA * residual_kwh / cycle_hours * 1000.0,
            STANDBY_W_MIN,
            STANDBY_W_MAX,
        )

    new_ratio = state.residual_ratio
    if state.cycle_gross_kwh >= RESIDUAL_RATIO_MIN_GROSS_KWH:
        observed_ratio = min(RESIDUAL_RATIO_MAX, abs(residual_kwh) / state.cycle_gross_kwh)
        new_ratio = _clamp(
            (1.0 - LEARN_BETA_FAST) * state.residual_ratio + LEARN_BETA_FAST * observed_ratio,
            RESIDUAL_RATIO_MIN,
            RESIDUAL_RATIO_MAX,
        )

    return replace(
        state,
        hot_fraction=sum(new_profile) / N_DAYPARTS,
        hot_fraction_profile=new_profile,
        standby_w=new_standby_w,
        residual_ratio=new_ratio,
    )


def _resolve_draw(
    state: TankState,
    params: TankParams,
    inputs: TickInputs,
    elapsed_min: float,
    hot_fraction: float,
) -> tuple[float, str, bool, float, float | None, str, float]:
    """Decide this tick's draw + water bookkeeping.

    Returns ``(draw_kwh, draw_source, tick_clean, metered_hot_liters,
    new_water_baseline_l, new_water_baseline_iso, new_pending_fallback_kwh)``.
    ``metered_hot_liters`` (0 unless the meter was valid) is what accumulates into
    the cycle so learning uses the same capped attribution as the deficit.
    """
    fallback = fallback_draw_kwh(
        inputs.e_base,
        inputs.e_draw_per_person,
        inputs.empty_house_factor,
        inputs.people_home,
        state.standby_w,
        elapsed_min,
    )
    reading = inputs.water_counter_l
    if reading is not None:
        span_min = _span_minutes(state.water_baseline_iso, inputs.now_iso, elapsed_min)
        delta, _ = water_delta(state.water_baseline_l, reading, span_min)
        if delta is not None:
            # Valid metered draw: charge the hot portion, netting off any fallback
            # already charged since the baseline (floored so it can't go negative).
            hot = hot_attributable_liters(delta, span_min)
            raw = draw_kwh_from_liters(hot, hot_fraction, params.setpoint_c, params.cold_in_c)
            draw = max(0.0, raw - state.pending_fallback_kwh)
            return draw, "meter", True, hot, reading, inputs.now_iso, 0.0
        # Misread (negative or impossibly fast): fall back this tick, re-baseline,
        # mark the cycle dirty, and drop the unreconcilable pending.
        return fallback, "fallback", False, 0.0, reading, inputs.now_iso, 0.0
    # No reading. A short gap is a blip — draw nothing and let the cumulative meter
    # catch up; only fall back (and accumulate pending) once the meter is stale.
    base_dt = _parse_iso(state.water_baseline_iso)
    now_dt = _parse_iso(inputs.now_iso)
    fresh = (
        state.water_baseline_l is not None
        and base_dt is not None
        and now_dt is not None
        and (now_dt - base_dt).total_seconds() < WATER_STALE_AFTER_S
    )
    if fresh:
        return (
            0.0,
            "none",
            True,
            0.0,
            state.water_baseline_l,
            state.water_baseline_iso,
            state.pending_fallback_kwh,
        )
    return (
        fallback,
        "fallback",
        False,
        0.0,
        state.water_baseline_l,
        state.water_baseline_iso,
        state.pending_fallback_kwh + fallback,
    )


def apply_tick(state: TankState, params: TankParams, inputs: TickInputs) -> TickResult:
    """Advance the tank estimate by one tick (the orchestrator).

    Integrates the energy balance (both ledgers), attributes the draw to the
    current daypart, applies standing loss and the post-trip relaxation, then
    either re-zeros at a 100 % anchor transition (learning from the cycle it
    closes) or carries the balance forward — *including* while the latch holds,
    so the % drifts down after a trip instead of being held at 100. Restart
    reconciliation is just the ordinary first tick after a gap: the counters are
    cumulative, so the deltas and the standby-over-the-gap all fall out of the
    same arithmetic.
    """
    capacity = capacity_kwh(params.volume_l, params.setpoint_c, params.cold_in_c)
    elapsed_min = max(0.0, inputs.elapsed_s) / 60.0
    bucket = daypart_bucket(_local_hour(inputs))
    hot_fraction = profile_of(state)[bucket]

    # Energy in (cumulative counter delta; rollback-tolerant, reset-safe).
    energy_in, new_energy_baseline = counter_delta(
        state.energy_baseline_kwh, inputs.energy_counter_kwh
    )

    # Draw + water bookkeeping (meter / fallback / none).
    (
        draw_kwh,
        draw_source,
        tick_clean,
        metered_hot_liters,
        new_water_baseline_l,
        new_water_baseline_iso,
        new_pending,
    ) = _resolve_draw(state, params, inputs, elapsed_min, hot_fraction)

    standby = standby_kwh(state.standby_w, elapsed_min)

    # Post-trip relaxation: the mixing loss the thermostat feels after a trip,
    # accrued between the previous tick and now (zero before the first anchor).
    relax = 0.0
    if state.last_anchor_iso:
        since_prev = _hours_between(state.last_anchor_iso, state.last_tick_iso)
        since_now = _hours_between(state.last_anchor_iso, inputs.now_iso)
        relax = post_trip_relax_kwh(state.hysteresis_kwh, since_prev, since_now)

    # Three ledgers, three jobs (see module docstring):
    # - control: clamped, counts the relaxation as a loss → conservative over-ask;
    # - learning: unclamped and relaxation-free → the trip residual is exactly the
    #   attribution error (trip-to-trip energy conservation has no mixing loss);
    # - relaxation balance: accrues after a trip, paid down first by energy in →
    #   the display declines after a trip and the re-heat brings it straight back.
    # Energy in is subtracted from BOTH the learning ledger and the relaxation
    # balance on purpose (two reviewers flagged it as a double count — it is not):
    # mixing redistributes heat, it does not remove it, so trip-to-trip energy
    # conservation says the learning ledger must see every delivered kWh. Charging
    # the first ``hysteresis_kwh`` of a heating run to the relaxation only was
    # tried on the replayed week: it turns the residuals' bias from +0.3 into
    # +1.1 kWh and their de-biased scatter from 1.03 into 1.22 kWh, i.e. a worse
    # model. The price is display-side only: while the relaxation is being paid
    # back the shown deficit falls ~2× as fast as energy arrives (up to
    # ``hysteresis_kwh`` in total), which is what lets the % converge on the
    # learning ledger before the trip instead of carrying a +3.6-point correction
    # into it.
    new_deficit = _clamp(state.deficit_kwh + draw_kwh + standby + relax - energy_in, 0.0, capacity)
    unclamped_before = (
        state.deficit_kwh if state.cycle_unclamped_kwh is None else state.cycle_unclamped_kwh
    )
    new_unclamped = _clamp(unclamped_before + draw_kwh + standby - energy_in, -capacity, capacity)
    new_relax = max(0.0, state.cycle_relax_kwh + relax - energy_in)

    # Display-only smoothing between counter steps: element on-time × rated power,
    # cleared whenever the authoritative counter actually moves.
    # It also clears when the element stops: the counter is authoritative, and a
    # burst it never registered (counter unavailable) must not stay subtracted.
    if energy_in > 0 or inputs.heating_on is not True:
        led_since = 0.0
    elif inputs.rated_power_kw:
        led_since = min(
            LED_SMOOTHING_CAP_KWH,
            state.led_kwh_since_counter + inputs.rated_power_kw * elapsed_min / 60.0,
        )
    else:
        led_since = state.led_kwh_since_counter

    # Cycle accumulators including this tick's contributions.
    acc_energy = state.cycle_energy_in_kwh + energy_in
    acc_liters = state.cycle_liters + metered_hot_liters
    acc_clean = state.cycle_clean and tick_clean
    acc_gross = state.cycle_gross_kwh + draw_kwh + standby + relax
    buckets = state.cycle_hot_liters_by_bucket
    if len(buckets) != N_DAYPARTS:
        buckets = _ZERO_PROFILE
    acc_buckets = tuple(
        liters + (metered_hot_liters if index == bucket else 0.0)
        for index, liters in enumerate(buckets)
    )

    # Tick-level bookkeeping applied on every branch.
    base = replace(
        state,
        energy_baseline_kwh=new_energy_baseline,
        water_baseline_l=new_water_baseline_l,
        water_baseline_iso=new_water_baseline_iso,
        pending_fallback_kwh=new_pending,
        last_tick_iso=inputs.now_iso,
        led_kwh_since_counter=led_since,
    )

    latched = latch_holds(state.anchor_latched, inputs.contactor_on, inputs.heating_on)
    transition = not latched and should_anchor(
        inputs.contactor_on,
        inputs.heating_on,
        inputs.contactor_on_for_s,
        inputs.heating_off_for_s,
    )

    if transition:
        # Anchor transition: close + learn the cycle (a no-op on the first anchor,
        # still uncalibrated), then pin the tank full and open a fresh cycle.
        cycle_hours = _hours_between(state.cycle_start_iso, inputs.now_iso)
        accumulated = replace(
            base,
            cycle_energy_in_kwh=acc_energy,
            cycle_liters=acc_liters,
            cycle_clean=acc_clean,
            cycle_gross_kwh=acc_gross,
            cycle_hot_liters_by_bucket=acc_buckets,
        )
        learned = learn_from_cycle(accumulated, params, cycle_hours, new_unclamped)
        new_state = replace(
            learned,
            deficit_kwh=0.0,
            cycle_unclamped_kwh=0.0,
            calibrated=True,
            anchor_latched=True,
            last_anchor_iso=inputs.now_iso,
            cycle_start_iso=inputs.now_iso,
            cycle_energy_in_kwh=0.0,
            cycle_liters=0.0,
            cycle_clean=True,
            cycle_gross_kwh=0.0,
            cycle_relax_kwh=0.0,
            cycle_hot_liters_by_bucket=_ZERO_PROFILE,
            led_kwh_since_counter=0.0,
        )
        shown = 0.0
        sigma = SIGMA_MIN_KWH
    else:
        # Ordinary integrating tick (latched or not): carry both ledgers and the
        # cycle forward. A latched tank keeps accruing standby/relaxation/draws so
        # the % drifts below 100 until the element re-engages or the contactor
        # drops — exactly what the thermostat is about to react to.
        new_state = replace(
            base,
            deficit_kwh=new_deficit,
            cycle_unclamped_kwh=new_unclamped,
            anchor_latched=latched,
            cycle_start_iso=state.cycle_start_iso or inputs.now_iso,
            cycle_energy_in_kwh=acc_energy,
            cycle_liters=acc_liters,
            cycle_clean=acc_clean,
            cycle_gross_kwh=acc_gross,
            cycle_relax_kwh=new_relax,
            cycle_hot_liters_by_bucket=acc_buckets,
        )
        sigma = uncertainty_kwh(state.residual_ratio, acc_gross)
        shown = _clamp(
            display_deficit_kwh(new_unclamped + new_relax - led_since, sigma), 0.0, capacity
        )

    return TickResult(
        state=new_state,
        soc=soc(shown, capacity),
        capacity_kwh=capacity,
        draw_source=draw_source,
        anchored=transition,
        latched=new_state.anchor_latched,
        energy_in_kwh=energy_in,
        draw_kwh=draw_kwh,
        standby_kwh=standby,
        deficit_shown_kwh=shown,
        uncertainty_kwh=sigma,
    )


def should_boost(
    state: TankState,
    soc_value: float,
    threshold_pct: float,
    now_iso: str,
    *,
    rearm_margin_pct: float = BOOST_REARM_MARGIN_PCT,
    min_interval_h: float = BOOST_MIN_INTERVAL_H,
) -> tuple[bool, TankState]:
    """Decide whether a low charge should trigger an early re-plan.

    Returns ``(fire, new_state)``. Firing re-runs the predict/push path so the
    scheduler books more heating before the tank empties. Guards, in order: never
    on an uncalibrated tank (the SoC is still the seeded guess); hysteresis re-arm
    once SoC recovers ``rearm_margin_pct`` above the threshold; fire only when
    below the threshold, armed, and at least ``min_interval_h`` since the last
    boost. The re-arm is applied even while uncalibrated so a tank that calibrates
    above the threshold is already armed for its first dip.
    """
    soc_pct = soc_value * 100.0
    new_state = state
    if soc_pct >= threshold_pct + rearm_margin_pct and not state.boost_armed:
        new_state = replace(state, boost_armed=True)
    if not new_state.calibrated:
        return False, new_state
    if soc_pct >= threshold_pct or not new_state.boost_armed:
        return False, new_state
    if new_state.last_boost_iso:
        last = _parse_iso(new_state.last_boost_iso)
        now = _parse_iso(now_iso)
        if last is not None and now is not None:
            if (now - last).total_seconds() / 3600.0 < min_interval_h:
                return False, new_state
    return True, replace(new_state, boost_armed=False, last_boost_iso=now_iso)


def deficit_minutes_from_kwh(deficit_kwh: float, rated_power_kw: float) -> float:
    """Convert a charge deficit (kWh) to heater runtime minutes at rated power.

    Mirrors ``predictor.kwh_to_minutes`` — the SoC-feedback path turns the measured
    deficit into the same minutes unit the scheduler target speaks.
    """
    if rated_power_kw <= 0:
        raise ValueError("rated_power_kw must be positive")
    return deficit_kwh / rated_power_kw * 60.0


def liters_at_temp(available_kwh: float, cold_in_c: float, mix_temp_c: float = MIX_TEMP_C) -> float:
    """Equivalent litres of usable water at a comfortable mixed tap temperature.

    Expresses the *available* charge (capacity − deficit) as litres a person would
    actually draw at ~40 °C, which is what the card shows. ``ΔT`` floored at 1 K.
    """
    delta_t = max(mix_temp_c - cold_in_c, 1.0)
    return max(0.0, available_kwh) / (delta_t * KWH_PER_LITER_KELVIN)


def showers_left(liters_40c: float, per_shower: float = SHOWER_LITERS_40C) -> float:
    """How many ~40 L showers the available litres cover."""
    if per_shower <= 0:
        return 0.0
    return max(0.0, liters_40c) / per_shower
