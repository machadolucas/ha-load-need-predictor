"""Pure hot-water tank state-of-charge model for Load Need Predictor.

**This module imports nothing from Home Assistant.** Like ``predictor.py`` and
``price_model.py`` it is loaded standalone (via ``importlib``) by the pure unit
tests, so it must stay dependency-free. The one non-obvious import is
``datetime`` — used only to parse fixed ISO timestamps passed in by the caller
(deterministic string arithmetic, no wall-clock access), which keeps the module
just as reproducible as the rest.

Why an energy balance (see the approved plan + CLAUDE.md): the water heater has
no "how full is the tank" sensor, and the daily *need* predictor already showed
that draw is stochastic. But three cumulative signals — delivered energy, the
cold-water meter, and the contactor/heating-element states — let us *integrate*
the tank's charge continuously and self-correct at natural checkpoints:

    deficit_kwh += draw + standing_loss − energy_in      # per tick, clamped [0, E_cap]
    SoC = clamp(1 − deficit / E_cap, 0, 1)

``deficit_kwh`` is the energy the tank is *below* "full at setpoint"; capacity is
``E_cap = volume × c × ΔT`` with ``ΔT = setpoint − cold_inlet`` (cross-checked on
the author's LVV: a fully depleted 300 L / 75 °C tank takes ~7–8 h at 3 kW ≈
22 kWh, matching a ~12 °C inlet).

**The anchor trick.** Integration drifts, so we re-zero it whenever physics hands
us ground truth: the contactor is commanded *on* but the heating element has gone
*idle* (its internal thermostat tripped) ⇒ the tank is physically full ⇒
``deficit = 0``. On the author's LVV this happens ~2×/week — sparse, but exact.
Anchors fire only on the *transition* (a latch dedupes the ticks while the
thermostat stays tripped) and require *sustained* states (duration thresholds), so
the sub-second ``unavailable`` blips both entities show never trigger a false
anchor.

**Learning, with guardrails.** Each anchor→anchor cycle closes the balance
identity ``energy_in − standby·Δt = hot_fraction · liters · ΔT · c``, from which we
EWMA-refine two parameters: ``hot_fraction`` (share of metered cold water that was
actually heated — seasonal, drifts down in summer) from cycles that metered enough
draw, and ``standby_w`` (standing loss) from near-zero-draw long cycles. Both are
hard-clamped so a bad cycle can't run them away, and learning is gated on the tank
being ``calibrated`` (never on the first anchor, whose cycle started from the
seeded 50 % guess so the identity doesn't hold) and the cycle being ``clean`` (no
meter fallback/misread ticks polluting the totals).

Every public function here is pure; ``apply_tick`` is the orchestrator the
HA-side ``tank_tracker`` calls once per tick, and ``dataclasses.replace`` is the
only way state changes (the dataclasses are frozen).
"""

from __future__ import annotations

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
SEED_HOT_FRACTION = 0.25
HOT_FRACTION_MIN, HOT_FRACTION_MAX = 0.05, 0.9
# standby_w: standing loss. Seed 70 W ≈ 1.68 kWh/day, matching the observed
# ~1.7 kWh gap between mean delivery and estimated draws.
SEED_STANDBY_W = 70.0
STANDBY_W_MIN, STANDBY_W_MAX = 20.0, 200.0

LEARN_BETA = 0.2  # EWMA weight per anchor cycle (~2/week → adapts over 2–4 weeks)
LEARN_MIN_LITERS = 50.0  # a cycle must meter ≥ this to move hot_fraction (signal)
STANDBY_MAX_LITERS = 10.0  # near-zero-draw cycle → its energy is (almost) all standby
STANDBY_MIN_HOURS = 12.0  # …and long enough that standby dominates the balance

# ── Water-meter attribution + guards ──────────────────────────────────────────
# Hot water only reaches taps + showers, whose combined flow tops out ~8 L/min;
# anything faster (garden hose, dishwasher/washing-machine fill) is cold-only and
# must not be charged to the tank. The cap is a *rate* over the span since the
# baseline, so it scales with however long the meter was quiet.
MAX_HOT_FLOW_LPM = 8.0
# Misread guard: even every tap open at once can't exceed ~30 L/min, so a larger
# implied rate is an OCR misread → drop the delta and re-baseline.
MAX_PLAUSIBLE_FLOW_LPM = 30.0
# A missing reading below this age is a blip: wait (draw 0), the cumulative meter
# catches up. Beyond it, fall back to the occupancy estimate.
WATER_STALE_AFTER_S = 900.0

# ── Anchor thresholds ─────────────────────────────────────────────────────────
# Sustained-state requirements: the element must have been idle ≥ 60 s (a real
# thermostat trip, not a blip) while the contactor has been commanded on ≥ 120 s
# (long enough that "on but idle" means full, not just switching on).
ANCHOR_MIN_HEATING_OFF_S = 60.0
ANCHOR_MIN_CONTACTOR_ON_S = 120.0

# ── Heating-active floor (the anchor's inverse) ───────────────────────────────
# An element actively drawing power means the thermostat is calling for heat —
# the tank is below setpoint by at least the hysteresis band, whatever the
# energy balance says. Floor the deficit accordingly (≈ 5 % of a 300 L tank) so
# the sensor can never read 100 % *while heating*; only a genuine
# commanded-on-but-idle anchor shows full. Sustained ≥ 60 s so detector blips
# can't inject the floor.
HEATING_MIN_DEFICIT_KWH = 1.0
HEATING_ACTIVE_MIN_S = 60.0

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

TANK_VERSION = "v1"


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

    ``deficit_kwh`` is the running energy-balance charge deficit. The baselines
    (energy counter, water counter, and *when* the water baseline was taken) make
    the deltas cumulative and lossless across restarts. The ``cycle_*`` fields
    accumulate the current anchor→anchor cycle so :func:`learn_from_cycle` can
    close the balance; ``pending_fallback_kwh`` is fallback draw charged since the
    last valid meter read, reconciled out of the next metered delta so an OCR
    dropout doesn't double-count. ``boost_armed`` / ``last_boost_iso`` back the
    low-charge boost hysteresis + rate limit.
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
    version: str = TANK_VERSION


@dataclass(frozen=True)
class TickInputs:
    """Everything one tick needs, read by the tracker from ``hass.states``.

    Cumulative counters may be ``None`` (sensor missing/unavailable). The
    contactor/heating booleans are tri-state (``None`` when unknown — never
    treated as off, so a missing sensor can't spuriously anchor). The trailing
    ``e_*`` fields are the load model's current occupancy params, used only for
    the fallback draw estimate.
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
    # ``heating_off_for_s`` therefore holds the time spent heating — which is
    # what the heating-active deficit floor checks.
    contactor_on_for_s: float | None
    heating_off_for_s: float | None
    people_home: int | None
    e_base: float
    e_draw_per_person: float
    empty_house_factor: float


@dataclass(frozen=True)
class TickResult:
    """The new state plus per-tick diagnostics (published + logged).

    ``draw_source`` records how the draw was attributed this tick — ``"meter"``
    (a valid cold-meter delta), ``"fallback"`` (occupancy estimate; meter stale or
    misread), or ``"none"`` (short blip, waiting for the meter to catch up).
    """

    state: TankState
    soc: float
    capacity_kwh: float
    draw_source: str
    anchored: bool
    energy_in_kwh: float
    draw_kwh: float
    standby_kwh: float


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


def counter_delta(baseline: float | None, reading: float | None) -> tuple[float, float | None]:
    """Consume a cumulative counter → ``(delta_since_baseline, new_baseline)``.

    A missing reading yields no delta and keeps the baseline (wait for the next
    read). A missing baseline or a *decrease* (counter reset / meter swap) yields
    no delta but re-baselines to the reading, so a reset never charges a huge
    phantom delta.
    """
    if reading is None:
        return 0.0, baseline
    if baseline is None or reading < baseline:
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
    over the span *since the baseline was taken* — in which case it is treated as
    a misread and we re-baseline. Because the rate is measured over the baseline
    span, a legitimate multi-litre delta that accumulated across a restart gap
    still passes (the span is large), while the same delta over a single tick is
    correctly rejected.
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

    Water above the hot-flow rate cap over the span is cold-only usage (garden
    hose, appliance fill) and is *not* charged to the tank.
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


def learn_from_cycle(state: TankState, params: TankParams, cycle_hours: float) -> TankState:
    """EWMA-refine ``hot_fraction`` / ``standby_w`` from a closed anchor cycle.

    Gated on a calibrated tank (the first cycle starts from the seeded guess, so
    the balance identity doesn't hold), a clean cycle (no fallback/misread ticks),
    and a positive duration. The two parameters are learned from disjoint cycle
    regimes so they never fight:

    - **hot_fraction** from cycles that metered ≥ ``LEARN_MIN_LITERS`` (enough draw
      to invert the identity for it), skipping any cycle whose delivered energy
      didn't even cover standby (numerator ≤ 0).
    - **standby_w** only from near-zero-draw (< ``STANDBY_MAX_LITERS``) *and* long
      (≥ ``STANDBY_MIN_HOURS``) cycles, where the delivered energy is essentially
      all standing loss.

    Mid-size cycles (10–50 L) match neither regime and teach nothing. Every update
    is hard-clamped to its guardrail band.
    """
    if not state.calibrated or not state.cycle_clean or cycle_hours <= 0:
        return state
    delta_t = max(params.setpoint_c - params.cold_in_c, 1.0)
    standby_over_cycle = state.standby_w * cycle_hours / 1000.0
    new_hot_fraction = state.hot_fraction
    new_standby_w = state.standby_w
    if state.cycle_liters >= LEARN_MIN_LITERS:
        numerator = state.cycle_energy_in_kwh - standby_over_cycle
        denom = state.cycle_liters * delta_t * KWH_PER_LITER_KELVIN
        if numerator > 0 and denom > 0:
            implied = numerator / denom
            new_hot_fraction = _clamp(
                (1.0 - LEARN_BETA) * state.hot_fraction + LEARN_BETA * implied,
                HOT_FRACTION_MIN,
                HOT_FRACTION_MAX,
            )
    elif state.cycle_liters < STANDBY_MAX_LITERS and cycle_hours >= STANDBY_MIN_HOURS:
        draw_est = draw_kwh_from_liters(
            state.cycle_liters, state.hot_fraction, params.setpoint_c, params.cold_in_c
        )
        implied_w = (state.cycle_energy_in_kwh - draw_est) / cycle_hours * 1000.0
        if implied_w > 0:
            new_standby_w = _clamp(
                (1.0 - LEARN_BETA) * state.standby_w + LEARN_BETA * implied_w,
                STANDBY_W_MIN,
                STANDBY_W_MAX,
            )
    if new_hot_fraction == state.hot_fraction and new_standby_w == state.standby_w:
        return state
    return replace(state, hot_fraction=new_hot_fraction, standby_w=new_standby_w)


def _resolve_draw(
    state: TankState, params: TankParams, inputs: TickInputs, elapsed_min: float
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
            raw = draw_kwh_from_liters(hot, state.hot_fraction, params.setpoint_c, params.cold_in_c)
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

    Integrates the energy balance, attributes the draw, applies standing loss,
    then either re-zeros at a 100 % anchor (learning from the cycle it closes) or
    carries the running deficit forward. Restart reconciliation is just the
    ordinary first tick after a gap: the counters are cumulative, so the deltas and
    the standby-over-the-gap all fall out of the same arithmetic.
    """
    capacity = capacity_kwh(params.volume_l, params.setpoint_c, params.cold_in_c)
    elapsed_min = max(0.0, inputs.elapsed_s) / 60.0

    # Energy in (cumulative counter delta; reset-safe → 0 + re-baseline).
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
    ) = _resolve_draw(state, params, inputs, elapsed_min)

    standby = standby_kwh(state.standby_w, elapsed_min)

    # Energy-balance deficit, bounded to the physical tank.
    new_deficit = _clamp(state.deficit_kwh + draw_kwh + standby - energy_in, 0.0, capacity)

    # Cycle accumulators including this tick's contributions.
    acc_energy = state.cycle_energy_in_kwh + energy_in
    acc_liters = state.cycle_liters + metered_hot_liters
    acc_clean = state.cycle_clean and tick_clean

    # Tick-level bookkeeping applied on every branch.
    base = replace(
        state,
        energy_baseline_kwh=new_energy_baseline,
        water_baseline_l=new_water_baseline_l,
        water_baseline_iso=new_water_baseline_iso,
        pending_fallback_kwh=new_pending,
        last_tick_iso=inputs.now_iso,
    )

    anchored = should_anchor(
        inputs.contactor_on,
        inputs.heating_on,
        inputs.contactor_on_for_s,
        inputs.heating_off_for_s,
    )

    if anchored and not state.anchor_latched:
        # Anchor transition: close + learn the cycle (a no-op on the first anchor,
        # still uncalibrated), then pin the tank full and open a fresh cycle.
        cycle_hours = _hours_between(state.cycle_start_iso, inputs.now_iso)
        accumulated = replace(
            base,
            cycle_energy_in_kwh=acc_energy,
            cycle_liters=acc_liters,
            cycle_clean=acc_clean,
        )
        learned = learn_from_cycle(accumulated, params, cycle_hours)
        new_state = replace(
            learned,
            deficit_kwh=0.0,
            calibrated=True,
            anchor_latched=True,
            last_anchor_iso=inputs.now_iso,
            cycle_start_iso=inputs.now_iso,
            cycle_energy_in_kwh=0.0,
            cycle_liters=0.0,
            cycle_clean=True,
        )
        result_deficit = 0.0
    elif anchored:
        # Still anchored (thermostat stays tripped): hold full, no re-learn, and
        # keep the (empty) cycle rolling forward so it starts fresh on release.
        new_state = replace(
            base,
            deficit_kwh=0.0,
            anchor_latched=True,
            cycle_start_iso=inputs.now_iso,
            cycle_energy_in_kwh=0.0,
            cycle_liters=0.0,
            cycle_clean=True,
        )
        result_deficit = 0.0
    else:
        # Ordinary integrating tick: carry the deficit, accumulate the cycle, and
        # clear the latch so the next anchor transition can fire.
        # The anchor's inverse: sustained active heating proves the tank is below
        # setpoint by at least the thermostat hysteresis, so the balance may not
        # read (near-)full — the seeds underestimate draws until calibrated, and
        # without this floor the clamp would show 100 % with the element running.
        if (
            inputs.heating_on is True
            and inputs.heating_off_for_s is not None
            and inputs.heating_off_for_s >= HEATING_ACTIVE_MIN_S
        ):
            new_deficit = max(new_deficit, min(HEATING_MIN_DEFICIT_KWH, capacity))
        new_state = replace(
            base,
            deficit_kwh=new_deficit,
            anchor_latched=False,
            cycle_start_iso=state.cycle_start_iso or inputs.now_iso,
            cycle_energy_in_kwh=acc_energy,
            cycle_liters=acc_liters,
            cycle_clean=acc_clean,
        )
        result_deficit = new_deficit

    return TickResult(
        state=new_state,
        soc=soc(result_deficit, capacity),
        capacity_kwh=capacity,
        draw_source=draw_source,
        anchored=anchored,
        energy_in_kwh=energy_in,
        draw_kwh=draw_kwh,
        standby_kwh=standby,
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
