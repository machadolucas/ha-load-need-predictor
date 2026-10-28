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

**…but not past a no-trip cut-off.** That curve is meant for *while heating*. If
the contactor opens with the element still drawing, the thermostat hasn't
tripped, so the tank is not full — a "beyond full" balance is then provably
wrong, and the curve's heavy tail would keep crediting it against every later
loss. A settled cut-off therefore floors the displayed and control deficits at
``σ`` and, when the contradiction exceeds the thermostat band, learns from it
(see ``CUTOFF_SETTLE_S``).

**Learning, with guardrails.** Each qualifying anchor→anchor cycle closes with a
*residual* — the unclamped deficit at the trip, which is exactly the model's error
over that cycle. Short cycles (< 4 h, the post-trip re-heats) teach
``hysteresis_kwh`` only. Every long cycle takes one preconditioned normalised-LMS
step that splits the residual between the daypart ``hot_fraction_profile`` (only
when it metered enough water; then ridge-pulled toward the profile mean so sparse
buckets can't run away), the base ``standby_w`` and its cold-weather slope
``standby_w_per_k`` — each in proportion to the energy it accounts for. Everything
is EWMA-slow and hard-clamped: the replay showed a fast learner chases the ±1 kWh
cycle noise and gets worse.

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
# ~1.7 kWh gap between mean delivery and estimated draws. Since v0.14.0 this is
# the *base* loss (ambient at/above ``STANDBY_REF_C``); the cap was raised from
# 200 W because that near-dry 47 h cycle needed ≥ 178 W to close.
SEED_STANDBY_W = 70.0
STANDBY_W_MIN, STANDBY_W_MAX = 20.0, 250.0
# Cold-weather standing loss: the author's tank sits in an exterior (NE) kitchen
# corner, so it loses more when it's cold outside. Modelled as a hinge on the
# configured outdoor temperature — ``standby_w + standby_w_per_k × max(0,
# STANDBY_REF_C − T)`` — with the slope learned like every other loss term. The
# 0 W/K seed is the old constant model: the slope only grows once cycles at
# different temperatures disagree with a constant. No temperature ⇒ no cold term.
STANDBY_REF_C = 15.0
SEED_STANDBY_W_PER_K = 0.0
STANDBY_W_PER_K_MIN, STANDBY_W_PER_K_MAX = 0.0, 15.0
STANDBY_EFFECTIVE_MAX_W = 400.0  # base + cold term, whatever the two learned
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
STANDBY_MAX_LITERS = 10.0  # a post-trip re-heat must be ~dry to teach the hysteresis
LEARN_MIN_CYCLE_HOURS = 4.0  # shorter cycles are post-trip re-heats: hysteresis only
HYSTERESIS_MIN_ENERGY_KWH = 0.2  # a re-heat must deliver this much to count as one
RESIDUAL_RATIO_MIN_GROSS_KWH = 0.5  # don't learn σ from cycles with ~no flow
# Prior scales for splitting a cycle residual across the loss terms (v0.14.0):
# one normalised-LMS step preconditioned by each parameter's prior spread, so a
# term takes a share ∝ (scale × its energy in the cycle)². A dry cycle teaches
# standby alone (as before), a wet one splits between the hot fractions and
# standby, and a cold one also moves the cold slope. Before this, standby only
# learned from ≥ 12 h cycles metering < 10 L — which never happened between the
# daily anchors, so standby sat at its seed for months while the hot fractions
# absorbed its error, until a near-dry 47 h cycle missed by > 3 kWh (2026-10-28).
PRIOR_HOT_FRACTION = 0.08
PRIOR_STANDBY_W = 30.0
PRIOR_STANDBY_W_PER_K = 3.0

# ── No-trip cut-offs ──────────────────────────────────────────────────────────
# When the contactor opens while the element is still heating, the thermostat has
# not tripped, so the tank is *not* full. If the learning ledger says it is past
# full by more than the thermostat band, the model has provably under-counted
# losses since the last anchor. That is evidence like an anchor, but one-sided:
# - the display and control ledgers are floored at the model's own σ, so the %
#   stops crediting a phantom surplus (the curve's 1/|u| tail otherwise shrank
#   every later loss to a few percent of its size — on 2026-10-28, 2 kWh of real
#   losses moved the display 0.6 points);
# - the learner takes a normal step on the excess beyond the band, and the
#   learning ledger is re-based to the corrected params, so the next anchor's
#   residual doesn't count the same error twice.
# The settle delay lets a lagging energy counter publish the run's last kWh first;
# a settle also waits while either counter is missing (its unmetered energy would
# otherwise read as a phantom surplus, be learned from, and be charged again on
# top of the floor once the reading returns). Cycles shorter than
# LEARN_MIN_CYCLE_HOURS are post-trip re-heats, where a balance past the first
# trip's level is physics, not error — they are neither floored nor learned from.
CUTOFF_SETTLE_S = 120.0
# The element stopping this much *before* the contactor opened means the
# thermostat tripped first (a full tank), not a cut-off.
CUTOFF_TRIP_MARGIN_S = 30.0
# A cut-off must be *seen*: a tick after a longer gap (HA was down; startup also
# resets every ``last_changed``) can't tell a trip from a cut-off.
CUTOFF_MAX_GAP_S = 180.0
# Beyond the learned thermostat band (a re-heated tank holds ~hysteresis more
# than at a first trip), a smaller contradiction is noise, not evidence.
CUTOFF_LEARN_MARGIN_KWH = 0.3
# An outdoor reading older than this is dropped (a removed or broken sensor must
# not keep a stale cold term running); until then a dropout holds the last value.
AMBIENT_HOLD_H = 6.0

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
# Slow meters: a meter that publishes a 50 L step every 10 min re-reports the same
# value on every tick in between, so "time since the previous read" (one tick)
# would present the whole step as one minute of flow — a misread for the 30 L/min
# guard, and clipped to 8 L by the hot-flow cap. The rate span may therefore
# stretch back over the meter's own change-to-change interval (its
# ``last_changed`` stamps), but only as far as the meter's learned publish
# cadence (``water_cadence_min``) and never past this window. The cadence starts
# at one tick and is raised only on *evidence of batching*: a step physically
# impossible over the current span (> MAX_PLAUSIBLE_FLOW_LPM — no plumbing does
# that, so it's an OCR misread or a batched publish) yet plausible over the
# meter's own change interval. The hot-flow cap is deliberately not the trigger:
# a > 8 L/min tick is a real hose/appliance fill and says nothing about batching.
# Why gate at all instead of always stretching: the replayed week (a responsive
# meter stepping ~2 L every ~30 s) showed stretching its 9–17 L evening steps
# attributes ~0.5 kWh/week more hot draw and worsens the trip residuals (rms
# 1.17 → 1.24 kWh) — for a fast meter the one-tick cap is right, and isolated
# small draws minutes apart must not loosen it. Time the meter was *not*
# observed at all (a dropout, a restart) always counts in full.
WATER_RATE_WINDOW_MIN = 15.0

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
# There is deliberately no "implausibly fast" energy-delta guard: after a restart
# a stale intermediate reading followed by the real one looks exactly like a jump,
# and silently discarding delivered energy is the failure the replay exists to
# catch. Entity swaps are handled by ``rebind_sources``, units by the tracker.

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
# …but never above 100 %: a high threshold (e.g. 90 + 15) could otherwise never
# re-arm. The control ledger reads exactly 100 at an anchor, so a trip re-arms.
BOOST_REARM_MAX_PCT = 100.0
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
    low-charge boost hysteresis + rate limit. ``water_changed_iso`` /
    ``water_cadence_min`` time the water meter's own steps (the slow-meter rate
    window) and ``*_source`` names the entity each baseline came from, so a
    reconfigured counter re-baselines. ``hot_fraction`` is kept as the mean
    of ``hot_fraction_profile`` (an empty profile means "all buckets = hot_fraction",
    which is how pre-v2 stored state migrates). ``standby_w`` is the base standing
    loss and ``standby_w_per_k`` its cold-weather slope (see :func:`standby_w_at`);
    ``cycle_cold_kh`` accumulates the cycle's K·h below ``STANDBY_REF_C`` as that
    slope's learning feature. ``heating_prev`` / ``cutoff_pending_iso`` detect and
    settle a no-trip cut-off, and ``cycle_display_lift_kwh`` is the floor it put
    under the displayed deficit (reset at an anchor).
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
    # When the water reading last *changed* ("" ⇒ unknown: the span falls back to
    # "since the last read") — the slow-meter rate window, see WATER_RATE_WINDOW_MIN.
    water_changed_iso: str = ""
    # The water meter's learned publish cadence (minutes; 0 ⇒ unknown ⇒ one-tick
    # spans): how far a delta's rate span may stretch. Raised only on evidence of
    # batched publishing, see :func:`_resolve_draw`.
    water_cadence_min: float = 0.0
    # Which entity each baseline came from ("" ⇒ unknown, adopted as-is): a
    # reconfigured counter must re-baseline, not dump its whole difference.
    energy_source: str = ""
    water_source: str = ""
    # v0.14.0 additions (defaulted: an older payload loads as "no slope, nothing
    # pending, no lift" — exactly the previous behaviour until the next events).
    standby_w_per_k: float = SEED_STANDBY_W_PER_K
    last_ambient_c: float | None = None  # last known ambient, held across a dropout
    last_ambient_iso: str = ""  # …for at most AMBIENT_HOLD_H after it was read
    cycle_cold_kh: float = 0.0
    heating_prev: bool | None = None  # last *known* element state (blips don't count)
    cutoff_pending_iso: str = ""  # a cut-off seen, waiting out CUTOFF_SETTLE_S
    last_cutoff_iso: str = ""
    cycle_display_lift_kwh: float = 0.0
    # The learning ledger level the last cut-off of this cycle learned down to
    # (None ⇒ none yet: the band edge). A later cut-off learns only from going
    # further past it — the same contradiction is never learned twice.
    cycle_cutoff_mark_kwh: float | None = None
    # When a *running* tracker first saw the water reading missing ("" ⇒ present).
    # Staleness is measured from here, not from the last read, so an HA restart
    # (whose downtime nobody observed) doesn't count as a meter outage.
    water_missing_since_iso: str = ""
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
    # When the water reading last changed on the meter's side (the entity's
    # ``last_changed``); ``None`` ⇒ unknown, the tick time stands in.
    water_changed_iso: str | None = None
    # Ambient (outdoor) temperature in °C for the cold-weather standing loss;
    # ``None`` ⇒ unknown, the last known value is held (none ever ⇒ no cold term).
    ambient_c: float | None = None


@dataclass(frozen=True)
class TickResult:
    """The new state plus per-tick diagnostics (published + logged).

    ``draw_source`` records how the draw was attributed this tick — ``"meter"``
    (a valid cold-meter delta), ``"fallback"`` (occupancy estimate; meter stale or
    misread), or ``"none"`` (short blip, waiting for the meter to catch up).
    ``anchored`` is True only on the anchor *transition* tick (the moment the tank
    is known full and the model learns); ``latched`` while the trip still holds.
    ``soc`` is derived from ``deficit_shown_kwh`` (the saturation-curve display
    deficit), not from the clamped control ``state.deficit_kwh``. ``cutoff`` is
    True on the tick a no-trip cut-off settled (floored, maybe learned) — like
    ``anchored``, a "persist now" signal. ``standby_w_now`` is the standing loss
    this tick actually used (base + cold term).
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
    cutoff: bool = False
    standby_w_now: float = SEED_STANDBY_W


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
    re-baseline. The caller's span (:func:`_water_span_minutes`) covers the
    whole unobserved gap plus, for a meter with a demonstrated slow publish
    cadence, its own change-to-change interval — so a delta that accumulated
    while HA was down, or behind a slow meter's cadence, still passes, while the
    same delta from a fast meter over a single tick is correctly rejected.
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
    hose, appliance fill) and is *not* charged to the tank. The span is the same
    one the misread guard uses — one tick for a fast meter (an 8 L/tick cap), a
    slow meter's own step interval (≤ ``WATER_RATE_WINDOW_MIN``), and the whole
    gap after a restart, so a delta that accumulated while HA was down is not
    clipped.
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


def cold_hinge_k(ambient_c: float | None) -> float:
    """Kelvin below ``STANDBY_REF_C`` — 0 when warmer, or when the ambient is unknown."""
    if ambient_c is None:
        return 0.0
    return max(0.0, STANDBY_REF_C - ambient_c)


def standby_w_at(state: TankState, ambient_c: float | None) -> float:
    """Standing loss (W) at an ambient temperature: base + learned cold slope.

    Only the sum is capped — each term is already clamped where it is learned.
    """
    return _clamp(
        state.standby_w + state.standby_w_per_k * cold_hinge_k(ambient_c),
        0.0,
        STANDBY_EFFECTIVE_MAX_W,
    )


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
    - **long**: one :func:`_attribute_residual` step splits the residual between
      the hot-fraction profile (only when the cycle metered ≥
      ``LEARN_MIN_LITERS``; then every bucket is also pulled ``PROFILE_RIDGE``
      toward the mean), the base ``standby_w`` and the cold slope
      ``standby_w_per_k`` — a dry cycle therefore teaches standby alone.

    Long cycles also update ``residual_ratio`` (|residual| per kWh of attributed
    flow) — the saturation curve's σ. Every update is EWMA-slow and hard-clamped.
    """
    if not state.calibrated or not state.cycle_clean or cycle_hours <= 0:
        return state
    standby_over_cycle = _standby_over_cycle_kwh(state, cycle_hours)

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

    learned, _ = _attribute_residual(state, params, cycle_hours, residual_kwh)
    new_ratio = state.residual_ratio
    if state.cycle_gross_kwh >= RESIDUAL_RATIO_MIN_GROSS_KWH:
        observed_ratio = min(RESIDUAL_RATIO_MAX, abs(residual_kwh) / state.cycle_gross_kwh)
        new_ratio = _clamp(
            (1.0 - LEARN_BETA_FAST) * state.residual_ratio + LEARN_BETA_FAST * observed_ratio,
            RESIDUAL_RATIO_MIN,
            RESIDUAL_RATIO_MAX,
        )
    return replace(learned, residual_ratio=new_ratio)


def learn_from_cutoff(
    state: TankState, params: TankParams, cycle_hours: float, ledger_kwh: float
) -> tuple[TankState, float]:
    """Learn from a no-trip cut-off → ``(new_state, extra_losses_kwh)``.

    The element was still heating when the contactor opened, so the tank is not
    full — yet ``ledger_kwh`` (the learning ledger) may say it is past full. Only
    the part beyond the thermostat band (``hysteresis_kwh`` +
    ``CUTOFF_LEARN_MARGIN_KWH``) is a contradiction, and it is a *lower bound*
    on the model's error, so the step uses just that excess: the same
    :func:`_attribute_residual` step an anchor takes, and ``residual_ratio`` may
    only grow (a bound can't show the model is better than it thought). Gated
    like an anchor: calibrated, clean, and a long cycle (a short one is a
    post-trip re-heat, where being past the first trip's level is normal).
    Within one cycle the threshold is ``cycle_cutoff_mark_kwh`` once a cut-off
    has learned: a later one only learns from going *further* past it (more
    energy delivered without a trip), never from the same contradiction again.

    ``extra_losses_kwh`` is how much more the corrected params say the cycle so
    far lost; the caller adds it to the learning ledger (and sets the mark to
    the re-based level) so the next anchor's residual is measured against the
    new params and the same error isn't learned twice.
    """
    if not state.calibrated or not state.cycle_clean or cycle_hours < LEARN_MIN_CYCLE_HOURS:
        return state, 0.0
    threshold = state.cycle_cutoff_mark_kwh
    if threshold is None:
        threshold = -(state.hysteresis_kwh + CUTOFF_LEARN_MARGIN_KWH)
    if ledger_kwh >= threshold:
        return state, 0.0
    excess = ledger_kwh - threshold
    learned, extra = _attribute_residual(state, params, cycle_hours, excess)
    learned = replace(learned, cycle_cutoff_mark_kwh=ledger_kwh + extra)
    if state.cycle_gross_kwh >= RESIDUAL_RATIO_MIN_GROSS_KWH:
        bound_ratio = min(RESIDUAL_RATIO_MAX, -excess / state.cycle_gross_kwh)
        if bound_ratio > state.residual_ratio:
            learned = replace(
                learned,
                residual_ratio=_clamp(
                    (1.0 - LEARN_BETA_FAST) * state.residual_ratio + LEARN_BETA_FAST * bound_ratio,
                    RESIDUAL_RATIO_MIN,
                    RESIDUAL_RATIO_MAX,
                ),
            )
    return learned, extra


def _standby_over_cycle_kwh(state: TankState, cycle_hours: float) -> float:
    """Standing loss the current params attribute to the cycle so far (kWh)."""
    return (state.standby_w * cycle_hours + state.standby_w_per_k * state.cycle_cold_kh) / 1000.0


def _attribute_residual(
    state: TankState, params: TankParams, cycle_hours: float, residual_kwh: float
) -> tuple[TankState, float]:
    """One preconditioned normalised-LMS step on a cycle residual.

    → ``(new_state, extra_losses_kwh)``. The cycle's modelled loss is linear in
    the learnable params — ``Σ hf_b·X_b + standby_w·H + standby_w_per_k·C`` (X_b
    the bucket's hot litres as kWh per unit fraction, H hours, C cold K·h, both
    ÷ 1000 for W→kWh) — so the step moves each param by ``−β·r·s²·x / Σ s²x²``
    with ``s`` its ``PRIOR_*`` scale. The total correction to the cycle's loss is
    always ``β·r`` (as the single-term learners had), split ∝ ``(s·x)²``: no
    litres ⇒ all standby, no cold hours ⇒ no slope change. The hot fractions only
    take part when the cycle metered ≥ ``LEARN_MIN_LITERS`` (fewer can't tell a
    shower from a toilet), and only then are they ridge-pulled toward their mean.
    ``extra_losses_kwh`` is recomputed from the *clamped* new params, so a clamp
    can't fake a correction.
    """
    delta_t = max(params.setpoint_c - params.cold_in_c, 1.0)
    buckets = state.cycle_hot_liters_by_bucket
    if len(buckets) != N_DAYPARTS or state.cycle_liters < LEARN_MIN_LITERS:
        buckets = _ZERO_PROFILE
    x_hot = [liters * delta_t * KWH_PER_LITER_KELVIN for liters in buckets]
    x_base = cycle_hours / 1000.0
    x_cold = state.cycle_cold_kh / 1000.0
    s_hot = PRIOR_HOT_FRACTION**2
    s_base = PRIOR_STANDBY_W**2
    s_cold = PRIOR_STANDBY_W_PER_K**2
    norm = s_hot * sum(x * x for x in x_hot) + s_base * x_base**2 + s_cold * x_cold**2
    if norm <= 0:
        return state, 0.0
    step = -LEARN_BETA * residual_kwh / norm

    profile = profile_of(state)
    new_profile = profile
    if any(x > 0 for x in x_hot):
        stepped = [hf + step * s_hot * x for hf, x in zip(profile, x_hot, strict=True)]
        mean = sum(stepped) / N_DAYPARTS
        new_profile = tuple(
            _clamp(hf - PROFILE_RIDGE * (hf - mean), HOT_FRACTION_MIN, HOT_FRACTION_MAX)
            for hf in stepped
        )
    new_base = _clamp(state.standby_w + step * s_base * x_base, STANDBY_W_MIN, STANDBY_W_MAX)
    new_slope = _clamp(
        state.standby_w_per_k + step * s_cold * x_cold, STANDBY_W_PER_K_MIN, STANDBY_W_PER_K_MAX
    )
    extra = (
        sum((new - old) * x for new, old, x in zip(new_profile, profile, x_hot, strict=True))
        + (new_base - state.standby_w) * x_base
        + (new_slope - state.standby_w_per_k) * x_cold
    )
    learned = replace(
        state,
        hot_fraction=sum(new_profile) / N_DAYPARTS,
        hot_fraction_profile=new_profile,
        standby_w=new_base,
        standby_w_per_k=new_slope,
    )
    return learned, extra


def _water_span_minutes(state: TankState, inputs: TickInputs, elapsed_min: float) -> float:
    """Minutes the current water delta is spread over, for the rate guards.

    Time since the last *read* always counts (nothing was observed then — a
    dropout or restart gap) and is the floor, so this is never stricter than the
    one-tick span. On top of that, the meter's own change-to-change interval
    (previous reading's change stamp → this reading's) counts up to its learned
    cadence floor (≤ ``WATER_RATE_WINDOW_MIN``), because a slow meter re-reports
    the same value until its next step. With no change stamp or no cadence yet
    it is the old "since the last read".
    """
    since_read = _span_minutes(state.water_baseline_iso, inputs.now_iso, elapsed_min)
    if not state.water_changed_iso or state.water_cadence_min <= 0:
        return since_read
    changed_now = inputs.water_changed_iso or inputs.now_iso
    between_changes = _span_minutes(state.water_changed_iso, changed_now, since_read)
    stretch = min(state.water_cadence_min, WATER_RATE_WINDOW_MIN)
    return max(since_read, min(between_changes, stretch))


def rebind_sources(state: TankState, energy_source: str, water_source: str) -> TankState:
    """Re-baseline any counter whose source entity changed since its baseline.

    Baselines are raw readings of *one* entity: a reconfigured counter (e.g. a
    powercalc ``_2`` re-add) reading higher would otherwise land its whole
    difference in a single tick — the tank reads full and the cycle closes on a
    bogus residual. A changed source therefore drops that baseline (the next
    reading is adopted with no delta) and dirties the cycle, whose balance now
    spans two meters. An empty stored source is state from before this was
    recorded: adopt the current one as-is, so an upgrade loses nothing.
    """
    updates: dict = {}
    if state.energy_source != energy_source:
        updates["energy_source"] = energy_source
        if state.energy_source:
            updates.update(energy_baseline_kwh=None, cycle_clean=False)
    if state.water_source != water_source:
        updates["water_source"] = water_source
        if state.water_source:
            updates.update(
                water_baseline_l=None,
                water_baseline_iso="",
                water_changed_iso="",
                pending_fallback_kwh=0.0,
                cycle_clean=False,
            )
    return replace(state, **updates) if updates else state


@dataclass(frozen=True)
class _WaterTick:
    """One tick's draw decision + the water meter's new bookkeeping.

    ``hot_liters`` (0 unless the meter was valid) is what accumulates into the
    cycle, so learning uses the same capped attribution as the deficit.
    """

    draw_kwh: float
    source: str
    clean: bool
    hot_liters: float
    baseline_l: float | None
    baseline_iso: str
    changed_iso: str
    cadence_min: float
    pending_kwh: float
    missing_since_iso: str = ""


def _resolve_draw(
    state: TankState,
    params: TankParams,
    inputs: TickInputs,
    elapsed_min: float,
    hot_fraction: float,
    standby_w: float,
) -> _WaterTick:
    """Decide this tick's draw + water bookkeeping (meter / fallback / none).

    Also trains the meter's publish cadence, from *validated* evidence only: a
    valid step arriving faster than the learned cadence lowers it (the meter just
    proved it publishes that quickly); a step impossible over the current span
    but plausible over the meter's own change interval is evidence of batched
    publishing and raises it. That evidencing step is itself still rejected — it
    can't yet be told from an OCR spike — and the meter's next step settles it.
    Adoptions, rollbacks/resets (negative deltas), steps implausible even over
    their own interval, and ordinary valid steps slower than the cadence (e.g.
    isolated small draws minutes apart) never raise it.
    """
    fallback = fallback_draw_kwh(
        inputs.e_base,
        inputs.e_draw_per_person,
        inputs.empty_house_factor,
        inputs.people_home,
        standby_w,
        elapsed_min,
    )
    reading = inputs.water_counter_l
    now_iso = inputs.now_iso
    cadence = state.water_cadence_min
    # The reading's own change time (meter-side), for the next delta's span.
    reading_changed_iso = inputs.water_changed_iso or now_iso
    if reading is not None and state.water_baseline_l is None:
        # First reading (fresh install, or a baseline dropped at an anchor /
        # source change): nothing to measure against yet, so adopt it with zero
        # draw. That is not a data-quality problem — the cycle stays clean (an
        # anchor that dropped the baseline already dirtied its own cycle). Any
        # pending fallback has no baseline to reconcile against → dropped.
        return _WaterTick(
            0.0, "none", True, 0.0, reading, now_iso, reading_changed_iso, cadence, 0.0
        )
    if reading is not None:
        span_min = _water_span_minutes(state, inputs, elapsed_min)
        delta, _ = water_delta(state.water_baseline_l, reading, span_min)
        # The meter's own interval for this step (None without a change stamp).
        between = (
            _span_minutes(state.water_changed_iso, reading_changed_iso, 0.0)
            if state.water_changed_iso
            else None
        )
        if delta is not None:
            # Valid metered draw: charge the hot portion, netting off any fallback
            # already charged since the baseline (floored so it can't go negative).
            hot = hot_attributable_liters(delta, span_min)
            raw = draw_kwh_from_liters(hot, hot_fraction, params.setpoint_c, params.cold_in_c)
            draw = max(0.0, raw - state.pending_fallback_kwh)
            if delta > 0 and between is not None and 0.0 < between < cadence:
                cadence = between  # a valid step this fast: the meter isn't that slow
            # Only a *change* moves the change stamp: an unchanged re-report must
            # not shrink a slow meter's next step into one tick of flow.
            changed_iso = (
                reading_changed_iso
                if delta > 0
                else (state.water_changed_iso or reading_changed_iso)
            )
            return _WaterTick(draw, "meter", True, hot, reading, now_iso, changed_iso, cadence, 0.0)
        raw_delta = reading - state.water_baseline_l
        if raw_delta > 0 and between is not None:
            own_span = min(between, WATER_RATE_WINDOW_MIN)
            if own_span > span_min and raw_delta <= MAX_PLAUSIBLE_FLOW_LPM * own_span:
                # Impossible over the current span, plausible over the meter's own
                # interval → it publishes in batches: learn that cadence.
                cadence = max(cadence, own_span)
        # Misread (negative or impossibly fast): fall back this tick, re-baseline,
        # mark the cycle dirty, and drop the unreconcilable pending.
        return _WaterTick(
            fallback, "fallback", False, 0.0, reading, now_iso, reading_changed_iso, cadence, 0.0
        )
    # No reading. A short gap is a blip — draw nothing and let the cumulative meter
    # catch up; only fall back (and accumulate pending) once the meter has been
    # *seen* missing for WATER_STALE_AFTER_S. A tick that follows an unobserved
    # gap (HA was down) restarts that clock: nobody watched the meter meanwhile,
    # and the cumulative reading reconciles the whole gap once it's back. Before
    # v0.14.0 the first tick after a restart — which often runs before the MQTT
    # meter is restored — fell back over the whole downtime and dirtied the cycle.
    unobserved_gap = inputs.elapsed_s >= WATER_STALE_AFTER_S
    missing_since = (
        state.water_missing_since_iso
        if state.water_missing_since_iso and not unobserved_gap
        else now_iso
    )
    missing_dt = _parse_iso(missing_since)
    now_dt = _parse_iso(now_iso)
    fresh = (
        state.water_baseline_l is not None
        and missing_dt is not None
        and now_dt is not None
        and (now_dt - missing_dt).total_seconds() < WATER_STALE_AFTER_S
    )
    if fresh:
        return _WaterTick(
            0.0,
            "none",
            True,
            0.0,
            state.water_baseline_l,
            state.water_baseline_iso,
            state.water_changed_iso,
            cadence,
            state.pending_fallback_kwh,
            missing_since,
        )
    if state.water_baseline_l is not None and state.water_baseline_iso:
        # Stale with a baseline: the fallback covers everything since the last
        # valid read, net of what it already charged — so the ticks spent waiting
        # (and a restart's downtime) are charged once the wait gives up, instead
        # of being lost if the meter never returns before the next anchor. When
        # it does return, the metered delta is netted against all of it.
        since_read = _span_minutes(state.water_baseline_iso, now_iso, elapsed_min)
        owed = fallback_draw_kwh(
            inputs.e_base,
            inputs.e_draw_per_person,
            inputs.empty_house_factor,
            inputs.people_home,
            standby_w,
            since_read,
        )
        fallback = max(0.0, owed - state.pending_fallback_kwh)
    return _WaterTick(
        fallback,
        "fallback",
        False,
        0.0,
        state.water_baseline_l,
        state.water_baseline_iso,
        state.water_changed_iso,
        cadence,
        state.pending_fallback_kwh + fallback,
        missing_since,
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
    same arithmetic. A no-trip cut-off (see ``CUTOFF_SETTLE_S``) is detected
    here, settled ``CUTOFF_SETTLE_S`` later, and then floors the display and
    control ledgers and may teach the learner.
    """
    capacity = capacity_kwh(params.volume_l, params.setpoint_c, params.cold_in_c)
    elapsed_min = max(0.0, inputs.elapsed_s) / 60.0
    bucket = daypart_bucket(_local_hour(inputs))
    hot_fraction = profile_of(state)[bucket]
    # A temperature dropout holds the last known value rather than snapping the
    # cold term to zero (that would read as a sudden warm spell) — but only for
    # AMBIENT_HOLD_H, so a removed sensor stops driving the cold term.
    if inputs.ambient_c is not None:
        ambient, ambient_iso = inputs.ambient_c, inputs.now_iso
    elif (
        state.last_ambient_c is not None
        and state.last_ambient_iso
        and _hours_between(state.last_ambient_iso, inputs.now_iso) < AMBIENT_HOLD_H
    ):
        ambient, ambient_iso = state.last_ambient_c, state.last_ambient_iso
    else:
        ambient, ambient_iso = None, ""
    standby_w_now = standby_w_at(state, ambient)

    # Energy in (cumulative counter delta; rollback-tolerant, reset-safe).
    energy_in, new_energy_baseline = counter_delta(
        state.energy_baseline_kwh, inputs.energy_counter_kwh
    )

    # Draw + water bookkeeping (meter / fallback / none).
    water = _resolve_draw(state, params, inputs, elapsed_min, hot_fraction, standby_w_now)
    draw_kwh, draw_source, tick_clean = water.draw_kwh, water.source, water.clean
    metered_hot_liters = water.hot_liters

    standby = standby_kwh(standby_w_now, elapsed_min)

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
    acc_cold = state.cycle_cold_kh + cold_hinge_k(ambient) * elapsed_min / 60.0

    # Tick-level bookkeeping applied on every branch.
    base = replace(
        state,
        energy_baseline_kwh=new_energy_baseline,
        water_baseline_l=water.baseline_l,
        water_baseline_iso=water.baseline_iso,
        water_changed_iso=water.changed_iso,
        water_cadence_min=water.cadence_min,
        water_missing_since_iso=water.missing_since_iso,
        pending_fallback_kwh=water.pending_kwh,
        last_tick_iso=inputs.now_iso,
        led_kwh_since_counter=led_since,
        last_ambient_c=ambient,
        last_ambient_iso=ambient_iso,
        heating_prev=_next_heating_prev(state.heating_prev, inputs),
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
        # A counter that is unavailable *at* the anchor has an unknown delta since
        # its last read, and that delta straddles the anchor: part of it belongs to
        # the cycle being closed (so this residual is incomplete → no learning)
        # and, were its baseline kept, all of it would be credited again to the
        # new cycle when the counter returns (a bogus residual next time). So drop
        # that baseline — the returning reading is adopted with no delta — and
        # open the new cycle dirty, since whatever flows before the return is
        # lost to its balance. The pending fallback is the closed cycle's (its
        # ledger was just re-zeroed), so it can never be netted against the new one.
        # This holds *whether or not a baseline exists*: an outage spanning
        # several anchors has none by the second one, and a cycle opened then
        # would otherwise look clean while recording zero delivered energy. So
        # cycles stay dirty until an anchor sees the counter back (a cycle opened
        # with the counter present has a valid baseline from its first tick).
        # With no counter configured the reading is always None and nothing
        # learns — correct, as that balance is missing a whole side.
        energy_gap = inputs.energy_counter_kwh is None
        water_gap = inputs.water_counter_l is None
        counter_gap = energy_gap or water_gap
        cycle_hours = _hours_between(state.cycle_start_iso, inputs.now_iso)
        accumulated = replace(
            base,
            cycle_energy_in_kwh=acc_energy,
            cycle_liters=acc_liters,
            cycle_clean=acc_clean and not counter_gap,
            cycle_gross_kwh=acc_gross,
            cycle_hot_liters_by_bucket=acc_buckets,
            cycle_cold_kh=acc_cold,
        )
        learned = learn_from_cycle(accumulated, params, cycle_hours, new_unclamped)
        if energy_gap:
            learned = replace(learned, energy_baseline_kwh=None)
        if water_gap:
            learned = replace(
                learned, water_baseline_l=None, water_baseline_iso="", water_changed_iso=""
            )
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
            cycle_clean=not counter_gap,
            cycle_gross_kwh=0.0,
            cycle_relax_kwh=0.0,
            cycle_hot_liters_by_bucket=_ZERO_PROFILE,
            cycle_cold_kh=0.0,
            cycle_display_lift_kwh=0.0,
            cycle_cutoff_mark_kwh=None,
            cutoff_pending_iso="",
            led_kwh_since_counter=0.0,
            pending_fallback_kwh=0.0,
        )
        shown = 0.0
        sigma = SIGMA_MIN_KWH
        cutoff_settled = False
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
            cycle_cold_kh=acc_cold,
        )
        # Both sides of the balance must be accounted for before a cut-off may
        # settle: a missing energy reading, or a water reading we're still
        # waiting on ("none"), would leave its energy out of the ledger.
        counters_ready = inputs.energy_counter_kwh is not None and (
            inputs.water_counter_l is not None or draw_source == "fallback"
        )
        new_state, cutoff_settled = _advance_cutoff(
            new_state, state, params, inputs, capacity, counters_ready
        )
        sigma = uncertainty_kwh(new_state.residual_ratio, acc_gross)
        display_ledger = (
            _ledger(new_state) + new_relax - led_since + new_state.cycle_display_lift_kwh
        )
        shown = _clamp(display_deficit_kwh(display_ledger, sigma), 0.0, capacity)

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
        cutoff=cutoff_settled,
        standby_w_now=standby_w_now,
    )


def _ledger(state: TankState) -> float:
    """The learning ledger (``cycle_unclamped_kwh``, or the control one pre-v2)."""
    return state.deficit_kwh if state.cycle_unclamped_kwh is None else state.cycle_unclamped_kwh


def _next_heating_prev(prev: bool | None, inputs: TickInputs) -> bool | None:
    """The element state cut-off detection remembers for the next tick.

    An unknown reading keeps the last known one. An element that went idle less
    than ``CUTOFF_TRIP_MARGIN_S`` ago with the contactor still on stays
    "heating" for now: the LED can drop a few seconds before the contactor opens
    (seen 22 s early on 2026-10-24), and recording that tick as "idle" would hide
    the cut-off the next tick sees. Idle for longer is a real trip (the anchor's
    business), and the margin check in :func:`is_cutoff` still tells the two apart.
    """
    if inputs.heating_on is None:
        return prev
    if (
        inputs.heating_on is False
        and prev is True
        and inputs.contactor_on is True
        and inputs.heating_off_for_s is not None
        and inputs.heating_off_for_s < CUTOFF_TRIP_MARGIN_S
    ):
        return True
    return inputs.heating_on


def is_cutoff(heating_prev: bool | None, inputs: TickInputs) -> bool:
    """True when the contactor opened while the element was still heating.

    Needs the element last *seen* heating, both now definitely off (an unknown
    reading never counts), and an observed transition: a tick after a gap longer
    than ``CUTOFF_MAX_GAP_S`` (HA was down — and startup resets every
    ``last_changed``, so the durations can't tell a trip from a cut-off) never
    counts. If the element went idle ``CUTOFF_TRIP_MARGIN_S`` before the
    contactor opened, the thermostat tripped first — a full tank whose contactor
    dropped before the anchor's sustain thresholds were met — so that is not a
    cut-off. The durations are each entity's time in its current state.
    """
    if heating_prev is not True or inputs.heating_on is not False:
        return False
    if inputs.contactor_on is not False or inputs.elapsed_s > CUTOFF_MAX_GAP_S:
        return False
    heating_off_s, contactor_off_s = inputs.heating_off_for_s, inputs.contactor_on_for_s
    if heating_off_s is not None and contactor_off_s is not None:
        if heating_off_s - contactor_off_s >= CUTOFF_TRIP_MARGIN_S:
            return False
    return True


def _advance_cutoff(
    new_state: TankState,
    prev: TankState,
    params: TankParams,
    inputs: TickInputs,
    capacity: float,
    counters_ready: bool,
) -> tuple[TankState, bool]:
    """Detect, settle and apply a no-trip cut-off → ``(state, settled_this_tick)``.

    Detection only starts the settle clock; a run that resumes inside it (the
    scheduler's slot-boundary re-toggle) cancels it, and its real end decides.
    A settle due while a counter is missing waits for it (``counters_ready``).
    A cut-off in a short cycle (a post-trip re-heat) is dropped: its balance may
    legitimately sit past the first trip's level. On settling,
    :func:`learn_from_cutoff` may correct the params, the learning ledger takes
    the extra losses they now explain (and the display lift gives up the same
    amount — the params now explain what it stood in for), then the displayed and
    control deficits are floored at the new σ: the tank isn't full, so it can't
    read as fuller than the model's own uncertainty allows.
    """
    pending = prev.cutoff_pending_iso
    if not pending and is_cutoff(prev.heating_prev, inputs):
        pending = inputs.now_iso
    if pending and inputs.heating_on is True:
        pending = ""
    if (
        not pending
        or _hours_between(pending, inputs.now_iso) * 3600.0 < CUTOFF_SETTLE_S
        or not counters_ready
    ):
        return replace(new_state, cutoff_pending_iso=pending), False

    cycle_hours = _hours_between(new_state.cycle_start_iso, inputs.now_iso)
    if cycle_hours < LEARN_MIN_CYCLE_HOURS:
        return replace(new_state, cutoff_pending_iso=""), False
    learned, extra = learn_from_cutoff(new_state, params, cycle_hours, _ledger(new_state))
    ledger = _clamp(_ledger(new_state) + extra, -capacity, capacity)
    lift = max(0.0, new_state.cycle_display_lift_kwh - extra)
    sigma = uncertainty_kwh(learned.residual_ratio, learned.cycle_gross_kwh)
    led_since = new_state.led_kwh_since_counter
    lift += max(0.0, sigma - (ledger + new_state.cycle_relax_kwh - led_since + lift))
    settled = replace(
        learned,
        cycle_unclamped_kwh=ledger,
        cycle_display_lift_kwh=lift,
        deficit_kwh=_clamp(max(new_state.deficit_kwh, sigma), 0.0, capacity),
        cutoff_pending_iso="",
        last_cutoff_iso=inputs.now_iso,
    )
    return settled, True


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
    once SoC recovers ``rearm_margin_pct`` above the threshold (capped at
    ``BOOST_REARM_MAX_PCT``, so a high threshold still re-arms at a full tank);
    fire only when
    below the threshold, armed, and at least ``min_interval_h`` since the last
    boost. The re-arm is applied even while uncalibrated so a tank that calibrates
    above the threshold is already armed for its first dip.
    """
    soc_pct = soc_value * 100.0
    new_state = state
    rearm_pct = min(threshold_pct + rearm_margin_pct, BOOST_REARM_MAX_PCT)
    if soc_pct >= rearm_pct and not state.boost_armed:
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
