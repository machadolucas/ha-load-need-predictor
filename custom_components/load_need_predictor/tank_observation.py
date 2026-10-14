"""Pure, diagnostic-only shower and appliance observations beside the tank model.

The control model is never passed here or changed. Completed, well-observed
events replace their baseline draw in a parallel energy ledger; everything else
retains the baseline. Bathroom humidity and ventilation form ONE corroborating
evidence family, since a fan commonly follows humidity automatically.
Appliance running signals collect gross household-flow profiles. Their separate
exploratory cold upper bound never changes the accepted shower candidate.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

KWH_PER_LITER_KELVIN = 4.186 / 3600.0
RETENTION_DAYS = 90
MAX_EVENTS = 2000
MAX_CYCLES = 500
MAX_APPLIANCE_RUNS = 1500
MAX_APPLIANCE_RUN_S = 24 * 3600.0
MAX_GAP_S = 180.0
QUIET_BOUNDARY_S = 120.0
MAX_EVENT_S = 3600.0
MAX_EVENT_FLOW_LPM = 15.0
MIN_EVENT_LITERS = 8.0
MIN_EVENT_DURATION_S = 120.0
HUMIDITY_RISE_PCT = 3.0
FEEDBACK_REPORTS = frozenset(("comfortable", "cooler_than_usual", "not_warm_enough"))


@dataclass(frozen=True)
class ObservationInputs:
    """One observed tick and the baseline model's *actual* accounting terms.

    ``previous_cycle_unclamped_kwh`` and ``previous_cycle_clean`` are captured
    BEFORE apply_tick, all other tank values AFTER. The old residual must be
    reconstructed before the anchor's reset can hide it. Signatures must cover
    configured sources and attribution settings, not mutable source states.
    """

    now_iso: str
    source_signature: str
    water_l: float | None
    motion: bool | None
    humidity: float | None
    ventilation: bool | None
    appliance_active: bool | None
    supply_temp_c: float | None = None
    target_temp_c: float = 40.0
    cold_in_c: float = 12.0
    baseline_draw_kwh: float = 0.0
    energy_in_kwh: float = 0.0
    standby_kwh: float = 0.0
    previous_cycle_unclamped_kwh: float = 0.0
    previous_cycle_clean: bool = True
    anchored: bool = False
    tank_capacity_kwh: float = 0.0
    tank_deficit_kwh: float = 0.0
    shown_deficit_kwh: float = 0.0
    uncertainty_kwh: float = 0.0
    draw_source: str = "meter"
    water_source_quality: str = "valid"
    energy_available: bool = True
    appliance_states: dict[str, bool | None] | None = None
    household_activity: bool | None = None
    appliance_boundaries_known: bool = True
    observation_restarted: bool = False


def _time(value: str) -> datetime:
    instant = datetime.fromisoformat(value)
    if instant.tzinfo is None:
        raise ValueError("Observation timestamps must include a timezone")
    return instant


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _invalidate(event: dict[str, Any], reason: str) -> None:
    event["complete"] = False
    reasons = event.setdefault("quality_reasons", [])
    if reason not in reasons:
        reasons.append(reason)


def _plausible_shower(event: dict[str, Any]) -> bool:
    """Keep event-level detail for plausible showers, not every toilet fill."""
    end = event.get("end", event["last_flow_iso"])
    return (
        event["motion_seen"]
        and event["water_l"] >= MIN_EVENT_LITERS
        and (_time(end) - _time(event["start"])).total_seconds() >= MIN_EVENT_DURATION_S
    )


def _finish(saved: dict[str, Any], end: str, reason: str | None = None) -> None:
    event = saved.pop("active_event", None)
    if not event:
        return
    if reason:
        _invalidate(event, reason)
    event["end"] = event["last_flow_iso"]
    duration = (_time(event["end"]) - _time(event["start"])).total_seconds()
    if duration < MIN_EVENT_DURATION_S or event["water_l"] < MIN_EVENT_LITERS:
        _invalidate(event, "short_or_small")
    if not event["motion_seen"]:
        _invalidate(event, "no_motion")
    if not event["corroborated"]:
        _invalidate(event, "no_environment_corroboration")
    confident = event["complete"] and not event["quality_reasons"]
    event["confidence"] = "confident" if confident else "ambiguous"
    event["kind"] = "likely_shower" if confident else "unclassified_draw"
    # The meter measures mixed shower water, not 75°C tank-outlet litres.
    event["mixed_demand_kwh"] = (
        event["water_l"]
        * KWH_PER_LITER_KELVIN
        * max(0.0, event["target_temp_c"] - event["cold_in_c"])
    )
    event["replacement_kwh"] = (
        event["mixed_demand_kwh"] if confident else event["baseline_draw_kwh"]
    )
    event["correction_kwh"] = event["replacement_kwh"] - event["baseline_draw_kwh"]
    event["completed_iso"] = end
    event["feedback"] = None
    if _plausible_shower(event):
        saved.setdefault("events", []).append(event)
    if confident:
        saved["last_likely_hot_draw_iso"] = event["end"]
        cycle = saved.get("cycle")
        if cycle:
            cycle["correction_kwh"] += event["correction_kwh"]
            cycle["confident_events"] += 1


def _prune(saved: dict[str, Any], now: datetime) -> None:
    cutoff = now - timedelta(days=RETENTION_DAYS)
    for key, limit in (("events", MAX_EVENTS), ("cycles", MAX_CYCLES)):
        saved[key] = [row for row in saved.get(key, []) if _time(row["end"]) >= cutoff][-limit:]
    last_hot = saved.get("last_likely_hot_draw_iso")
    if last_hot and _time(last_hot) < cutoff:
        saved.pop("last_likely_hot_draw_iso", None)
    appliances = saved.get("appliance_observations")
    if appliances:
        appliances["runs"] = [
            row for row in appliances.get("runs", []) if _time(row["end"]) >= cutoff
        ][-MAX_APPLIANCE_RUNS:]
        appliances["daily_bounds"] = [
            row for row in appliances.get("daily_bounds", []) if _time(row["end"]) >= cutoff
        ][-RETENTION_DAYS:]
        appliances["active_runs"] = {
            source: run
            for source, run in appliances.get("active_runs", {}).items()
            if _time(run["start"]) >= cutoff
        }


def prune_observation(saved: dict[str, Any], now_iso: str) -> dict[str, Any]:
    """Enforce wall-clock retention even when tracking has been disabled."""
    result = copy.deepcopy(saved)
    now = _time(now_iso)
    _prune(result, now)
    cutoff = now - timedelta(days=RETENTION_DAYS)
    for key in ("active_event", "cycle"):
        if row := result.get(key):
            if _time(row["start"]) < cutoff:
                result.pop(key)
    if result.get("last_iso") and _time(result["last_iso"]) < cutoff:
        # Expired sensor snapshots must not be revived as a new event baseline.
        result = {
            key: result[key]
            for key in ("version", "mode", "events", "cycles", "appliance_observations")
            if key in result
        }
        if appliances := result.get("appliance_observations"):
            appliances["last_states"] = {}
            appliances["active_runs"] = {}
    return result


def _appliance_states(inputs: ObservationInputs) -> dict[str, bool | None]:
    return {
        str(source): value if isinstance(value, bool) else None
        for source, value in (inputs.appliance_states or {}).items()
    }


def _run_flag(run: dict[str, Any], reason: str, *, incomplete: bool = False) -> None:
    if reason not in run["quality_flags"]:
        run["quality_flags"].append(reason)
    if incomplete:
        run["complete"] = False


def _close_appliance_run(
    store: dict[str, Any], source: str, now_iso: str, reason: str | None = None
) -> None:
    run = store["active_runs"].pop(source)
    if reason:
        _run_flag(run, reason, incomplete=True)
    if run["gross_household_l"] <= 0:
        _run_flag(run, "no_metered_water")
    run["interior_profile_usable"] = run["observed_interior_household_l"] > 0
    run["end"] = now_iso
    run["attribution"] = "gross_household_water_during_run_not_appliance_fill"
    store["runs"].append(run)
    # Completion must trigger persistence even after the retained-list cap.
    store["sequence"] += 1


def _advance_appliances(
    saved: dict[str, Any],
    inputs: ObservationInputs,
    *,
    elapsed: float,
    delta: float | None,
    disruption: str | None,
) -> tuple[float, float]:
    """Collect run interiors and a *possible*, never attributed, cold upper bound.

    A washer/dishwasher running signal is not a filling signal. Even otherwise
    quiet, singly-active windows establish only an upper bound, not ownership.
    Gross run litres may overlap across sources and must never be summed as
    device consumption. Only the separate daily/cycle bound counts each meter
    interval once, and only against that interval's existing baseline draw.
    """
    states = _appliance_states(inputs)
    existing = saved.get("appliance_observations")
    if not states and not existing:
        return 0.0, 0.0
    store = saved.get("appliance_observations")
    if not isinstance(store, dict):
        store = {}
        saved["appliance_observations"] = store
    for key, default in (
        ("active_runs", {}),
        ("runs", []),
        ("last_states", {}),
        ("sequence", 0),
        ("daily_bounds", []),
    ):
        store.setdefault(key, default)
    previous = store["last_states"]
    previous_iso = saved.get("last_iso", inputs.now_iso)
    now = _time(inputs.now_iso)
    valid_interval = disruption is None and 0 < elapsed <= MAX_GAP_S
    if inputs.observation_restarted:
        valid_interval = False
    positive_delta = max(0.0, delta or 0.0) if valid_interval else 0.0
    active_sources = {source for source, value in states.items() if value is True}
    previous_active = {source for source, value in previous.items() if value is True}
    unknown_source = any(value is None for value in states.values())
    changed_sources = {source for source in states if previous.get(source) != states[source]}
    shower_overlap = inputs.motion is not False or bool(
        saved.get("active_event", {}).get("motion_seen")
    )
    rejection = []
    if not valid_interval:
        rejection.append(disruption or "interval_unobserved")
    if not inputs.appliance_boundaries_known or changed_sources:
        rejection.append("run_boundary_uncertain")
    if len(active_sources) != 1 or active_sources != previous_active:
        rejection.append("not_one_stable_running_source")
    if unknown_source:
        rejection.append("appliance_state_unknown")
    if inputs.household_activity is not False:
        rejection.append("household_active_or_unknown")
    if shower_overlap:
        rejection.append("shower_active_or_unknown")
    if inputs.draw_source != "meter":
        rejection.append("baseline_draw_unobserved")
    bound_l = positive_delta if not rejection else 0.0
    bound_kwh = max(0.0, inputs.baseline_draw_kwh) if bound_l > 0 else 0.0

    for source in list(store["active_runs"]):
        run = store["active_runs"][source]
        if source not in states:
            _close_appliance_run(store, source, inputs.now_iso, "source_removed")
        elif disruption in ("sources_changed", "observation_gap") or inputs.observation_restarted:
            _close_appliance_run(store, source, inputs.now_iso, "restart_or_gap")
        elif (now - _time(run["start"])).total_seconds() > MAX_APPLIANCE_RUN_S:
            _close_appliance_run(store, source, inputs.now_iso, "run_too_long")

    for source in sorted(active_sources):
        if source in store["active_runs"]:
            continue
        observed_start = (
            valid_interval and previous.get(source) is False and inputs.appliance_boundaries_known
        )
        store["sequence"] += 1
        store["active_runs"][source] = {
            "id": f"{inputs.now_iso}/appliance/{store['sequence']}",
            "source": source,
            "source_signature": inputs.source_signature,
            "start": previous_iso if observed_start else inputs.now_iso,
            "complete": observed_start,
            "quality_flags": [] if observed_start else ["start_unobserved"],
            "gross_household_l": 0.0,
            "observed_interior_household_l": 0.0,
            "observed_interior_seconds": 0.0,
            "household_flow_episodes": 0,
            "last_flow_iso": None,
            "episode_l": 0.0,
            "max_household_flow_episode_l": 0.0,
            "exploratory_cold_upper_liters": 0.0,
            "exploratory_max_draw_reduction_kwh": 0.0,
        }

    for source, run in list(store["active_runs"].items()):
        state = states.get(source)
        previous_state = previous.get(source)
        if not valid_interval:
            _run_flag(run, disruption or "restart_or_gap", incomplete=True)
        if not inputs.appliance_boundaries_known:
            _run_flag(run, "boundary_unobserved", incomplete=True)
        if state is None or unknown_source:
            _run_flag(run, "appliance_state_unknown", incomplete=True)
        if len(active_sources | previous_active) > 1:
            _run_flag(run, "appliance_overlap")
        if inputs.household_activity is True:
            _run_flag(run, "household_activity_overlap")
        elif inputs.household_activity is None:
            _run_flag(run, "household_activity_unknown")
        if shower_overlap:
            _run_flag(run, "shower_active_or_unknown")
        if source in changed_sources:
            _run_flag(run, "state_transition_interval")
        # Runs retain gross contemporaneous household flow, including their
        # uncertain edge intervals, but never turn it into appliance litres.
        if state is True or previous_state is True or state is None:
            run["gross_household_l"] += positive_delta
            if (
                valid_interval
                and state is True
                and previous_state is True
                and inputs.appliance_boundaries_known
            ):
                run["observed_interior_household_l"] += positive_delta
                run["observed_interior_seconds"] += elapsed
            if positive_delta > 0:
                last_flow = run["last_flow_iso"]
                if last_flow is None or (now - _time(last_flow)).total_seconds() > QUIET_BOUNDARY_S:
                    run["household_flow_episodes"] += 1
                    run["episode_l"] = 0.0
                run["episode_l"] += positive_delta
                run["max_household_flow_episode_l"] = max(
                    run["max_household_flow_episode_l"], run["episode_l"]
                )
                run["last_flow_iso"] = inputs.now_iso
        if state is True and bound_l > 0:
            run["exploratory_cold_upper_liters"] += bound_l
            run["exploratory_max_draw_reduction_kwh"] += bound_kwh
        if state is False:
            _close_appliance_run(store, source, inputs.now_iso)

    day = now.date().isoformat()
    daily_rows = store["daily_bounds"]
    if not daily_rows or daily_rows[-1]["day"] != day:
        daily_rows.append(
            {
                "day": day,
                "start": inputs.now_iso,
                "end": inputs.now_iso,
                "exploratory_cold_upper_liters": 0.0,
                "exploratory_max_draw_reduction_kwh": 0.0,
                "observed_intervals": 0,
                "rejection_counts": {},
            }
        )
    daily = daily_rows[-1]
    daily["end"] = inputs.now_iso
    daily["observed_intervals"] += int(valid_interval)
    daily["exploratory_cold_upper_liters"] += bound_l
    daily["exploratory_max_draw_reduction_kwh"] += bound_kwh
    if positive_delta > 0:
        for reason in rejection:
            counts = daily["rejection_counts"]
            counts[reason] = counts.get(reason, 0) + 1
    store["last_states"] = states
    return bound_l, bound_kwh


def advance_observation(saved: dict[str, Any] | None, inputs: ObservationInputs) -> dict[str, Any]:
    """Return a separate JSON-compatible shadow state; never mutate the input.

    Long gaps, counter resets and configuration changes restart observations.
    Their intervals keep baseline draw and invalidate the current anchor cycle.
    A fresh anchor starts a synchronized cycle, including after HA restarts.
    """
    result = copy.deepcopy(saved or {})
    now = _time(inputs.now_iso)
    previous_iso = result.get("last_iso")
    elapsed = (now - _time(previous_iso)).total_seconds() if previous_iso else 0.0
    previous_water = _number(result.get("water_l"))
    water = _number(inputs.water_l)
    # A timer and a manual refresh can sample the same instant. They must not
    # manufacture a gap, double-count energy, or destroy an observed boundary.
    if (
        previous_iso
        and elapsed == 0
        and result.get("source_signature") == inputs.source_signature
        and previous_water == water
        and not inputs.observation_restarted
        and (
            inputs.appliance_states is None
            or result.get("appliance_observations", {}).get("last_states", {})
            == _appliance_states(inputs)
        )
    ):
        return result
    humidity = _number(inputs.humidity)
    delta = water - previous_water if water is not None and previous_water is not None else None
    disruption = None
    if previous_iso and result.get("source_signature") != inputs.source_signature:
        disruption = "sources_changed"
    elif previous_iso and inputs.observation_restarted:
        disruption = "observation_gap"
    elif previous_iso and (elapsed <= 0 or elapsed > MAX_GAP_S):
        disruption = "observation_gap"
    elif delta is not None and delta < 0:
        disruption = "meter_reset"
    elif water is None or (previous_iso and previous_water is None):
        disruption = "water_unavailable"
    elif inputs.water_source_quality != "valid":
        disruption = "water_quality"
    elif delta is not None and elapsed > 0:
        if delta / (elapsed / 60.0) > MAX_EVENT_FLOW_LPM:
            disruption = "batched_or_overlapping_flow"

    cycle = result.get("cycle")
    residual_before_reset = (
        inputs.previous_cycle_unclamped_kwh
        + inputs.baseline_draw_kwh
        + inputs.standby_kwh
        - inputs.energy_in_kwh
    )
    saturated = inputs.tank_capacity_kwh > 0 and (
        abs(inputs.previous_cycle_unclamped_kwh) >= inputs.tank_capacity_kwh
        or abs(residual_before_reset) >= inputs.tank_capacity_kwh
    )
    if disruption:
        _finish(result, inputs.now_iso, disruption)
        result["quiet_since_iso"] = inputs.now_iso
        if cycle:
            _invalidate(cycle, disruption)
        delta = None
    cold_bound_l, cold_bound_kwh = _advance_appliances(
        result, inputs, elapsed=elapsed, delta=delta, disruption=disruption
    )
    clean_tick = (
        inputs.previous_cycle_clean
        and inputs.draw_source == "meter"
        and inputs.energy_available
        and disruption is None
    )
    if cycle:
        cycle["clean"] = cycle["clean"] and clean_tick
        if saturated:
            _invalidate(cycle, "baseline_saturated")
        reasons = cycle.setdefault("quality_reasons", [])
        for condition, reason in (
            (not inputs.previous_cycle_clean, "baseline_cycle_dirty"),
            (inputs.draw_source != "meter", "baseline_draw_unobserved"),
            (not inputs.energy_available, "energy_unavailable"),
        ):
            if condition and reason not in reasons:
                reasons.append(reason)
        for field, value in (
            ("baseline_draw_kwh", inputs.baseline_draw_kwh),
            ("energy_in_kwh", inputs.energy_in_kwh),
            ("standby_kwh", inputs.standby_kwh),
            ("water_l", max(0.0, delta or 0.0)),
        ):
            cycle[field] = cycle.get(field, 0.0) + value
        cycle["exploratory_cold_upper_liters"] = (
            cycle.get("exploratory_cold_upper_liters", 0.0) + cold_bound_l
        )
        cycle["exploratory_max_draw_reduction_kwh"] = (
            cycle.get("exploratory_max_draw_reduction_kwh", 0.0) + cold_bound_kwh
        )

    if not previous_iso:
        result["quiet_since_iso"] = inputs.now_iso
    flowing = delta is not None and delta > 0 and disruption is None
    event = result.get("active_event")
    if flowing and event is None:
        # A dry lead-in establishes that we saw the event's start, rather than
        # joining an already-running shower after an observation restart.
        quiet_iso = result.get("quiet_since_iso", inputs.now_iso)
        observed_boundary = (
            _time(previous_iso or inputs.now_iso) - _time(quiet_iso)
        ).total_seconds() >= QUIET_BOUNDARY_S
        result["event_sequence"] = int(result.get("event_sequence", 0)) + 1
        event = {
            "id": f"{inputs.now_iso}/{result['event_sequence']}",
            "start": previous_iso or inputs.now_iso,
            "last_flow_iso": inputs.now_iso,
            "water_l": 0.0,
            "baseline_draw_kwh": 0.0,
            "target_temp_c": inputs.target_temp_c,
            "cold_in_c": inputs.cold_in_c,
            "humidity_start": _number(result.get("humidity")),
            "motion_seen": False,
            "corroborated": False,
            "complete": observed_boundary,
            "quality_reasons": [] if observed_boundary else ["start_unobserved"],
            "supply_temp_c": _number(inputs.supply_temp_c),
            "pipe_idle_before_minutes": (
                max(
                    0.0,
                    (
                        _time(previous_iso or inputs.now_iso)
                        - _time(result["last_likely_hot_draw_iso"])
                    ).total_seconds()
                    / 60.0,
                )
                if result.get("last_likely_hot_draw_iso")
                else None
            ),
            "start_soc_pct": (
                100.0 * (1.0 - inputs.tank_deficit_kwh / inputs.tank_capacity_kwh)
                if inputs.tank_capacity_kwh > 0
                else None
            ),
        }
        result["active_event"] = event
    if event:
        if flowing:
            event["water_l"] += delta
            event["baseline_draw_kwh"] += inputs.baseline_draw_kwh
            event["last_flow_iso"] = inputs.now_iso
        event["motion_seen"] = event["motion_seen"] or inputs.motion is True
        humidity_start = event["humidity_start"]
        humidity_rise = (
            humidity is not None
            and humidity_start is not None
            and humidity - humidity_start >= HUMIDITY_RISE_PCT
        )
        ventilation_started = inputs.ventilation is True and result.get("ventilation") is False
        event["corroborated"] = event["corroborated"] or humidity_rise or ventilation_started
        if inputs.motion is None or inputs.appliance_active is None:
            _invalidate(event, "source_unknown")
        if inputs.appliance_active is True or any(
            value is True for value in _appliance_states(inputs).values()
        ):
            _invalidate(event, "appliance_overlap")
        if any(value is None for value in _appliance_states(inputs).values()):
            _invalidate(event, "appliance_source_unknown")
        if inputs.appliance_states and not inputs.appliance_boundaries_known:
            _invalidate(event, "appliance_boundary_unobserved")
        if inputs.draw_source != "meter":
            _invalidate(event, "fallback_draw")
        event_age = (now - _time(event["start"])).total_seconds()
        quiet_age = (now - _time(event["last_flow_iso"])).total_seconds()
        if event_age > MAX_EVENT_S:
            _finish(result, inputs.now_iso, "event_too_long")
        elif quiet_age >= QUIET_BOUNDARY_S:
            _finish(result, inputs.now_iso)
    if not flowing:
        if result.get("was_flowing"):
            result["quiet_since_iso"] = previous_iso or inputs.now_iso
    else:
        result["quiet_since_iso"] = inputs.now_iso

    if inputs.anchored:
        # A mixed-water event split by an anchor cannot retrospectively put all
        # its correction into either side. Discard it and both partial sides.
        anchor_mid_event = bool(result.get("active_event"))
        if anchor_mid_event:
            _finish(result, inputs.now_iso, "anchor_mid_event")
            if cycle:
                _invalidate(cycle, "anchor_mid_event")
        if cycle:
            baseline_residual = residual_before_reset
            if inputs.tank_capacity_kwh > 0:
                baseline_residual = max(
                    -inputs.tank_capacity_kwh, min(inputs.tank_capacity_kwh, baseline_residual)
                )
            result.setdefault("cycles", []).append(
                {
                    **cycle,
                    "end": inputs.now_iso,
                    "complete": cycle["complete"] and not anchor_mid_event,
                    "baseline_residual_kwh": baseline_residual,
                    "shadow_residual_kwh": baseline_residual + cycle["correction_kwh"],
                    "exploratory_min_residual_kwh": (
                        baseline_residual
                        + cycle["correction_kwh"]
                        - cycle.get("exploratory_max_draw_reduction_kwh", 0.0)
                    ),
                    "anchor_raw_deficit_kwh": inputs.tank_deficit_kwh,
                    "anchor_shown_deficit_kwh": inputs.shown_deficit_kwh,
                    "anchor_uncertainty_kwh": inputs.uncertainty_kwh,
                }
            )
        result["cycle"] = {
            "start": inputs.now_iso,
            "correction_kwh": 0.0,
            "confident_events": 0,
            "complete": not anchor_mid_event,
            "clean": True,
            "synchronized": True,
            "source_signature": inputs.source_signature,
            "baseline_draw_kwh": 0.0,
            "energy_in_kwh": 0.0,
            "standby_kwh": 0.0,
            "water_l": 0.0,
            "quality_reasons": ["anchor_mid_event"] if anchor_mid_event else [],
            "exploratory_cold_upper_liters": 0.0,
            "exploratory_max_draw_reduction_kwh": 0.0,
        }

    result.update(
        {
            "version": 1,
            "mode": "shadow",
            "last_iso": inputs.now_iso,
            "source_signature": inputs.source_signature,
            "water_l": water,
            "humidity": humidity,
            "ventilation": inputs.ventilation,
            "was_flowing": flowing,
            "latest": {
                "supply_temp_c": _number(inputs.supply_temp_c),
                "supply_temperature_role": "log_only",
                "baseline_deficit_kwh": inputs.tank_deficit_kwh,
                "shown_deficit_kwh": inputs.shown_deficit_kwh,
                "uncertainty_kwh": inputs.uncertainty_kwh,
                "shadow_correction_kwh": result.get("cycle", {}).get("correction_kwh", 0.0),
                "accepted_appliance_cold_attribution_kwh": 0.0,
            },
        }
    )
    _prune(result, now)
    return result


def record_feedback(
    saved: dict[str, Any] | None, *, event_id: str, report: str, now_iso: str
) -> dict[str, Any]:
    """Attach an explicit user's report to exactly one retained completed event."""
    if report not in FEEDBACK_REPORTS:
        raise ValueError("Unknown shower feedback report")
    now = _time(now_iso)
    result = copy.deepcopy(saved or {})
    _prune(result, now)
    for event in result.get("events", []):
        if event["id"] == event_id and _plausible_shower(event):
            event["feedback"] = report
            event["feedback_iso"] = now_iso
            return result
    raise ValueError("Shower event ID is not retained or not completed")


def summarize_observation(saved: dict[str, Any]) -> dict[str, Any]:
    """Small sensor/diagnostic view; absent comfort reports never imply safety."""
    events = saved.get("events", [])
    reports = [event["feedback"] for event in events if event.get("feedback")]
    risk = "unknown"
    if len(reports) >= 5:
        risk = (
            "reported_shortfalls"
            if "not_warm_enough" in reports
            else "reported_cooler"
            if "cooler_than_usual" in reports
            else "no_shortfall_reported"
        )
    last_hot = saved.get("last_likely_hot_draw_iso")
    last_tick = saved.get("last_iso")
    return {
        "mode": "shadow",
        "automatic_promotion": False,
        "event_count": len(events),
        "confident_event_count": sum(e["confidence"] == "confident" for e in events),
        "comfort_report_count": len(reports),
        "comfort_risk": risk,
        "comfort_risk_basis": "self_reports_only_not_temperature_validation",
        "feedback_counts": {report: reports.count(report) for report in sorted(FEEDBACK_REPORTS)},
        "pipe_idle_feedback": _pipe_feedback(events),
        "pipe_idle_minutes": (
            max(0.0, (_time(last_tick) - _time(last_hot)).total_seconds() / 60.0)
            if last_hot and last_tick
            else None
        ),
        "pipe_idle_basis": "since_last_likely_hot_draw",
        "pipe_idle_role": "diagnostic_only",
        "recent_showers": [
            {
                key: event.get(key)
                for key in ("id", "start", "end", "water_l", "confidence", "feedback")
            }
            for event in reversed(events[-10:])
            if _plausible_shower(event)
        ],
        "latest": copy.deepcopy(saved.get("latest", {})),
        "evaluation": evaluate_observations(saved),
        "appliance_profiles": _appliance_summary(saved),
    }


def _appliance_summary(saved: dict[str, Any]) -> dict[str, Any]:
    store = saved.get("appliance_observations", {})
    runs = store.get("runs", [])
    active = store.get("active_runs", {})
    configured = bool(store.get("last_states"))
    profiles: dict[str, dict[str, Any]] = {}
    for run in runs:
        source = run["source"]
        profile = profiles.setdefault(
            source,
            {
                "run_count": 0,
                "complete_run_count": 0,
                "runs_with_observed_interior": 0,
                "runs_with_metered_water": 0,
                "runs_with_usable_interior": 0,
                "gross_household_l": 0.0,
                "observed_interior_household_l": 0.0,
                "observed_interior_minutes": 0.0,
                "household_flow_episodes": 0,
                "quality_flag_counts": {},
            },
        )
        profile["run_count"] += 1
        profile["complete_run_count"] += int(run["complete"])
        profile["runs_with_observed_interior"] += int(run["observed_interior_seconds"] > 0)
        profile["runs_with_metered_water"] += int(run["gross_household_l"] > 0)
        profile["runs_with_usable_interior"] += int(run["interior_profile_usable"])
        profile["gross_household_l"] += run["gross_household_l"]
        profile["observed_interior_household_l"] += run["observed_interior_household_l"]
        profile["observed_interior_minutes"] += run["observed_interior_seconds"] / 60.0
        profile["household_flow_episodes"] += run["household_flow_episodes"]
        for flag in run["quality_flags"]:
            counts = profile["quality_flag_counts"]
            counts[flag] = counts.get(flag, 0) + 1
    for profile in profiles.values():
        metered_runs = profile["runs_with_metered_water"]
        profile["mean_gross_household_l_per_metered_run"] = (
            profile["gross_household_l"] / metered_runs if metered_runs else None
        )
        profile["cold_volume_consistency_status"] = "not_established"
    daily = store.get("daily_bounds", [])
    return {
        "status": "observational_only" if configured or runs or active else "not_configured",
        "configured": configured,
        "cold_attribution_status": "unvalidated",
        "bound_basis": "possible_not_attributed",
        "gross_liters_overlap_between_sources": True,
        "automatic_promotion": False,
        "accepted_cold_attribution_kwh": 0.0,
        "per_source": profiles,
        "recent_runs": [copy.deepcopy(run) for run in reversed(runs[-10:])],
        "active_run_count": len(active),
        "exploratory_cold_upper_liters": sum(row["exploratory_cold_upper_liters"] for row in daily),
        "exploratory_max_draw_reduction_kwh": sum(
            row["exploratory_max_draw_reduction_kwh"] for row in daily
        ),
    }


def _pipe_feedback(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fixed descriptive gap groups; no fitted comfort or pipe-temperature model."""
    groups: dict[str, dict[str, Any]] = {}
    for event in events:
        if not event.get("feedback"):
            continue
        gap = event.get("pipe_idle_before_minutes")
        bucket = (
            "unknown"
            if gap is None
            else "under_2h"
            if gap < 120
            else "2_to_6h"
            if gap < 360
            else "at_least_6h"
        )
        row = groups.setdefault(
            bucket,
            {
                "gap_bucket": bucket,
                "report_count": 0,
                **{report: 0 for report in sorted(FEEDBACK_REPORTS)},
            },
        )
        row["report_count"] += 1
        row[event["feedback"]] += 1
    return list(groups.values())


def _metrics(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = [row[field] for row in rows]
    return {
        "rmse_kwh": math.sqrt(sum(value * value for value in values) / len(values)),
        "bias_kwh": sum(values) / len(values),
        "mae_kwh": sum(abs(value) for value in values) / len(values),
    }


def evaluate_observations(saved: dict[str, Any]) -> dict[str, Any]:
    """Chronological holdout comparison, never a command to promote control.

    The formula has no fitted parameters. The early half is an observation
    development window and the later half is held out. A changed source setup
    starts its own evidence window instead of borrowing acceptance from old data.
    """
    signature = saved.get("source_signature")
    rows = sorted(
        (
            row
            for row in saved.get("cycles", [])
            if row.get("complete")
            and row.get("clean")
            and row.get("synchronized")
            and row.get("source_signature") == signature
            and _number(row.get("baseline_residual_kwh")) is not None
            and _number(row.get("shadow_residual_kwh")) is not None
        ),
        key=lambda row: _time(row["end"]),
    )
    report: dict[str, Any] = {
        "status": "insufficient_evidence",
        "eligible_cycles": len(rows),
        "automatic_promotion": False,
        "comfort_risk": "not_inferred_from_anchor_residuals",
        "appliance_exploration": evaluate_appliance_exploration(saved),
    }
    if len(rows) < 12:
        return report
    heldout = rows[len(rows) // 2 :]
    report["heldout_cycles"] = len(heldout)
    report["heldout_start"] = heldout[0]["start"]
    report["heldout_confident_events"] = sum(r.get("confident_events", 0) for r in heldout)
    if report["heldout_confident_events"] < 3:
        return report
    baseline = _metrics(heldout, "baseline_residual_kwh")
    candidate = _metrics(heldout, "shadow_residual_kwh")
    report.update(
        {
            "baseline": baseline,
            "candidate": candidate,
            "status": (
                "passes_shadow_comparison"
                if candidate["rmse_kwh"] < baseline["rmse_kwh"]
                and abs(candidate["bias_kwh"]) <= abs(baseline["bias_kwh"]) + 1e-12
                else "fails_shadow_comparison"
            ),
        }
    )
    return report


def evaluate_appliance_exploration(saved: dict[str, Any]) -> dict[str, Any]:
    """Show an exploratory residual envelope; even improved error proves no fill.

    The upper-bound assumption removes at most the eligible interval's existing
    draw. It is intentionally excluded from ``passes_shadow_comparison`` and
    cannot validate any per-program appliance volume or cold-water ownership.
    """
    profiles = _appliance_summary(saved)
    report: dict[str, Any] = {
        "status": "unvalidated_upper_bound"
        if profiles["status"] == "observational_only"
        else "not_configured",
        "automatic_promotion": False,
        "accepted_cold_attribution_kwh": 0.0,
        "bound_basis": "possible_not_attributed",
        "exploratory_cold_upper_liters": profiles["exploratory_cold_upper_liters"],
        "exploratory_max_draw_reduction_kwh": profiles["exploratory_max_draw_reduction_kwh"],
    }
    rows = sorted(
        (
            row
            for row in saved.get("cycles", [])
            if row.get("complete")
            and row.get("clean")
            and row.get("synchronized")
            and row.get("source_signature") == saved.get("source_signature")
            and _number(row.get("baseline_residual_kwh")) is not None
            and _number(row.get("shadow_residual_kwh")) is not None
        ),
        key=lambda row: _time(row["end"]),
    )
    report["eligible_cycles"] = len(rows)
    if len(rows) >= 12:
        heldout = rows[len(rows) // 2 :]
        exploratory_rows = [
            {
                "residual": row["shadow_residual_kwh"]
                - max(0.0, row.get("exploratory_max_draw_reduction_kwh", 0.0))
            }
            for row in heldout
        ]
        report["heldout_cycles"] = len(heldout)
        report["baseline"] = _metrics(heldout, "baseline_residual_kwh")
        report["hypothetical_min_residual"] = _metrics(exploratory_rows, "residual")
        report["interpretation"] = "hypothetical_bound_not_appliance_attribution_acceptance"
    return report
