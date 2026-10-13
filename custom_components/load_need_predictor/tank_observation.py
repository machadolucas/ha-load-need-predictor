"""Pure, diagnostic-only shower attribution alongside the existing tank model.

The control model is never passed here or changed. Completed, well-observed
events replace their baseline draw in a parallel energy ledger; everything else
retains the baseline. Bathroom humidity and ventilation form ONE corroborating
evidence family, since a fan commonly follows humidity automatically.
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
            key: result[key] for key in ("version", "mode", "events", "cycles") if key in result
        }
    return result


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
    ):
        return result
    humidity = _number(inputs.humidity)
    delta = water - previous_water if water is not None and previous_water is not None else None
    disruption = None
    if previous_iso and result.get("source_signature") != inputs.source_signature:
        disruption = "sources_changed"
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
        if inputs.appliance_active is True:
            _invalidate(event, "appliance_overlap")
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
