"""Occupancy sampling — duration-aware over the trailing/upcoming day.

Occupancy is the only feature with real day-ahead leverage, and it has no
long-term statistics — so we read it from short-term history here and the
predictor self-logs it.

Rather than a point-in-time snapshot (which would miss "out at a meeting at
14:00 but home to shower in the evening"), we measure *duration*:

- **Residents:** count the people who were ``home`` for at least
  ``RESIDENT_MIN_HOME_HOURS`` over the trailing ``RESIDENT_WINDOW_HOURS`` — i.e.
  the people actually living here that day, regardless of a daytime absence.
- **Guests:** weight by how long the guest visit is. A short visit (dinner,
  ``< GUEST_LONG_HOURS``) draws little hot water (~half a person); a long visit
  (sauna + showers) is treated as ``GUEST_LONG_FACTOR`` guest-equivalents — and
  we deliberately over-provision so guests never run the tank cold.

``person`` states are not binary (``home``, ``not_home``, or a zone name like
``Tampere``); only ``home`` counts as present. Missing data / no recorder falls
back to the instantaneous state so a sensor or recorder outage never under-serves
the tank.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime, timedelta
from functools import partial

from homeassistant.const import STATE_HOME, STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)

# Residents: home for at least this many of the trailing window's hours.
RESIDENT_WINDOW_HOURS = 24
RESIDENT_MIN_HOME_HOURS = 12

# Guests: look this far ahead for a visit; a visit at/above the long threshold
# counts as the "long" weight, otherwise the "short" weight.
GUEST_LOOKAHEAD_HOURS = 24
GUEST_LONG_HOURS = 6.0
GUEST_SHORT_FACTOR = 0.5
GUEST_LONG_FACTOR = 2.0
# Used only when the calendar's event list can't be read but it reports "on".
GUEST_FALLBACK_FACTOR = 1.0


# ── instantaneous helpers (fallbacks) ────────────────────────────────────────


def count_people_home(hass: HomeAssistant, person_entities: Iterable[str]) -> int:
    """Count residents whose ``person`` entity is ``home`` right now."""
    count = 0
    for entity_id in person_entities:
        state = hass.states.get(entity_id)
        if state is None or state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            count += 1  # conservative: assume present rather than under-serve
        elif state.state == STATE_HOME:
            count += 1
    return count


def guests_active(hass: HomeAssistant, calendar_entity: str | None) -> bool:
    """True when the guests calendar currently reports an active event."""
    if not calendar_entity:
        return False
    state = hass.states.get(calendar_entity)
    return state is not None and state.state == STATE_ON


# ── pure duration math (unit-tested directly) ────────────────────────────────


def home_seconds(states: list, start: datetime, end: datetime) -> float:
    """Seconds spent in ``home`` across ``[start, end]`` given ordered states.

    Each state holds ``.state`` + ``.last_changed``; a state spans from its
    ``last_changed`` (clamped to ``start``) until the next state's, or ``end``.
    """
    ordered = sorted(states, key=lambda s: s.last_changed)
    total = 0.0
    for i, state in enumerate(ordered):
        seg_start = max(state.last_changed, start)
        seg_end = ordered[i + 1].last_changed if i + 1 < len(ordered) else end
        if seg_end > seg_start and state.state == STATE_HOME:
            total += (seg_end - seg_start).total_seconds()
    return total


def event_hours(event: dict) -> float:
    """Duration of a calendar event in hours; all-day events count as a full day."""
    start = dt_util.parse_datetime(str(event.get("start", "")))
    end = dt_util.parse_datetime(str(event.get("end", "")))
    if start and end and end > start:
        return (end - start).total_seconds() / 3600.0
    return 24.0  # all-day (date-only) event → treat as a long visit


def guest_factor(
    max_event_hours: float,
    *,
    long_hours: float = GUEST_LONG_HOURS,
    short_factor: float = GUEST_SHORT_FACTOR,
    long_factor: float = GUEST_LONG_FACTOR,
) -> float:
    """Map the longest guest visit to a guest-equivalent weight."""
    if max_event_hours <= 0:
        return 0.0
    return long_factor if max_event_hours >= long_hours else short_factor


# ── history / calendar reads ─────────────────────────────────────────────────


async def _async_history(
    hass: HomeAssistant, entities: list[str], start: datetime, end: datetime
) -> dict | None:
    """Per-entity state history over ``[start, end]``; None if recorder absent."""
    try:
        from homeassistant.components.recorder import get_instance
        from homeassistant.components.recorder.history import get_significant_states
    except ImportError:
        return None
    try:
        instance = get_instance(hass)
    except KeyError:
        _LOGGER.debug("Recorder unavailable; using instantaneous occupancy")
        return None
    try:
        return await instance.async_add_executor_job(
            partial(
                get_significant_states,
                hass,
                start,
                end,
                entities,
                include_start_time_state=True,
                significant_changes_only=False,
                no_attributes=True,
            )
        )
    except Exception as err:  # noqa: BLE001 - any recorder hiccup → instantaneous fallback
        _LOGGER.debug("History read failed; using instantaneous occupancy: %s", err)
        return None


async def async_count_residents_home(
    hass: HomeAssistant,
    person_entities: Iterable[str],
    *,
    window_hours: int = RESIDENT_WINDOW_HOURS,
    min_home_hours: float = RESIDENT_MIN_HOME_HOURS,
) -> int:
    """Count residents home for ≥ ``min_home_hours`` over the trailing window."""
    entities = list(person_entities)
    if not entities:
        return 0
    end = dt_util.utcnow()
    start = end - timedelta(hours=window_hours)
    history = await _async_history(hass, entities, start, end)
    if history is None:
        return count_people_home(hass, entities)  # recorder down → instantaneous

    min_seconds = min_home_hours * 3600.0
    count = 0
    for entity_id in entities:
        states = history.get(entity_id)
        if not states:
            # No history for this resident → fall back to its current state.
            current = hass.states.get(entity_id)
            if current is None or current.state in (STATE_UNAVAILABLE, STATE_UNKNOWN, STATE_HOME):
                count += 1
            continue
        if home_seconds(states, start, end) >= min_seconds:
            count += 1
    return count


async def async_guest_equivalents(
    hass: HomeAssistant,
    calendar_entity: str | None,
    *,
    lookahead_hours: int = GUEST_LOOKAHEAD_HOURS,
) -> float:
    """Guest-equivalent weight from the longest guest visit in the lookahead window."""
    if not calendar_entity:
        return 0.0
    start = dt_util.now()
    end = start + timedelta(hours=lookahead_hours)
    try:
        response = await hass.services.async_call(
            "calendar",
            "get_events",
            {
                "entity_id": calendar_entity,
                "start_date_time": start.isoformat(),
                "end_date_time": end.isoformat(),
            },
            blocking=True,
            return_response=True,
        )
    except HomeAssistantError as err:
        # Can't enumerate events → fall back to the on/off state.
        _LOGGER.debug("Could not read guest events from %s: %s", calendar_entity, err)
        return GUEST_FALLBACK_FACTOR if guests_active(hass, calendar_entity) else 0.0

    events = (response or {}).get(calendar_entity, {}).get("events", [])
    longest = max((event_hours(ev) for ev in events), default=0.0)
    return guest_factor(longest)
