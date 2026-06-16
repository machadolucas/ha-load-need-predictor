"""Live occupancy sampling.

Occupancy is the only feature with real day-ahead leverage, and it has no
long-term statistics — so we read it live here and the predictor self-logs it.
``person`` states are not binary (they can be ``home``, ``not_home`` or a zone
name like ``Tampere``); only ``home`` counts as present for hot-water purposes.
Unknown/unavailable is treated as *present* so a sensor outage never under-serves
the tank.
"""

from __future__ import annotations

from collections.abc import Iterable

from homeassistant.const import STATE_HOME, STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant


def count_people_home(hass: HomeAssistant, person_entities: Iterable[str]) -> int:
    """Count residents currently home (conservative on missing data)."""
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
