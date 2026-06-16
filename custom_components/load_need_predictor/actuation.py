"""Push the predicted target into the Load Scheduler.

The only output path implemented in load_scheduler v0.1.0 is writing its target
``number`` via ``number.set_value``. This must never raise: if the scheduler
isn't installed yet, or the entity isn't there, we log and report failure so the
predictor keeps publishing its own sensor and simply retries next cycle.
"""

from __future__ import annotations

import logging

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from .const import SCHEDULER_SET_SERVICE, SCHEDULER_SET_SERVICE_DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_push_target(
    hass: HomeAssistant, number_entity_id: str | None, minutes: int | None
) -> bool:
    """Set the scheduler target. Returns True on success, False (no raise) otherwise."""
    if not number_entity_id or minutes is None:
        return False
    if hass.states.get(number_entity_id) is None:
        _LOGGER.warning(
            "Scheduler target %s is unavailable (is Load Scheduler installed?); "
            "skipping push — the prediction is still published",
            number_entity_id,
        )
        return False
    try:
        await hass.services.async_call(
            SCHEDULER_SET_SERVICE_DOMAIN,
            SCHEDULER_SET_SERVICE,
            {"entity_id": number_entity_id, "value": minutes},
            blocking=True,
        )
    except (HomeAssistantError, ValueError) as err:
        _LOGGER.warning("Failed to push target to %s: %s", number_entity_id, err)
        return False
    return True
