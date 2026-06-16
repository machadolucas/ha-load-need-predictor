"""The Load Need Predictor integration.

A hub config entry holds the global prediction/capture schedule; one config
*subentry* per load (the water heater today, other flexible loads later) carries
that load's sensors and its link to a Load Scheduler target.

The predictor answers *how much* a load needs to run each day and pushes that to
the Load Scheduler, which decides *when*. See ``CLAUDE.md`` for the model and the
data findings behind it.

M0 scaffold: this module is intentionally minimal. The coordinator, the daily
jobs and the sensor platform are wired in from M2/M3.
"""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Load Need Predictor from the hub config entry (scaffold)."""
    _LOGGER.debug("Load Need Predictor setup (scaffold) for entry %s", entry.entry_id)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload the hub config entry (scaffold)."""
    return True
