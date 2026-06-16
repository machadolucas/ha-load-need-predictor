"""The Load Need Predictor integration.

A hub config entry holds the global prediction/capture schedule + the
coordinator; one config *subentry* per load (the water heater today, other
flexible loads later) carries that load's sensors and its link to a Load
Scheduler target.

The predictor answers *how much* a load needs to run each day and pushes that to
the Load Scheduler, which decides *when*. See ``CLAUDE.md`` for the model and the
data findings behind it.
"""

from __future__ import annotations

import logging

from homeassistant.core import HomeAssistant

from .const import PLATFORMS
from .coordinator import LoadNeedPredictorConfigEntry, LoadNeedPredictorCoordinator
from .jobs import PredictorJobs

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: LoadNeedPredictorConfigEntry) -> bool:
    """Set up Load Need Predictor from the hub config entry."""
    coordinator = LoadNeedPredictorCoordinator(hass, entry)
    await coordinator.async_load_runtime()
    await coordinator.async_config_entry_first_refresh()
    entry.runtime_data = coordinator

    # Schedule the two daily jobs (predict+push, capture+log).
    jobs = PredictorJobs(hass, coordinator)
    jobs.async_start()
    entry.async_on_unload(jobs.async_shutdown)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_reload_entry))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: LoadNeedPredictorConfigEntry) -> bool:
    """Unload the hub config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def _async_reload_entry(hass: HomeAssistant, entry: LoadNeedPredictorConfigEntry) -> None:
    """Reload on options/subentry changes (picks up added/removed loads)."""
    await hass.config_entries.async_reload(entry.entry_id)
