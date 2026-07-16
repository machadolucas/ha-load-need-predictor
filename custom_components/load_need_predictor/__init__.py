"""The Load Need Predictor integration.

A hub config entry holds the global prediction/capture schedule + two
coordinators; one config *subentry* per capability:

- **load** subentries predict *how much* a load needs to run and push that to the
  Load Scheduler (which decides *when*);
- a **price_forecast** subentry estimates electricity prices *beyond* Nord Pool's
  day-ahead horizon, so the scheduler can defer an expensive day to a
  forecast-cheaper one.

See ``CLAUDE.md`` for the models and the data findings behind them.
"""

from __future__ import annotations

import logging

from homeassistant.core import HomeAssistant

from .const import PLATFORMS
from .coordinator import LoadNeedPredictorCoordinator
from .forecast_coordinator import PriceForecastCoordinator
from .frontend import async_register_card
from .jobs import PredictorJobs
from .runtime import LoadNeedPredictorConfigEntry, RuntimeData
from .tank_tracker import TankTracker

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: LoadNeedPredictorConfigEntry) -> bool:
    """Set up Load Need Predictor from the hub config entry."""
    # Serve + register the dashboard card (once per HA process; safe on reload).
    await async_register_card(hass)

    load = LoadNeedPredictorCoordinator(hass, entry)
    await load.async_load_runtime()
    await load.async_config_entry_first_refresh()

    forecast = PriceForecastCoordinator(hass, entry)
    await forecast.async_load_runtime()
    await forecast.async_config_entry_first_refresh()

    # The tank tracker shares the load coordinator's persisted state (it owns the
    # tank dict) and drives its own 60 s tick.
    tank = TankTracker(hass, entry, load)

    entry.runtime_data = RuntimeData(load=load, forecast=forecast, tank=tank)

    # Both daily capabilities run on the hub's two daily wall-clock times.
    jobs = PredictorJobs(hass, entry)
    jobs.async_start()
    entry.async_on_unload(jobs.async_shutdown)

    # The tank tracker ticks on its own interval (registered only when a load
    # opted in), independent of the daily jobs.
    tank.async_start()
    entry.async_on_unload(tank.async_shutdown_ticker)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_reload_entry))

    # Build the price forecast once on setup so the sensor is populated
    # immediately (and after every restart), not only at the next predict time.
    # Backgrounded so a slow statistics fit never delays/fails setup.
    if forecast.has_loads:
        entry.async_create_background_task(
            hass, forecast.async_build_forecast(), "lnp-initial-forecast"
        )
    # One backgrounded tick seeds the tank sensor + performs restart
    # reconciliation without delaying setup.
    if tank.has_tanks:
        entry.async_create_background_task(hass, tank.async_tick(), "lnp-initial-tank-tick")
    return True


async def async_unload_entry(hass: HomeAssistant, entry: LoadNeedPredictorConfigEntry) -> bool:
    """Unload the hub config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def _async_reload_entry(hass: HomeAssistant, entry: LoadNeedPredictorConfigEntry) -> None:
    """Reload on options/subentry changes (picks up added/removed loads)."""
    await hass.config_entries.async_reload(entry.entry_id)
