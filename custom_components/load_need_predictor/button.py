"""Manual "run now" buttons.

A per-load **Predict now** button recomputes the load's prediction and pushes it
to the Load Scheduler immediately — useful to apply fresh values without waiting
for the daily predict time (e.g. occupancy changed, or tomorrow's prices arrived
early/late). A per-forecast **Update forecast now** button rebuilds the
beyond-horizon price forecast on demand.
"""

from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import SUBENTRY_TYPE_LOAD, SUBENTRY_TYPE_PRICE_FORECAST
from .entity import ForecastEntity, PredictorEntity
from .runtime import LoadNeedPredictorConfigEntry


async def async_setup_entry(
    hass: HomeAssistant,
    entry: LoadNeedPredictorConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Add a 'run now' button to each load and price-forecast subentry."""
    runtime = entry.runtime_data
    for subentry_id, subentry in entry.subentries.items():
        if subentry.subentry_type == SUBENTRY_TYPE_LOAD:
            async_add_entities(
                [PredictNowButton(runtime.load, subentry_id, subentry)],
                config_subentry_id=subentry_id,
            )
        elif (
            subentry.subentry_type == SUBENTRY_TYPE_PRICE_FORECAST and runtime.forecast is not None
        ):
            async_add_entities(
                [ForecastNowButton(runtime.forecast, subentry_id, subentry)],
                config_subentry_id=subentry_id,
            )


class PredictNowButton(PredictorEntity, ButtonEntity):
    """Recompute this load's prediction and push it to the scheduler now."""

    _attr_icon = "mdi:play-circle"

    def __init__(self, coordinator, subentry_id, subentry) -> None:
        super().__init__(coordinator, subentry_id, subentry, "predict_now")

    async def async_press(self) -> None:
        await self.coordinator.async_predict_and_push(only=self._subentry_id)


class ForecastNowButton(ForecastEntity, ButtonEntity):
    """Rebuild this load's beyond-horizon price forecast now."""

    _attr_icon = "mdi:refresh"

    def __init__(self, coordinator, subentry_id, subentry) -> None:
        super().__init__(coordinator, subentry_id, subentry, "forecast_now")

    async def async_press(self) -> None:
        await self.coordinator.async_build_forecast(only=self._subentry_id)
