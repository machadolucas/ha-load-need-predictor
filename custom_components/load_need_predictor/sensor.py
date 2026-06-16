"""Per-load sensors: the forecast plus evaluation metrics.

Six sensors per load. ``predicted_runtime`` is the headline value (and the
forward-compatible "target source" the scheduler can point at); the rest expose
the actual delivery and how well the model is tracking it, satisfying the
"log predictions vs. actual" requirement directly in the UI.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import UnitOfEnergy, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import SUBENTRY_TYPE_LOAD
from .coordinator import LoadNeedPredictorConfigEntry, LoadResult
from .entity import PredictorEntity


@dataclass(frozen=True, kw_only=True)
class PredictorSensorDescription(SensorEntityDescription):
    """A sensor description plus how to pull its value from a LoadResult."""

    value_fn: Callable[[LoadResult], float | int | None]


SENSORS: tuple[PredictorSensorDescription, ...] = (
    PredictorSensorDescription(
        key="predicted_runtime",
        translation_key="predicted_runtime",
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.MINUTES,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:timer-cog",
        value_fn=lambda r: r.predicted_minutes,
    ),
    PredictorSensorDescription(
        key="predicted_energy",
        translation_key="predicted_energy",
        native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:lightning-bolt",
        value_fn=lambda r: r.predicted_kwh,
    ),
    PredictorSensorDescription(
        key="last_delivered",
        translation_key="last_delivered",
        native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:water-boiler",
        value_fn=lambda r: r.last_delivered_kwh,
    ),
    PredictorSensorDescription(
        key="prediction_error",
        translation_key="prediction_error",
        native_unit_of_measurement=UnitOfTime.MINUTES,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:target",
        value_fn=lambda r: r.prediction_error_minutes,
    ),
    PredictorSensorDescription(
        key="rolling_mae",
        translation_key="rolling_mae",
        native_unit_of_measurement=UnitOfTime.MINUTES,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:chart-bell-curve",
        value_fn=lambda r: r.rolling_mae_minutes,
    ),
    PredictorSensorDescription(
        key="sample_count",
        translation_key="sample_count",
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:counter",
        value_fn=lambda r: r.sample_count,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: LoadNeedPredictorConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the sensor suite for each load subentry."""
    coordinator = entry.runtime_data
    for subentry_id, subentry in entry.subentries.items():
        if subentry.subentry_type != SUBENTRY_TYPE_LOAD:
            continue
        async_add_entities(
            [
                PredictorSensor(coordinator, subentry_id, subentry, description)
                for description in SENSORS
            ],
            config_subentry_id=subentry_id,
        )


class PredictorSensor(PredictorEntity, SensorEntity):
    """One published value from the per-load LoadResult."""

    entity_description: PredictorSensorDescription

    def __init__(self, coordinator, subentry_id, subentry, description) -> None:
        super().__init__(coordinator, subentry_id, subentry, description.key)
        self.entity_description = description

    @property
    def native_value(self) -> float | int | None:
        result = self._result
        if result is None:
            return None
        return self.entity_description.value_fn(result)
