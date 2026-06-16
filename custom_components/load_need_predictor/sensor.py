"""Per-subentry sensors.

Load subentries publish the runtime forecast + its evaluation metrics; the
price-forecast subentry publishes one sensor whose attributes carry the
scheduler-shaped ``data_today`` slot list, plus forecast-accuracy metrics.
``predicted_runtime`` and the price-forecast sensor are the headline values; the
rest expose actual-vs-predicted so the models can be evaluated from the UI.
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
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, SUBENTRY_TYPE_LOAD, SUBENTRY_TYPE_PRICE_FORECAST
from .entity import PredictorEntity
from .forecast_coordinator import ForecastResult, PriceForecastCoordinator
from .runtime import LoadNeedPredictorConfigEntry

_EUR_PER_KWH = "€/kWh"


# ── Load sensors ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, kw_only=True)
class PredictorSensorDescription(SensorEntityDescription):
    """A load-sensor description plus how to pull its value from a LoadResult."""

    value_fn: Callable


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


# ── Price-forecast sensors ───────────────────────────────────────────────────


@dataclass(frozen=True, kw_only=True)
class ForecastSensorDescription(SensorEntityDescription):
    """A forecast-metric description plus how to pull it from a ForecastResult."""

    value_fn: Callable[[ForecastResult], float | int | None]


FORECAST_VALUE_SENSORS: tuple[ForecastSensorDescription, ...] = (
    ForecastSensorDescription(
        key="forecast_error",
        translation_key="forecast_error",
        native_unit_of_measurement=_EUR_PER_KWH,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:target",
        value_fn=lambda r: r.last_error,
    ),
    ForecastSensorDescription(
        key="forecast_mae",
        translation_key="forecast_mae",
        native_unit_of_measurement=_EUR_PER_KWH,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:chart-bell-curve",
        value_fn=lambda r: r.forecast_mae,
    ),
    ForecastSensorDescription(
        key="forecast_samples",
        translation_key="forecast_samples",
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:counter",
        value_fn=lambda r: r.forecast_samples,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: LoadNeedPredictorConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up sensors for each subentry (loads and the price forecast)."""
    runtime = entry.runtime_data
    for subentry_id, subentry in entry.subentries.items():
        if subentry.subentry_type == SUBENTRY_TYPE_LOAD:
            async_add_entities(
                [
                    PredictorSensor(runtime.load, subentry_id, subentry, description)
                    for description in SENSORS
                ],
                config_subentry_id=subentry_id,
            )
        elif (
            subentry.subentry_type == SUBENTRY_TYPE_PRICE_FORECAST and runtime.forecast is not None
        ):
            entities: list[SensorEntity] = [
                PriceForecastSensor(runtime.forecast, subentry_id, subentry)
            ]
            entities.extend(
                ForecastValueSensor(runtime.forecast, subentry_id, subentry, description)
                for description in FORECAST_VALUE_SENSORS
            )
            async_add_entities(entities, config_subentry_id=subentry_id)


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


class _ForecastEntity(CoordinatorEntity[PriceForecastCoordinator]):
    """Base for the price-forecast subentry's sensors (one subentry = one device)."""

    _attr_has_entity_name = True

    def __init__(self, coordinator, subentry_id, subentry, key) -> None:
        super().__init__(coordinator)
        self._subentry_id = subentry_id
        self._attr_unique_id = f"{subentry_id}_{key}"
        self._attr_translation_key = key
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, subentry_id)},
            name=subentry.title,
            manufacturer="Load Need Predictor",
            entry_type=DeviceEntryType.SERVICE,
        )

    @property
    def _result(self) -> ForecastResult | None:
        if self.coordinator.data is None:
            return None
        return self.coordinator.data.get(self._subentry_id)


class PriceForecastSensor(_ForecastEntity, SensorEntity):
    """The headline forecast: state = mean predicted price; attrs = scheduler slots."""

    _attr_native_unit_of_measurement = _EUR_PER_KWH
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:cash-clock"

    def __init__(self, coordinator, subentry_id, subentry) -> None:
        super().__init__(coordinator, subentry_id, subentry, "price_forecast")

    @property
    def native_value(self) -> float | None:
        result = self._result
        return result.mean_buy if result else None

    @property
    def extra_state_attributes(self) -> dict:
        result = self._result
        if result is None:
            return {}
        # `data_today` is the contract the Load Scheduler parses; the rest is
        # human-facing context.
        return {
            "data_today": result.slots,
            "status": result.status,
            "days": result.days,
            "model_samples": result.model_samples,
            "fit_mae_eur_kwh": result.fit_mae,
            "forecast_mae_eur_kwh": result.forecast_mae,
        }


class ForecastValueSensor(_ForecastEntity, SensorEntity):
    """One forecast-accuracy metric from the ForecastResult."""

    entity_description: ForecastSensorDescription

    def __init__(self, coordinator, subentry_id, subentry, description) -> None:
        super().__init__(coordinator, subentry_id, subentry, description.key)
        self.entity_description = description

    @property
    def native_value(self) -> float | int | None:
        result = self._result
        if result is None:
            return None
        return self.entity_description.value_fn(result)
