"""Shared base for Load Need Predictor per-load entities."""

from __future__ import annotations

from homeassistant.config_entries import ConfigSubentry
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import LoadNeedPredictorCoordinator, LoadResult
from .forecast_coordinator import ForecastResult, PriceForecastCoordinator
from .tank_tracker import TankResult, TankTracker


def _device_info(subentry_id: str, subentry: ConfigSubentry) -> DeviceInfo:
    """One subentry = one device."""
    return DeviceInfo(
        identifiers={(DOMAIN, subentry_id)},
        name=subentry.title,
        manufacturer="Load Need Predictor",
        entry_type=DeviceEntryType.SERVICE,
    )


class PredictorEntity(CoordinatorEntity[LoadNeedPredictorCoordinator]):
    """Base entity for one load (one subentry = one device)."""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: LoadNeedPredictorCoordinator,
        subentry_id: str,
        subentry: ConfigSubentry,
        key: str,
    ) -> None:
        super().__init__(coordinator)
        self._subentry_id = subentry_id
        self._attr_unique_id = f"{subentry_id}_{key}"
        self._attr_translation_key = key
        self._attr_device_info = _device_info(subentry_id, subentry)

    @property
    def _result(self) -> LoadResult | None:
        """This load's current result (None before the first refresh)."""
        if self.coordinator.data is None:
            return None
        return self.coordinator.data.get(self._subentry_id)


class ForecastEntity(CoordinatorEntity[PriceForecastCoordinator]):
    """Base entity for a price-forecast subentry (one subentry = one device)."""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: PriceForecastCoordinator,
        subentry_id: str,
        subentry: ConfigSubentry,
        key: str,
    ) -> None:
        super().__init__(coordinator)
        self._subentry_id = subentry_id
        self._attr_unique_id = f"{subentry_id}_{key}"
        self._attr_translation_key = key
        self._attr_device_info = _device_info(subentry_id, subentry)

    @property
    def _result(self) -> ForecastResult | None:
        """This forecast's current result (None before the first refresh)."""
        if self.coordinator.data is None:
            return None
        return self.coordinator.data.get(self._subentry_id)


class TankEntity(CoordinatorEntity[TankTracker]):
    """Base entity for a load's tank charge sensor.

    Uses the same ``_device_info`` as the load's other sensors, so the tank
    charge sensor joins that same device rather than spawning a new one.
    """

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: TankTracker,
        subentry_id: str,
        subentry: ConfigSubentry,
        key: str,
    ) -> None:
        super().__init__(coordinator)
        self._subentry_id = subentry_id
        self._attr_unique_id = f"{subentry_id}_{key}"
        self._attr_translation_key = key
        self._attr_device_info = _device_info(subentry_id, subentry)

    @property
    def _result(self) -> TankResult | None:
        """This load's current tank result (None before the first tick)."""
        if self.coordinator.data is None:
            return None
        return self.coordinator.data.get(self._subentry_id)
