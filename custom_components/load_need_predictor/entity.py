"""Shared base for Load Need Predictor per-load entities."""

from __future__ import annotations

from homeassistant.config_entries import ConfigSubentry
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import LoadNeedPredictorCoordinator, LoadResult


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
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, subentry_id)},
            name=subentry.title,
            manufacturer="Load Need Predictor",
            entry_type=DeviceEntryType.SERVICE,
        )

    @property
    def _result(self) -> LoadResult | None:
        """This load's current result (None before the first refresh)."""
        if self.coordinator.data is None:
            return None
        return self.coordinator.data.get(self._subentry_id)
