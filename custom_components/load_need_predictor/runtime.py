"""Runtime container stored on the config entry.

Holds both coordinators — the load-need predictor and the optional price
forecaster. Kept in its own module (with the coordinators imported only under
``TYPE_CHECKING``) so the typed ``ConfigEntry`` alias can be shared without a
circular import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntry

if TYPE_CHECKING:
    from .coordinator import LoadNeedPredictorCoordinator
    from .forecast_coordinator import PriceForecastCoordinator
    from .tank_tracker import TankTracker


@dataclass
class RuntimeData:
    """What lives on ``entry.runtime_data``."""

    load: LoadNeedPredictorCoordinator
    forecast: PriceForecastCoordinator | None = None
    tank: TankTracker | None = None


type LoadNeedPredictorConfigEntry = ConfigEntry[RuntimeData]
