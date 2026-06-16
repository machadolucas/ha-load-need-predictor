"""Per-load configuration model.

Maps a config ``ConfigSubentry``'s ``data`` dict to a frozen, validated
``LoadConfig``. Reads only plain dict values (no live Home Assistant state), so
it is trivially testable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .const import (
    CONF_DELIVERED_ENERGY_ENTITY,
    CONF_DELIVERED_RUNTIME_ENTITY,
    CONF_FIT_DAYS,
    CONF_FORECAST_DAYS,
    CONF_GUESTS_CALENDAR_ENTITY,
    CONF_MAX_MINUTES,
    CONF_MIN_MINUTES,
    CONF_NAME,
    CONF_OUTDOOR_TEMP_ENTITY,
    CONF_PERSON_ENTITIES,
    CONF_PRICE_ENTITY,
    CONF_RATED_POWER_KW,
    CONF_SUPPLY_TEMP_ENTITY,
    CONF_TARGET_NUMBER_ENTITY,
    CONF_TEMP_HISTORY_ENTITY,
    CONF_WATER_TOTAL_ENTITY,
    CONF_WEATHER_ENTITY,
    CONF_WIND_ENTITY,
    DEFAULT_FIT_DAYS,
    DEFAULT_FORECAST_DAYS,
    DEFAULT_MAX_MINUTES,
    DEFAULT_MIN_MINUTES,
    DEFAULT_RATED_POWER_KW,
)


@dataclass(frozen=True)
class LoadConfig:
    """Immutable view of one load's configuration."""

    name: str
    # Output: the Load Scheduler number whose value we set (None → publish-only).
    target_number_entity: str | None
    # Delivery feedback (training target) + optional runtime cross-check.
    delivered_energy_entity: str | None
    delivered_runtime_entity: str | None
    rated_power_kw: float
    # Occupancy drivers.
    person_entities: tuple[str, ...]
    guests_calendar_entity: str | None
    # Log-only context (recorded, not used by the v1 model).
    supply_temp_entity: str | None
    outdoor_temp_entity: str | None
    water_total_entity: str | None
    # Output clamp (minutes/day).
    min_minutes: float
    max_minutes: float


def _as_tuple(value) -> tuple[str, ...]:
    """Normalise an EntitySelector(multiple) value to a tuple of entity ids."""
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def load_config_from_data(data: Mapping) -> LoadConfig:
    """Build a :class:`LoadConfig` from a subentry's ``data`` mapping."""
    return LoadConfig(
        name=str(data.get(CONF_NAME, "")),
        target_number_entity=data.get(CONF_TARGET_NUMBER_ENTITY),
        delivered_energy_entity=data.get(CONF_DELIVERED_ENERGY_ENTITY),
        delivered_runtime_entity=data.get(CONF_DELIVERED_RUNTIME_ENTITY),
        rated_power_kw=float(data.get(CONF_RATED_POWER_KW, DEFAULT_RATED_POWER_KW)),
        person_entities=_as_tuple(data.get(CONF_PERSON_ENTITIES)),
        guests_calendar_entity=data.get(CONF_GUESTS_CALENDAR_ENTITY),
        supply_temp_entity=data.get(CONF_SUPPLY_TEMP_ENTITY),
        outdoor_temp_entity=data.get(CONF_OUTDOOR_TEMP_ENTITY),
        water_total_entity=data.get(CONF_WATER_TOTAL_ENTITY),
        min_minutes=float(data.get(CONF_MIN_MINUTES, DEFAULT_MIN_MINUTES)),
        max_minutes=float(data.get(CONF_MAX_MINUTES, DEFAULT_MAX_MINUTES)),
    )


@dataclass(frozen=True)
class PriceForecastConfig:
    """Immutable view of a price-forecast subentry's configuration."""

    name: str
    price_entity: str | None  # actual buy price (€/kWh): fit target + evaluation
    wind_entity: str | None  # wind production forecast (series attribute)
    weather_entity: str | None  # daily temperature forecast source
    temp_history_entity: str | None  # actual outdoor temp for fitting
    forecast_days: int
    fit_days: int


def price_forecast_config_from_data(data: Mapping) -> PriceForecastConfig:
    """Build a :class:`PriceForecastConfig` from a subentry's ``data`` mapping."""
    return PriceForecastConfig(
        name=str(data.get(CONF_NAME, "")),
        price_entity=data.get(CONF_PRICE_ENTITY),
        wind_entity=data.get(CONF_WIND_ENTITY),
        weather_entity=data.get(CONF_WEATHER_ENTITY),
        temp_history_entity=data.get(CONF_TEMP_HISTORY_ENTITY),
        forecast_days=int(data.get(CONF_FORECAST_DAYS, DEFAULT_FORECAST_DAYS)),
        fit_days=int(data.get(CONF_FIT_DAYS, DEFAULT_FIT_DAYS)),
    )
