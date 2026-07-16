"""Per-load configuration model.

Maps a config ``ConfigSubentry``'s ``data`` dict to a frozen, validated
``LoadConfig``. Reads only plain dict values (no live Home Assistant state), so
it is trivially testable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .const import (
    CONF_CONTROLLED_SWITCH_ENTITY,
    CONF_DEFICIT_CAP_MINUTES,
    CONF_DELIVERED_ENERGY_ENTITY,
    CONF_DELIVERED_RUNTIME_ENTITY,
    CONF_FIT_DAYS,
    CONF_FORECAST_DAYS,
    CONF_GUESTS_CALENDAR_ENTITY,
    CONF_HEATING_ACTIVE_ENTITY,
    CONF_MAX_MINUTES,
    CONF_MIN_MINUTES,
    CONF_NAME,
    CONF_OUTDOOR_TEMP_ENTITY,
    CONF_PERSON_ENTITIES,
    CONF_PRICE_ENTITY,
    CONF_RATED_POWER_KW,
    CONF_SUPPLY_TEMP_ENTITY,
    CONF_TANK_BOOST_SOC_PCT,
    CONF_TANK_COLD_IN_C,
    CONF_TANK_SETPOINT_C,
    CONF_TANK_VOLUME_L,
    CONF_TARGET_NUMBER_ENTITY,
    CONF_TEMP_HISTORY_ENTITY,
    CONF_WATER_TOTAL_ENTITY,
    CONF_WEATHER_ENTITY,
    CONF_WIND_ENTITY,
    DEFAULT_DEFICIT_CAP_FACTOR,
    DEFAULT_FIT_DAYS,
    DEFAULT_FORECAST_DAYS,
    DEFAULT_MAX_MINUTES,
    DEFAULT_MIN_MINUTES,
    DEFAULT_RATED_POWER_KW,
    DEFAULT_TANK_COLD_IN_C,
    DEFAULT_TANK_SETPOINT_C,
    DEFAULT_TANK_VOLUME_L,
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
    # Tank state-of-charge (opt-in: tracked only when heating_active_entity is set).
    heating_active_entity: str | None
    tank_volume_l: float
    tank_setpoint_c: float
    tank_cold_in_c: float
    tank_boost_soc_pct: float | None  # None → low-charge boost disabled
    # Output clamp (minutes/day).
    min_minutes: float
    max_minutes: float
    # Deficit carryover: the controlled switch whose on-time measures runtime
    # actually delivered (None → carryover disabled), and the backlog cap.
    controlled_switch_entity: str | None
    deficit_cap_minutes: float


def _as_tuple(value) -> tuple[str, ...]:
    """Normalise an EntitySelector(multiple) value to a tuple of entity ids."""
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def load_config_from_data(data: Mapping) -> LoadConfig:
    """Build a :class:`LoadConfig` from a subentry's ``data`` mapping."""
    max_minutes = float(data.get(CONF_MAX_MINUTES, DEFAULT_MAX_MINUTES))
    cap = data.get(CONF_DEFICIT_CAP_MINUTES)
    deficit_cap_minutes = (
        float(cap) if cap not in (None, "") else DEFAULT_DEFICIT_CAP_FACTOR * max_minutes
    )
    boost = data.get(CONF_TANK_BOOST_SOC_PCT)
    tank_boost_soc_pct = float(boost) if boost not in (None, "") else None
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
        heating_active_entity=data.get(CONF_HEATING_ACTIVE_ENTITY),
        tank_volume_l=float(data.get(CONF_TANK_VOLUME_L, DEFAULT_TANK_VOLUME_L)),
        tank_setpoint_c=float(data.get(CONF_TANK_SETPOINT_C, DEFAULT_TANK_SETPOINT_C)),
        tank_cold_in_c=float(data.get(CONF_TANK_COLD_IN_C, DEFAULT_TANK_COLD_IN_C)),
        tank_boost_soc_pct=tank_boost_soc_pct,
        min_minutes=float(data.get(CONF_MIN_MINUTES, DEFAULT_MIN_MINUTES)),
        max_minutes=max_minutes,
        controlled_switch_entity=data.get(CONF_CONTROLLED_SWITCH_ENTITY),
        deficit_cap_minutes=deficit_cap_minutes,
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
