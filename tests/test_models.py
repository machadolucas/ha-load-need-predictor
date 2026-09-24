"""Tests for the per-load config mapping (logic-only; reads plain dicts)."""

from __future__ import annotations

import pytest

from custom_components.load_need_predictor.const import (
    CONF_DELIVERED_ENERGY_ENTITY,
    CONF_HEATING_ACTIVE_ENTITY,
    CONF_NAME,
    CONF_PERSON_ENTITIES,
    CONF_PRICE_SERIES_ENTITY,
    CONF_RATED_POWER_KW,
    CONF_TANK_BOOST_SOC_PCT,
    CONF_TANK_COLD_IN_C,
    CONF_TANK_SETPOINT_C,
    CONF_TANK_VOLUME_L,
    CONF_USE_WATTCAST,
    CONF_VAT_PCT,
    CONF_WATTCAST_ZONE,
    DEFAULT_FIT_DAYS,
    DEFAULT_FORECAST_DAYS,
    DEFAULT_MAX_MINUTES,
    DEFAULT_MIN_MINUTES,
    DEFAULT_RATED_POWER_KW,
    DEFAULT_TANK_COLD_IN_C,
    DEFAULT_TANK_SETPOINT_C,
    DEFAULT_TANK_VOLUME_L,
    DEFAULT_WATTCAST_ZONE,
)
from custom_components.load_need_predictor.models import (
    load_config_from_data,
    price_forecast_config_from_data,
)


def test_full_config_maps_all_fields():
    cfg = load_config_from_data(
        {
            CONF_NAME: "LVV",
            "target_number_entity": "number.lvv_target",
            CONF_DELIVERED_ENERGY_ENTITY: "sensor.energy",
            CONF_RATED_POWER_KW: 2.5,
            CONF_PERSON_ENTITIES: ["person.a", "person.b"],
            "min_minutes": 30,
            "max_minutes": 200,
        }
    )
    assert cfg.name == "LVV"
    assert cfg.target_number_entity == "number.lvv_target"
    assert cfg.delivered_energy_entity == "sensor.energy"
    assert cfg.rated_power_kw == 2.5
    assert cfg.person_entities == ("person.a", "person.b")
    assert cfg.min_minutes == 30
    assert cfg.max_minutes == 200


def test_defaults_applied_when_absent():
    cfg = load_config_from_data({CONF_NAME: "LVV"})
    assert cfg.rated_power_kw == DEFAULT_RATED_POWER_KW
    assert cfg.min_minutes == DEFAULT_MIN_MINUTES
    assert cfg.max_minutes == DEFAULT_MAX_MINUTES
    assert cfg.person_entities == ()
    assert cfg.target_number_entity is None
    # Tank fields default to the seeded LVV constants; opt-in fields are unset.
    assert cfg.heating_active_entity is None
    assert cfg.tank_volume_l == DEFAULT_TANK_VOLUME_L
    assert cfg.tank_setpoint_c == DEFAULT_TANK_SETPOINT_C
    assert cfg.tank_cold_in_c == DEFAULT_TANK_COLD_IN_C
    assert cfg.tank_boost_soc_pct is None


def test_single_person_entity_normalised_to_tuple():
    cfg = load_config_from_data({CONF_NAME: "LVV", CONF_PERSON_ENTITIES: "person.solo"})
    assert cfg.person_entities == ("person.solo",)


def test_tank_fields_round_trip():
    cfg = load_config_from_data(
        {
            CONF_NAME: "LVV",
            CONF_HEATING_ACTIVE_ENTITY: "binary_sensor.led",
            CONF_TANK_VOLUME_L: 300,
            CONF_TANK_SETPOINT_C: 75,
            CONF_TANK_COLD_IN_C: 12,
            CONF_TANK_BOOST_SOC_PCT: 20,
        }
    )
    assert cfg.heating_active_entity == "binary_sensor.led"
    assert cfg.tank_volume_l == 300.0
    assert cfg.tank_setpoint_c == 75.0
    assert cfg.tank_cold_in_c == 12.0
    assert cfg.tank_boost_soc_pct == 20.0


def test_tank_boost_empty_string_disables():
    cfg = load_config_from_data({CONF_NAME: "LVV", CONF_TANK_BOOST_SOC_PCT: ""})
    assert cfg.tank_boost_soc_pct is None


# ── price-forecast subentry mapping ──────────────────────────────────────────


def test_price_forecast_defaults_for_pre_wattcast_subentry():
    # A v0.9 subentry dict: none of the Wattcast keys, the old 3-day horizon
    # stored explicitly. It must keep working, with Wattcast switched on.
    cfg = price_forecast_config_from_data(
        {
            CONF_NAME: "LVV",
            "price_entity": "sensor.electricity_price",
            "wind_entity": "sensor.wind",
            "weather_entity": "weather.home",
            "temp_history_entity": "sensor.temp",
            "forecast_days": 3,
        }
    )
    assert cfg.use_wattcast is True
    assert cfg.wattcast_zone == DEFAULT_WATTCAST_ZONE == "FI"
    assert cfg.vat == pytest.approx(0.255)
    assert cfg.price_series_entity is None
    assert cfg.forecast_days == 3  # an explicit stored value wins
    assert cfg.fit_days == DEFAULT_FIT_DAYS


def test_price_forecast_minimal_defaults():
    cfg = price_forecast_config_from_data({CONF_NAME: "LVV", "price_entity": "sensor.p"})
    assert cfg.forecast_days == DEFAULT_FORECAST_DAYS == 7
    assert cfg.wind_entity is None
    assert cfg.weather_entity is None
    assert cfg.temp_history_entity is None
    assert cfg.use_wattcast is True


def test_price_forecast_new_fields():
    cfg = price_forecast_config_from_data(
        {
            CONF_NAME: "LVV",
            "price_entity": "sensor.p",
            CONF_USE_WATTCAST: False,
            CONF_WATTCAST_ZONE: "EE",
            CONF_PRICE_SERIES_ENTITY: "sensor.nordpool",
            CONF_VAT_PCT: 24.0,
            "forecast_days": 9.0,  # NumberSelector hands back floats
        }
    )
    assert cfg.use_wattcast is False
    assert cfg.wattcast_zone == "EE"
    assert cfg.price_series_entity == "sensor.nordpool"
    assert cfg.vat == pytest.approx(0.24)
    assert cfg.forecast_days == 9 and isinstance(cfg.forecast_days, int)
