"""Tests for the per-load config mapping (logic-only; reads plain dicts)."""

from __future__ import annotations

from custom_components.load_need_predictor.const import (
    CONF_DELIVERED_ENERGY_ENTITY,
    CONF_HEATING_ACTIVE_ENTITY,
    CONF_NAME,
    CONF_PERSON_ENTITIES,
    CONF_RATED_POWER_KW,
    CONF_TANK_BOOST_SOC_PCT,
    CONF_TANK_COLD_IN_C,
    CONF_TANK_SETPOINT_C,
    CONF_TANK_VOLUME_L,
    DEFAULT_MAX_MINUTES,
    DEFAULT_MIN_MINUTES,
    DEFAULT_RATED_POWER_KW,
    DEFAULT_TANK_COLD_IN_C,
    DEFAULT_TANK_SETPOINT_C,
    DEFAULT_TANK_VOLUME_L,
)
from custom_components.load_need_predictor.models import load_config_from_data


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
