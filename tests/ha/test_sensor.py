"""Sensor attributes that feed the dashboard card's rationale.

The card reads the *load* breakdown off the ``predicted_runtime`` sensor and the
*forecast* coefficients off the ``price_forecast`` sensor, so these assert the
contract those attributes form.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.load_need_predictor.const import (
    DOMAIN,
    SUBENTRY_TYPE_LOAD,
    SUBENTRY_TYPE_PRICE_FORECAST,
)

_FC = "custom_components.load_need_predictor.forecast_coordinator"


async def _setup(hass: HomeAssistant) -> MockConfigEntry:
    hass.states.async_set("person.a", "home")
    hass.states.async_set("person.b", "home")
    hass.states.async_set("number.lvv_target", "0")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"name": "Predictor", "predict_time": "14:00:00", "capture_time": "23:55:00"},
        subentries_data=[
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_LOAD,
                title="LVV water heater",
                unique_id=None,
                data={
                    "name": "LVV water heater",
                    "target_number_entity": "number.lvv_target",
                    "rated_power_kw": 3.0,
                    "person_entities": ["person.a", "person.b"],
                },
            ),
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_PRICE_FORECAST,
                title="LVV",
                unique_id=None,
                data={"name": "LVV", "price_entity": "sensor.price"},
            ),
        ],
    )
    entry.add_to_hass(hass)
    # Don't let setup's forecast build reach the (absent) weather/recorder.
    with patch(f"{_FC}.PriceForecastCoordinator.async_build_forecast", new=AsyncMock()):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    return entry


def _sensor_id(hass: HomeAssistant, subentry_id: str, key: str) -> str:
    eid = er.async_get(hass).async_get_entity_id("sensor", DOMAIN, f"{subentry_id}_{key}")
    assert eid is not None, f"sensor.{key} not registered"
    return eid


def _load_id(entry: MockConfigEntry) -> str:
    return next(s for s, sub in entry.subentries.items() if sub.subentry_type == "load")


def _forecast_id(entry: MockConfigEntry) -> str:
    return next(s for s, sub in entry.subentries.items() if sub.subentry_type == "price_forecast")


async def test_runtime_sensor_exposes_breakdown_and_metrics(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    state = hass.states.get(_sensor_id(hass, _load_id(entry), "predicted_runtime"))
    assert state is not None

    bd = state.attributes["breakdown"]
    for key in (
        "people_home",
        "e_base",
        "gain",
        "occupancy_factor",
        "predicted_kwh",
        "predicted_minutes",
        "clamped",
    ):
        assert key in bd, key
    assert bd["people_home"] == 2  # both persons home
    assert bd["predicted_minutes"] == int(state.state)

    metrics = state.attributes["metrics"]
    assert metrics["sample_count"] == 0  # no observations folded in yet


async def test_other_load_sensors_carry_no_breakdown(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    state = hass.states.get(_sensor_id(hass, _load_id(entry), "predicted_energy"))
    assert state is not None
    assert "breakdown" not in state.attributes


async def test_price_forecast_sensor_exposes_coefficients(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    state = hass.states.get(_sensor_id(hass, _forecast_id(entry), "price_forecast"))
    assert state is not None
    assert "fitted" in state.attributes
    assert "coefficients" in state.attributes
    # No fit yet (build was mocked out) → seed formula, no coefficients.
    assert state.attributes["fitted"] is False
    assert state.attributes["coefficients"] is None


async def test_non_tank_load_has_no_tank_sensor(hass: HomeAssistant) -> None:
    # The base setup's load has no heating detector → no tank charge sensor.
    entry = await _setup(hass)
    eid = er.async_get(hass).async_get_entity_id("sensor", DOMAIN, f"{_load_id(entry)}_tank_soc")
    assert eid is None


_TANK_LOAD = {
    "name": "LVV",
    "target_number_entity": "number.lvv_target",
    "delivered_energy_entity": "sensor.lvv_energy",
    "rated_power_kw": 3.0,
    "person_entities": ["person.a"],
    "heating_active_entity": "binary_sensor.led",
    "water_total_entity": "sensor.water",
    "tank_volume_l": 300,
    "tank_setpoint_c": 75,
    "tank_cold_in_c": 12,
}

# The documented tank-charge attribute keys the card reads (HA adds its own on top).
_TANK_ATTRS = {
    "deficit_kwh",
    "capacity_kwh",
    "hot_fraction",
    "standby_w",
    "calibrated",
    "last_full",
    "draw_source",
    "liters_40c",
    "showers_left",
}
_HA_MANAGED = {"state_class", "unit_of_measurement", "icon", "friendly_name", "device_class"}


async def test_tank_soc_attribute_contract(hass: HomeAssistant) -> None:
    hass.states.async_set("person.a", "home")
    hass.states.async_set("number.lvv_target", "0")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"name": "Predictor", "predict_time": "14:00:00", "capture_time": "23:55:00"},
        subentries_data=[
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_LOAD, title="LVV", unique_id=None, data=_TANK_LOAD
            )
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    sid = _load_id(entry)
    await entry.runtime_data.tank.async_tick()
    await hass.async_block_till_done()

    state = hass.states.get(_sensor_id(hass, sid, "tank_soc"))
    assert state is not None
    assert _TANK_ATTRS <= set(state.attributes)  # every documented key present
    # …and nothing extra beyond the documented keys + HA-managed ones.
    assert set(state.attributes) - _TANK_ATTRS <= _HA_MANAGED
    assert "breakdown" not in state.attributes  # not a load runtime sensor
