"""The 'predict now' / 'update forecast now' buttons."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_mock_service

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
                title="LVV",
                unique_id=None,
                data={
                    "name": "LVV",
                    "target_number_entity": "number.lvv_target",
                    "delivered_energy_entity": "sensor.lvv_energy",
                    "rated_power_kw": 3.0,
                    "person_entities": ["person.a", "person.b"],
                },
            ),
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_PRICE_FORECAST,
                title="LVV",
                unique_id=None,
                data={
                    "name": "LVV",
                    "price_entity": "sensor.price",
                    "wind_entity": "sensor.wind",
                    "weather_entity": "weather.home",
                    "temp_history_entity": "sensor.temp",
                },
            ),
        ],
    )
    entry.add_to_hass(hass)
    # Avoid the setup-time forecast build hitting the (absent) weather/recorder.
    with patch(f"{_FC}.PriceForecastCoordinator.async_build_forecast", new=AsyncMock()):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    return entry


def _button_id(hass: HomeAssistant, subentry_id: str, key: str) -> str:
    eid = er.async_get(hass).async_get_entity_id("button", DOMAIN, f"{subentry_id}_{key}")
    assert eid is not None, f"button.{key} not registered"
    return eid


async def test_buttons_exist(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    load_id = next(s for s, sub in entry.subentries.items() if sub.subentry_type == "load")
    fc_id = next(s for s, sub in entry.subentries.items() if sub.subentry_type == "price_forecast")
    assert hass.states.get(_button_id(hass, load_id, "predict_now")) is not None
    assert hass.states.get(_button_id(hass, fc_id, "forecast_now")) is not None


async def test_predict_now_pushes_target(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    load_id = next(s for s, sub in entry.subentries.items() if sub.subentry_type == "load")
    calls = async_mock_service(hass, "number", "set_value")

    await hass.services.async_call(
        "button",
        "press",
        {"entity_id": _button_id(hass, load_id, "predict_now")},
        blocking=True,
    )
    await hass.async_block_till_done()

    # 2 people → 7.4 kWh → 150 min pushed immediately.
    assert len(calls) == 1
    assert calls[0].data["value"] == 150
    assert entry.runtime_data.load.training[load_id][-1]["predicted_minutes"] == 150


async def test_forecast_now_triggers_build(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    fc_id = next(s for s, sub in entry.subentries.items() if sub.subentry_type == "price_forecast")
    forecast = entry.runtime_data.forecast
    with patch.object(forecast, "async_build_forecast", new=AsyncMock()) as build:
        await hass.services.async_call(
            "button",
            "press",
            {"entity_id": _button_id(hass, fc_id, "forecast_now")},
            blocking=True,
        )
        await hass.async_block_till_done()
    build.assert_awaited_once_with(only=fc_id)


async def test_predict_only_restricts_to_one_subentry(hass: HomeAssistant) -> None:
    # The `only` filter must touch just its own load, not every load.
    entry = await _setup(hass)
    coordinator = entry.runtime_data.load
    load_id = next(s for s, sub in entry.subentries.items() if sub.subentry_type == "load")
    async_mock_service(hass, "number", "set_value")

    await coordinator.async_predict_and_push(only="nonexistent")
    assert load_id not in coordinator.training  # nothing logged for the real load

    await coordinator.async_predict_and_push(only=load_id)
    assert coordinator.training[load_id]  # now it ran
