"""Config-flow tests: the hub flow and the per-load subentry wizard."""

from __future__ import annotations

import pytest
from homeassistant.config_entries import SOURCE_USER, ConfigSubentryData
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType, InvalidData
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.load_need_predictor.const import (
    DOMAIN,
    SUBENTRY_TYPE_LOAD,
    SUBENTRY_TYPE_PRICE_FORECAST,
)


async def test_hub_user_flow_creates_entry(hass: HomeAssistant) -> None:
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
    assert result["type"] == FlowResultType.FORM

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {"name": "Predictor", "predict_time": "14:00:00", "capture_time": "23:55:00"},
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"]["predict_time"] == "14:00:00"
    assert result["data"]["capture_time"] == "23:55:00"


async def test_hub_reconfigure(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(domain=DOMAIN, data={"name": "Predictor", "predict_time": "14:00:00"})
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    result = await entry.start_reconfigure_flow(hass)
    assert result["type"] == FlowResultType.FORM
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"name": "Predictor", "predict_time": "15:30:00"}
    )
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.data["predict_time"] == "15:30:00"


async def test_add_load_subentry(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(domain=DOMAIN, data={"name": "Predictor"})
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_LOAD), context={"source": SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            "name": "LVV",
            "delivered_energy_entity": "sensor.lvv_energy",
            "rated_power_kw": 3.0,
            "heating_active_entity": "binary_sensor.led",
            "tank_volume_l": 300,
            "tank_setpoint_c": 75,
            "tank_cold_in_c": 12,
            "tank_boost_soc_pct": 20,
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "LVV"
    await hass.async_block_till_done()
    assert len(entry.subentries) == 1
    subentry = next(iter(entry.subentries.values()))
    assert subentry.data["heating_active_entity"] == "binary_sensor.led"
    assert subentry.data["tank_volume_l"] == 300
    assert subentry.data["tank_setpoint_c"] == 75
    assert subentry.data["tank_cold_in_c"] == 12
    assert subentry.data["tank_boost_soc_pct"] == 20


async def test_add_price_forecast_subentry(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(domain=DOMAIN, data={"name": "Predictor"})
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PRICE_FORECAST), context={"source": SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            "name": "LVV forecast",
            "price_entity": "sensor.electricity_price",
            "wind_entity": "sensor.wind",
            "weather_entity": "weather.home",
            "temp_history_entity": "sensor.temp",
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "LVV forecast"


async def _start_forecast_flow(hass: HomeAssistant):
    entry = MockConfigEntry(domain=DOMAIN, data={"name": "Predictor"})
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PRICE_FORECAST), context={"source": SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM
    return entry, result


async def test_price_forecast_subentry_minimal_uses_wattcast_defaults(
    hass: HomeAssistant,
) -> None:
    # Wind/weather/temperature only feed the local fallback now → optional.
    entry, result = await _start_forecast_flow(hass)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"name": "LVV", "price_entity": "sensor.electricity_price"}
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    data = result["data"]
    assert data["use_wattcast"] is True
    assert data["wattcast_zone"] == "FI"
    assert data["vat_pct"] == 25.5
    assert data["forecast_days"] == 7
    for key in ("wind_entity", "weather_entity", "temp_history_entity", "price_series_entity"):
        assert key not in data
    await hass.async_block_till_done()  # the reload after adding the subentry


async def test_price_forecast_subentry_all_new_fields(hass: HomeAssistant) -> None:
    entry, result = await _start_forecast_flow(hass)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            "name": "LVV",
            "price_entity": "sensor.electricity_price",
            "price_series_entity": "sensor.nordpool_fi_day_ahead",
            "use_wattcast": False,
            "wattcast_zone": "EE",
            "vat_pct": 24,
            "forecast_days": 9,  # the new maximum
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    data = result["data"]
    assert data["price_series_entity"] == "sensor.nordpool_fi_day_ahead"
    assert data["use_wattcast"] is False  # a False toggle survives _clean()
    assert data["wattcast_zone"] == "EE"
    assert data["vat_pct"] == 24
    assert data["forecast_days"] == 9
    await hass.async_block_till_done()


@pytest.mark.parametrize(
    "bad",
    [{"forecast_days": 10}, {"wattcast_zone": "SE3"}, {"vat_pct": 60}],
)
async def test_price_forecast_subentry_rejects_out_of_range(hass: HomeAssistant, bad) -> None:
    entry, result = await _start_forecast_flow(hass)
    with pytest.raises(InvalidData):
        await hass.config_entries.subentries.async_configure(
            result["flow_id"], {"name": "LVV", "price_entity": "sensor.p", **bad}
        )


async def test_price_forecast_reconfigure_keeps_new_fields(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"name": "Predictor"},
        subentries_data=[
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_PRICE_FORECAST,
                title="LVV",
                unique_id=None,
                # A pre-v0.10 subentry: none of the Wattcast keys.
                data={
                    "name": "LVV",
                    "price_entity": "sensor.p",
                    "wind_entity": "sensor.wind",
                    "weather_entity": "weather.home",
                    "temp_history_entity": "sensor.temp",
                    "forecast_days": 3,
                },
            )
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    sid = next(iter(entry.subentries))

    result = await entry.start_subentry_reconfigure_flow(hass, sid)
    assert result["type"] == FlowResultType.FORM
    # The form offers Wattcast on by default for an old subentry.
    schema = {str(k): k for k in result["data_schema"].schema}
    assert schema["use_wattcast"].default() is True
    assert schema["forecast_days"].default() == 3

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            "name": "LVV",
            "price_entity": "sensor.p",
            "price_series_entity": "sensor.nordpool",
            "use_wattcast": True,
            "wattcast_zone": "FI",
            "vat_pct": 25.5,
            "forecast_days": 7,
        },
    )
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    await hass.async_block_till_done()
    data = entry.subentries[sid].data
    assert data["price_series_entity"] == "sensor.nordpool"
    assert data["forecast_days"] == 7
    # Clearing the optional local-fallback inputs drops them.
    assert "wind_entity" not in data
