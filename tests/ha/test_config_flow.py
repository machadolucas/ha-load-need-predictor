"""Config-flow tests: the hub flow and the per-load subentry wizard."""

from __future__ import annotations

from homeassistant.config_entries import SOURCE_USER
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.load_need_predictor.const import DOMAIN, SUBENTRY_TYPE_LOAD


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
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "LVV"
    await hass.async_block_till_done()
    assert len(entry.subentries) == 1
