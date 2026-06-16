"""Diagnostics dump: includes model state, redacts entity ids."""

from __future__ import annotations

from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.load_need_predictor.const import DOMAIN, SUBENTRY_TYPE_LOAD
from custom_components.load_need_predictor.diagnostics import (
    async_get_config_entry_diagnostics,
)

_LOAD_DATA = {
    "name": "LVV",
    "delivered_energy_entity": "sensor.lvv_energy",
    "person_entities": ["person.a"],
    "rated_power_kw": 3.0,
}


async def _setup(hass: HomeAssistant):
    hass.states.async_set("person.a", "home")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"name": "Predictor", "predict_time": "14:00:00"},
        subentries_data=[
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_LOAD, title="LVV", unique_id=None, data=_LOAD_DATA
            )
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


async def test_diagnostics_structure_and_redaction(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    diag = await async_get_config_entry_diagnostics(hass, entry)

    assert set(diag) == {"hub", "loads", "forecasts"}
    sid = next(iter(diag["loads"]))
    load = diag["loads"][sid]

    # Model + result are present for debugging.
    assert "gain" in load["model"]
    # One person home → 3.0 + 2.2 = 5.2 kWh → /3 kW ×60 = 104 → rounded to 105 min.
    assert load["result"]["predicted_minutes"] == 105

    # Entity ids are redacted; the friendly name is not.
    assert load["config"]["delivered_energy_entity"] != "sensor.lvv_energy"
    assert load["config"]["name"] == "LVV"
