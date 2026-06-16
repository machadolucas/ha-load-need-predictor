"""Integration setup: a hub + one load produces sensors and a forecast."""

from __future__ import annotations

from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.load_need_predictor.const import DOMAIN, SUBENTRY_TYPE_LOAD

_LOAD_DATA = {
    "name": "LVV",
    "delivered_energy_entity": "sensor.lvv_energy",
    "rated_power_kw": 3.0,
    "person_entities": ["person.a", "person.b"],
    "min_minutes": 40,
    "max_minutes": 240,
}


async def _setup(hass: HomeAssistant, *, people_home: bool, load_data: dict | None = None):
    presence = "home" if people_home else "not_home"
    hass.states.async_set("person.a", presence)
    hass.states.async_set("person.b", presence)
    hass.states.async_set("sensor.lvv_energy", "1000", {"state_class": "total_increasing"})
    subentry = ConfigSubentryData(
        subentry_type=SUBENTRY_TYPE_LOAD,
        title="LVV",
        unique_id=None,
        data=load_data or _LOAD_DATA,
    )
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"name": "Predictor", "predict_time": "14:00:00", "capture_time": "23:55:00"},
        subentries_data=[subentry],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


def _entity_id(hass: HomeAssistant, subentry_id: str, key: str) -> str:
    reg = er.async_get(hass)
    eid = reg.async_get_entity_id("sensor", DOMAIN, f"{subentry_id}_{key}")
    assert eid is not None, f"sensor.{key} not registered"
    return eid


async def test_setup_creates_all_sensors(hass: HomeAssistant) -> None:
    entry = await _setup(hass, people_home=True)
    subentry_id = next(iter(entry.subentries))
    for key in (
        "predicted_runtime",
        "predicted_energy",
        "last_delivered",
        "prediction_error",
        "rolling_mae",
        "sample_count",
    ):
        _entity_id(hass, subentry_id, key)


async def test_forecast_for_two_people(hass: HomeAssistant) -> None:
    entry = await _setup(hass, people_home=True)
    subentry_id = next(iter(entry.subentries))
    # 2 people: 3.0 + 2×2.2 = 7.4 kWh → /3 kW ×60 = 148 → rounded to 150 min.
    runtime = hass.states.get(_entity_id(hass, subentry_id, "predicted_runtime"))
    assert runtime.state == "150"
    energy = hass.states.get(_entity_id(hass, subentry_id, "predicted_energy"))
    assert float(energy.state) == 7.4
    # No data logged yet → confidence/eval reflect the cold start.
    assert hass.states.get(_entity_id(hass, subentry_id, "sample_count")).state == "0"


async def test_empty_house_hits_safety_floor(hass: HomeAssistant) -> None:
    entry = await _setup(hass, people_home=False)
    subentry_id = next(iter(entry.subentries))
    # Nobody home: 0.4×3.0 = 1.2 kWh → 24 min → clamped up to the 40-min floor (45).
    runtime = hass.states.get(_entity_id(hass, subentry_id, "predicted_runtime"))
    assert runtime.state == "45"


async def test_reload_preserves_entities(hass: HomeAssistant) -> None:
    entry = await _setup(hass, people_home=True)
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    subentry_id = next(iter(entry.subentries))
    assert hass.states.get(_entity_id(hass, subentry_id, "predicted_runtime")).state == "150"
