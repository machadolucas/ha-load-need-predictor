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


async def test_reload_within_save_debounce_keeps_learning(hass: HomeAssistant) -> None:
    """Unload flushes the load Store — no manual save, reload right away.

    Regression: only the forecast Store was flushed, so a reload inside the
    10 s debounce (every subentry add/edit reloads) lost the day's learning.
    """
    from unittest.mock import AsyncMock, patch

    from pytest_homeassistant_custom_component.common import async_mock_service

    entry = await _setup(hass, people_home=True)
    coordinator = entry.runtime_data.load
    sid = next(iter(entry.subentries))
    async_mock_service(hass, "number", "set_value")
    await coordinator.async_predict_and_push()
    with patch(
        "custom_components.load_need_predictor.coordinator.async_daily_delivered_kwh",
        new=AsyncMock(return_value=6.9),
    ):
        await coordinator.async_capture_and_log()
    learned = coordinator.models[sid]
    assert learned.sample_count == 1

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    reloaded = entry.runtime_data.load
    assert reloaded is not coordinator
    assert reloaded.models[sid].sample_count == 1
    assert reloaded.models[sid].gain == learned.gain


_TANK_LOAD = {
    **_LOAD_DATA,
    "target_number_entity": "number.lvv_target",
    "controlled_switch_entity": "switch.lvv",
    "heating_active_entity": "binary_sensor.lvv_heating",
}


async def test_failed_unload_restores_tank_tick_and_writes(hass: HomeAssistant) -> None:
    from unittest.mock import AsyncMock, patch

    from custom_components.load_need_predictor import async_unload_entry

    hass.states.async_set("binary_sensor.lvv_heating", "off")
    hass.states.async_set("switch.lvv", "off")
    entry = await _setup(hass, people_home=True, load_data=_TANK_LOAD)
    runtime = entry.runtime_data
    assert runtime.tank._unsub is not None  # ticking

    with (
        patch.object(
            hass.config_entries, "async_unload_platforms", new=AsyncMock(return_value=False)
        ),
        patch.object(runtime.load, "async_flush", side_effect=RuntimeError("disk full")),
    ):
        assert await async_unload_entry(hass, entry) is False  # flush error didn't abort

    assert runtime.tank._unsub is not None  # tank tracking resumed
    assert runtime.load._closing is False


async def test_published_target_survives_reload(hass: HomeAssistant) -> None:
    """After a capture moves the gain, a reload shows what was *pushed*."""
    from unittest.mock import AsyncMock, patch

    from pytest_homeassistant_custom_component.common import async_mock_service

    data = {**_LOAD_DATA, "target_number_entity": "number.lvv_target"}
    hass.states.async_set("number.lvv_target", "0")
    entry = await _setup(hass, people_home=True, load_data=data)
    coordinator = entry.runtime_data.load
    sid = next(iter(entry.subentries))
    calls = async_mock_service(hass, "number", "set_value")
    await coordinator.async_predict_and_push()
    pushed = calls[-1].data["value"]
    with patch(
        "custom_components.load_need_predictor.coordinator.async_daily_delivered_kwh",
        new=AsyncMock(return_value=14.0),
    ):
        await coordinator.async_capture_and_log()
    assert coordinator.models[sid].gain > 1.0  # a live recompute would now differ

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    reloaded = entry.runtime_data.load
    assert reloaded.data[sid].predicted_minutes == pushed
    assert reloaded.data[sid].rationale["target_minutes"] == pushed

    # Retarget the load → the restored cache no longer describes its scheduler.
    subentry = entry.subentries[sid]
    hass.config_entries.async_update_subentry(
        entry, subentry, data={**subentry.data, "target_number_entity": "number.other"}
    )
    await hass.async_block_till_done()
    retargeted = entry.runtime_data.load
    assert sid not in retargeted._published
    assert retargeted.data[sid].predicted_minutes != pushed  # live, with the new gain
