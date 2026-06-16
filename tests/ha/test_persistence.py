"""Persistence: serialization round-trip and Store reload."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_mock_service

from custom_components.load_need_predictor.const import DOMAIN, STORAGE_VERSION, SUBENTRY_TYPE_LOAD
from custom_components.load_need_predictor.persistence import model_from_dict, model_to_dict
from custom_components.load_need_predictor.predictor import ModelState

_LOAD_DATA = {
    "name": "LVV",
    "target_number_entity": "number.lvv_target",
    "delivered_energy_entity": "sensor.lvv_energy",
    "rated_power_kw": 3.0,
    "person_entities": ["person.a", "person.b"],
}


def test_model_dict_round_trip():
    state = ModelState(e_base=3.3, e_draw_per_person=2.1, gain=1.15, sample_count=7)
    restored = model_from_dict(model_to_dict(state))
    assert restored == state


def test_model_from_dict_defaults_on_missing():
    # An empty/partial payload falls back to seeds (forward-compatible).
    assert model_from_dict(None) == ModelState()
    assert model_from_dict({"gain": 1.3}).gain == 1.3
    assert model_from_dict({"gain": 1.3}).e_base == ModelState().e_base


async def _setup(hass: HomeAssistant):
    hass.states.async_set("person.a", "home")
    hass.states.async_set("person.b", "home")
    hass.states.async_set("number.lvv_target", "0")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"name": "Predictor"},
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


async def test_learned_state_survives_reload(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    coordinator = entry.runtime_data.load
    sid = next(iter(coordinator.load_configs()))

    async_mock_service(hass, "number", "set_value")
    await coordinator.async_predict_and_push()
    with patch(
        "custom_components.load_need_predictor.coordinator.async_daily_delivered_kwh",
        new=AsyncMock(return_value=6.9),
    ):
        await coordinator.async_capture_and_log()
    learned_gain = coordinator.models[sid].gain
    assert coordinator.models[sid].sample_count == 1

    # Flush immediately (bypass the 10s debounce) and reload the entry.
    await coordinator._store.async_save_now(coordinator._runtime_snapshot())
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    reloaded = entry.runtime_data.load
    assert reloaded.models[sid].sample_count == 1
    assert reloaded.models[sid].gain == learned_gain
    assert reloaded.training[sid][-1]["actual_kwh"] == 6.9


async def test_load_runtime_reads_existing_store(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    coordinator = entry.runtime_data.load
    sid = next(iter(coordinator.load_configs()))

    # Pre-seed the Store with a learned model, then re-load runtime.
    store: Store[dict] = Store(hass, STORAGE_VERSION, f"{DOMAIN}.{entry.entry_id}")
    await store.async_save(
        {
            sid: {
                "model": {"e_base": 3.5, "gain": 1.2, "sample_count": 42},
                "training": [{"date": "2026-06-15", "actual_kwh": 7.0}],
                "eval": [10.0, 20.0],
            }
        }
    )
    coordinator.models.clear()
    await coordinator.async_load_runtime()

    assert coordinator.models[sid].gain == 1.2
    assert coordinator.models[sid].sample_count == 42
    assert coordinator.training[sid][-1]["actual_kwh"] == 7.0
    assert coordinator.eval_errors[sid] == [10.0, 20.0]
