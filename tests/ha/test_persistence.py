"""Persistence: serialization round-trip and Store reload."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_mock_service

from custom_components.load_need_predictor.const import DOMAIN, STORAGE_VERSION, SUBENTRY_TYPE_LOAD
from custom_components.load_need_predictor.persistence import (
    model_from_dict,
    model_to_dict,
    tank_from_dict,
    tank_to_dict,
)
from custom_components.load_need_predictor.predictor import ModelState
from custom_components.load_need_predictor.tank_model import TankState

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


def test_tank_dict_round_trip():
    state = TankState(
        deficit_kwh=5.5,
        hot_fraction=0.3,
        standby_w=90.0,
        calibrated=True,
        energy_baseline_kwh=7610.0,
        water_baseline_l=412350.0,
        cycle_liters=42.0,
        last_anchor_iso="2026-07-16T10:00:00+00:00",
    )
    assert tank_from_dict(tank_to_dict(state)) == state


def test_tank_from_dict_none_and_defaults():
    # Missing/empty → None (nothing stored); a partial payload keeps the rest at
    # its dataclass default, and the nullable baselines stay None.
    assert tank_from_dict(None) is None
    assert tank_from_dict({}) is None
    partial = tank_from_dict({"deficit_kwh": 4.0, "calibrated": True})
    assert partial.deficit_kwh == 4.0
    assert partial.calibrated is True
    assert partial.hot_fraction == TankState(deficit_kwh=0.0).hot_fraction
    assert partial.energy_baseline_kwh is None


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


async def test_tank_state_survives_reload(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    coordinator = entry.runtime_data.load
    sid = next(iter(coordinator.load_configs()))

    # The tank state is owned by the load coordinator; the snapshot includes it.
    coordinator.tanks[sid] = TankState(
        deficit_kwh=7.0, hot_fraction=0.28, standby_w=85.0, calibrated=True
    )
    await coordinator._store.async_save_now(coordinator._runtime_snapshot())
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    reloaded = entry.runtime_data.load
    assert sid in reloaded.tanks
    assert reloaded.tanks[sid].deficit_kwh == 7.0
    assert reloaded.tanks[sid].calibrated is True


async def test_load_runtime_without_tank_key(hass: HomeAssistant) -> None:
    # A legacy payload (pre-tank) has no "tank" key → loads fine, no tank state.
    entry = await _setup(hass)
    coordinator = entry.runtime_data.load
    sid = next(iter(coordinator.load_configs()))

    store: Store[dict] = Store(hass, STORAGE_VERSION, f"{DOMAIN}.{entry.entry_id}")
    await store.async_save({sid: {"model": {"gain": 1.1}, "training": [], "eval": []}})
    coordinator.tanks.clear()
    await coordinator.async_load_runtime()

    assert sid not in coordinator.tanks
    assert coordinator.models[sid].gain == 1.1
