"""Source warnings, Repairs lifecycle, and genuine registry renames."""

from __future__ import annotations

from datetime import timedelta

import pytest
from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import issue_registry as ir
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.load_need_predictor.const import DOMAIN, SUBENTRY_TYPE_LOAD
from custom_components.load_need_predictor.models import load_config_from_data
from custom_components.load_need_predictor.source_health import (
    SOURCE_ISSUE_GRACE,
    SourceHealthMonitor,
    appliance_state,
    async_delete_stale_source_issues,
    source_health_warnings,
)


def _entry(hass, **sources):
    hass.states.async_set("sensor.energy", "100", {"unit_of_measurement": "kWh"})
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={},
        subentries_data=[
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_LOAD,
                title="Tank",
                unique_id=None,
                data={"name": "Tank", "delivered_energy_entity": "sensor.energy", **sources},
            )
        ],
    )
    entry.add_to_hass(hass)
    return entry


def _issues(hass):
    return {
        key: value
        for key, value in ir.async_get(hass).issues.items()
        if key[0] == DOMAIN and key[1].startswith("source_unhealthy_")
    }


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("detecting", True),
        ("rinsing", True),
        ("power_off", False),
        ("end", False),
        ("unavailable", None),
        ("mystery", None),
    ],
)
def test_washer_status_semantics(state, expected):
    assert appliance_state(state) is expected


async def test_missing_appliance_warns_even_when_another_is_running(hass: HomeAssistant):
    cfg = load_config_from_data(
        {"cold_appliance_entities": ["sensor.old_washer", "binary_sensor.dishwasher"]}
    )
    hass.states.async_set("binary_sensor.dishwasher", "on")
    assert source_health_warnings(hass, cfg) == [
        "Configured source sensor.old_washer: entity is missing"
    ]


async def test_all_configured_context_sources_are_checked(hass: HomeAssistant):
    cfg = load_config_from_data(
        {
            "shower_motion_entity": "binary_sensor.motion",
            "shower_humidity_entity": "sensor.humidity",
            "shower_ventilation_entity": "input_boolean.vent",
            "cold_appliance_entities": ["sensor.washer"],
            "appliance_household_activity_entities": ["binary_sensor.kitchen"],
        }
    )
    hass.states.async_set("binary_sensor.motion", "unavailable")
    hass.states.async_set("sensor.humidity", "nan")
    hass.states.async_set("input_boolean.vent", "unknown")
    hass.states.async_set("sensor.washer", "mystery")
    warnings = source_health_warnings(hass, cfg)
    assert len(warnings) == 5
    assert any("sensor.washer: run state is unsupported" in value for value in warnings)
    assert any("binary_sensor.kitchen: entity is missing" in value for value in warnings)
    assert source_health_warnings(hass, load_config_from_data({})) == []


async def test_counter_unit_health_matches_tracker_aliases(hass: HomeAssistant):
    cfg = load_config_from_data(
        {"delivered_energy_entity": "sensor.energy", "water_total_entity": "sensor.water"}
    )
    hass.states.async_set("sensor.energy", "10", {"unit_of_measurement": "Wh"})
    hass.states.async_set("sensor.water", "100", {"unit_of_measurement": "litres"})
    assert source_health_warnings(hass, cfg) == []
    hass.states.async_set("sensor.water", "100", {"unit_of_measurement": "bananas"})
    assert len(source_health_warnings(hass, cfg)) == 1


async def test_humidity_requires_relative_humidity_unit_and_range(hass: HomeAssistant):
    cfg = load_config_from_data({"shower_humidity_entity": "sensor.humidity"})
    hass.states.async_set("sensor.humidity", "50", {"unit_of_measurement": "%"})
    assert source_health_warnings(hass, cfg) == []
    hass.states.async_set("sensor.humidity", "50", {"unit_of_measurement": "g/m³"})
    assert len(source_health_warnings(hass, cfg)) == 1
    hass.states.async_set("sensor.humidity", "101", {"unit_of_measurement": "%"})
    assert len(source_health_warnings(hass, cfg)) == 1


async def test_repairs_grace_recovery_and_recurrence(hass: HomeAssistant):
    entry = _entry(hass, cold_appliance_entities=["sensor.washer"])
    monitor = SourceHealthMonitor(hass, entry)
    now = dt_util.utcnow()
    monitor._check(now)
    assert not _issues(hass)
    monitor._check(now + SOURCE_ISSUE_GRACE - timedelta(seconds=1))
    assert not _issues(hass)
    monitor._check(now + SOURCE_ISSUE_GRACE)
    assert len(_issues(hass)) == 1
    hass.states.async_set("sensor.washer", "standby")
    monitor._check(now + SOURCE_ISSUE_GRACE + timedelta(seconds=1))
    assert not _issues(hass)
    hass.states.async_set("sensor.washer", "unavailable")
    monitor._check(now + SOURCE_ISSUE_GRACE + timedelta(seconds=2))
    assert not _issues(hass)  # a recovered source earns a new transient grace


async def test_reconfiguration_and_entry_removal_clear_repairs(hass: HomeAssistant):
    entry = _entry(hass, cold_appliance_entities=["sensor.old_washer"])
    monitor = SourceHealthMonitor(hass, entry)
    now = dt_util.utcnow()
    monitor._check(now)
    monitor._check(now + SOURCE_ISSUE_GRACE)
    assert len(_issues(hass)) == 1
    subentry = next(iter(entry.subentries.values()))
    hass.config_entries.async_update_subentry(
        entry, subentry, data={**subentry.data, "cold_appliance_entities": []}
    )
    monitor._check(now + SOURCE_ISSUE_GRACE)
    assert not _issues(hass)
    hass.config_entries.async_update_subentry(
        entry, subentry, data={**subentry.data, "cold_appliance_entities": ["sensor.new_washer"]}
    )
    monitor._check(now + SOURCE_ISSUE_GRACE)
    monitor._check(now + 2 * SOURCE_ISSUE_GRACE)
    assert len(_issues(hass)) == 1
    async_delete_stale_source_issues(hass, exclude_entry_id=entry.entry_id)
    assert not _issues(hass)


async def test_registry_rename_updates_selected_refs_without_guessing_replacements(
    hass: HomeAssistant,
):
    registry = er.async_get(hass)
    washer = registry.async_get_or_create("sensor", "test", "washer", suggested_object_id="washer")
    entry = _entry(
        hass,
        cold_appliance_entities=[washer.entity_id],
        delivered_runtime_entity=washer.entity_id,
    )
    monitor = SourceHealthMonitor(hass, entry)
    monitor.async_start()
    try:
        registry.async_update_entity(washer.entity_id, new_entity_id="sensor.washer_renamed")
        await hass.async_block_till_done()
        subentry = next(iter(entry.subentries.values()))
        assert subentry.data["cold_appliance_entities"] == ["sensor.washer_renamed"]
        assert subentry.data["delivered_runtime_entity"] == "sensor.washer_renamed"
        registry.async_get_or_create(
            "sensor", "test", "washer_status", suggested_object_id="washer_current_status"
        )
        registry.async_remove("sensor.washer_renamed")
        await hass.async_block_till_done()
        assert subentry.data["cold_appliance_entities"] == ["sensor.washer_renamed"]
    finally:
        monitor.async_stop()


async def test_existing_persistent_issue_survives_monitor_reload(hass: HomeAssistant):
    entry = _entry(hass, cold_appliance_entities=["sensor.washer"])
    now = dt_util.utcnow()
    monitor = SourceHealthMonitor(hass, entry)
    monitor._check(now)
    monitor._check(now + SOURCE_ISSUE_GRACE)
    assert len(_issues(hass)) == 1
    reloaded = SourceHealthMonitor(hass, entry)
    reloaded._check(now + SOURCE_ISSUE_GRACE)
    assert len(_issues(hass)) == 1
