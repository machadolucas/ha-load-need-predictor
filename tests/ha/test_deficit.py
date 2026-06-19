"""Deficit carryover: a skipped/under-run cycle is made up the next day, and the
gain only learns from clean (fully-delivered, no-backlog) cycles.

``_commanded_since`` (predict path) and ``async_commanded_minutes`` (capture
path) are patched so a test controls how much "actually ran" without a recorder.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_mock_service

from custom_components.load_need_predictor.const import DOMAIN, SUBENTRY_TYPE_LOAD

_STATS = "custom_components.load_need_predictor.coordinator.async_daily_delivered_kwh"
_CMD = "custom_components.load_need_predictor.coordinator.async_commanded_minutes"

_LOAD_DATA = {
    "name": "LVV",
    "target_number_entity": "number.lvv_target",
    "delivered_energy_entity": "sensor.lvv_energy",
    "controlled_switch_entity": "switch.lvv",
    "rated_power_kw": 3.0,
    "person_entities": ["person.a", "person.b"],
    "min_minutes": 40,
    "max_minutes": 480,
}


async def _setup(hass: HomeAssistant):
    hass.states.async_set("person.a", "home")
    hass.states.async_set("person.b", "home")
    hass.states.async_set("number.lvv_target", "0")
    hass.states.async_set("switch.lvv", "off")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"name": "Predictor", "predict_time": "14:00:00", "capture_time": "23:55:00"},
        subentries_data=[
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_LOAD, title="LVV", unique_id=None, data=_LOAD_DATA
            )
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry, entry.runtime_data.load


async def test_skip_accumulates_into_next_target(hass: HomeAssistant) -> None:
    entry, coord = await _setup(hass)
    calls = async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))

    # Two predicts with nothing running between them: the first opens a cycle
    # (no prior cycle to close), the second closes the skipped one → backlog.
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=0.0)):
        await coord.async_predict_and_push()
        assert calls[-1].data["value"] == 150  # plain need (2 people → 7.4 kWh)
        await coord.async_predict_and_push()

    # Need (150) + carried backlog (~148) → a clearly larger pushed target.
    assert calls[-1].data["value"] > 150
    assert coord.models[sid].deficit_minutes > 100
    assert coord.data[sid].deficit_minutes is not None

    # Run the full ask this cycle → the backlog clears.
    ran = float(calls[-1].data["value"])
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=ran)):
        await coord.async_predict_and_push()
    assert coord.models[sid].deficit_minutes == 0.0


async def test_missing_recorder_keeps_backlog_zero(hass: HomeAssistant) -> None:
    entry, coord = await _setup(hass)
    calls = async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))

    # No commanded reading available → never close → backlog stays 0, behaviour
    # is the plain daily predictor.
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        await coord.async_predict_and_push()
        await coord.async_predict_and_push()
    assert coord.models[sid].deficit_minutes == 0.0
    assert calls[-1].data["value"] == 150


async def test_gain_learns_on_clean_cycle(hass: HomeAssistant) -> None:
    entry, coord = await _setup(hass)
    async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))

    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        await coord.async_predict_and_push()  # pushes 150, opens a cycle, no backlog

    # Full delivery (commanded ≈ ask) with no backlog → clean → the gain learns.
    with (
        patch(_STATS, new=AsyncMock(return_value=6.9)),
        patch(_CMD, new=AsyncMock(return_value=150.0)),
    ):
        await coord.async_capture_and_log()

    row = coord.training[sid][-1]
    assert row["clean_cycle"] is True
    assert coord.models[sid].sample_count == 1
    assert coord.models[sid].gain < 1.0  # 6.9 < 7.4 predicted


async def test_gain_not_learned_when_scheduler_underran(hass: HomeAssistant) -> None:
    entry, coord = await _setup(hass)
    async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))

    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        await coord.async_predict_and_push()  # pushes 150

    # The scheduler ran far less than asked (a price/solar defer): the meter
    # under-reads demand, so the day is not clean and the gain must not move.
    before = coord.models[sid].gain
    with (
        patch(_STATS, new=AsyncMock(return_value=3.0)),
        patch(_CMD, new=AsyncMock(return_value=20.0)),
    ):
        await coord.async_capture_and_log()

    row = coord.training[sid][-1]
    assert row["clean_cycle"] is False
    assert coord.models[sid].sample_count == 0
    assert coord.models[sid].gain == before


async def test_gain_not_learned_while_backlog_active(hass: HomeAssistant) -> None:
    entry, coord = await _setup(hass)
    async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))

    # Build a backlog by skipping a cycle.
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=0.0)):
        await coord.async_predict_and_push()
        await coord.async_predict_and_push()
    assert coord.models[sid].deficit_minutes > 30  # backlog in play

    # Even a full, valid delivery isn't a clean demand sample while recovering.
    before = coord.models[sid].gain
    with (
        patch(_STATS, new=AsyncMock(return_value=6.0)),
        patch(_CMD, new=AsyncMock(return_value=300.0)),
    ):
        await coord.async_capture_and_log()

    row = coord.training[sid][-1]
    assert row["clean_cycle"] is False
    assert coord.models[sid].sample_count == 0
    assert coord.models[sid].gain == before
