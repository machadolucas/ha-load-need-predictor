"""Reading delivered energy + commanded runtime from the recorder."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.core import HomeAssistant, State
from homeassistant.util import dt as dt_util

from custom_components.load_need_predictor.statistics_source import (
    async_commanded_minutes,
    async_daily_delivered_kwh,
)

_GET_INSTANCE = "homeassistant.components.recorder.get_instance"


async def test_none_when_recorder_not_set_up(hass: HomeAssistant) -> None:
    # No recorder in this test → get_instance raises KeyError → graceful None.
    result = await async_daily_delivered_kwh(hass, "sensor.e", dt_util.start_of_local_day())
    assert result is None


async def test_sums_daily_change(hass: HomeAssistant) -> None:
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock(
        return_value={"sensor.e": [{"change": 3.0}, {"change": 2.5}]}
    )
    with patch(_GET_INSTANCE, return_value=instance):
        result = await async_daily_delivered_kwh(hass, "sensor.e", dt_util.start_of_local_day())
    assert result == 5.5


async def test_none_when_no_series(hass: HomeAssistant) -> None:
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock(return_value={})
    with patch(_GET_INSTANCE, return_value=instance):
        result = await async_daily_delivered_kwh(hass, "sensor.e", dt_util.start_of_local_day())
    assert result is None


async def test_none_when_changes_all_missing(hass: HomeAssistant) -> None:
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock(return_value={"sensor.e": [{"change": None}]})
    with patch(_GET_INSTANCE, return_value=instance):
        result = await async_daily_delivered_kwh(hass, "sensor.e", dt_util.start_of_local_day())
    assert result is None


# ── async_commanded_minutes (switch on-time over an arbitrary window) ─────────


async def test_commanded_none_when_recorder_not_set_up(hass: HomeAssistant) -> None:
    start = dt_util.start_of_local_day()
    result = await async_commanded_minutes(hass, "switch.x", start, start + timedelta(hours=6))
    assert result is None


async def test_commanded_none_for_empty_window(hass: HomeAssistant) -> None:
    # end <= start is rejected before touching the recorder.
    start = dt_util.start_of_local_day()
    assert await async_commanded_minutes(hass, "switch.x", start, start) is None


async def test_commanded_sums_on_time(hass: HomeAssistant) -> None:
    start = dt_util.start_of_local_day()
    end = start + timedelta(hours=2)
    states = [
        State("switch.x", "off", last_changed=start),
        State("switch.x", "on", last_changed=start + timedelta(minutes=30)),
        State("switch.x", "off", last_changed=start + timedelta(minutes=90)),
    ]
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock(return_value={"switch.x": states})
    with patch(_GET_INSTANCE, return_value=instance):
        result = await async_commanded_minutes(hass, "switch.x", start, end)
    assert result == 60.0  # on from +30 to +90 = 60 minutes


async def test_commanded_clips_trailing_on_state_to_window_end(hass: HomeAssistant) -> None:
    start = dt_util.start_of_local_day()
    end = start + timedelta(hours=1)
    # On since before the window opened and never turned off → counts start→end.
    states = [State("switch.x", "on", last_changed=start - timedelta(hours=3))]
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock(return_value={"switch.x": states})
    with patch(_GET_INSTANCE, return_value=instance):
        result = await async_commanded_minutes(hass, "switch.x", start, end)
    assert result == 60.0


async def test_commanded_none_when_no_history(hass: HomeAssistant) -> None:
    start = dt_util.start_of_local_day()
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock(return_value={})
    with patch(_GET_INSTANCE, return_value=instance):
        result = await async_commanded_minutes(hass, "switch.x", start, start + timedelta(hours=1))
    assert result is None
