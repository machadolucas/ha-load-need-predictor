"""Reading delivered energy from long-term statistics (recorder)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.load_need_predictor.statistics_source import async_daily_delivered_kwh

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
