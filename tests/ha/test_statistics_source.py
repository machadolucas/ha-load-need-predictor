"""Reading delivered energy + commanded runtime from the recorder."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
from homeassistant.components.recorder.statistics import statistic_during_period
from homeassistant.core import HomeAssistant, State
from homeassistant.util import dt as dt_util

from custom_components.load_need_predictor.statistics_source import (
    async_commanded_minutes,
    async_daily_delivered_kwh,
    async_statistic_change,
    capture_window,
)

_GET_INSTANCE = "homeassistant.components.recorder.get_instance"


def _window():
    end = dt_util.now()
    return end - timedelta(hours=24), end


async def test_none_when_recorder_not_set_up(hass: HomeAssistant) -> None:
    # No recorder in this test → get_instance raises KeyError → graceful None.
    result = await async_daily_delivered_kwh(hass, "sensor.e", *_window())
    assert result is None


async def test_reads_trailing_window_change_in_kwh(hass: HomeAssistant) -> None:
    """The singular statistic_during_period over the exact (unaligned) window.

    It blends 5-min short-term stats, so the 23:00–24:00 hour a 23:55 capture
    used to drop is counted; ``units`` normalises a Wh meter to kWh.
    """
    start, end = _window()
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock(return_value={"change": 5.5})
    with patch(_GET_INSTANCE, return_value=instance):
        result = await async_daily_delivered_kwh(hass, "sensor.e", start, end)
    assert result == 5.5
    func, _hass, a_start, a_end, stat_id, types, units = (
        instance.async_add_executor_job.call_args.args
    )
    assert func is statistic_during_period
    assert (a_start, a_end, stat_id) == (start, end, "sensor.e")
    assert types == {"change"}
    assert units == {"energy": "kWh"}


async def test_generic_change_keeps_native_unit(hass: HomeAssistant) -> None:
    # The water meter's delta is log-only context in its own unit (m³).
    start, end = _window()
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock(return_value={"change": 0.3})
    with patch(_GET_INSTANCE, return_value=instance):
        result = await async_statistic_change(hass, "sensor.water", start, end)
    assert result == 0.3
    assert instance.async_add_executor_job.call_args.args[-1] is None


async def test_none_when_no_statistic(hass: HomeAssistant) -> None:
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock(return_value={})
    with patch(_GET_INSTANCE, return_value=instance):
        result = await async_daily_delivered_kwh(hass, "sensor.e", *_window())
    assert result is None


async def test_none_when_change_missing(hass: HomeAssistant) -> None:
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock(return_value={"change": None})
    with patch(_GET_INSTANCE, return_value=instance):
        result = await async_daily_delivered_kwh(hass, "sensor.e", *_window())
    assert result is None


async def test_none_when_recorder_query_raises(hass: HomeAssistant) -> None:
    # A DB hiccup must degrade to "unknown", never escape into the coordinator.
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock(side_effect=RuntimeError("db locked"))
    with patch(_GET_INSTANCE, return_value=instance):
        assert await async_daily_delivered_kwh(hass, "sensor.e", *_window()) is None
        start, end = _window()
        assert await async_commanded_minutes(hass, "switch.x", start, end) is None


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


# ── capture_window (deterministic, compiled, contiguous, DST-safe) ─────────────

_HEL = ZoneInfo("Europe/Helsinki")


@pytest.fixture
def helsinki():
    previous = dt_util.get_default_time_zone()
    dt_util.set_default_time_zone(_HEL)
    yield
    dt_util.set_default_time_zone(previous)


def _hel(*args) -> datetime:
    return datetime(*args, tzinfo=_HEL)


def test_capture_window_ends_at_last_compiled_bucket(helsinki) -> None:
    # 23:55:00 → the 23:50–23:55 short-term stat isn't compiled until ~23:55:10.
    start, end = capture_window(_hel(2026, 9, 27, 23, 55, 0))
    assert end == dt_util.as_utc(_hel(2026, 9, 27, 23, 50))
    assert start == dt_util.as_utc(_hel(2026, 9, 26, 23, 50))
    assert end - start == timedelta(hours=24)  # elapsed UTC time on a normal day
    assert end.tzinfo is not None and end.utcoffset() == timedelta(0)


def test_capture_window_is_deterministic_for_unaligned_now(helsinki) -> None:
    # Seconds/µs never shift the bucket; a lock wait of a minute or two doesn't
    # either, so consecutive captures keep tiling.
    _, end = capture_window(_hel(2026, 9, 27, 23, 57, 42, 123456))
    assert end == dt_util.as_utc(_hel(2026, 9, 27, 23, 55))
    assert end.second == 0 and end.microsecond == 0


def test_consecutive_captures_tile(helsinki) -> None:
    _, yesterday_end = capture_window(_hel(2026, 9, 26, 23, 55))
    start, _ = capture_window(_hel(2026, 9, 27, 23, 55))
    assert start == yesterday_end


def test_capture_window_dst_fall_back_day_tiles_with_25_hours(helsinki) -> None:
    # 2026-10-25: EEST → EET at 04:00. Captures at a fixed local 23:55 are 25 h
    # apart that day; a fixed 24 h window would leave a 1-hour hole.
    _, yesterday_end = capture_window(_hel(2026, 10, 24, 23, 55))
    start, end = capture_window(_hel(2026, 10, 25, 23, 55))
    assert start == yesterday_end
    assert end - start == timedelta(hours=25)
    assert end == datetime(2026, 10, 25, 21, 50, tzinfo=dt_util.UTC)


def test_capture_window_dst_spring_forward_day_is_23_hours(helsinki) -> None:
    _, yesterday_end = capture_window(_hel(2026, 3, 28, 23, 55))
    start, end = capture_window(_hel(2026, 3, 29, 23, 55))
    assert start == yesterday_end
    assert end - start == timedelta(hours=23)
