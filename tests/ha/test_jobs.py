"""The daily loop: predict→push→log and capture→calibrate."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, patch

from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_mock_service

from custom_components.load_need_predictor.const import DOMAIN, SUBENTRY_TYPE_LOAD
from custom_components.load_need_predictor.jobs import PredictorJobs, _parse_time

_STATS = "custom_components.load_need_predictor.coordinator.async_daily_delivered_kwh"

_LOAD_DATA = {
    "name": "LVV",
    "target_number_entity": "number.lvv_target",
    "delivered_energy_entity": "sensor.lvv_energy",
    "rated_power_kw": 3.0,
    "person_entities": ["person.a", "person.b"],
    "min_minutes": 40,
    "max_minutes": 240,
}


async def _setup(hass: HomeAssistant, *, with_target: bool = True):
    from pytest_homeassistant_custom_component.common import MockConfigEntry

    hass.states.async_set("person.a", "home")
    hass.states.async_set("person.b", "home")
    if with_target:
        hass.states.async_set("number.lvv_target", "0")
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


def test_parse_time():
    assert _parse_time("14:00:00") == (14, 0, 0)
    assert _parse_time("23:55") == (23, 55, 0)
    assert _parse_time("9") == (9, 0, 0)


async def test_predict_pushes_target_and_logs(hass: HomeAssistant) -> None:
    entry, coordinator = await _setup(hass)
    calls = async_mock_service(hass, "number", "set_value")

    await coordinator.async_predict_and_push()
    await hass.async_block_till_done()

    sid = next(iter(coordinator.load_configs()))
    assert len(calls) == 1
    assert calls[0].data["value"] == 150  # 2 people → 7.4 kWh → 150 min
    row = coordinator.training[sid][-1]
    assert row["predicted_minutes"] == 150
    assert row["actual_kwh"] is None  # not captured yet
    assert coordinator.data[sid].last_push_ok is True


async def test_capture_logs_actual_and_calibrates(hass: HomeAssistant) -> None:
    entry, coordinator = await _setup(hass)
    async_mock_service(hass, "number", "set_value")
    await coordinator.async_predict_and_push()

    with patch(_STATS, new=AsyncMock(return_value=6.9)):
        await coordinator.async_capture_and_log()
    await hass.async_block_till_done()

    sid = next(iter(coordinator.load_configs()))
    row = coordinator.training[sid][-1]
    assert row["actual_kwh"] == 6.9
    assert row["actual_minutes"] == 138  # 6.9 / 3 * 60
    assert row["abs_error_minutes"] == 12  # |150 - 138|
    assert row["data_quality"] is True
    # The model learned one observation; the gain nudged toward 6.9/7.4 < 1.
    assert coordinator.models[sid].sample_count == 1
    assert coordinator.models[sid].gain < 1.0
    # Sensors reflect the captured actual.
    assert coordinator.data[sid].last_delivered_kwh == 6.9
    assert coordinator.data[sid].prediction_error_minutes == 12
    assert coordinator.data[sid].rolling_mae_minutes == 12.0


async def test_capture_ignores_invalid_delivery(hass: HomeAssistant) -> None:
    entry, coordinator = await _setup(hass)
    async_mock_service(hass, "number", "set_value")
    await coordinator.async_predict_and_push()

    with patch(_STATS, new=AsyncMock(return_value=0.05)):  # meter-reset / outlier
        await coordinator.async_capture_and_log()

    sid = next(iter(coordinator.load_configs()))
    row = coordinator.training[sid][-1]
    assert row["data_quality"] is False
    assert coordinator.models[sid].sample_count == 0  # not learned from
    assert coordinator.data[sid].rolling_mae_minutes is None  # nothing in the ring


async def test_capture_without_prediction_records_actual_only(hass: HomeAssistant) -> None:
    entry, coordinator = await _setup(hass)
    with patch(_STATS, new=AsyncMock(return_value=6.0)):
        await coordinator.async_capture_and_log()

    sid = next(iter(coordinator.load_configs()))
    row = coordinator.training[sid][-1]
    assert row["actual_kwh"] == 6.0
    assert row.get("predicted_minutes") is None
    assert coordinator.models[sid].sample_count == 0  # no prediction → no calibration
    assert coordinator.data[sid].last_delivered_kwh == 6.0


async def test_predict_resilient_without_scheduler(hass: HomeAssistant) -> None:
    entry, coordinator = await _setup(hass, with_target=False)  # no number.lvv_target
    calls = async_mock_service(hass, "number", "set_value")

    await coordinator.async_predict_and_push()

    sid = next(iter(coordinator.load_configs()))
    assert len(calls) == 0  # nothing pushed
    assert coordinator.data[sid].last_push_ok is False
    # …but the prediction is still computed, logged and published.
    assert coordinator.training[sid][-1]["predicted_minutes"] == 150
    assert coordinator.data[sid].predicted_minutes == 150


async def test_jobs_register_and_cancel_listeners(hass: HomeAssistant) -> None:
    entry, coordinator = await _setup(hass)
    jobs = PredictorJobs(hass, entry)
    jobs.async_start()
    assert len(jobs._unsubs) == 2
    jobs.async_shutdown()
    assert jobs._unsubs == []


async def test_job_handlers_delegate_to_coordinator(hass: HomeAssistant) -> None:
    entry, coordinator = await _setup(hass)
    jobs = PredictorJobs(hass, entry)
    with (
        patch.object(coordinator, "async_predict_and_push", new=AsyncMock()) as predict,
        patch.object(coordinator, "async_capture_and_log", new=AsyncMock()) as capture,
    ):
        await jobs._handle_predict(None)
        await jobs._handle_capture(None)
    predict.assert_awaited_once()
    capture.assert_awaited_once()


async def test_capture_reads_trailing_24h_window(hass: HomeAssistant) -> None:
    """The capture window is [now − 24 h, now], not the calendar day.

    Regression: a calendar-day read at 23:55 missed the not-yet-compiled
    23:00–24:00 hour, and the next day's read was a different day.
    """
    entry, coordinator = await _setup(hass)
    stats = AsyncMock(return_value=6.9)
    with patch(_STATS, new=stats):
        await coordinator.async_capture_and_log()
    _hass, _entity, start, end = stats.call_args.args
    assert end - start == timedelta(hours=24)
    sid = next(iter(coordinator.load_configs()))
    assert coordinator.training[sid][-1]["date"] == dt_util.as_local(end).date().isoformat()


async def test_one_load_failure_does_not_block_others(hass: HomeAssistant) -> None:
    entry, coordinator = await _setup(hass)
    calls = async_mock_service(hass, "number", "set_value")
    sid = next(iter(coordinator.load_configs()))
    with (
        patch.object(coordinator, "async_build_snapshot", side_effect=RuntimeError("db")),
        patch.object(coordinator, "async_persist") as persist,
    ):
        ok = await coordinator.async_predict_and_push()
        await coordinator.async_capture_and_log()
    assert ok is False  # the failed load counts as a failed push
    assert len(calls) == 0
    assert persist.call_count == 2  # persist still runs after each job
    # The failing load kept its previous result; the coordinator stays healthy.
    assert coordinator.last_update_success is True
    assert coordinator.data[sid].predicted_minutes == 150


async def test_recorder_failure_keeps_sensors_available(hass: HomeAssistant) -> None:
    entry, coordinator = await _setup(hass)
    sid = next(iter(coordinator.load_configs()))
    with patch.object(coordinator, "_result_for", side_effect=RuntimeError("db")):
        await coordinator.async_refresh()
    assert coordinator.last_update_success is True
    assert coordinator.data[sid].predicted_minutes == 150  # previous result kept


async def test_job_steps_are_isolated(hass: HomeAssistant) -> None:
    """A load failure never skips the price-forecast step (and vice versa)."""
    entry, coordinator = await _setup(hass)
    forecast = entry.runtime_data.forecast
    jobs = PredictorJobs(hass, entry)
    boom = AsyncMock(side_effect=RuntimeError("boom"))
    with (
        patch.object(coordinator, "async_predict_and_push", new=boom),
        patch.object(coordinator, "async_capture_and_log", new=boom),
        patch.object(forecast, "async_build_forecast", new=AsyncMock()) as build,
        patch.object(forecast, "async_evaluate", new=AsyncMock()) as evaluate,
    ):
        await jobs._handle_predict(None)
        await jobs._handle_capture(None)
    build.assert_awaited_once()
    evaluate.assert_awaited_once()

    with (
        patch.object(coordinator, "async_predict_and_push", new=AsyncMock()) as predict,
        patch.object(forecast, "async_build_forecast", new=boom),
    ):
        await jobs._handle_predict(None)  # must not raise
    predict.assert_awaited_once()
