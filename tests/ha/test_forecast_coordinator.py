"""Price-forecast coordinator: build → publish slots, and evaluate vs actual."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, patch

from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant, SupportsResponse
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.load_need_predictor.const import DOMAIN, SUBENTRY_TYPE_PRICE_FORECAST

_MOD = "custom_components.load_need_predictor.forecast_coordinator"


async def _setup(hass: HomeAssistant, forecast_days: int = 2):
    base = dt_util.start_of_local_day() + timedelta(days=1)  # local midnight tomorrow
    # Wind series: hourly 2.0 GW covering the next ~3 local days.
    data = [
        [int(dt_util.as_utc(base + timedelta(hours=h)).timestamp() * 1000), 2.0] for h in range(72)
    ]
    hass.states.async_set("sensor.wind", "2000", {"series": [{"data": data}]})

    async def _forecast(call):
        return {
            "weather.home": {
                "forecast": [
                    {
                        "datetime": dt_util.as_utc(base).isoformat(),
                        "temperature": -5,
                        "templow": -5,
                    },
                    {
                        "datetime": dt_util.as_utc(base + timedelta(days=1)).isoformat(),
                        "temperature": 0,
                        "templow": 0,
                    },
                    {
                        "datetime": dt_util.as_utc(base + timedelta(days=2)).isoformat(),
                        "temperature": 3,
                        "templow": 3,
                    },
                ]
            }
        }

    hass.services.async_register(
        "weather", "get_forecasts", _forecast, supports_response=SupportsResponse.ONLY
    )
    hass.states.async_set("sensor.price", "0.1", {"state_class": "measurement"})
    hass.states.async_set("sensor.temp", "5", {"state_class": "measurement"})

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"name": "P", "predict_time": "14:00:00", "capture_time": "23:55:00"},
        subentries_data=[
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_PRICE_FORECAST,
                title="LVV",
                unique_id=None,
                data={
                    "name": "LVV",
                    "price_entity": "sensor.price",
                    "wind_entity": "sensor.wind",
                    "weather_entity": "weather.home",
                    "temp_history_entity": "sensor.temp",
                    "forecast_days": forecast_days,
                },
            )
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry, entry.runtime_data.forecast


async def test_forecast_builds_on_setup(hass: HomeAssistant) -> None:
    # No explicit build call: setup itself should populate the forecast (so the
    # sensor isn't blank until the next predict time / after a restart).
    entry, fc = await _setup(hass)
    sid = next(iter(fc.forecast_configs()))
    assert fc.slots.get(sid)  # non-empty
    assert fc.data[sid].status == "ok"


async def test_build_produces_scheduler_shaped_slots(hass: HomeAssistant) -> None:
    entry, fc = await _setup(hass, forecast_days=2)
    with patch(f"{_MOD}.async_fit_rows", new=AsyncMock(return_value=[])):  # no history → seed
        await fc.async_build_forecast()
    sid = next(iter(fc.forecast_configs()))
    slots = fc.slots[sid]
    assert len(slots) == 48  # 2 days × 24 h
    first = slots[0]
    assert set(first) == {"start", "end", "buy"}
    start = dt_util.parse_datetime(first["start"])
    assert start.tzinfo is not None  # tz-aware, as the scheduler requires
    assert first["buy"] > 0
    # Beyond-horizon: the first slot is tomorrow's local midnight.
    assert start == dt_util.start_of_local_day() + timedelta(days=1)


async def test_build_uses_fitted_model_when_history_present(hass: HomeAssistant) -> None:
    entry, fc = await _setup(hass)
    grid = [(float(t), 2.0, 0.10) for t in range(-20, 21, 2)]  # ≥ MIN_FIT_ROWS, varied temp
    with patch(f"{_MOD}.async_fit_rows", new=AsyncMock(return_value=grid)):
        await fc.async_build_forecast()
    sid = next(iter(fc.forecast_configs()))
    assert fc.models[sid] is not None
    assert fc.models[sid].n == len(grid)
    assert fc.data[sid].model_samples == len(grid)


async def test_sensor_publishes_data_today(hass: HomeAssistant) -> None:
    entry, fc = await _setup(hass)
    with patch(f"{_MOD}.async_fit_rows", new=AsyncMock(return_value=[])):
        await fc.async_build_forecast()
    await hass.async_block_till_done()
    sid = next(iter(fc.forecast_configs()))
    reg = er.async_get(hass)
    eid = reg.async_get_entity_id("sensor", DOMAIN, f"{sid}_price_forecast")
    assert eid is not None
    state = hass.states.get(eid)
    data_today = state.attributes["data_today"]
    assert isinstance(data_today, list) and data_today
    assert {"start", "end", "buy"} <= set(data_today[0])
    assert state.attributes["status"] == "ok"


async def test_evaluate_reconciles_past_forecast(hass: HomeAssistant) -> None:
    entry, fc = await _setup(hass)
    sid = next(iter(fc.forecast_configs()))
    yesterday = dt_util.start_of_local_day() - timedelta(days=1)
    fc.log[sid] = [
        {
            "date": yesterday.date().isoformat(),
            "bucket_ms": int(yesterday.timestamp() * 1000),
            "predicted": 0.10,
            "actual": None,
            "abs_error": None,
        }
    ]
    with patch(f"{_MOD}.async_daily_price_mean", new=AsyncMock(return_value=0.13)):
        await fc.async_evaluate()
    row = fc.log[sid][0]
    assert row["actual"] == 0.13
    assert row["abs_error"] == 0.03  # |0.10 - 0.13|
    assert fc.data[sid].forecast_samples == 1
    assert fc.data[sid].last_actual == 0.13
    assert fc.data[sid].forecast_mae == 0.03


async def test_evaluate_skips_unrealised_future_day(hass: HomeAssistant) -> None:
    entry, fc = await _setup(hass)
    sid = next(iter(fc.forecast_configs()))
    tomorrow = dt_util.start_of_local_day() + timedelta(days=1)
    fc.log[sid] = [
        {
            "date": tomorrow.date().isoformat(),
            "bucket_ms": int(tomorrow.timestamp() * 1000),
            "predicted": 0.10,
            "actual": None,
            "abs_error": None,
        }
    ]
    with patch(f"{_MOD}.async_daily_price_mean", new=AsyncMock(return_value=0.13)) as mean:
        await fc.async_evaluate()
    mean.assert_not_awaited()  # the day hasn't finished yet
    assert fc.log[sid][0]["actual"] is None
