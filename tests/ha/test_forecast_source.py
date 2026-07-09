"""Reading the forecast inputs: wind series, temp forecast, LTS fit rows."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.core import HomeAssistant, SupportsResponse
from homeassistant.util import dt as dt_util

from custom_components.load_need_predictor import forecast_source as fs

_GET_INSTANCE = "homeassistant.components.recorder.get_instance"


async def test_wind_series_parsed_to_gw(hass: HomeAssistant) -> None:
    t0 = 1781629200000  # epoch ms
    hass.states.async_set(
        "sensor.wind",
        "2138",
        {"series": [{"name": "wind", "data": [[t0, 2.1], [t0 + 3600000, 3.0]]}]},
    )
    series = await fs.async_wind_series_gw(hass, "sensor.wind")
    assert len(series) == 2
    assert series[0][1] == 2.1  # already GW
    assert series[0][0] == datetime.fromtimestamp(t0 / 1000, tz=UTC)


async def test_wind_series_missing_or_malformed(hass: HomeAssistant) -> None:
    assert await fs.async_wind_series_gw(hass, "sensor.absent") == []
    hass.states.async_set("sensor.wind", "x", {})  # no series attr
    assert await fs.async_wind_series_gw(hass, "sensor.wind") == []
    hass.states.async_set("sensor.wind2", "x", {"series": [{"data": [["bad"], [1]]}]})
    assert await fs.async_wind_series_gw(hass, "sensor.wind2") == []


async def test_wind_canonical_forecast_parsed_and_normalised_to_gw(hass: HomeAssistant) -> None:
    hass.states.async_set(
        "sensor.wind",
        "2138",
        {
            "forecast": [
                {
                    "start": "2026-06-18T00:00:00+00:00",
                    "end": "2026-06-18T01:00:00+00:00",
                    "value": 2100.0,
                },
                {
                    "start": "2026-06-18T01:00:00+00:00",
                    "end": "2026-06-18T02:00:00+00:00",
                    "value": 3000.0,
                },
            ],
            "unit": "MW",
            "source": "wind_forecast_fi",
        },
    )
    series = await fs.async_wind_series_gw(hass, "sensor.wind")
    assert len(series) == 2
    assert series[0] == (datetime(2026, 6, 18, tzinfo=UTC), 2.1)
    assert series[1] == (datetime(2026, 6, 18, 1, tzinfo=UTC), 3.0)


async def test_wind_canonical_forecast_skips_malformed_entries(hass: HomeAssistant) -> None:
    hass.states.async_set(
        "sensor.wind",
        "2138",
        {
            "forecast": [
                {"start": "2026-06-18T00:00:00+00:00", "value": 2100.0},  # good
                {"end": "2026-06-18T02:00:00+00:00", "value": 3000.0},  # missing start
                {"start": "not-a-date", "value": 3000.0},  # unparseable date
                {"start": "2026-06-18T03:00:00+00:00", "value": "not-a-number"},  # bad value
                "not-a-dict",  # wrong shape entirely
            ],
        },
    )
    series = await fs.async_wind_series_gw(hass, "sensor.wind")
    assert series == [(datetime(2026, 6, 18, tzinfo=UTC), 2.1)]


async def test_wind_canonical_forecast_wins_over_legacy_series(hass: HomeAssistant) -> None:
    t0 = 1781629200000  # epoch ms
    hass.states.async_set(
        "sensor.wind",
        "2138",
        {
            "forecast": [{"start": "2026-06-18T00:00:00+00:00", "value": 2100.0}],
            "series": [{"name": "wind", "data": [[t0, 9.9]]}],
        },
    )
    series = await fs.async_wind_series_gw(hass, "sensor.wind")
    assert series == [(datetime(2026, 6, 18, tzinfo=UTC), 2.1)]


async def test_wind_empty_canonical_forecast_falls_back_to_legacy_series(
    hass: HomeAssistant,
) -> None:
    t0 = 1781629200000  # epoch ms
    hass.states.async_set(
        "sensor.wind",
        "2138",
        {
            "forecast": [],  # present but empty → fall back
            "series": [{"name": "wind", "data": [[t0, 2.1]]}],
        },
    )
    series = await fs.async_wind_series_gw(hass, "sensor.wind")
    assert series == [(datetime.fromtimestamp(t0 / 1000, tz=UTC), 2.1)]


async def test_daily_wind_means_gw(hass: HomeAssistant) -> None:
    base = dt_util.start_of_local_day() + timedelta(days=1)  # local midnight tomorrow
    series = [
        (dt_util.as_utc(base + timedelta(hours=2)), 2.0),
        (dt_util.as_utc(base + timedelta(hours=4)), 4.0),
        (dt_util.as_utc(base + timedelta(days=1, hours=2)), 1.0),
    ]
    means = fs.daily_wind_means_gw(series)
    assert means[base.date()] == 3.0  # (2+4)/2
    assert means[(base + timedelta(days=1)).date()] == 1.0


async def test_daily_temp_forecast(hass: HomeAssistant) -> None:
    async def _forecast(call):
        return {
            "weather.home": {
                "forecast": [
                    {"datetime": "2026-06-18T00:00:00+00:00", "temperature": 5, "templow": -5},
                    {"datetime": "2026-06-19T00:00:00+00:00", "temperature": 10},
                ]
            }
        }

    hass.services.async_register(
        "weather", "get_forecasts", _forecast, supports_response=SupportsResponse.ONLY
    )
    out = await fs.async_daily_temp_forecast(hass, "weather.home")
    # Derive the expected local-date keys (the test harness tz isn't Helsinki).
    d1 = dt_util.as_local(dt_util.parse_datetime("2026-06-18T00:00:00+00:00")).date()
    d2 = dt_util.as_local(dt_util.parse_datetime("2026-06-19T00:00:00+00:00")).date()
    assert out[d1] == 0.0  # (5 + -5)/2
    assert out[d2] == 10.0  # no templow → high only


async def test_daily_temp_forecast_handles_missing_entity(hass: HomeAssistant) -> None:
    # No such service registered → call raises → graceful empty dict.
    assert await fs.async_daily_temp_forecast(hass, "weather.absent") == {}


async def test_fit_rows_aligns_and_normalises_wind(hass: HomeAssistant) -> None:
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock(
        return_value={
            "sensor.price": [
                {"start": 1000, "mean": 0.10},
                {"start": 2000, "mean": 0.20},
                {"start": 3000, "mean": 0.30},  # no temp/wind that day → dropped
            ],
            "sensor.temp": [{"start": 1000, "mean": -10.0}, {"start": 2000, "mean": 5.0}],
            "sensor.wind": [{"start": 1000, "mean": 2000.0}, {"start": 2000, "mean": 3000.0}],
        }
    )
    with patch(_GET_INSTANCE, return_value=instance):
        rows = await fs.async_fit_rows(hass, "sensor.price", "sensor.temp", "sensor.wind", days=365)
    # wind MW → GW (÷1000); only the two aligned days survive.
    assert sorted(rows) == [(-10.0, 2.0, 0.10), (5.0, 3.0, 0.20)]


async def test_fit_rows_empty_without_recorder(hass: HomeAssistant) -> None:
    rows = await fs.async_fit_rows(hass, "sensor.price", "sensor.temp", "sensor.wind", days=365)
    assert rows == []


async def test_daily_price_mean_returns_bucket(hass: HomeAssistant) -> None:
    day_start = dt_util.start_of_local_day()
    bucket = int(day_start.timestamp() * 1000)
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock(
        return_value={"sensor.price": [{"start": bucket, "mean": 0.12}]}
    )
    with patch(_GET_INSTANCE, return_value=instance):
        result = await fs.async_daily_price_mean(hass, "sensor.price", day_start)
    assert result == 0.12


async def test_daily_price_mean_none_when_empty(hass: HomeAssistant) -> None:
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock(return_value={})
    with patch(_GET_INSTANCE, return_value=instance):
        result = await fs.async_daily_price_mean(hass, "sensor.price", dt_util.start_of_local_day())
    assert result is None
