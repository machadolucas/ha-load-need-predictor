"""Read the inputs for the price forecast from Home Assistant.

Three sources, all reaching ~72 h (unlike Nord Pool's day-ahead horizon):

- **Future wind** — the ``finland_wind_forecast_average_fmi`` sensor carries an
  hourly series in ``attributes.series[0].data`` as ``[epoch_ms, GW]`` pairs.
- **Future temperature** — a daily forecast via the ``weather.get_forecasts``
  service (``temperature``/``templow`` per day).
- **History for fitting** — daily-mean long-term statistics for the price, the
  temperature and the wind sensors.

Unit note: the wind sensor's *state / LTS is in MW* (~2138) while its *series
attribute is in GW* (~2.1). We normalise everything to **GW** here so the model
sees one consistent scale.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)

# The wind series is GW; the wind state/LTS is MW. Divide MW by this to get GW.
_MW_PER_GW = 1000.0


async def async_wind_series_gw(hass: HomeAssistant, entity_id: str) -> list[tuple[datetime, float]]:
    """Future hourly wind production (UTC datetime, GW) from the sensor's series."""
    state = hass.states.get(entity_id)
    if state is None:
        return []
    series = state.attributes.get("series")
    if not isinstance(series, list) or not series:
        return []
    data = series[0].get("data") if isinstance(series[0], dict) else None
    if not isinstance(data, list):
        return []
    out: list[tuple[datetime, float]] = []
    for point in data:
        try:
            ts_ms, value = point[0], point[1]
            out.append((datetime.fromtimestamp(ts_ms / 1000, tz=UTC), float(value)))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def daily_wind_means_gw(series: list[tuple[datetime, float]]) -> dict[date, float]:
    """Mean wind (GW) per *local* calendar day from an hourly series."""
    buckets: dict[date, list[float]] = {}
    for when, value in series:
        buckets.setdefault(dt_util.as_local(when).date(), []).append(value)
    return {day: sum(vals) / len(vals) for day, vals in buckets.items()}


async def async_daily_temp_forecast(hass: HomeAssistant, weather_entity: str) -> dict[date, float]:
    """Daily mean temperature (°C) forecast keyed by local date.

    Uses ``weather.get_forecasts`` (the supported replacement for the deprecated
    ``forecast`` attribute). Mean = (high + low)/2 when a low is present.
    """
    try:
        response = await hass.services.async_call(
            "weather",
            "get_forecasts",
            {"entity_id": weather_entity, "type": "daily"},
            blocking=True,
            return_response=True,
        )
    except HomeAssistantError as err:
        _LOGGER.warning("Could not read temperature forecast from %s: %s", weather_entity, err)
        return {}
    forecasts = (response or {}).get(weather_entity, {}).get("forecast", [])
    out: dict[date, float] = {}
    for entry in forecasts:
        when = dt_util.parse_datetime(entry.get("datetime", ""))
        temp = entry.get("temperature")
        if when is None or temp is None:
            continue
        low = entry.get("templow")
        mean = (float(temp) + float(low)) / 2 if low is not None else float(temp)
        out[dt_util.as_local(when).date()] = mean
    return out


async def _daily_means(
    hass: HomeAssistant, entity_ids: list[str], start: datetime
) -> dict[str, dict[int, float]]:
    """Daily-mean statistics for several entities → {entity: {bucket_ms: mean}}."""
    try:
        from homeassistant.components.recorder import get_instance
        from homeassistant.components.recorder.statistics import statistics_during_period
    except ImportError:
        return {}
    try:
        instance = get_instance(hass)
    except KeyError:
        _LOGGER.debug("Recorder not available; cannot fit price model")
        return {}

    stats = await instance.async_add_executor_job(
        statistics_during_period,
        hass,
        start,
        None,
        set(entity_ids),
        "day",
        None,
        {"mean"},
    )
    result: dict[str, dict[int, float]] = {}
    for entity_id, series in stats.items():
        per_day: dict[int, float] = {}
        for row in series:
            mean = row.get("mean")
            if mean is not None:
                per_day[int(row["start"])] = float(mean)
        result[entity_id] = per_day
    return result


async def async_fit_rows(
    hass: HomeAssistant,
    price_entity: str,
    temp_entity: str,
    wind_entity: str,
    days: int,
) -> list[tuple[float, float, float]]:
    """Aligned daily ``(temp °C, wind GW, price €/kWh)`` rows from LTS for fitting."""
    start = dt_util.start_of_local_day() - timedelta(days=days)
    stats = await _daily_means(hass, [price_entity, temp_entity, wind_entity], start)
    price = stats.get(price_entity, {})
    temp = stats.get(temp_entity, {})
    wind = stats.get(wind_entity, {})
    rows: list[tuple[float, float, float]] = []
    for bucket, price_mean in price.items():
        if bucket in temp and bucket in wind:
            rows.append((temp[bucket], wind[bucket] / _MW_PER_GW, price_mean))
    return rows


async def async_daily_price_mean(
    hass: HomeAssistant, price_entity: str, day_start: datetime
) -> float | None:
    """Mean realised price (€/kWh) for the local day at ``day_start`` (for evaluation)."""
    stats = await _daily_means(hass, [price_entity], day_start)
    per_day = stats.get(price_entity, {})
    if not per_day:
        return None
    target_ms = int(day_start.timestamp() * 1000)
    # Prefer the exact bucket; fall back to the only/earliest bucket in range.
    if target_ms in per_day:
        return per_day[target_ms]
    return per_day[min(per_day)]
