"""Read the inputs for the price forecast from Home Assistant.

Three sources, all reaching ~72 h (unlike Nord Pool's day-ahead horizon):

- **Future wind** — the expected wind entity is a ``wind_forecast_fi`` sensor
  carrying its canonical hourly series in ``attributes.forecast`` as
  ``[{"start": <ISO8601 tz-aware>, "end": ..., "value": <MW>}, ...]``. The
  legacy ``finland_wind_forecast_average_fmi`` REST sensor's
  ``attributes.series[0].data`` as ``[epoch_ms, GW]`` pairs is kept as a
  fallback for entities that haven't migrated yet.
- **Future temperature** — a daily forecast via the ``weather.get_forecasts``
  service (``temperature``/``templow`` per day).
- **History for fitting** — daily-mean long-term statistics for the price, the
  temperature and the wind sensors.

Plus the primary source, the **Wattcast** spot forecast (``wattcast.eu``, free,
no key): one HTTP GET per hour at most, parsed by the pure ``spot_forecast``
module. The fetch never raises — failures come back as a :class:`WattcastFetch`
with the error (and any ``Retry-After``) so the coordinator can back off and
keep serving its cache.

Unit note: the wind sensor's *state / LTS is in MW* (~2138) — both the new and
legacy sensors keep MW there — while the *forecast series is normalised to
GW* here so the model sees one consistent scale.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import aiohttp
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import dt as dt_util

from .const import WATTCAST_HOURS, WATTCAST_TIMEOUT_S, WATTCAST_URL
from .spot_forecast import WattcastSeries, parse_wattcast

_LOGGER = logging.getLogger(__name__)

# The wind series is GW; the wind state/LTS is MW. Divide MW by this to get GW.
_MW_PER_GW = 1000.0


def _parse_canonical_forecast_gw(forecast: list) -> list[tuple[datetime, float]]:
    """Parse ``wind_forecast_fi``'s canonical ``forecast`` attribute to (UTC, GW)."""
    out: list[tuple[datetime, float]] = []
    for entry in forecast:
        try:
            when = dt_util.parse_datetime(entry["start"])
            if when is None:
                continue
            value = float(entry["value"])
        except (TypeError, ValueError, KeyError):
            continue
        out.append((dt_util.as_utc(when), value / _MW_PER_GW))
    return out


def _parse_legacy_series_gw(attributes: Mapping) -> list[tuple[datetime, float]]:
    """Parse the legacy REST sensor's ``series[0].data`` [epoch_ms, GW] pairs."""
    series = attributes.get("series")
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


async def async_wind_series_gw(hass: HomeAssistant, entity_id: str) -> list[tuple[datetime, float]]:
    """Future hourly wind production (UTC datetime, GW) from the wind entity.

    Prefers the canonical ``wind_forecast_fi`` ``forecast`` attribute (MW,
    normalised to GW); falls back to the legacy REST sensor's GW ``series``.
    """
    state = hass.states.get(entity_id)
    if state is None:
        return []
    forecast = state.attributes.get("forecast")
    if isinstance(forecast, list) and forecast:
        return _parse_canonical_forecast_gw(forecast)
    return _parse_legacy_series_gw(state.attributes)


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


# ── Hourly statistics (intraday shape + hourly evaluation) ──────────────────


async def async_hourly_price_rows(
    hass: HomeAssistant, price_entity: str, start: datetime, end: datetime | None = None
) -> list[tuple[datetime, float]]:
    """Hourly-mean realised price rows ``(local hour start, €/kWh)`` from LTS."""
    try:
        from homeassistant.components.recorder import get_instance
        from homeassistant.components.recorder.statistics import statistics_during_period
    except ImportError:
        return []
    try:
        instance = get_instance(hass)
    except KeyError:
        return []
    stats = await instance.async_add_executor_job(
        statistics_during_period, hass, start, end, {price_entity}, "hour", None, {"mean"}
    )
    rows: list[tuple[datetime, float]] = []
    for row in stats.get(price_entity, []):
        mean = row.get("mean")
        if mean is None:
            continue
        rows.append((dt_util.as_local(datetime.fromtimestamp(row["start"], tz=UTC)), float(mean)))
    return rows


# ── Wattcast ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WattcastFetch:
    """Outcome of one fetch: a parsed series, or why there isn't one."""

    series: WattcastSeries | None
    error: str | None = None
    retry_after_s: int | None = None  # from a 429's Retry-After header


async def async_fetch_wattcast(hass: HomeAssistant, zone: str) -> WattcastFetch:
    """GET the 15-min forecast for ``zone``. Never raises."""
    session = async_get_clientsession(hass)
    params = {"zone": zone, "resolution": "15min", "hours": str(WATTCAST_HOURS)}
    try:
        async with session.get(
            WATTCAST_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=WATTCAST_TIMEOUT_S),
            headers={"Accept": "application/json"},
        ) as resp:
            if resp.status == 429:
                retry = resp.headers.get("Retry-After")
                return WattcastFetch(
                    None,
                    "rate limited (HTTP 429)",
                    int(retry) if retry and retry.isdigit() else None,
                )
            if resp.status != 200:
                return WattcastFetch(None, f"HTTP {resp.status}")
            payload = await resp.json(content_type=None)
    except TimeoutError:
        return WattcastFetch(None, "timeout")
    except (aiohttp.ClientError, ValueError) as err:
        return WattcastFetch(None, f"{type(err).__name__}: {err}")
    series = parse_wattcast(payload)
    if series is None:
        return WattcastFetch(None, "unparseable response")
    return WattcastFetch(series)


# ── Real price series (Nord Pool-shaped slot lists) ─────────────────────────

_SERIES_ATTRS = ("data_yesterday", "data_today", "data_tomorrow")


def real_price_slots(state: State | None) -> list[tuple[datetime, datetime, float]]:
    """``(start UTC, end UTC, buy €/kWh)`` from a slot-list price entity's attrs."""
    if state is None:
        return []
    out: list[tuple[datetime, datetime, float]] = []
    for attr in _SERIES_ATTRS:
        items = state.attributes.get(attr)
        if not isinstance(items, list):
            continue
        for item in items:
            try:
                start = dt_util.parse_datetime(str(item["start"]))
                end = dt_util.parse_datetime(str(item["end"])) if item.get("end") else None
                buy = float(item["buy"])
            except (TypeError, ValueError, KeyError):
                continue
            if start is None or start.tzinfo is None:
                continue
            start = dt_util.as_utc(start)
            end = dt_util.as_utc(end) if end is not None else start + timedelta(minutes=15)
            out.append((start, end, buy))
    out.sort(key=lambda row: row[0])
    return out


def retail_pairs(
    real: list[tuple[datetime, datetime, float]], series: WattcastSeries | None
) -> list[tuple[int, float, float]]:
    """``(start ms, spot €/MWh, buy €/kWh)`` where a real buy meets a settled spot."""
    if series is None or not real:
        return []
    spot = {int(p.start.timestamp() * 1000): p.eur_mwh for p in series.known}
    step_ms = series.slot_minutes * 60_000
    pairs: list[tuple[int, float, float]] = []
    for start, _end, buy in real:
        ms = int(start.timestamp() * 1000)
        # An hourly Wattcast series covers four real quarters.
        value = spot.get(ms)
        if value is None and step_ms > 900_000:
            value = spot.get(ms - ms % step_ms)
        if value is not None:
            pairs.append((ms, value, buy))
    return pairs


def current_pair(
    state: State | None, series: WattcastSeries | None, now: datetime
) -> tuple[int, float, float] | None:
    """Fallback pair: the buy-price sensor's current value vs the spot now."""
    if state is None or series is None:
        return None
    try:
        buy = float(state.state)
    except (TypeError, ValueError):
        return None
    step = timedelta(minutes=series.slot_minutes)
    for point in series.known:
        if point.start <= now < point.start + step:
            ms = int(point.start.timestamp() * 1000)
            return (ms, point.eur_mwh, buy)
    return None
