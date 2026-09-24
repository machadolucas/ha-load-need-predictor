"""Wattcast path of the price forecast: fetch, mapping, cache, backoff, eval.

- ``forecast_source.async_fetch_wattcast`` is driven through ``aioclient_mock``
  (never the network).
- The coordinator tests reuse the autouse ``no_wattcast_network`` mock from
  ``conftest.py`` (patched at ``forecast_coordinator.async_fetch_wattcast``) and
  just swap its ``return_value``. Time is frozen with ``freezer`` at the
  fixture's capture instant (2026-09-24 08:42 UTC = 11:42 Helsinki) and the HA
  time zone is set to Europe/Helsinki so local-day boundaries are realistic.
- The synthetic payload mirrors the real API shape: ``known`` = the CET delivery
  day of 2026-09-24 (01:00 → 01:00 local), ``forecast`` = contiguous quarters
  from there. A fake Nord Pool entity carries the *same* quarters with
  ``buy = 1.255·spot/1000 + 0.087`` (VAT 25.5 % + a flat margin), so the learned
  mapping has a known answer.
"""

from __future__ import annotations

import copy
import json
import math
import pathlib
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant, State, SupportsResponse
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import issue_registry as ir
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)

from custom_components.load_need_predictor.const import (
    DOMAIN,
    SUBENTRY_TYPE_PRICE_FORECAST,
    WATTCAST_URL,
)
from custom_components.load_need_predictor.forecast_coordinator import _next_issue_fetch
from custom_components.load_need_predictor.forecast_source import (
    WattcastFetch,
    async_fetch_wattcast,
    current_pair,
    real_price_slots,
    retail_pairs,
)
from custom_components.load_need_predictor.sensor import PriceForecastSensor
from custom_components.load_need_predictor.spot_forecast import parse_wattcast

_MOD = "custom_components.load_need_predictor.forecast_coordinator"
_FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "wattcast_fi_15min.json"

_NOW = datetime(2026, 9, 24, 8, 42, tzinfo=UTC)  # 11:42 local (EEST, +03:00)
_KNOWN_START = datetime(2026, 9, 23, 22, 0, tzinfo=UTC)  # CET day → 01:00 local
_KNOWN_Q = 96
_FC_START = _KNOWN_START + timedelta(minutes=15 * _KNOWN_Q)  # 2026-09-25 01:00 local
_Q = timedelta(minutes=15)
_SLOPE, _MARGIN = 1.255, 0.087
_RAW_DELTA = -10.0  # p50Raw = p50 − 10 (as if every slot carried an LLM adjustment)


# ── synthetic payload / states ───────────────────────────────────────────────


def _spot(i: int) -> float:
    # Varies within every hour (the mapping fit demeans per hour bucket) and
    # stays positive (single-slope regime).
    return round(40 + 25 * math.sin(i / 5) + (i % 4) * 3, 2)


def _z(when: datetime) -> str:
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


def _payload(forecast_quarters: int = 672, made_at: datetime | None = None) -> dict:
    made = made_at or _NOW - timedelta(minutes=17)
    known = []
    for i in range(_KNOWN_Q):
        t = _KNOWN_START + i * _Q
        spot = _spot(i)
        known.append(
            {"ts": int(t.timestamp()), "startsAt": _z(t), "eurMwh": spot, "ctKwh": spot / 10}
        )
    forecast = []
    for i in range(forecast_quarters):
        t = _FC_START + i * _Q
        p50 = _spot(_KNOWN_Q + i)
        forecast.append(
            {
                "ts": int(t.timestamp()),
                "startsAt": _z(t),
                "k": 1 + i // 96,
                "p10": round(p50 - 15, 2),
                "p50": p50,
                "p90": round(p50 + 30, 2),
                "p50CtKwh": p50 / 10,
                "p50Raw": round(p50 + _RAW_DELTA, 2),
            }
        )
    return {
        "zone": "FI",
        "resolution": "15min",
        "unit": "EUR/MWh",
        "now": int(_NOW.timestamp()),
        "madeAt": int(made.timestamp()),
        "madeAtIso": _z(made),
        "known": known,
        "forecast": forecast,
    }


def _ok(**kwargs) -> WattcastFetch:
    return WattcastFetch(parse_wattcast(_payload(**kwargs)))


_FAIL = WattcastFetch(None, "timeout")


def _buy(spot: float) -> float:
    return round(_SLOPE * spot / 1000 + _MARGIN, 5)


def _slot_items(start: datetime, spots: list[float]) -> list[dict]:
    items = []
    for i, spot in enumerate(spots):
        t = start + i * _Q
        items.append(
            {
                "start": dt_util.as_local(t).isoformat(),
                "end": dt_util.as_local(t + _Q).isoformat(),
                "buy": _buy(spot),
                "sell": round(spot / 1000, 5),
            }
        )
    return items


def _nordpool_attrs() -> dict:
    return {"data_today": _slot_items(_KNOWN_START, [_spot(i) for i in range(_KNOWN_Q)])}


def _local_inputs(hass: HomeAssistant) -> None:
    """Weather + wind so the local fallback covers ~12 days from today."""
    today_utc = dt_util.as_utc(dt_util.start_of_local_day())
    data = [[int((today_utc + timedelta(hours=h)).timestamp() * 1000), 2.0] for h in range(24 * 12)]
    hass.states.async_set("sensor.wind", "2000", {"series": [{"data": data}]})

    async def _forecast(call):
        return {
            "weather.home": {
                "forecast": [
                    {
                        "datetime": (today_utc + timedelta(days=d)).isoformat(),
                        "temperature": 5 + d,
                        "templow": 5 + d,
                    }
                    for d in range(12)
                ]
            }
        }

    hass.services.async_register(
        "weather", "get_forecasts", _forecast, supports_response=SupportsResponse.ONLY
    )


_LOCAL = {"wind_entity": "sensor.wind", "weather_entity": "weather.home"}


@pytest.fixture
def wattcast(no_wattcast_network):
    """The coordinator's fetch mock, defaulting to a good synthetic response."""
    no_wattcast_network.return_value = _ok()
    return no_wattcast_network


async def _setup(
    hass: HomeAssistant,
    freezer,
    *,
    local: bool = False,
    series_entity: bool = True,
    **data,
):
    await hass.config.async_set_time_zone("Europe/Helsinki")
    freezer.move_to(_NOW)
    hass.states.async_set("sensor.price", "0.14", {"state_class": "measurement"})
    hass.states.async_set("sensor.nordpool", "0.1", _nordpool_attrs())
    sub = {"name": "LVV", "price_entity": "sensor.price"}
    if series_entity:
        sub["price_series_entity"] = "sensor.nordpool"
    if local:
        _local_inputs(hass)
        sub.update(_LOCAL)
    sub.update(data)
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"name": "P", "predict_time": "14:00:00", "capture_time": "23:55:00"},
        subentries_data=[
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_PRICE_FORECAST, title="LVV", unique_id=None, data=sub
            )
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    fc = entry.runtime_data.forecast
    return entry, fc, next(iter(fc.forecast_configs()))


def _sensor_id(hass: HomeAssistant, sid: str, key: str) -> str:
    eid = er.async_get(hass).async_get_entity_id("sensor", DOMAIN, f"{sid}_{key}")
    assert eid is not None, f"sensor {key} not registered"
    return eid


def _local_iso(when: datetime) -> str:
    return dt_util.as_local(when).isoformat()


# ── async_fetch_wattcast (aioclient_mock) ────────────────────────────────────


async def test_fetch_parses_fixture_with_expected_query(hass: HomeAssistant, aioclient_mock):
    aioclient_mock.get(WATTCAST_URL, json=json.loads(_FIXTURE.read_text()))
    outcome = await async_fetch_wattcast(hass, "FI")
    assert outcome.error is None
    series = outcome.series
    assert series is not None
    assert series.slot_minutes == 15
    assert len(series.known) == 8
    assert len(series.forecast) == 24
    assert series.made_at == datetime(2026, 9, 24, 8, 25, tzinfo=UTC)
    assert series.known[0].start == datetime(2026, 9, 23, 21, 0, tzinfo=UTC)
    assert series.known[0].eur_mwh == 35.5
    # LLM-adjusted slot keeps its raw median; unadjusted ones fall back to p50.
    assert series.forecast[-1].p50_raw == 24.07
    assert series.forecast[0].p50_raw == series.forecast[0].p50

    assert aioclient_mock.call_count == 1
    _method, url, _data, _headers = aioclient_mock.mock_calls[0]
    assert str(url).startswith(WATTCAST_URL)
    assert dict(url.query) == {"zone": "FI", "resolution": "15min", "hours": "216"}


async def test_fetch_passes_zone(hass: HomeAssistant, aioclient_mock):
    aioclient_mock.get(WATTCAST_URL, json=_payload())
    outcome = await async_fetch_wattcast(hass, "EE")
    assert outcome.series is not None
    assert aioclient_mock.mock_calls[0][1].query["zone"] == "EE"


@pytest.mark.parametrize(
    ("header", "expected"),
    [({"Retry-After": "120"}, 120), ({}, None), ({"Retry-After": "Wed, 21 Oct 2026"}, None)],
)
async def test_fetch_429_reports_retry_after(hass: HomeAssistant, aioclient_mock, header, expected):
    aioclient_mock.get(WATTCAST_URL, status=429, headers=header, text="slow down")
    outcome = await async_fetch_wattcast(hass, "FI")
    assert outcome.series is None
    assert "429" in outcome.error
    assert outcome.retry_after_s == expected


async def test_fetch_http_error(hass: HomeAssistant, aioclient_mock):
    aioclient_mock.get(WATTCAST_URL, status=500, text="boom")
    outcome = await async_fetch_wattcast(hass, "FI")
    assert outcome.series is None
    assert outcome.error == "HTTP 500"
    assert outcome.retry_after_s is None


@pytest.mark.parametrize(
    "exc",
    [TimeoutError(), aiohttp.ServerTimeoutError("read"), aiohttp.ConnectionTimeoutError("c")],
)
async def test_fetch_timeout(hass: HomeAssistant, aioclient_mock, exc):
    # asyncio's own timeout (the ClientTimeout) and aiohttp's timeout errors
    # (which subclass TimeoutError) all read as a plain timeout.
    aioclient_mock.get(WATTCAST_URL, exc=exc)
    outcome = await async_fetch_wattcast(hass, "FI")
    assert outcome == WattcastFetch(None, "timeout")


async def test_fetch_client_error(hass: HomeAssistant, aioclient_mock):
    aioclient_mock.get(WATTCAST_URL, exc=aiohttp.ClientConnectionError("refused"))
    outcome = await async_fetch_wattcast(hass, "FI")
    assert outcome.series is None
    assert outcome.error.startswith("ClientConnectionError")


async def test_fetch_bad_json(hass: HomeAssistant, aioclient_mock):
    aioclient_mock.get(WATTCAST_URL, text="<html>maintenance</html>")
    outcome = await async_fetch_wattcast(hass, "FI")
    assert outcome.series is None
    assert outcome.error  # a decode error, reported not raised


async def test_fetch_unusable_payload(hass: HomeAssistant, aioclient_mock):
    aioclient_mock.get(WATTCAST_URL, json={"zone": "FI", "known": [], "forecast": []})
    outcome = await async_fetch_wattcast(hass, "FI")
    assert outcome == WattcastFetch(None, "unparseable response")


# ── Nord Pool-shaped helpers ─────────────────────────────────────────────────


def _np_item(start: str, end: str | None, buy) -> dict:
    item = {"start": start, "buy": buy}
    if end is not None:
        item["end"] = end
    return item


def test_real_price_slots_merges_lists_and_skips_garbage() -> None:
    state = State(
        "sensor.nordpool",
        "0.1",
        {
            "data_tomorrow": [_np_item("2026-09-25T01:00:00+03:00", None, 0.2)],
            "data_today": [
                _np_item("2026-09-24T01:15:00+03:00", "2026-09-24T01:30:00+03:00", "0.11"),
                _np_item("2026-09-24T01:00:00+03:00", "2026-09-24T01:15:00+03:00", 0.1),
                {"start": "2026-09-24T01:30:00+03:00"},  # no buy
                _np_item("2026-09-24T01:45:00", None, 0.1),  # naive → ambiguous, skipped
                _np_item("not a date", None, 0.1),
            ],
            "data_yesterday": "not a list",
        },
    )
    rows = real_price_slots(state)
    assert [r[0] for r in rows] == [
        datetime(2026, 9, 23, 22, 0, tzinfo=UTC),
        datetime(2026, 9, 23, 22, 15, tzinfo=UTC),
        datetime(2026, 9, 24, 22, 0, tzinfo=UTC),
    ]
    assert rows[1][2] == 0.11  # numeric strings accepted
    # A missing end defaults to one quarter.
    assert rows[2][1] - rows[2][0] == _Q
    assert all(r[0].tzinfo == UTC for r in rows)
    assert real_price_slots(None) == []


def test_retail_pairs_match_by_start() -> None:
    series = parse_wattcast(_payload(forecast_quarters=4))
    real = [
        (_KNOWN_START, _KNOWN_START + _Q, 0.1),
        (_KNOWN_START + _Q, _KNOWN_START + 2 * _Q, 0.12),
        (_FC_START, _FC_START + _Q, 0.3),  # no settled spot there → no pair
    ]
    pairs = retail_pairs(real, series)
    assert pairs == [
        (int(_KNOWN_START.timestamp() * 1000), _spot(0), 0.1),
        (int((_KNOWN_START + _Q).timestamp() * 1000), _spot(1), 0.12),
    ]
    assert retail_pairs(real, None) == []
    assert retail_pairs([], series) == []


def test_retail_pairs_hourly_series_covers_quarters() -> None:
    hour = datetime(2026, 9, 24, 10, 0, tzinfo=UTC)
    series = parse_wattcast(
        {"resolution": "hour", "known": [{"ts": int(hour.timestamp()), "eurMwh": 50.0}]}
    )
    assert series.slot_minutes == 60
    real = [(hour + i * _Q, hour + (i + 1) * _Q, 0.1 + i / 100) for i in range(5)]
    pairs = retail_pairs(real, series)
    assert [p[1] for p in pairs] == [50.0] * 4  # the 5th quarter is the next hour
    assert [p[2] for p in pairs] == pytest.approx([0.1, 0.11, 0.12, 0.13])


def test_current_pair() -> None:
    series = parse_wattcast(_payload(forecast_quarters=4))
    now = _KNOWN_START + timedelta(minutes=37)  # inside quarter #2
    pair = current_pair(State("sensor.price", "0.0915"), series, now)
    assert pair == (int((_KNOWN_START + 2 * _Q).timestamp() * 1000), _spot(2), 0.0915)
    assert current_pair(State("sensor.price", "unavailable"), series, now) is None
    assert current_pair(None, series, now) is None
    assert current_pair(State("sensor.price", "0.1"), None, now) is None
    # Outside the settled range (e.g. in the forecast part) → no pair.
    assert current_pair(State("sensor.price", "0.1"), series, _FC_START + _Q) is None


# ── coordinator: fetch → published slots ─────────────────────────────────────


async def test_success_publishes_15min_wattcast_slots(hass: HomeAssistant, freezer, wattcast):
    entry, fc, sid = await _setup(hass, freezer)
    wattcast.assert_awaited_once()
    assert wattcast.await_args.args[1] == "FI"

    slots = fc.slots[sid]
    # From the first unsettled slot (real prices end 01:00 local tomorrow) to
    # local midnight after day 7 (today = day 0): 7 days − 1 h = 668 quarters.
    assert fc.known_until[sid] == _FC_START
    assert slots[0]["start"] == "2026-09-25T01:00:00+03:00"
    assert slots[-1]["end"] == "2026-10-02T00:00:00+03:00"
    assert len(slots) == 668
    for prev, cur in zip(slots, slots[1:], strict=False):
        assert prev["end"] == cur["start"]  # contiguous
    assert {s["src"] for s in slots} == {"wattcast"}
    assert set(slots[0]) == {"start", "end", "buy", "p10", "p90", "src"}
    assert all(s["p10"] <= s["buy"] <= s["p90"] for s in slots)

    result = fc.data[sid]
    assert result.status == "ok"
    assert result.source == "wattcast"
    assert dt_util.parse_datetime(result.known_until) == _FC_START
    assert result.stale is False
    assert result.cache_age_h == 0
    assert dt_util.parse_datetime(result.wattcast_made_at) == datetime(
        2026, 9, 24, 8, 25, tzinfo=UTC
    )
    assert result.fetch_error is None


async def test_start_is_now_without_real_prices(hass: HomeAssistant, freezer, wattcast):
    # Nothing marks where real prices end → the start is just "now" floored.
    entry, fc, sid = await _setup(hass, freezer, series_entity=False, forecast_days=1)
    wattcast.return_value = WattcastFetch(
        parse_wattcast({**_payload(), "known": []})  # no settled spot at all
    )
    freezer.tick(timedelta(minutes=20))  # past the button's 15-min refetch guard
    await fc.async_build_forecast(only=sid, fetch=True)
    assert wattcast.await_count == 2
    # No known[] → no real end; start = 08:30Z floored, and Wattcast's forecast
    # only begins 22:00Z, so the published series begins there.
    assert fc.known_until[sid] is None
    assert fc.data[sid].known_until is None
    assert fc.slots[sid][0]["start"] == _local_iso(_FC_START)
    assert fc.slots[sid][-1]["end"] == "2026-09-26T00:00:00+03:00"


async def test_real_prices_in_the_past_do_not_pin_start(hass: HomeAssistant, freezer, wattcast):
    # A stale series entity (its last slot already over) → start = now floored.
    entry, fc, sid = await _setup(hass, freezer, local=True, forecast_days=1)
    hass.states.async_set(
        "sensor.nordpool",
        "0.1",
        {"data_today": _slot_items(_KNOWN_START, [_spot(i) for i in range(8)])},
    )
    freezer.tick(timedelta(minutes=5))
    await fc.async_tick()  # real end moved (backwards) → rebuild
    assert fc.known_until[sid] == _KNOWN_START + 8 * _Q
    # 08:47Z floored → 08:45Z = 11:45 local; the local model fills today until
    # Wattcast's forecast takes over at 01:00 tomorrow.
    assert fc.slots[sid][0]["start"] == "2026-09-24T11:45:00+03:00"
    assert fc.slots[sid][0]["src"] == "local"
    first_wc = next(s for s in fc.slots[sid] if s["src"] == "wattcast")
    assert first_wc["start"] == _local_iso(_FC_START)


async def test_mapping_learned_from_series_pairs(hass: HomeAssistant, freezer, wattcast):
    entry, fc, sid = await _setup(hass, freezer)
    mapping = fc.mapping[sid]
    assert mapping.n == _KNOWN_Q
    assert len(fc.pairs[sid]) == _KNOWN_Q
    assert mapping.slope_pos == pytest.approx(_SLOPE, abs=5e-3)
    assert mapping.base == pytest.approx(_MARGIN, abs=5e-4)
    assert mapping.mae < 1e-4
    # The published buy is the mapped p50.
    first = fc.slots[sid][0]
    expected = _SLOPE * _spot(_KNOWN_Q) / 1000 + _MARGIN
    assert first["buy"] == pytest.approx(expected, abs=2e-4)

    attr = fc.data[sid].retail_mapping
    assert set(attr) == {"slope_pos", "slope_neg", "base", "n", "mae"}
    assert attr["n"] == _KNOWN_Q


async def test_mapping_seed_with_few_fixture_pairs(hass: HomeAssistant, freezer, wattcast):
    # The trimmed real fixture has only 8 settled quarters: below the 24-pair
    # threshold the slope stays VAT-only and just the intercept is learned.
    fixture = json.loads(_FIXTURE.read_text())
    series = parse_wattcast(fixture)
    wattcast.return_value = WattcastFetch(series)
    await hass.config.async_set_time_zone("Europe/Helsinki")
    freezer.move_to(_NOW)
    items = [
        {
            "start": _local_iso(p.start),
            "end": _local_iso(p.start + _Q),
            "buy": _buy(p.eur_mwh),
        }
        for p in series.known
    ]
    entry, fc, sid = await _setup(hass, freezer)
    hass.states.async_set("sensor.nordpool", "0.1", {"data_today": items})
    fc.pairs[sid] = []
    fc.fetch[sid].next_fetch = _NOW  # force a fetch → pairs + refit
    await fc.async_tick()
    mapping = fc.mapping[sid]
    assert mapping.n == 8
    assert mapping.slope_pos == pytest.approx(1.255)
    assert mapping.slope_neg == pytest.approx(1.255)
    assert mapping.base == pytest.approx(_MARGIN, abs=1e-5)
    assert mapping.offsets == {}


async def test_mapping_falls_back_to_current_price_pair(hass: HomeAssistant, freezer, wattcast):
    # No real-price series: one pair per fetch from the buy sensor's current
    # value vs the settled spot covering "now"; real prices end at known_end.
    entry, fc, sid = await _setup(hass, freezer, series_entity=False)
    idx = int((_NOW - _KNOWN_START) // _Q)  # the quarter holding 08:42Z
    assert fc.pairs[sid] == [[int((_KNOWN_START + idx * _Q).timestamp() * 1000), _spot(idx), 0.14]]
    mapping = fc.mapping[sid]
    assert mapping.n == 1
    assert mapping.slope_pos == pytest.approx(1.255)
    assert mapping.base == pytest.approx(0.14 - 1.255 * _spot(idx) / 1000)
    assert fc.known_until[sid] == _FC_START  # Wattcast's known_end


async def test_fallback_to_local_beyond_wattcast_coverage(hass: HomeAssistant, freezer, wattcast):
    wattcast.return_value = _ok(forecast_quarters=192)  # → 2026-09-27 01:00 local
    entry, fc, sid = await _setup(hass, freezer, local=True, forecast_days=3)
    slots = fc.slots[sid]
    coverage_end = _local_iso(_FC_START + 192 * _Q)
    assert coverage_end == "2026-09-27T01:00:00+03:00"
    srcs = [s["src"] for s in slots]
    assert srcs == ["wattcast"] * 192 + ["local"] * 92  # 01:00 → midnight = 23 h
    first_local = slots[192]
    assert first_local["start"] == coverage_end
    assert "p10" not in first_local and "p90" not in first_local
    assert slots[-1]["end"] == "2026-09-28T00:00:00+03:00"
    for prev, cur in zip(slots, slots[1:], strict=False):
        assert prev["end"] == cur["start"]

    days = {d["date"]: d for d in fc.data[sid].days}
    assert list(days) == ["2026-09-25", "2026-09-26", "2026-09-27"]
    assert days["2026-09-25"]["src"] == "wattcast"
    assert days["2026-09-27"]["src"] == "local"  # 92 local vs 4 wattcast quarters
    for day in days.values():
        assert {"date", "mean", "min", "max", "src", "slots"} <= set(day)
        assert day["min"] <= day["mean"] <= day["max"]
        # The local model's inputs are attached where it had them.
        assert day["wind_src"] == "forecast"
        assert "temp" in day and "wind_gw" in day


async def test_use_wattcast_false_never_fetches(hass: HomeAssistant, freezer, wattcast):
    entry, fc, sid = await _setup(hass, freezer, local=True, use_wattcast=False)
    for _ in range(3):
        freezer.tick(timedelta(hours=1))
        async_fire_time_changed(hass)  # the 5-min due-check ticker
        await hass.async_block_till_done()
        await fc.async_tick()
    await fc.async_build_forecast(only=sid, fetch=True)  # even the button
    wattcast.assert_not_awaited()
    assert fc.data[sid].source == "local"
    assert {s["src"] for s in fc.slots[sid]} == {"local"}
    # The real-price series still marks where the forecast starts.
    assert fc.slots[sid][0]["start"] == "2026-09-25T01:00:00+03:00"


# ── cache, staleness, backoff, repair issue ──────────────────────────────────


async def test_failure_keeps_cache_and_marks_stale(hass: HomeAssistant, freezer, wattcast):
    entry, fc, sid = await _setup(hass, freezer)
    cached = fc.wattcast[sid]
    before = copy.deepcopy(fc.slots[sid])
    wattcast.return_value = _FAIL

    freezer.tick(timedelta(minutes=61))
    await fc.async_tick()
    assert wattcast.await_count == 2
    assert fc.wattcast[sid] is cached  # never dropped on failure
    assert fc.slots[sid] == before
    assert fc.data[sid].stale is False  # ~1 h old: not yet a missed re-issue
    assert fc.data[sid].fetch_error == "timeout"

    while fc.data[sid].cache_age_h <= 2.0:
        freezer.move_to(fc.fetch[sid].next_fetch)
        await fc.async_tick()
    await hass.async_block_till_done()
    result = fc.data[sid]
    assert result.stale is True
    assert fc.wattcast[sid] is cached
    assert {s["src"] for s in fc.slots[sid]} == {"wattcast"}

    state = hass.states.get(_sensor_id(hass, sid, "price_forecast"))
    assert state.attributes["stale"] is True
    assert state.attributes["cache_age_h"] > 2
    assert state.attributes["fetch_error"] == "timeout"
    assert state.attributes["source"] == "wattcast"


async def test_backoff_ladder(hass: HomeAssistant, freezer, wattcast):
    wattcast.return_value = _FAIL
    entry, fc, sid = await _setup(hass, freezer)
    state = fc.fetch[sid]
    assert wattcast.await_count == 1
    assert state.failures == 1
    assert state.failing_since == _NOW
    assert state.next_fetch == _NOW + timedelta(minutes=5)

    # Not due yet → no request.
    freezer.tick(timedelta(minutes=4))
    await fc.async_tick()
    assert wattcast.await_count == 1

    steps = []
    for _ in range(5):
        due = state.next_fetch
        freezer.move_to(due)
        await fc.async_tick()
        steps.append(int((state.next_fetch - due).total_seconds() // 60))
    assert steps == [15, 30, 60, 60, 60]  # 5 → 15 → 30 → 60, then 60 repeats
    assert wattcast.await_count == 6
    assert state.failing_since == _NOW  # consecutive-failure clock not reset

    # A success resets to the hourly cadence.
    wattcast.return_value = _ok()
    freezer.move_to(state.next_fetch)
    now = dt_util.utcnow()
    await fc.async_tick()
    assert state.failures == 0
    assert state.failing_since is None
    assert state.last_error is None
    assert state.next_fetch == now + timedelta(minutes=60)
    assert fc.fetched_at[sid] == now


@pytest.mark.parametrize(("retry_after", "delay"), [(7200, 120), (60, 5)])
async def test_retry_after_honoured(hass: HomeAssistant, freezer, wattcast, retry_after, delay):
    # Retry-After wins when it is longer than the ladder step, never shorter.
    wattcast.return_value = WattcastFetch(None, "rate limited (HTTP 429)", retry_after)
    entry, fc, sid = await _setup(hass, freezer)
    assert fc.fetch[sid].next_fetch == _NOW + timedelta(minutes=delay)
    assert fc.fetch[sid].last_error == "rate limited (HTTP 429)"


async def test_repair_issue_after_six_hours_cleared_on_success(
    hass: HomeAssistant, freezer, wattcast
):
    wattcast.return_value = _FAIL
    entry, fc, sid = await _setup(hass, freezer)
    registry = ir.async_get(hass)
    issue_id = f"wattcast_unreachable_{sid}"

    while True:
        due = fc.fetch[sid].next_fetch
        freezer.move_to(due)
        await fc.async_tick()
        issue = registry.async_get_issue(DOMAIN, issue_id)
        if due - _NOW < timedelta(hours=6):
            assert issue is None, f"issue raised early at +{due - _NOW}"
        else:
            break
    assert issue is not None
    assert issue.translation_key == "wattcast_unreachable"
    assert issue.severity == ir.IssueSeverity.WARNING
    assert issue.is_fixable is False
    assert issue.translation_placeholders["zone"] == "FI"
    assert issue.translation_placeholders["last_success"] == "never"
    assert issue.translation_placeholders["error"] == "timeout"

    wattcast.return_value = _ok()
    freezer.move_to(fc.fetch[sid].next_fetch)
    await fc.async_tick()
    assert registry.async_get_issue(DOMAIN, issue_id) is None
    assert {s["src"] for s in fc.slots[sid]} == {"wattcast"}


async def test_cache_restored_after_reload_without_fetch(hass: HomeAssistant, freezer, wattcast):
    entry, fc, sid = await _setup(hass, freezer)
    assert wattcast.await_count == 1
    await fc._store.async_save_now(fc._runtime_snapshot())

    freezer.tick(timedelta(minutes=30))
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    new = entry.runtime_data.forecast
    assert new is not fc
    assert wattcast.await_count == 1  # cache < 60 min old → no request on startup

    assert new.wattcast[sid].to_dict() == fc.wattcast[sid].to_dict()
    assert new.fetched_at[sid] == _NOW
    # Next request just after the following Wattcast issue (HH:32).
    assert new.fetch[sid].next_fetch == _next_issue_fetch(_NOW)
    assert new.mapping[sid].to_dict() == fc.mapping[sid].to_dict()
    assert new.pairs[sid] == fc.pairs[sid]
    assert new.slots[sid] and {s["src"] for s in new.slots[sid]} == {"wattcast"}
    assert new.data[sid].cache_age_h == 0.5
    assert new.data[sid].source == "wattcast"


async def test_reload_fetches_when_cache_is_old(hass: HomeAssistant, freezer, wattcast):
    entry, fc, sid = await _setup(hass, freezer)
    await fc._store.async_save_now(fc._runtime_snapshot())
    freezer.tick(timedelta(minutes=61))
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    assert wattcast.await_count == 2
    new = entry.runtime_data.forecast
    assert new.fetched_at[sid] == _NOW + timedelta(minutes=61)


async def test_failure_after_restart_serves_persisted_cache(hass: HomeAssistant, freezer, wattcast):
    entry, fc, sid = await _setup(hass, freezer)
    await fc._store.async_save_now(fc._runtime_snapshot())
    wattcast.return_value = _FAIL
    freezer.tick(timedelta(hours=3))
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    new = entry.runtime_data.forecast
    assert wattcast.await_count == 2  # tried, failed
    assert new.wattcast[sid] is not None
    assert {s["src"] for s in new.slots[sid]} == {"wattcast"}
    assert new.data[sid].stale is True


async def test_ticker_drives_hourly_fetch(hass: HomeAssistant, freezer, wattcast):
    entry, fc, sid = await _setup(hass, freezer)
    assert wattcast.await_count == 1
    freezer.tick(timedelta(minutes=5))
    async_fire_time_changed(hass)
    await hass.async_block_till_done()
    assert wattcast.await_count == 1  # due-check only
    freezer.tick(timedelta(minutes=56))
    async_fire_time_changed(hass)
    await hass.async_block_till_done()
    assert wattcast.await_count == 2


async def test_button_refetch_is_rate_limited(hass: HomeAssistant, freezer, wattcast):
    entry, fc, sid = await _setup(hass, freezer)
    freezer.tick(timedelta(minutes=20))
    await fc.async_build_forecast(only=sid, fetch=True)
    assert wattcast.await_count == 2  # cache ≥ 15 min → forced refresh
    freezer.tick(timedelta(minutes=5))
    await fc.async_build_forecast(only=sid, fetch=True)
    assert wattcast.await_count == 2  # too soon → served from cache


async def test_one_request_per_zone_per_tick(hass: HomeAssistant, freezer, wattcast):
    await hass.config.async_set_time_zone("Europe/Helsinki")
    freezer.move_to(_NOW)
    hass.states.async_set("sensor.price", "0.14")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"name": "P", "predict_time": "14:00:00", "capture_time": "23:55:00"},
        subentries_data=[
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_PRICE_FORECAST,
                title=name,
                unique_id=None,
                data={"name": name, "price_entity": "sensor.price"},
            )
            for name in ("A", "B")
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert wattcast.await_count == 1
    fc = entry.runtime_data.forecast
    assert all(fc.wattcast[sid] is not None for sid in fc.forecast_configs())


# ── evaluation: per-source snapshots + scores ────────────────────────────────


async def test_snapshot_logs_each_source(hass: HomeAssistant, freezer, wattcast):
    entry, fc, sid = await _setup(hass, freezer, local=True)
    log = {e["date"]: e for e in fc.log[sid]}
    # Today is (almost) all real/past → no snapshot; tomorrow has 23 forecast hours.
    assert "2026-09-24" not in log
    entry_ = log["2026-09-25"]
    assert set(entry_["hourly"]) == {"wattcast", "wattcast_raw", "local"}
    for vec in entry_["hourly"].values():
        assert len(vec) == 24
        assert vec[0] is None  # 00:00–01:00 local is still real (CET day)
        assert all(v is not None for v in vec[1:])
    assert set(entry_["daily"]) == {"wattcast", "wattcast_raw", "local"}
    # p50Raw = p50 − 10 €/MWh → the raw variant is ≈ 1.255 c/kWh cheaper.
    diff = entry_["daily"]["wattcast"] - entry_["daily"]["wattcast_raw"]
    assert diff == pytest.approx(10 * _SLOPE / 1000, abs=2e-4)
    assert entry_["src"] == "wattcast"
    assert entry_["predicted"] == entry_["daily"]["wattcast"]
    assert entry_["actual"] is None
    # Local-day bucket start for the evaluator.
    assert entry_["bucket_ms"] == int(datetime(2026, 9, 24, 21, 0, tzinfo=UTC).timestamp() * 1000)


async def test_snapshot_frozen_once_day_becomes_real(hass: HomeAssistant, freezer, wattcast):
    entry, fc, sid = await _setup(hass, freezer, local=True)
    before = copy.deepcopy(next(e for e in fc.log[sid] if e["date"] == "2026-09-25"))

    # Tomorrow's day-ahead prices publish (CET day → until 01:00 the day after).
    attrs = _nordpool_attrs()
    attrs["data_tomorrow"] = _slot_items(_FC_START, [_spot(_KNOWN_Q + i) for i in range(96)])
    hass.states.async_set("sensor.nordpool", "0.1", attrs)
    freezer.tick(timedelta(minutes=5))
    await fc.async_tick()  # real end moved → rebuild without a fetch
    assert wattcast.await_count == 1

    assert fc.known_until[sid] == _FC_START + timedelta(days=1)
    assert fc.slots[sid][0]["start"] == "2026-09-26T01:00:00+03:00"
    after = next(e for e in fc.log[sid] if e["date"] == "2026-09-25")
    assert after == before  # the pre-publication forecast is what gets scored


def _past_entry(day: datetime, **extra) -> dict:
    return {
        "date": day.date().isoformat(),
        "bucket_ms": int(day.timestamp() * 1000),
        "predicted": None,
        "actual": None,
        "abs_error": None,
        **extra,
    }


async def test_evaluate_scores_each_source(hass: HomeAssistant, freezer, wattcast):
    entry, fc, sid = await _setup(hass, freezer)
    yesterday = dt_util.start_of_local_day() - timedelta(days=1)
    fc.log[sid] = [
        _past_entry(
            yesterday,
            predicted=0.10,
            src="wattcast",
            hourly={"wattcast": [0.10] * 24, "wattcast_raw": [0.125] * 24, "local": [0.15] * 24},
            daily={"wattcast": 0.10, "wattcast_raw": 0.125, "local": 0.15},
        )
    ]
    y_utc = dt_util.as_utc(yesterday)
    rows = [(dt_util.as_local(y_utc + timedelta(hours=h)), 0.11) for h in range(24)]
    with patch(f"{_MOD}.async_hourly_price_rows", new=AsyncMock(return_value=rows)) as hourly:
        await fc.async_evaluate()
    await hass.async_block_till_done()
    hourly.assert_awaited_once()
    assert hourly.await_args.args[1] == "sensor.price"

    row = fc.log[sid][0]
    assert row["actual"] == 0.11
    assert row["abs_error"] == 0.01
    assert row["scores"] == {"wattcast": 0.01, "wattcast_raw": 0.015, "local": 0.04}
    assert row["daily_err"] == {"wattcast": 0.01, "wattcast_raw": 0.015, "local": 0.04}

    result = fc.data[sid]
    assert result.hourly_mae == 0.01
    assert result.forecast_mae == 0.01
    assert result.mae_by_source == {
        "local": {"hourly_mae": 0.04, "daily_mae": 0.04, "n": 1},
        "wattcast": {"hourly_mae": 0.01, "daily_mae": 0.01, "n": 1},
        "wattcast_raw": {"hourly_mae": 0.015, "daily_mae": 0.015, "n": 1},
    }
    assert float(hass.states.get(_sensor_id(hass, sid, "forecast_hourly_mae")).state) == 0.01
    attrs = hass.states.get(_sensor_id(hass, sid, "price_forecast")).attributes
    assert attrs["mae_by_source"]["local"]["hourly_mae"] == 0.04
    assert attrs["forecast_hourly_mae_eur_kwh"] == 0.01


async def test_evaluate_falls_back_to_daily_mean(hass: HomeAssistant, freezer, wattcast):
    # No hourly statistics → the daily LTS mean still scores the daily error.
    entry, fc, sid = await _setup(hass, freezer)
    yesterday = dt_util.start_of_local_day() - timedelta(days=1)
    fc.log[sid] = [
        _past_entry(
            yesterday,
            predicted=0.10,
            src="wattcast",
            hourly={"wattcast": [0.10] * 24},
            daily={"wattcast": 0.10},
        )
    ]
    with (
        patch(f"{_MOD}.async_hourly_price_rows", new=AsyncMock(return_value=[])),
        patch(f"{_MOD}.async_daily_price_mean", new=AsyncMock(return_value=0.13)),
    ):
        await fc.async_evaluate()
    row = fc.log[sid][0]
    assert row["actual"] == 0.13
    assert "scores" not in row  # no hourly truth → no hourly score
    assert row["daily_err"] == {"wattcast": 0.03}
    assert fc.data[sid].hourly_mae is None


@pytest.mark.parametrize(
    ("scores", "primary"),
    [
        ({"wattcast": 0.05, "wattcast_raw": 0.04, "local": 0.01}, "local"),
        ({"wattcast": 0.05, "wattcast_raw": 0.01, "local": 0.04}, "wattcast_raw"),
        ({"wattcast": 0.01, "wattcast_raw": 0.04, "local": 0.04}, "wattcast"),
    ],
)
async def test_primary_follows_best_scored_source(
    hass: HomeAssistant, freezer, wattcast, scores, primary
):
    entry, fc, sid = await _setup(hass, freezer, local=True)
    start = dt_util.start_of_local_day()
    past = [
        _past_entry(start - timedelta(days=d), predicted=0.1, actual=0.1, scores=dict(scores))
        for d in range(8, 1, -1)  # 7 scored days
    ]
    fc.log[sid] = past + fc.log[sid]
    await fc.async_tick(force_rebuild=True)
    assert fc.primary[sid] == primary
    assert fc.data[sid].source == primary
    slots = fc.slots[sid]
    if primary == "local":
        assert {s["src"] for s in slots} == {"local"}
        return
    assert {s["src"] for s in slots} == {"wattcast"}
    point = fc.wattcast[sid].forecast[0]
    median = point.p50_raw if primary == "wattcast_raw" else point.p50
    local = dt_util.parse_datetime(slots[0]["start"])
    assert slots[0]["buy"] == round(fc.mapping[sid].apply(median, local), 5)


async def test_primary_needs_seven_scored_days(hass: HomeAssistant, freezer, wattcast):
    entry, fc, sid = await _setup(hass, freezer, local=True)
    start = dt_util.start_of_local_day()
    fc.log[sid] = [
        _past_entry(
            start - timedelta(days=d),
            predicted=0.1,
            actual=0.1,
            scores={"wattcast": 0.05, "local": 0.01},
        )
        for d in range(7, 1, -1)  # only 6 days
    ] + fc.log[sid]
    await fc.async_tick(force_rebuild=True)
    assert fc.primary[sid] == "wattcast"


# ── sensor contract ──────────────────────────────────────────────────────────


async def test_price_forecast_sensor_attributes(hass: HomeAssistant, freezer, wattcast):
    entry, fc, sid = await _setup(hass, freezer, local=True)
    state = hass.states.get(_sensor_id(hass, sid, "price_forecast"))
    attrs = state.attributes
    for key in (
        "data_today",
        "status",
        "source",
        "known_until",
        "stale",
        "wattcast_made_at",
        "cache_age_h",
        "fetch_error",
        "days",
        "retail_mapping",
        "mae_by_source",
        "model_samples",
        "fit_mae_eur_kwh",
        "forecast_mae_eur_kwh",
        "forecast_hourly_mae_eur_kwh",
        "fitted",
        "coefficients",
    ):
        assert key in attrs, key
    assert attrs["source"] == "wattcast"
    assert dt_util.parse_datetime(attrs["known_until"]) == _FC_START
    assert attrs["stale"] is False
    assert attrs["cache_age_h"] == 0
    assert attrs["mae_by_source"] == {}
    assert set(attrs["retail_mapping"]) == {"slope_pos", "slope_neg", "base", "n", "mae"}
    assert len(attrs["data_today"]) == 668
    assert attrs["data_today"][0]["start"] == "2026-09-25T01:00:00+03:00"
    assert {"date", "mean", "min", "max", "src"} <= set(attrs["days"][0])
    assert "Wattcast" in attrs["attribution"]
    # The state is the mean of the published series.
    buys = [s["buy"] for s in attrs["data_today"]]
    assert float(state.state) == pytest.approx(sum(buys) / len(buys), abs=1e-5)


def test_big_attributes_are_unrecorded() -> None:
    # ~670 slots is far past the recorder's 16 KB attribute cap.
    unrecorded = PriceForecastSensor._unrecorded_attributes
    assert {"data_today", "days", "mae_by_source", "retail_mapping"} <= unrecorded
    assert "source" not in unrecorded and "stale" not in unrecorded


async def test_hourly_mae_sensor_exists(hass: HomeAssistant, freezer, wattcast):
    entry, fc, sid = await _setup(hass, freezer)
    state = hass.states.get(_sensor_id(hass, sid, "forecast_hourly_mae"))
    assert state is not None
    assert state.state == "unknown"  # nothing scored yet
    assert state.attributes["unit_of_measurement"] == "€/kWh"


# ── DST (EU clocks go back 2026-10-25 04:00 EEST → 03:00 EET) ────────────────


def _forecast_from(start: datetime, quarters: int) -> WattcastFetch:
    payload = {
        "resolution": "15min",
        "madeAt": int(start.timestamp()),
        "known": [],
        "forecast": [
            {"ts": int((start + i * _Q).timestamp()), "k": 1, "p50": _spot(i)}
            for i in range(quarters)
        ],
    }
    return WattcastFetch(parse_wattcast(payload))


async def test_dst_fall_back_day_has_100_quarters(hass: HomeAssistant, freezer, wattcast):
    now = datetime(2026, 10, 23, 9, 0, tzinfo=UTC)  # 12:00 EEST
    await hass.config.async_set_time_zone("Europe/Helsinki")
    freezer.move_to(now)
    wattcast.return_value = _forecast_from(now, 5 * 96)
    entry, fc, sid = await _setup(hass, freezer, series_entity=False, forecast_days=3)
    freezer.move_to(now)  # _setup re-freezes at _NOW; undo before rebuilding
    fc.fetch[sid].next_fetch = now
    await fc.async_tick()

    slots = fc.slots[sid]
    days = {d["date"]: d["slots"] for d in fc.data[sid].days}
    assert days == {"2026-10-23": 48, "2026-10-24": 96, "2026-10-25": 100, "2026-10-26": 96}
    assert slots[-1]["end"] == "2026-10-27T00:00:00+02:00"  # local midnight, EET
    for prev, cur in zip(slots, slots[1:], strict=False):
        assert prev["end"] == cur["start"]
    # The repeated 03:00 hour shows up with both offsets, in order.
    starts = [s["start"] for s in slots if s["start"].startswith("2026-10-25T03:00")]
    assert starts == ["2026-10-25T03:00:00+03:00", "2026-10-25T03:00:00+02:00"]
    # The snapshot still has 24 wall-clock hours (the repeat merged).
    entry_ = next(e for e in fc.log[sid] if e["date"] == "2026-10-25")
    assert len(entry_["hourly"]["wattcast"]) == 24
    assert all(v is not None for v in entry_["hourly"]["wattcast"])


async def test_evaluate_covers_the_whole_25_hour_day(hass: HomeAssistant, freezer, wattcast):
    entry, fc, sid = await _setup(hass, freezer)
    freezer.move_to(datetime(2026, 10, 26, 21, 0, tzinfo=UTC))
    day = dt_util.as_local(datetime(2026, 10, 24, 21, 0, tzinfo=UTC))  # 25 Oct 00:00 EEST
    fc.log[sid] = [
        _past_entry(
            day,
            predicted=0.11,
            src="wattcast",
            hourly={"wattcast": [0.11] * 24},
            daily={"wattcast": 0.11},
        )
    ]
    # 25 hourly LTS rows for the local day (hour 23 local = 21:00Z) + a
    # neighbour on each side; the mock honours the requested window.
    first = datetime(2026, 10, 24, 20, 0, tzinfo=UTC)
    rows = []
    for h in range(27):
        when = dt_util.as_local(first + timedelta(hours=h))
        rows.append((when, 0.20 if (when.date() == day.date() and when.hour == 23) else 0.11))

    async def _rows(hass_, entity, start, end=None):
        return [(w, v) for w, v in rows if start <= w < end]

    with patch(f"{_MOD}.async_hourly_price_rows", new=_rows):
        await fc.async_evaluate()
    row = fc.log[sid][0]
    # 23:00–24:00 local is part of the day: only it misses, by 0.09.
    assert row["scores"]["wattcast"] == round(0.09 / 24, 5)


async def test_button_does_not_override_failure_backoff(hass: HomeAssistant, freezer, wattcast):
    entry, fc, sid = await _setup(hass, freezer)
    wattcast.return_value = _FAIL
    freezer.tick(timedelta(hours=1))
    await fc.async_tick()  # due → fails → backing off
    calls = wattcast.await_count
    backoff_until = fc.fetch[sid].next_fetch
    assert backoff_until > dt_util.utcnow()
    await fc.async_build_forecast(only=sid, fetch=True)
    assert wattcast.await_count == calls  # the button respected the backoff
    assert fc.fetch[sid].next_fetch == backoff_until


async def test_fetches_align_to_issue_minute(hass: HomeAssistant, freezer, wattcast):
    entry, fc, sid = await _setup(hass, freezer)
    nxt = fc.fetch[sid].next_fetch
    assert nxt.minute == 32 and nxt > _NOW and nxt - _NOW <= timedelta(hours=1)


async def test_disabling_wattcast_clears_its_state(hass: HomeAssistant, freezer, wattcast):
    entry, fc, sid = await _setup(hass, freezer)
    wattcast.return_value = _FAIL
    for _ in range(2):  # first failure, then one ≥ 6 h later
        freezer.tick(timedelta(hours=7))
        fc.fetch[sid].next_fetch = dt_util.utcnow()
        await fc.async_tick()
    issue_id = f"wattcast_unreachable_{sid}"
    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id) is not None
    subentry = entry.subentries[sid]
    hass.config_entries.async_update_subentry(
        entry, subentry, data={**subentry.data, "use_wattcast": False}
    )
    await hass.async_block_till_done()
    new = entry.runtime_data.forecast
    await new.async_tick(force_rebuild=True)
    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id) is None
    result = new.data[sid]
    assert result.stale is False and result.cache_age_h is None and result.fetch_error is None
