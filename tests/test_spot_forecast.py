"""Unit tests for the pure spot-forecast module (no Home Assistant).

Loaded standalone via ``importlib`` like ``test_price_model.py``.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import random
import sys
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "custom_components"
    / "load_need_predictor"
    / "spot_forecast.py"
)
_spec = importlib.util.spec_from_file_location("lnp_spot_forecast", _PATH)
sf = importlib.util.module_from_spec(_spec)
sys.modules["lnp_spot_forecast"] = sf
_spec.loader.exec_module(sf)

_FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "wattcast_fi_15min.json"
HEL = ZoneInfo("Europe/Helsinki")
VAT = 0.255


def _payload() -> dict:
    return json.loads(_FIXTURE.read_text())


def _utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def _true_buy(spot: float, local: datetime) -> float:
    """A deterministic retail formula: VAT-scaled positive spot, unscaled
    negative spot, 8.7 c margin, +3 c daytime transfer 07–22 Mon–Sat."""
    buy = 1.255 * max(spot, 0.0) / 1000 + 1.0 * min(spot, 0.0) / 1000 + 0.087
    if 7 <= local.hour < 22 and local.weekday() < 6:
        buy += 0.03
    return buy


def _pairs(days: int = 10, seed: int = 1) -> list[tuple[datetime, float, float]]:
    rng = random.Random(seed)
    start = datetime(2026, 9, 1, tzinfo=HEL)  # a Tuesday
    out = []
    t = start.astimezone(UTC)
    for _ in range(days * 96):
        local = t.astimezone(HEL)
        spot = rng.uniform(-25.0, 140.0)
        out.append((local, spot, _true_buy(spot, local)))
        t += timedelta(minutes=15)
    return out


def _series(points, slot_minutes: int = 15, known=()) -> object:
    """A WattcastSeries from ``(utc start, p10, p50, p90[, raw])`` tuples."""
    fc = []
    for p in points:
        start, p10, p50, p90 = p[:4]
        raw = p[4] if len(p) > 4 else p50
        fc.append(sf.ForecastPoint(start, 1, p10, p50, p90, raw))
    return sf.WattcastSeries(
        made_at=_utc(2026, 9, 24, 8),
        slot_minutes=slot_minutes,
        known=tuple(known),
        forecast=tuple(fc),
    )


# ── bucket keys ──────────────────────────────────────────────────────────────


def test_bucket_key_daytypes():
    assert sf.bucket_key(datetime(2026, 9, 24, 7, 30, tzinfo=HEL)) == "wd:07"  # Thursday
    assert sf.bucket_key(datetime(2026, 9, 26, 0, 0, tzinfo=HEL)) == "sat:00"
    assert sf.bucket_key(datetime(2026, 9, 27, 23, 45, tzinfo=HEL)) == "sun:23"
    assert sf.DAYTYPES == ("wd", "sat", "sun")


# ── parse ────────────────────────────────────────────────────────────────────


def test_parse_real_fixture():
    series = sf.parse_wattcast(_payload())
    assert series is not None
    assert series.slot_minutes == 15
    assert series.made_at == _utc(2026, 9, 24, 8, 25)
    assert len(series.known) == 8
    assert len(series.forecast) == 24
    assert series.known[0] == sf.SpotPoint(_utc(2026, 9, 23, 21, 0), 35.5)
    assert series.known[1].eur_mwh == 41.46
    assert series.known_end == _utc(2026, 9, 23, 23, 0)
    assert series.coverage_end == _utc(2026, 9, 28, 4, 0)
    first = series.forecast[0]
    assert first.start == _utc(2026, 9, 24, 22, 0)
    assert (first.k, first.p10, first.p50, first.p90) == (1, 7.75, 19.63, 48.1)
    assert first.p50_raw == first.p50  # unadjusted slot → raw = p50
    adjusted = [p for p in series.forecast if p.p50_raw != p.p50]
    assert len(adjusted) >= 2
    assert adjusted[0].start == _utc(2026, 9, 28, 3, 0)
    assert adjusted[0].p50 == 25.87
    assert adjusted[0].p50_raw == 15.87
    # Sorted, tz-aware UTC starts throughout.
    starts = [p.start for p in series.forecast]
    assert starts == sorted(starts)
    assert all(s.tzinfo is not None and s.utcoffset() == timedelta(0) for s in starts)


@pytest.mark.parametrize(
    "payload",
    [None, [], "garbage", 42, {}, {"known": [], "forecast": []}, {"known": "x", "forecast": 5}],
)
def test_parse_garbage_returns_none(payload):
    assert sf.parse_wattcast(payload) is None


def test_parse_skips_bad_items_and_defaults_optionals():
    payload = {
        "resolution": "15min",
        "known": [
            "nope",
            {"startsAt": "2026-09-24T00:00:00Z"},  # no price
            {"ts": "bad", "startsAt": "not a date", "eurMwh": 1.0},  # no time
            {"startsAt": "2026-09-24T00:15:00Z", "eurMwh": float("nan")},
            {"startsAt": "2026-09-24T00:30:00Z", "ctKwh": 4.0},  # ct only → 40 €/MWh
            {"startsAt": "2026-09-24T00:45:00Z", "eurMwh": True},  # bool isn't a price
        ],
        "forecast": [
            {"startsAt": "2026-09-24T01:00:00Z", "p50": None},
            {"startsAt": "2026-09-24T01:15:00Z", "p50": 20.0},  # p10/p90/k/raw missing
            {"ts": 1790213400, "p50": "21.5", "p10": 10, "p90": 30, "k": 2},
        ],
    }
    series = sf.parse_wattcast(payload)
    assert series is not None
    assert series.made_at is None
    assert series.known == (sf.SpotPoint(_utc(2026, 9, 24, 0, 30), 40.0),)
    assert len(series.forecast) == 2
    p = series.forecast[0]
    assert (p.k, p.p10, p.p50, p.p90, p.p50_raw) == (0, 20.0, 20.0, 20.0, 20.0)
    q = series.forecast[1]
    assert q.start == datetime.fromtimestamp(1790213400, UTC)
    assert (q.k, q.p10, q.p50, q.p90) == (2, 10.0, 21.5, 30.0)


def test_parse_resolution_hour_and_inferred():
    base = {
        "forecast": [
            {"startsAt": "2026-09-24T01:00:00Z", "p50": 1.0},
            {"startsAt": "2026-09-24T02:00:00Z", "p50": 2.0},
        ]
    }
    assert sf.parse_wattcast({**base, "resolution": "hour"}).slot_minutes == 60
    assert sf.parse_wattcast(base).slot_minutes == 60  # inferred from spacing
    assert sf.parse_wattcast(_payload() | {"resolution": None}).slot_minutes == 15


def test_parse_made_at_iso_fallback():
    payload = _payload()
    del payload["madeAt"]
    assert sf.parse_wattcast(payload).made_at == _utc(2026, 9, 24, 8, 25)


# ── series serialization ─────────────────────────────────────────────────────


def test_series_round_trip():
    series = sf.parse_wattcast(_payload())
    data = series.to_dict()
    json.dumps(data)  # JSON-safe
    assert sf.WattcastSeries.from_dict(data) == series
    assert sf.WattcastSeries.from_dict(json.loads(json.dumps(data))) == series


def test_series_round_trip_without_made_at():
    series = _series([(_utc(2026, 9, 24, 0), 1.0, 2.0, 3.0)])
    series = sf.WattcastSeries(None, 15, series.known, series.forecast)
    assert sf.WattcastSeries.from_dict(series.to_dict()) == series


@pytest.mark.parametrize(
    "data",
    [
        None,
        "x",
        {},
        {"slot_minutes": 7, "known": [[1790197200, 1.0]]},
        {"slot_minutes": 15, "known": [], "forecast": []},
        {"slot_minutes": 15, "known": [["x", 1.0], [1], None], "forecast": [[1, 2]]},
    ],
)
def test_series_from_dict_garbage(data):
    assert sf.WattcastSeries.from_dict(data) is None


def test_series_from_dict_skips_bad_rows():
    data = sf.parse_wattcast(_payload()).to_dict()
    data["forecast"].append([1790213400, 1, "bad", 2.0, 3.0, 2.0])
    data["known"].append(["bad", 1.0])
    restored = sf.WattcastSeries.from_dict(data)
    assert len(restored.forecast) == 24
    assert len(restored.known) == 8


def test_coverage_none_when_empty():
    series = sf.WattcastSeries(None, 15, (), ())
    assert series.coverage_end is None
    assert series.known_end is None


# ── retail mapping ───────────────────────────────────────────────────────────


def test_seed_mapping():
    m = sf.seed_mapping(VAT)
    assert (m.slope_pos, m.slope_neg, m.offsets, m.base, m.n, m.mae) == (
        1.255,
        1.255,
        {},
        0.0,
        0,
        None,
    )
    local = datetime(2026, 9, 24, 12, tzinfo=HEL)
    assert m.apply(100.0, local) == pytest.approx(0.1255)
    assert m.apply(-10.0, local) == pytest.approx(-0.01255)


def test_apply_uses_bucket_offset_else_base():
    m = sf.RetailMapping(1.0, 0.5, {"wd:12": 0.1}, 0.05, 30, 0.0)
    assert m.apply(20.0, datetime(2026, 9, 24, 12, 30, tzinfo=HEL)) == pytest.approx(0.12)
    assert m.apply(-20.0, datetime(2026, 9, 27, 12, tzinfo=HEL)) == pytest.approx(0.04)


def test_mapping_round_trip_and_garbage():
    m = sf.fit_retail_mapping(_pairs(), VAT)
    assert sf.RetailMapping.from_dict(m.to_dict()) == m
    assert sf.RetailMapping.from_dict(json.loads(json.dumps(m.to_dict()))) == m
    assert sf.RetailMapping.from_dict(None) is None
    assert sf.RetailMapping.from_dict({"slope_pos": 1}) is None
    tolerant = sf.RetailMapping.from_dict(
        {"slope_pos": 1, "slope_neg": 1, "base": 0, "n": 3, "offsets": {"wd:01": "x", 5: 1}}
    )
    assert tolerant.offsets == {}
    assert tolerant.mae is None


def test_fit_empty_is_seed():
    assert sf.fit_retail_mapping([], VAT) == sf.seed_mapping(VAT)


def test_fit_few_pairs_keeps_seed_slopes_median_base():
    pairs = _pairs()[:10]
    m = sf.fit_retail_mapping(pairs, VAT)
    assert m.slope_pos == m.slope_neg == pytest.approx(1.255)
    assert m.offsets == {}
    assert m.n == 10
    expected = sorted(b - 1.255 * s / 1000 for _, s, b in pairs)
    assert m.base == pytest.approx((expected[4] + expected[5]) / 2)
    assert m.mae is not None and m.mae >= 0


def test_fit_recovers_deterministic_formula():
    pairs = _pairs(days=10)
    assert sum(1 for _, s, _ in pairs if s < 0) >= sf.MIN_NEGATIVE_PAIRS
    m = sf.fit_retail_mapping(pairs, VAT)
    assert m.n == len(pairs)
    assert m.slope_pos == pytest.approx(1.255, abs=1e-4)
    assert m.slope_neg == pytest.approx(1.0, abs=1e-4)
    assert m.mae < 1e-4
    # Night / weekday day / Saturday day / Sunday day, positive and negative spot.
    for local in (
        datetime(2026, 9, 8, 3, 15, tzinfo=HEL),
        datetime(2026, 9, 8, 12, 0, tzinfo=HEL),
        datetime(2026, 9, 5, 9, 45, tzinfo=HEL),
        datetime(2026, 9, 6, 12, 0, tzinfo=HEL),
        datetime(2026, 9, 6, 22, 0, tzinfo=HEL),
    ):
        for spot in (-15.0, 0.0, 60.0, 250.0):
            assert m.apply(spot, local) == pytest.approx(_true_buy(spot, local), abs=1e-4)


def test_fit_few_negatives_share_positive_slope():
    pairs = [(dt, abs(s), _true_buy(abs(s), dt)) for dt, s, _ in _pairs(days=5)]
    m = sf.fit_retail_mapping(pairs, VAT)
    assert m.slope_pos == pytest.approx(1.255, abs=1e-4)
    assert m.slope_neg == m.slope_pos


def test_fit_constant_spot_falls_back_to_seed_slope():
    start = datetime(2026, 9, 1, tzinfo=HEL)
    pairs = [(start + timedelta(minutes=15 * i), 50.0, 0.15) for i in range(48)]
    m = sf.fit_retail_mapping(pairs, VAT)
    assert m.slope_pos == pytest.approx(1.255)
    assert m.apply(50.0, start) == pytest.approx(0.15, abs=1e-9)


# ── intraday shape ───────────────────────────────────────────────────────────


def _hourly_rows(days: int, start: date = date(2026, 9, 1)) -> list[tuple[datetime, float]]:
    rows = []
    for d in range(days):
        day = start + timedelta(days=d)
        level = 0.05 + 0.01 * d  # the daily level varies; the shape doesn't
        for h in range(24):
            dt = datetime(day.year, day.month, day.day, h, tzinfo=HEL)
            rows.append((dt, level + (0.04 if 8 <= h < 20 else -0.04)))
    return rows


def test_shape_needs_min_days():
    assert sf.fit_intraday_shape(_hourly_rows(2)) == {}
    assert sf.fit_intraday_shape(_hourly_rows(2), min_days=2) != {}


def test_shape_ignores_incomplete_days():
    rows = _hourly_rows(3)
    rows = [r for r in rows if not (r[0].day == 3 and r[0].hour >= 12)]
    assert sf.fit_intraday_shape(rows) == {}


def test_shape_recovers_offsets_and_applies():
    shape = sf.fit_intraday_shape(_hourly_rows(14))
    # Every daytype of every seen hour is present (weekend inherits/shrinks).
    assert len(shape) == 72
    assert shape["wd:12"] == pytest.approx(0.04, abs=1e-9)
    assert shape["sun:03"] == pytest.approx(-0.04, abs=1e-9)
    assert sum(shape[f"wd:{h:02d}"] for h in range(24)) == pytest.approx(0.0, abs=1e-9)
    local = datetime(2026, 10, 1, 12, tzinfo=HEL)
    assert sf.apply_shape(shape, 0.10, local) == pytest.approx(0.14)
    assert sf.apply_shape({}, 0.10, local) == 0.10
    assert sf.apply_shape(None, 0.10, local) == 0.10


def test_shape_bucket_shrinks_toward_hour_mean():
    rows = _hourly_rows(14)
    # One Sunday with a spike at 10:00; the bucket must not take it at face value.
    rows = [(dt, p + (0.24 if dt.weekday() == 6 and dt.hour == 10 else 0.0)) for dt, p in rows]
    shape = sf.fit_intraday_shape(rows)
    spike_dev = 0.24 * 23 / 24  # the spike's own deviation from its day's mean
    assert shape["wd:10"] < shape["sun:10"] < 0.04 + spike_dev


# ── build_slots ──────────────────────────────────────────────────────────────


def test_build_slots_merges_wattcast_then_local():
    series = sf.parse_wattcast(_payload())
    mapping = sf.seed_mapping(VAT)
    slots = sf.build_slots(
        start=_utc(2026, 9, 24, 22, 7),  # floored to 22:00
        end=_utc(2026, 9, 25, 5, 0),
        tz=HEL,
        wattcast=series,
        variant="wattcast",
        mapping=mapping,
        local_daily={date(2026, 9, 25): 0.08},
        shape=None,
    )
    assert len(slots) == 28
    assert [s["src"] for s in slots] == ["wattcast"] * 20 + ["local"] * 8
    first = slots[0]
    assert first["start"] == "2026-09-25T01:00:00+03:00"
    assert first["end"] == "2026-09-25T01:15:00+03:00"
    assert first["buy"] == round(1.255 * 19.63 / 1000, 5)
    assert first["p10"] == round(1.255 * 7.75 / 1000, 5)
    assert first["p90"] == round(1.255 * 48.1 / 1000, 5)
    assert slots[20]["start"] == "2026-09-25T06:00:00+03:00"
    assert slots[20]["buy"] == 0.08
    assert "p10" not in slots[20] and "p90" not in slots[20]
    for s in slots[:20]:
        assert s["p10"] <= s["buy"] <= s["p90"]


def test_build_slots_raw_variant_and_band_ordering():
    series = sf.parse_wattcast(_payload())
    kwargs = dict(
        start=_utc(2026, 9, 28, 3, 0),
        end=_utc(2026, 9, 28, 4, 0),
        tz=HEL,
        wattcast=series,
        mapping=sf.seed_mapping(VAT),
        local_daily={},
        shape=None,
    )
    adj = sf.build_slots(variant="wattcast", **kwargs)
    raw = sf.build_slots(variant="wattcast_raw", **kwargs)
    assert len(adj) == len(raw) == 4
    assert adj[0]["buy"] == round(1.255 * 25.87 / 1000, 5)
    assert raw[0]["buy"] == round(1.255 * 15.87 / 1000, 5)
    assert raw[0]["src"] == "wattcast"
    # A raw median outside the adjusted band still yields p10 ≤ buy ≤ p90.
    odd = _series([(_utc(2026, 9, 28, 3), 20.0, 30.0, 40.0, 5.0)])
    slot = sf.build_slots(variant="wattcast_raw", **(kwargs | {"wattcast": odd}))[0]
    assert slot["p10"] <= slot["buy"] <= slot["p90"]
    assert slot["p10"] == slot["buy"] == round(1.255 * 5.0 / 1000, 5)


def test_build_slots_hourly_series_fills_quarters():
    series = _series(
        [(_utc(2026, 9, 24, 10), 10.0, 20.0, 30.0), (_utc(2026, 9, 24, 11), 40.0, 50.0, 60.0)],
        slot_minutes=60,
    )
    slots = sf.build_slots(
        start=_utc(2026, 9, 24, 10, 30),
        end=_utc(2026, 9, 24, 13, 0),
        tz=HEL,
        wattcast=series,
        variant="wattcast",
        mapping=sf.seed_mapping(VAT),
        local_daily={},
        shape=None,
    )
    assert len(slots) == 6  # 10:30, 10:45, 11:00–11:45; 12:xx has no source
    assert [s["buy"] for s in slots] == [
        round(1.255 * v / 1000, 5) for v in (20, 20, 50, 50, 50, 50)
    ]


def test_build_slots_pure_local_with_shape():
    shape = sf.fit_intraday_shape(_hourly_rows(14))
    slots = sf.build_slots(
        start=datetime(2026, 10, 1, tzinfo=HEL),
        end=datetime(2026, 10, 2, tzinfo=HEL),
        tz=HEL,
        wattcast=None,
        variant="wattcast",
        mapping=sf.seed_mapping(VAT),
        local_daily={date(2026, 10, 1): 0.10},
        shape=shape,
    )
    assert len(slots) == 96
    assert {s["src"] for s in slots} == {"local"}
    assert slots[0]["buy"] == pytest.approx(0.06)
    assert slots[48]["buy"] == pytest.approx(0.14)


@pytest.mark.parametrize(("day", "quarters"), [(date(2026, 10, 25), 100), (date(2026, 3, 29), 92)])
def test_build_slots_dst_days(day, quarters):
    start = datetime(day.year, day.month, day.day, tzinfo=HEL)
    nxt = day + timedelta(days=1)
    slots = sf.build_slots(
        start=start,
        end=datetime(nxt.year, nxt.month, nxt.day, tzinfo=HEL),
        tz=HEL,
        wattcast=None,
        variant="wattcast",
        mapping=sf.seed_mapping(VAT),
        local_daily={day: 0.1},
        shape=None,
    )
    assert len(slots) == quarters
    assert slots[0]["start"] == start.isoformat()
    assert slots[-1]["end"] == datetime(nxt.year, nxt.month, nxt.day, tzinfo=HEL).isoformat()
    # Contiguous in real time even across the offset change.
    starts = [datetime.fromisoformat(s["start"]) for s in slots]
    assert all(b - a == timedelta(minutes=15) for a, b in zip(starts, starts[1:], strict=False))
    assert all(a["end"] == b["start"] for a, b in zip(slots, slots[1:], strict=False))


# ── hourly vectors + metrics ─────────────────────────────────────────────────


def test_hourly_vectors_fall_back_merges_repeated_hour():
    # Wattcast covering the whole 25-hour day with a distinct value per quarter.
    start = datetime(2026, 10, 25, tzinfo=HEL).astimezone(UTC)
    points = [(start + timedelta(minutes=15 * i), 0.0, float(i), 1000.0) for i in range(100)]
    mapping = sf.RetailMapping(1.0, 1.0, {}, 0.0, 0, None)  # buy = spot / 1000
    slots = sf.build_slots(
        start=start,
        end=start + timedelta(minutes=15 * 100),
        tz=HEL,
        wattcast=_series(points),
        variant="wattcast",
        mapping=mapping,
        local_daily={},
        shape=None,
    )
    vecs = sf.hourly_vectors(slots, HEL)
    assert list(vecs) == ["2026-10-25"]
    vec = vecs["2026-10-25"]
    assert len(vec) == 24 and all(v is not None for v in vec)
    # Local 03:00 happens twice: quarters 12–19 all land in hour 3.
    assert vec[3] == pytest.approx(sum(range(12, 20)) / 8 / 1000)
    assert vec[4] == pytest.approx(sum(range(20, 24)) / 4 / 1000)


def test_hourly_vectors_spring_forward_gap_and_multi_day():
    day = date(2026, 3, 29)
    slots = sf.build_slots(
        start=datetime(2026, 3, 28, 22, tzinfo=HEL),
        end=datetime(2026, 3, 30, tzinfo=HEL),
        tz=HEL,
        wattcast=None,
        variant="wattcast",
        mapping=sf.seed_mapping(VAT),
        local_daily={day: 0.1, date(2026, 3, 28): 0.2},
        shape=None,
    )
    vecs = sf.hourly_vectors(slots + [{"start": "junk", "buy": 1}, {"buy": 1}, "x"], HEL)
    assert list(vecs) == ["2026-03-28", "2026-03-29"]
    assert vecs["2026-03-28"][:22] == [None] * 22
    assert vecs["2026-03-28"][22:] == [0.2, 0.2]
    assert vecs["2026-03-29"][3] is None  # 03:00 local doesn't exist that day
    assert sum(v is not None for v in vecs["2026-03-29"]) == 23


def test_hourly_mae_and_daily_mean():
    assert sf.hourly_mae([1.0, None, 3.0, 4.0], [2.0, 5.0, None, 4.5]) == pytest.approx(0.75)
    assert sf.hourly_mae([None, 1.0], [2.0, None]) is None
    assert sf.hourly_mae([], []) is None
    assert sf.daily_mean([1.0, None, 3.0]) == 2.0
    assert sf.daily_mean([None, None]) is None
    assert sf.daily_mean([]) is None


# ── select_primary ───────────────────────────────────────────────────────────


def test_select_primary_defaults_without_history():
    assert sf.SOURCES == ("wattcast", "wattcast_raw", "local")
    assert sf.select_primary({}, available=sf.SOURCES) == "wattcast"
    assert sf.select_primary({}, available=("local",)) == "local"
    assert sf.select_primary({}, available=()) == "local"
    # Too little history is the same as none.
    scores = {"local": [0.001] * 6, "wattcast": [0.05] * 3}
    assert sf.select_primary(scores, available=sf.SOURCES) == "wattcast"


def test_select_primary_picks_lowest_recent_mean():
    scores = {
        "wattcast": [0.02] * 10,
        "wattcast_raw": [0.018] * 10,
        "local": [0.03] * 10,
    }
    assert sf.select_primary(scores, available=sf.SOURCES) == "wattcast_raw"
    # Unavailable sources don't compete even with the best score.
    assert sf.select_primary(scores, available=("wattcast", "local")) == "wattcast"


def test_select_primary_uses_window():
    # Local was awful long ago but is best over the last 14 days.
    scores = {"wattcast": [0.02] * 30, "local": [0.5] * 16 + [0.01] * 14}
    assert sf.select_primary(scores, available=sf.SOURCES) == "local"
    assert sf.select_primary(scores, available=sf.SOURCES, window=30) == "wattcast"


def test_select_primary_tie_prefers_source_order():
    scores = {"local": [0.02] * 7, "wattcast": [0.02] * 7}
    assert sf.select_primary(scores, available=("local", "wattcast")) == "wattcast"


def test_select_primary_min_days():
    scores = {"wattcast": [0.05] * 7, "local": [0.01] * 5}
    assert sf.select_primary(scores, available=sf.SOURCES) == "wattcast"
    assert sf.select_primary(scores, available=sf.SOURCES, min_days=5) == "local"


def test_no_homeassistant_import():
    assert "homeassistant" not in _PATH.read_text()
