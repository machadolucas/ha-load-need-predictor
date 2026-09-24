"""Pure spot-price forecast → retail-price series.

**No Home Assistant imports** — loaded standalone (via ``importlib``) by the pure
unit tests, like ``price_model.py``.

Wattcast publishes a probabilistic *spot* forecast (€/MWh, ex-VAT, UTC slot
starts, p10/p50/p90) roughly a week ahead. The scheduler, however, compares
slots in the user's *all-in* retail price (€/kWh: VAT + margin + possibly a
time-of-use transfer tariff). So this module:

1. parses the Wattcast payload tolerantly (``parse_wattcast``) and caches it
   compactly (``WattcastSeries.to_dict``) — the API allows ≤ 1 request/hour, so
   the cache must survive restarts;
2. learns the spot → retail mapping from paired history (``fit_retail_mapping``):
   a piecewise slope (negative spot is often *not* VAT-scaled, so it gets its
   own slope) plus an intercept per (daytype, local hour) bucket — that bucket
   grid is exactly the shape of a day/night transfer tariff;
3. learns an intraday shape for the local (weather-regression) fallback, which
   only predicts a *daily* mean (``fit_intraday_shape``);
4. merges the two into 15-min ``{start, end, buy}`` slots, DST-safe
   (``build_slots``), and scores sources per day so the best one is published
   (``hourly_vectors`` / ``hourly_mae`` / ``select_primary``).

All time stepping is done in UTC; local time is only used for bucketing and for
the ISO strings the scheduler reads, so 23- and 25-hour days fall out naturally.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, tzinfo
from typing import Any

# Mon–Fri / Saturday / Sunday. Finnish transfer tariffs split weekdays from
# Sunday (and sometimes Saturday), so three daytypes cover the real contracts
# without fragmenting 14 days of history into 7 sparse weekday buckets.
DAYTYPES = ("wd", "sat", "sun")

SOURCES = ("wattcast", "wattcast_raw", "local")

SLOT = timedelta(minutes=15)

# Below this many (spot, buy) pairs the bucket grid can't be trusted: keep the
# VAT-only slopes and fit just a single intercept.
MIN_MAPPING_PAIRS = 24

# A separate negative-spot slope needs a handful of negative quarters to be
# identifiable; with fewer it would be fit to noise, so reuse the positive one.
MIN_NEGATIVE_PAIRS = 8

# Upper bound of the offset shrinkage λ (shrink factor n / (n + λ)). The
# effective λ scales with how noisy the level is (see ``_shrunk_level``): a
# deterministic tariff is recovered exactly, a noisy one is pulled toward the
# parent level as if ~4 extra pairs of "no deviation" had been seen.
SHRINK_LAMBDA = 4.0

# Coordinate-descent rounds between slopes and offsets. The slopes start from
# the within-bucket estimate, which is already unbiased, so few are needed.
FIT_ROUNDS = 3

# Anti-garbage guardrails on the learned slopes (VAT-scaled ≈ 1.255; a contract
# that floors negative spot at 0 gives slope_neg ≈ 0).
SLOPE_POS_RANGE = (0.5, 3.0)
SLOPE_NEG_RANGE = (0.0, 3.0)

# Intraday shape: a day needs most of its hours to give a meaningful
# "price − day mean"; buckets shrink toward the hour's all-daytype mean.
SHAPE_MIN_HOURS = 20
SHAPE_LAMBDA = 4.0


def _daytype(local_dt: datetime) -> str:
    wd = local_dt.weekday()
    if wd < 5:
        return "wd"
    return "sat" if wd == 5 else "sun"


def bucket_key(local_dt: datetime) -> str:
    """``"<daytype>:<HH>"`` for a *local* datetime (the caller converts)."""
    return f"{_daytype(local_dt)}:{local_dt.hour:02d}"


def _num(value: Any) -> float | None:
    """A finite float, or None (bools and NaN/inf are not prices)."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _rows(value: Any) -> Sequence[Any]:
    """Only real lists count as item arrays (a string or number would iterate
    into nonsense or raise)."""
    return value if isinstance(value, (list, tuple)) else ()


def _parse_time(epoch: Any, iso: Any) -> datetime | None:
    """UTC datetime from an epoch-seconds value, falling back to an ISO string."""
    ts = _num(epoch)
    if ts is not None:
        try:
            return datetime.fromtimestamp(ts, UTC)
        except (OverflowError, OSError, ValueError):
            pass
    if isinstance(iso, str) and iso:
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except ValueError:
            return None
        # A naive ISO here would be ambiguous; Wattcast always sends UTC.
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    return None


def _epoch(dt: datetime) -> int:
    return int(dt.timestamp())


# ── Wattcast series ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SpotPoint:
    """A settled day-ahead spot slot (``start`` tz-aware UTC, €/MWh ex-VAT)."""

    start: datetime
    eur_mwh: float


@dataclass(frozen=True)
class ForecastPoint:
    """A forecast spot slot. ``p50_raw`` is the pre-LLM-adjustment median
    (equal to ``p50`` on unadjusted slots), kept so the raw model can be scored
    separately from Wattcast's adjusted one."""

    start: datetime
    k: int
    p10: float
    p50: float
    p90: float
    p50_raw: float


@dataclass(frozen=True)
class WattcastSeries:
    made_at: datetime | None
    slot_minutes: int
    known: tuple[SpotPoint, ...]
    forecast: tuple[ForecastPoint, ...]

    @property
    def coverage_end(self) -> datetime | None:
        """End of the last forecast slot (None without a forecast)."""
        if not self.forecast:
            return None
        return self.forecast[-1].start + timedelta(minutes=self.slot_minutes)

    @property
    def known_end(self) -> datetime | None:
        """End of the last settled slot (None without known prices)."""
        if not self.known:
            return None
        return self.known[-1].start + timedelta(minutes=self.slot_minutes)

    def to_dict(self) -> dict:
        # Rows as positional lists keep a week of quarters (~700 rows) small in
        # the persisted cache; epoch seconds avoid re-parsing ISO strings.
        return {
            "made_at": _epoch(self.made_at) if self.made_at else None,
            "slot_minutes": self.slot_minutes,
            "known": [[_epoch(p.start), p.eur_mwh] for p in self.known],
            "forecast": [
                [_epoch(p.start), p.k, p.p10, p.p50, p.p90, p.p50_raw] for p in self.forecast
            ],
        }

    @classmethod
    def from_dict(cls, data: Any) -> WattcastSeries | None:
        if not isinstance(data, Mapping):
            return None
        slot = data.get("slot_minutes")
        if slot not in (15, 60):
            return None
        known: dict[datetime, SpotPoint] = {}
        for row in _rows(data.get("known")):
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            start, value = _parse_time(row[0], None), _num(row[1])
            if start is not None and value is not None:
                known[start] = SpotPoint(start, value)
        forecast: dict[datetime, ForecastPoint] = {}
        for row in _rows(data.get("forecast")):
            if not isinstance(row, (list, tuple)) or len(row) < 6:
                continue
            start = _parse_time(row[0], None)
            k = _num(row[1])
            vals = [_num(v) for v in row[2:6]]
            if start is None or k is None or any(v is None for v in vals):
                continue
            forecast[start] = ForecastPoint(start, int(k), *vals)
        if not known and not forecast:
            return None
        made = data.get("made_at")
        return cls(
            made_at=_parse_time(made, None) if made is not None else None,
            slot_minutes=int(slot),
            known=tuple(known[s] for s in sorted(known)),
            forecast=tuple(forecast[s] for s in sorted(forecast)),
        )


def _slot_minutes(payload: Mapping, starts: Sequence[datetime]) -> int:
    resolution = payload.get("resolution")
    if isinstance(resolution, str):
        res = resolution.strip().lower()
        if res in ("15min", "15m", "quarter"):
            return 15
        if res in ("hour", "hourly", "60min", "1h", "60m"):
            return 60
    # Unknown/missing label: infer from the tightest spacing between slots.
    ordered = sorted(set(starts))
    gaps = [(b - a).total_seconds() for a, b in zip(ordered, ordered[1:], strict=False)]
    return 60 if gaps and min(gaps) >= 3600 else 15


def parse_wattcast(payload: Any) -> WattcastSeries | None:
    """Parse a Wattcast ``/v1/forecast`` response.

    Tolerant by design (it's a free, unofficial API): missing optional keys get
    defaults, malformed items are skipped, and only a payload with no usable
    item at all returns ``None`` so the caller keeps its cached series.
    """
    if not isinstance(payload, Mapping):
        return None

    known: dict[datetime, SpotPoint] = {}
    for item in _rows(payload.get("known")):
        if not isinstance(item, Mapping):
            continue
        start = _parse_time(item.get("ts"), item.get("startsAt"))
        value = _num(item.get("eurMwh"))
        if value is None:
            # ct/kWh is the same number /10 — accept it if that's all we got.
            ct = _num(item.get("ctKwh"))
            value = ct * 10.0 if ct is not None else None
        if start is not None and value is not None:
            known.setdefault(start, SpotPoint(start, value))

    forecast: dict[datetime, ForecastPoint] = {}
    for item in _rows(payload.get("forecast")):
        if not isinstance(item, Mapping):
            continue
        start = _parse_time(item.get("ts"), item.get("startsAt"))
        p50 = _num(item.get("p50"))
        if start is None or p50 is None:
            continue
        p10 = _num(item.get("p10"))
        p90 = _num(item.get("p90"))
        raw = _num(item.get("p50Raw"))  # present only on LLM-adjusted slots
        k = _num(item.get("k"))
        forecast.setdefault(
            start,
            ForecastPoint(
                start=start,
                k=int(k) if k is not None else 0,
                p10=p10 if p10 is not None else p50,
                p50=p50,
                p90=p90 if p90 is not None else p50,
                p50_raw=raw if raw is not None else p50,
            ),
        )

    if not known and not forecast:
        return None
    return WattcastSeries(
        made_at=_parse_time(payload.get("madeAt"), payload.get("madeAtIso")),
        slot_minutes=_slot_minutes(payload, [*known, *forecast]),
        known=tuple(known[s] for s in sorted(known)),
        forecast=tuple(forecast[s] for s in sorted(forecast)),
    )


# ── Spot → retail mapping ────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetailMapping:
    """spot €/MWh → all-in buy €/kWh.

    ``buy = slope_pos·max(s,0)/1000 + slope_neg·min(s,0)/1000 + offset(bucket)``;
    a bucket with no learned offset uses ``base``.
    """

    slope_pos: float
    slope_neg: float
    offsets: dict[str, float]
    base: float
    n: int
    mae: float | None

    def apply(self, eur_mwh: float, local_dt: datetime) -> float:
        spot = eur_mwh / 1000.0
        offset = self.offsets.get(bucket_key(local_dt), self.base)
        return self.slope_pos * max(spot, 0.0) + self.slope_neg * min(spot, 0.0) + offset

    def to_dict(self) -> dict:
        return {
            "slope_pos": self.slope_pos,
            "slope_neg": self.slope_neg,
            "offsets": dict(self.offsets),
            "base": self.base,
            "n": self.n,
            "mae": self.mae,
        }

    @classmethod
    def from_dict(cls, data: Any) -> RetailMapping | None:
        if not isinstance(data, Mapping):
            return None
        slope_pos = _num(data.get("slope_pos"))
        slope_neg = _num(data.get("slope_neg"))
        base = _num(data.get("base"))
        n = _num(data.get("n"))
        if slope_pos is None or slope_neg is None or base is None or n is None:
            return None
        offsets: dict[str, float] = {}
        raw = data.get("offsets")
        if isinstance(raw, Mapping):
            for key, value in raw.items():
                val = _num(value)
                if isinstance(key, str) and val is not None:
                    offsets[key] = val
        return cls(
            slope_pos=slope_pos,
            slope_neg=slope_neg,
            offsets=offsets,
            base=base,
            n=int(n),
            mae=_num(data.get("mae")),
        )


def seed_mapping(vat: float) -> RetailMapping:
    """VAT-only mapping for the cold start (no margin, no tariff known yet)."""
    slope = 1.0 + vat
    return RetailMapping(slope_pos=slope, slope_neg=slope, offsets={}, base=0.0, n=0, mae=None)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(max(value, bounds[0]), bounds[1])


def _lstsq(cols: Sequence[Sequence[float]], y: Sequence[float]) -> list[float] | None:
    """No-intercept least squares for 1–2 regressors (closed-form normal eqs)."""
    if len(cols) == 1:
        (x,) = cols
        sxx = sum(v * v for v in x)
        if sxx < 1e-18:
            return None
        return [sum(a * b for a, b in zip(x, y, strict=True)) / sxx]
    x1, x2 = cols
    s11 = sum(v * v for v in x1)
    s22 = sum(v * v for v in x2)
    s12 = sum(a * b for a, b in zip(x1, x2, strict=True))
    s1y = sum(a * b for a, b in zip(x1, y, strict=True))
    s2y = sum(a * b for a, b in zip(x2, y, strict=True))
    det = s11 * s22 - s12 * s12
    # Relative singularity test: the regressors are ~0.01–0.1 in €/kWh units.
    if abs(det) <= 1e-12 * max(s11 * s22, 1e-300):
        return None
    return [(s1y * s22 - s2y * s12) / det, (s2y * s11 - s1y * s12) / det]


def _demean(values: Sequence[float], groups: Sequence[str]) -> list[float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for v, g in zip(values, groups, strict=True):
        sums[g] = sums.get(g, 0.0) + v
        counts[g] = counts.get(g, 0) + 1
    return [v - sums[g] / counts[g] for v, g in zip(values, groups, strict=True)]


def _shrunk_level(resid: Sequence[float], groups: Sequence[str]) -> dict[str, float]:
    """Per-group mean of ``resid``, shrunk toward 0 by ``n / (n + λ)``.

    λ is ``SHRINK_LAMBDA`` scaled by the level's noise share σ²_w / (σ²_w + τ²)
    (within-group residual variance vs. the variance of the group means). A
    real tariff makes residuals group-constant (σ²_w → 0), so it's recovered
    without bias; genuinely noisy groups (or ones carrying no real deviation)
    are pulled back toward the parent level, which is the point of the
    hierarchy when a bucket has only a couple of pairs.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r, g in zip(resid, groups, strict=True):
        sums[g] = sums.get(g, 0.0) + r
        counts[g] = counts.get(g, 0) + 1
    means = {g: sums[g] / counts[g] for g in sums}
    n_total = len(resid)
    dof = n_total - len(means)
    within = (
        sum((r - means[g]) ** 2 for r, g in zip(resid, groups, strict=True)) / dof
        if dof > 0
        else 0.0
    )
    raw_between = sum(counts[g] * means[g] ** 2 for g in means) / n_total if n_total else 0.0
    # The group means themselves carry σ²_w/n of noise; don't count it as signal.
    between = max(0.0, raw_between - within * len(means) / max(n_total, 1))
    noise_share = within / (within + between) if within + between > 1e-30 else 0.0
    lam = SHRINK_LAMBDA * noise_share
    return {g: means[g] * counts[g] / (counts[g] + lam) for g in means}


def _fit_offsets(
    resid: Sequence[float], hours: Sequence[str], buckets: Sequence[str]
) -> tuple[float, dict[str, float]]:
    """Hierarchical intercepts: base (global mean) + hour dev + daytype×hour dev.

    Returns ``(base, offsets)``; ``offsets`` is filled for every daytype of every
    hour that was seen, so an unseen daytype inherits its hour's level instead
    of dropping to the global base.
    """
    base = sum(resid) / len(resid)
    r1 = [r - base for r in resid]
    hour_dev = _shrunk_level(r1, hours)
    r2 = [r - hour_dev[h] for r, h in zip(r1, hours, strict=True)]
    bucket_dev = _shrunk_level(r2, buckets)
    offsets: dict[str, float] = {}
    for hour, dev in hour_dev.items():
        for daytype in DAYTYPES:
            key = f"{daytype}:{hour}"
            offsets[key] = base + dev + bucket_dev.get(key, 0.0)
    return base, offsets


def fit_retail_mapping(pairs: Sequence[tuple[datetime, float, float]], vat: float) -> RetailMapping:
    """Learn the spot → retail mapping from ``(local start, spot €/MWh, buy €/kWh)``.

    With too few pairs, keep the VAT slopes and fit only a robust (median)
    intercept. Otherwise the slopes are first estimated *within* buckets (spot
    demeaned per bucket, so a tariff that happens to coincide with expensive
    daytime hours can't leak into the slope), then refined by a few rounds of
    coordinate descent against the hierarchical offsets.
    """
    clean = [
        (dt, s, b)
        for dt, s, b in pairs
        if isinstance(dt, datetime) and _num(s) is not None and _num(b) is not None
    ]
    seed = 1.0 + vat
    n = len(clean)
    if n < MIN_MAPPING_PAIRS:
        if not clean:
            return seed_mapping(vat)
        base = _median([b - seed * s / 1000.0 for _, s, b in clean])
        mae = sum(abs(b - seed * s / 1000.0 - base) for _, s, b in clean) / n
        return RetailMapping(slope_pos=seed, slope_neg=seed, offsets={}, base=base, n=n, mae=mae)

    pos = [max(s, 0.0) / 1000.0 for _, s, _ in clean]
    neg = [min(s, 0.0) / 1000.0 for _, s, _ in clean]
    y = [float(b) for _, _, b in clean]
    buckets = [bucket_key(dt) for dt, _, _ in clean]
    hours = [key.split(":")[1] for key in buckets]
    split = sum(1 for v in neg if v < 0.0) >= MIN_NEGATIVE_PAIRS
    # Single-slope mode regresses on the signed spot so both halves share it.
    regressors = [pos, neg] if split else [[p + q for p, q in zip(pos, neg, strict=True)]]

    def to_slopes(coef: list[float] | None, fallback: tuple[float, float]) -> tuple[float, float]:
        if coef is None:
            return fallback
        if split:
            return _clamp(coef[0], SLOPE_POS_RANGE), _clamp(coef[1], SLOPE_NEG_RANGE)
        slope = _clamp(coef[0], SLOPE_POS_RANGE)
        return slope, slope

    within = _lstsq([_demean(x, buckets) for x in regressors], _demean(y, buckets))
    if within is None:
        # Every bucket holds a single pair (or spot is constant within them):
        # pooled regression with one intercept instead.
        pooled = _lstsq([_demean(x, ["all"] * n) for x in regressors], _demean(y, ["all"] * n))
        slopes = to_slopes(pooled, (seed, seed))
    else:
        slopes = to_slopes(within, (seed, seed))

    def residual(sp: tuple[float, float]) -> list[float]:
        return [yi - sp[0] * p - sp[1] * q for yi, p, q in zip(y, pos, neg, strict=True)]

    base, offsets = _fit_offsets(residual(slopes), hours, buckets)
    for _ in range(FIT_ROUNDS):
        after = [yi - offsets[k] for yi, k in zip(y, buckets, strict=True)]
        slopes = to_slopes(_lstsq(regressors, after), slopes)
        base, offsets = _fit_offsets(residual(slopes), hours, buckets)

    mapping = RetailMapping(
        slope_pos=slopes[0], slope_neg=slopes[1], offsets=offsets, base=base, n=n, mae=None
    )
    mae = sum(abs(mapping.apply(s, dt) - b) for dt, s, b in clean) / n
    return RetailMapping(
        slope_pos=slopes[0], slope_neg=slopes[1], offsets=offsets, base=base, n=n, mae=mae
    )


# ── Intraday shape (for the daily-mean local forecast) ───────────────────────


def fit_intraday_shape(
    rows: Sequence[tuple[datetime, float]], min_days: int = 3
) -> dict[str, float]:
    """Per-bucket offset of the hourly all-in price from its local day's mean.

    ``rows`` = ``(local hour start, hourly mean €/kWh)``. Only near-complete days
    count (a half day's mean would bias every deviation). Each daytype×hour
    offset is shrunk toward the hour's all-daytype mean, so a bucket seen on one
    Sunday doesn't dictate every Sunday. ``{}`` until ``min_days`` days exist.
    """
    by_day: dict[date, list[tuple[datetime, float]]] = {}
    for dt, price in rows:
        value = _num(price)
        if isinstance(dt, datetime) and value is not None:
            by_day.setdefault(dt.date(), []).append((dt, value))
    days = [vals for vals in by_day.values() if len({dt.hour for dt, _ in vals}) >= SHAPE_MIN_HOURS]
    if len(days) < min_days:
        return {}

    hour_sum: dict[int, float] = {}
    hour_n: dict[int, int] = {}
    bucket_sum: dict[str, float] = {}
    bucket_n: dict[str, int] = {}
    for vals in days:
        mean = sum(v for _, v in vals) / len(vals)
        for dt, value in vals:
            dev = value - mean
            hour_sum[dt.hour] = hour_sum.get(dt.hour, 0.0) + dev
            hour_n[dt.hour] = hour_n.get(dt.hour, 0) + 1
            key = bucket_key(dt)
            bucket_sum[key] = bucket_sum.get(key, 0.0) + dev
            bucket_n[key] = bucket_n.get(key, 0) + 1

    shape: dict[str, float] = {}
    for hour, total in hour_sum.items():
        hour_mean = total / hour_n[hour]
        for daytype in DAYTYPES:
            key = f"{daytype}:{hour:02d}"
            n = bucket_n.get(key, 0)
            if n:
                bucket_mean = bucket_sum[key] / n
                shape[key] = hour_mean + (bucket_mean - hour_mean) * n / (n + SHAPE_LAMBDA)
            else:
                shape[key] = hour_mean
    return shape


def apply_shape(shape: Mapping[str, float] | None, daily_price: float, local_dt: datetime) -> float:
    """Daily mean + the bucket's intraday offset (flat when no shape)."""
    if not shape:
        return daily_price
    return daily_price + shape.get(bucket_key(local_dt), 0.0)


# ── Slot building ────────────────────────────────────────────────────────────


def _floor_quarter(dt: datetime) -> datetime:
    utc = dt.astimezone(UTC)
    return utc.replace(minute=utc.minute - utc.minute % 15, second=0, microsecond=0)


def _aware(dt: datetime, tz: tzinfo) -> datetime:
    return dt.replace(tzinfo=tz) if dt.tzinfo is None else dt


def build_slots(
    *,
    start: datetime,
    end: datetime,
    tz: tzinfo,
    wattcast: WattcastSeries | None,
    variant: str,
    mapping: RetailMapping,
    local_daily: Mapping[date, float],
    shape: Mapping[str, float] | None,
) -> list[dict]:
    """15-min retail slots over ``[start, end)``: Wattcast where it covers,
    the local daily forecast (shaped) beyond it, nothing where neither does.

    Stepping is in UTC, so a DST day yields 92 or 100 quarters, not 96. An
    hourly Wattcast series fills all four quarters of each hour. ``variant``
    only picks which median is mapped (``"wattcast_raw"`` → ``p50_raw``).
    """
    use_raw = variant == "wattcast_raw"
    cover: dict[int, ForecastPoint] = {}
    step = 900
    if wattcast is not None:
        step = wattcast.slot_minutes * 60
        cover = {_epoch(p.start): p for p in wattcast.forecast}

    out: list[dict] = []
    t = _floor_quarter(_aware(start, tz))
    stop = _aware(end, tz).astimezone(UTC)
    while t < stop:
        local = t.astimezone(tz)
        item: dict[str, Any] = {
            "start": local.isoformat(),
            "end": (t + SLOT).astimezone(tz).isoformat(),
        }
        epoch = _epoch(t)
        point = cover.get(epoch - epoch % step)
        if point is not None:
            buy = mapping.apply(point.p50_raw if use_raw else point.p50, local)
            # The mapping is monotone, but the raw median can sit outside the
            # adjusted band; keep the invariant p10 ≤ buy ≤ p90 for consumers.
            p10 = min(mapping.apply(point.p10, local), buy)
            p90 = max(mapping.apply(point.p90, local), buy)
            item.update(buy=round(buy, 5), p10=round(p10, 5), p90=round(p90, 5), src="wattcast")
        else:
            daily = local_daily.get(local.date())
            value = _num(daily)
            if value is None:
                t += SLOT
                continue
            item.update(buy=round(apply_shape(shape, value, local), 5), src="local")
        out.append(item)
        t += SLOT
    return out


# ── Scoring + source selection ───────────────────────────────────────────────


def hourly_vectors(slots: Sequence[dict], tz: tzinfo) -> dict[str, list[float | None]]:
    """Local date ISO → 24 hourly mean buys (None for an hour with no slots).

    Indexing by local wall-clock hour makes a forecast and the realised prices
    comparable hour-for-hour; the DST fall-back hour's 8 quarters merge into
    one value and the spring-forward hour is simply None.
    """
    sums: dict[str, list[float]] = {}
    counts: dict[str, list[int]] = {}
    for slot in slots:
        if not isinstance(slot, Mapping):
            continue
        buy = _num(slot.get("buy"))
        raw = slot.get("start")
        if buy is None or not isinstance(raw, str):
            continue
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            continue
        local = _aware(dt, tz).astimezone(tz)
        key = local.date().isoformat()
        if key not in sums:
            sums[key] = [0.0] * 24
            counts[key] = [0] * 24
        sums[key][local.hour] += buy
        counts[key][local.hour] += 1
    return {
        key: [s / c if c else None for s, c in zip(sums[key], counts[key], strict=True)]
        for key in sorted(sums)
    }


def hourly_mae(pred: Sequence[float | None], actual: Sequence[float | None]) -> float | None:
    """Mean |pred − actual| over hours where both exist (None if none do)."""
    errors = [
        abs(p - a)
        for p, a in zip(pred, actual, strict=False)
        if _num(p) is not None and _num(a) is not None
    ]
    return sum(errors) / len(errors) if errors else None


def daily_mean(vec: Sequence[float | None]) -> float | None:
    values = [v for v in vec if _num(v) is not None]
    return sum(values) / len(values) if values else None


def select_primary(
    scores: Mapping[str, Sequence[float]],
    *,
    available: Collection[str],
    window: int = 14,
    min_days: int = 7,
) -> str:
    """The source with the lowest recent mean hourly MAE.

    Only sources that are available right now *and* have ``min_days`` of scores
    compete (a two-day lucky streak shouldn't flip the published series). Ties
    go to the earlier entry of ``SOURCES``. With no qualified source, prefer
    the adjusted Wattcast forecast when present, else the local fallback.
    """
    order = {src: i for i, src in enumerate(SOURCES)}
    best: tuple[float, int, str] | None = None
    for src in available:
        history = [v for v in scores.get(src) or () if _num(v) is not None]
        if len(history) < min_days:
            continue
        recent = history[-window:] if window > 0 else history
        candidate = (sum(recent) / len(recent), order.get(src, len(order)), src)
        if best is None or candidate < best:
            best = candidate
    if best is not None:
        return best[2]
    return "wattcast" if "wattcast" in available else "local"
