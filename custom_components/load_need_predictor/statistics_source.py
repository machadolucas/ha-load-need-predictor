"""Read delivered energy + commanded runtime from the recorder.

Raw history isn't retained long here, so the training target — the energy
delivered over the capture window — comes from the recorder's ``change``
statistic on the ``total_increasing`` energy sensor (the
``leddetector_water_heater_energy`` for the LVV). Every helper degrades to
``None`` when the recorder isn't available (recorder is a soft dep) *or* the
query fails — a DB hiccup must never take the sensors or the daily jobs down.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from functools import partial

from homeassistant.core import HomeAssistant, State
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)

# States that count as the load actually drawing (the contactor is closed). A
# switch reports "on"/"off"; we also accept a few synonyms and any positive
# numeric reading so a power/relay sensor works too. Everything else (incl.
# "unknown"/"unavailable") counts as off.
_ON_STATES = frozenset({"on", "true", "heat", "heating", "active", "open"})


def _is_on(value: str) -> bool:
    """True when a recorded state string means the load was running."""
    text = str(value).strip().lower()
    if text in _ON_STATES:
        return True
    try:
        return float(text) > 0.0
    except (TypeError, ValueError):
        return False


# The recorder compiles a 5-min short-term stat a few seconds *after* its period
# ends; ending the window this far back guarantees the last bucket exists.
_COMPILE_LAG = timedelta(minutes=1)


def capture_window(now: datetime) -> tuple[datetime, datetime]:
    """The capture's ``[start, end]`` (UTC), deterministic and already compiled.

    ``end`` is ``now − 1 min`` floored to the 5-minute short-term grid (a 23:55:00
    capture ends at 23:50 — the 23:50–23:55 bucket isn't written until ~23:55:10,
    and the singular stats API silently returns whatever exists). Flooring also
    drops the seconds/µs of ``now`` so the baseline bucket never shifts.

    ``start`` is the same local wall-clock time on the previous day, so daily
    captures at a fixed local time tile the timeline exactly — including across
    DST, where that span is 23 or 25 elapsed hours (a fixed 24 h would leave a
    1-hour gap or overlap between consecutive captures). The arithmetic is done
    explicitly on the local date, then converted back to UTC.
    """
    utc = dt_util.as_utc(now) - _COMPILE_LAG
    end = utc.replace(second=0, microsecond=0)
    end -= timedelta(minutes=end.minute % 5)
    local_end = dt_util.as_local(end)
    # Naive wall-clock minus one day, re-attached to the zone: zoneinfo then
    # resolves that date's own UTC offset.
    wall = local_end.replace(tzinfo=None) - timedelta(days=1)
    start = dt_util.as_utc(wall.replace(tzinfo=local_end.tzinfo))
    return start, end


async def async_statistic_change(
    hass: HomeAssistant,
    entity_id: str,
    start: datetime,
    end: datetime,
    units: dict[str, str] | None = None,
) -> float | None:
    """The statistic's ``change`` over ``[start, end]``, or ``None``.

    Uses the singular ``statistic_during_period``, which blends the 5-minute
    short-term stats into the hourly ones — so an unaligned window (e.g. the
    trailing 24 h ending at 23:55) is read in full, including the current hour
    that the hourly compiler hasn't reached yet. ``units`` converts to a display
    unit per unit class (``{"energy": "kWh"}`` normalises a Wh meter); ``None``
    keeps the sensor's native unit.
    """
    if not entity_id or end <= start:
        return None
    try:
        from homeassistant.components.recorder import get_instance
        from homeassistant.components.recorder.statistics import statistic_during_period
    except ImportError:  # recorder not installed at all
        return None

    try:
        instance = get_instance(hass)
    except KeyError:  # recorder not set up in this instance
        _LOGGER.debug("Recorder not available; cannot read statistics for %s", entity_id)
        return None

    try:
        stat = await instance.async_add_executor_job(
            statistic_during_period, hass, start, end, entity_id, {"change"}, units
        )
    except Exception:  # noqa: BLE001 - a recorder/DB failure degrades to "unknown"
        _LOGGER.warning("Reading the %s statistic failed", entity_id, exc_info=True)
        return None
    change = (stat or {}).get("change")
    return None if change is None else float(change)


async def async_daily_delivered_kwh(
    hass: HomeAssistant, entity_id: str, start: datetime, end: datetime
) -> float | None:
    """Energy (kWh) delivered over ``[start, end]`` — the capture's training target.

    The capture job passes the trailing 24 h ending at capture time; that keeps
    the 23:00–24:00 hour that a calendar-day read would drop (at a 23:55 capture
    the day's last hour isn't compiled yet, and the next day's read is a
    different day, so it was never counted). Normalised to kWh whatever the
    meter's own energy unit.
    """
    return await async_statistic_change(hass, entity_id, start, end, {"energy": "kWh"})


def _on_minutes(states: list[State], start: datetime, end: datetime) -> float:
    """Minutes the recorded entity spent ON within ``[start, end)``.

    Each state holds until the next change; the final state holds to ``end``.
    Segments are clipped to the window so a state that began before ``start``
    only counts from ``start`` onward.
    """
    on_seconds = 0.0
    n = len(states)
    for i, st in enumerate(states):
        seg_start = max(st.last_changed, start)
        seg_end = states[i + 1].last_changed if i + 1 < n else end
        seg_end = min(seg_end, end)
        if seg_end <= seg_start:
            continue
        if _is_on(st.state):
            on_seconds += (seg_end - seg_start).total_seconds()
    return on_seconds / 60.0


async def async_commanded_minutes(
    hass: HomeAssistant, entity_id: str, start: datetime, end: datetime
) -> float | None:
    """Minutes ``entity_id`` was switched ON over ``[start, end)``, from history.

    This is the runtime actually delivered to the load — the controlled
    switch/contactor's on-time from *any* source (the scheduler, a manual boost,
    a comfort automation) — which the deficit accounting compares against what
    was asked for the cycle. It reads raw state-change history rather than the
    daily statistic so the window can be an arbitrary predict→predict span.

    Returns ``None`` when the recorder is unavailable or there's no history for
    the entity, so the caller skips the cycle rather than reading a false zero.
    """
    if not entity_id or end <= start:
        return None
    try:
        from homeassistant.components.recorder import get_instance
        from homeassistant.components.recorder.history import state_changes_during_period
    except ImportError:  # recorder not installed at all
        return None

    try:
        instance = get_instance(hass)
    except KeyError:  # recorder not set up in this instance
        _LOGGER.debug("Recorder not available; cannot read commanded runtime")
        return None

    try:
        history = await instance.async_add_executor_job(
            partial(
                state_changes_during_period,
                hass,
                start,
                end,
                entity_id,
                include_start_time_state=True,
                no_attributes=True,
            )
        )
    except Exception:  # noqa: BLE001 - a recorder/DB failure degrades to "unknown"
        _LOGGER.warning("Reading the %s history failed", entity_id, exc_info=True)
        return None
    states = (history or {}).get(entity_id)
    if not states:
        return None
    return _on_minutes(states, start, end)
