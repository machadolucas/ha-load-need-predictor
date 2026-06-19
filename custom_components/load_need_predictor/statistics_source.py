"""Read delivered energy from Home Assistant long-term statistics.

Raw history isn't retained long here, so the training target — the day's
delivered energy — comes from the recorder's daily ``change`` statistic on the
``total_increasing`` energy sensor (the ``leddetector_water_heater_energy`` for
the LVV). This is the only place the integration touches the recorder, and it
degrades to ``None`` when the recorder isn't available (recorder is a soft dep).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from functools import partial

from homeassistant.core import HomeAssistant, State

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


async def async_daily_delivered_kwh(
    hass: HomeAssistant, entity_id: str, day_start: datetime
) -> float | None:
    """Return the energy delivered during the local calendar day at ``day_start``.

    Uses the ``change`` aggregation of the daily statistic, summed defensively
    in case the window spans more than one bucket. Returns ``None`` if the
    recorder is unavailable or there's no statistic for that day.
    """
    try:
        from homeassistant.components.recorder import get_instance
        from homeassistant.components.recorder.statistics import statistics_during_period
    except ImportError:  # recorder not installed at all
        return None

    try:
        instance = get_instance(hass)
    except KeyError:  # recorder not set up in this instance
        _LOGGER.debug("Recorder not available; cannot read delivered energy")
        return None

    end = day_start + timedelta(days=1)
    stats = await instance.async_add_executor_job(
        statistics_during_period,
        hass,
        day_start,
        end,
        {entity_id},
        "day",
        None,
        {"change"},
    )
    series = stats.get(entity_id)
    if not series:
        return None
    total = 0.0
    seen = False
    for row in series:
        change = row.get("change")
        if change is not None:
            total += float(change)
            seen = True
    return total if seen else None


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
    states = history.get(entity_id)
    if not states:
        return None
    return _on_minutes(states, start, end)
