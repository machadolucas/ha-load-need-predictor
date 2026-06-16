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

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


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
