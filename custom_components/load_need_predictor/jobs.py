"""The two daily jobs, scheduled on the hub's wall-clock times.

- **Predict + push** (afternoon, after tomorrow's prices publish): recompute the
  forecast and push the target to the scheduler.
- **Capture + log** (late evening): read the day's actual delivery, complete the
  training row, calibrate, and refresh the evaluation metrics.

M2 wires the scheduling and the predict-side refresh. The push (``actuation``)
and the capture/calibration (``statistics_source``) are filled in at M3 — the
hooks below are deliberately thin so the timing is testable on its own.
"""

from __future__ import annotations

import logging

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_time_change

from .const import CONF_CAPTURE_TIME, CONF_PREDICT_TIME, DEFAULT_CAPTURE_TIME, DEFAULT_PREDICT_TIME
from .coordinator import LoadNeedPredictorCoordinator

_LOGGER = logging.getLogger(__name__)


def _parse_time(value: str) -> tuple[int, int, int]:
    """Parse a ``HH:MM[:SS]`` string into (hour, minute, second)."""
    parts = [int(p) for p in value.split(":")]
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


class PredictorJobs:
    """Owns the daily time-change listeners for one hub."""

    def __init__(self, hass: HomeAssistant, coordinator: LoadNeedPredictorCoordinator) -> None:
        self.hass = hass
        self.coordinator = coordinator
        self._unsubs: list[callable] = []

    @callback
    def async_start(self) -> None:
        """Register the predict and capture time-change listeners."""
        data = self.coordinator.config_entry.data
        ph, pm, ps = _parse_time(data.get(CONF_PREDICT_TIME, DEFAULT_PREDICT_TIME))
        ch, cm, cs = _parse_time(data.get(CONF_CAPTURE_TIME, DEFAULT_CAPTURE_TIME))
        self._unsubs.append(
            async_track_time_change(self.hass, self._handle_predict, hour=ph, minute=pm, second=ps)
        )
        self._unsubs.append(
            async_track_time_change(self.hass, self._handle_capture, hour=ch, minute=cm, second=cs)
        )

    @callback
    def async_shutdown(self) -> None:
        """Cancel all listeners (on unload)."""
        while self._unsubs:
            self._unsubs.pop()()

    async def _handle_predict(self, now) -> None:
        """Predict + push the target for each load."""
        _LOGGER.debug("Predict job firing at %s", now)
        await self.coordinator.async_predict_and_push()

    async def _handle_capture(self, now) -> None:
        """Capture actuals + log + calibrate for each load."""
        _LOGGER.debug("Capture job firing at %s", now)
        await self.coordinator.async_capture_and_log()
