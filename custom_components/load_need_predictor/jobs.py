"""The two daily jobs, scheduled on the hub's wall-clock times.

- **Predict** (afternoon, after tomorrow's prices publish): each load predicts +
  pushes its target, and the price forecaster (re)fits and publishes the
  beyond-horizon slots.
- **Capture** (late evening): each load captures its actual delivery + calibrates,
  and the price forecaster reconciles past forecasts against realised prices.

Both capabilities share these two times; each coordinator does its own work.
"""

from __future__ import annotations

import logging

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_time_change

from .const import CONF_CAPTURE_TIME, CONF_PREDICT_TIME, DEFAULT_CAPTURE_TIME, DEFAULT_PREDICT_TIME
from .runtime import LoadNeedPredictorConfigEntry

_LOGGER = logging.getLogger(__name__)


def _parse_time(value: str) -> tuple[int, int, int]:
    """Parse a ``HH:MM[:SS]`` string into (hour, minute, second)."""
    parts = [int(p) for p in value.split(":")]
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


class PredictorJobs:
    """Owns the daily time-change listeners for one hub."""

    def __init__(self, hass: HomeAssistant, entry: LoadNeedPredictorConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self._unsubs: list[callable] = []

    @callback
    def async_start(self) -> None:
        """Register the predict and capture time-change listeners."""
        data = self.entry.data
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
        """Predict + push each load, and (re)build the price forecast."""
        _LOGGER.debug("Predict job firing at %s", now)
        runtime = self.entry.runtime_data
        await runtime.load.async_predict_and_push()
        if runtime.forecast is not None:
            await runtime.forecast.async_build_forecast()

    async def _handle_capture(self, now) -> None:
        """Capture + calibrate each load, and evaluate past forecasts."""
        _LOGGER.debug("Capture job firing at %s", now)
        runtime = self.entry.runtime_data
        await runtime.load.async_capture_and_log()
        if runtime.forecast is not None:
            await runtime.forecast.async_evaluate()
