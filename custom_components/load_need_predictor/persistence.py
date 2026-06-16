"""Store-backed persistence for model state and the self-logged training set.

Everything the predictor learns lives in a ``Store`` under ``.storage/`` (so it
is part of Home Assistant backups): per load, the current :class:`ModelState`,
the capped list of daily training rows (the dataset HA's statistics can't keep,
because it includes occupancy), and a rolling ring of recent errors for the
evaluation sensors.

On-disk shape::

    {
      "<subentry_id>": {
        "model":    {e_base, e_draw_per_person, guest_bonus, gain, ...},
        "training": [ {date, people_home, predicted_kwh, actual_kwh, ...}, ... ],
        "eval":     [error_minutes, ...]
      },
      ...
    }
"""

from __future__ import annotations

from collections.abc import Callable

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.storage import Store

from .const import DOMAIN, SAVE_DELAY, STORAGE_VERSION
from .predictor import ModelState


class PredictorStore:
    """Thin wrapper over ``Store`` keyed per hub config entry."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        self._store: Store[dict] = Store(hass, STORAGE_VERSION, f"{DOMAIN}.{entry_id}")

    async def async_load(self) -> dict:
        """Return the saved ``{subentry_id: {...}}`` mapping (empty if none)."""
        return await self._store.async_load() or {}

    @callback
    def async_schedule_save(self, snapshot: Callable[[], dict]) -> None:
        """Debounced save; ``snapshot`` is called when the write actually fires."""
        self._store.async_delay_save(snapshot, SAVE_DELAY)


def model_to_dict(state: ModelState) -> dict:
    """Serialise a :class:`ModelState` for the Store."""
    return {
        "e_base": state.e_base,
        "e_draw_per_person": state.e_draw_per_person,
        "guest_bonus": state.guest_bonus,
        "gain": state.gain,
        "empty_house_factor": state.empty_house_factor,
        "sample_count": state.sample_count,
        "version": state.version,
    }


def model_from_dict(data: dict | None) -> ModelState:
    """Rebuild a :class:`ModelState`, falling back to the seeded defaults.

    Unknown/absent keys take their dataclass defaults, so an older stored shape
    still loads (forward-compatible).
    """
    if not data:
        return ModelState()
    defaults = ModelState()
    return ModelState(
        e_base=float(data.get("e_base", defaults.e_base)),
        e_draw_per_person=float(data.get("e_draw_per_person", defaults.e_draw_per_person)),
        guest_bonus=float(data.get("guest_bonus", defaults.guest_bonus)),
        gain=float(data.get("gain", defaults.gain)),
        empty_house_factor=float(data.get("empty_house_factor", defaults.empty_house_factor)),
        sample_count=int(data.get("sample_count", defaults.sample_count)),
        version=str(data.get("version", defaults.version)),
    )
