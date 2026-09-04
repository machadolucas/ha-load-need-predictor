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
        "eval":     [error_minutes, ...],
        "tank":     {deficit_kwh, hot_fraction, standby_w, calibrated, ...}
      },
      ...
    }

The ``tank`` key is additive (present only for loads with tank tracking enabled),
so ``STORAGE_VERSION`` stays 1 — an older file without it loads fine.
"""

from __future__ import annotations

from collections.abc import Callable

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.storage import Store

from .const import DOMAIN, SAVE_DELAY, STORAGE_VERSION
from .predictor import ModelState
from .tank_model import TankState


class PredictorStore:
    """Thin wrapper over ``Store`` keyed per hub config entry."""

    def __init__(self, hass: HomeAssistant, entry_id: str, suffix: str = "") -> None:
        # The load coordinator uses suffix="" (one file per hub); the price
        # forecaster uses a distinct suffix so the two don't clobber each other.
        self._store: Store[dict] = Store(hass, STORAGE_VERSION, f"{DOMAIN}.{entry_id}{suffix}")

    async def async_load(self) -> dict:
        """Return the saved ``{subentry_id: {...}}`` mapping (empty if none)."""
        return await self._store.async_load() or {}

    @callback
    def async_schedule_save(self, snapshot: Callable[[], dict]) -> None:
        """Debounced save; ``snapshot`` is called when the write actually fires."""
        self._store.async_delay_save(snapshot, SAVE_DELAY)

    async def async_save_now(self, data: dict) -> None:
        """Write immediately, bypassing the debounce (shutdown / explicit flush)."""
        await self._store.async_save(data)


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
        "deficit_minutes": state.deficit_minutes,
        "pending_owed_minutes": state.pending_owed_minutes,
        "cycle_start_iso": state.cycle_start_iso,
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
        deficit_minutes=float(data.get("deficit_minutes", defaults.deficit_minutes)),
        pending_owed_minutes=float(data.get("pending_owed_minutes", defaults.pending_owed_minutes)),
        cycle_start_iso=str(data.get("cycle_start_iso", defaults.cycle_start_iso)),
    )


def tank_to_dict(state: TankState) -> dict:
    """Serialise a :class:`TankState` for the Store (the ``"tank"`` key)."""
    return {
        "deficit_kwh": state.deficit_kwh,
        "hot_fraction": state.hot_fraction,
        "standby_w": state.standby_w,
        "calibrated": state.calibrated,
        "anchor_latched": state.anchor_latched,
        "energy_baseline_kwh": state.energy_baseline_kwh,
        "water_baseline_l": state.water_baseline_l,
        "water_baseline_iso": state.water_baseline_iso,
        "pending_fallback_kwh": state.pending_fallback_kwh,
        "last_tick_iso": state.last_tick_iso,
        "last_anchor_iso": state.last_anchor_iso,
        "last_boost_iso": state.last_boost_iso,
        "boost_armed": state.boost_armed,
        "cycle_start_iso": state.cycle_start_iso,
        "cycle_energy_in_kwh": state.cycle_energy_in_kwh,
        "cycle_liters": state.cycle_liters,
        "cycle_clean": state.cycle_clean,
        # v2: tuples become JSON lists; ``cycle_unclamped_kwh`` is genuinely
        # nullable (None = "same as the clamped deficit", the pre-v2 shape).
        "hot_fraction_profile": list(state.hot_fraction_profile),
        "hysteresis_kwh": state.hysteresis_kwh,
        "residual_ratio": state.residual_ratio,
        "cycle_unclamped_kwh": state.cycle_unclamped_kwh,
        "cycle_gross_kwh": state.cycle_gross_kwh,
        "cycle_relax_kwh": state.cycle_relax_kwh,
        "cycle_hot_liters_by_bucket": list(state.cycle_hot_liters_by_bucket),
        "led_kwh_since_counter": state.led_kwh_since_counter,
        "version": state.version,
    }


def tank_from_dict(data: dict | None) -> TankState | None:
    """Rebuild a :class:`TankState`, or ``None`` when nothing is stored.

    Same defaults-tolerant style as :func:`model_from_dict`: every field falls
    back to its dataclass default, so an older stored shape still loads (a pre-v2
    payload has none of the ``hot_fraction_profile`` / ``cycle_*`` v2 keys and
    gets the seeds — an empty profile means "flat at ``hot_fraction``" and a
    ``None`` ``cycle_unclamped_kwh`` means "start the learning ledger from the
    clamped deficit"). The cumulative baselines and ``cycle_unclamped_kwh`` are
    genuinely nullable, so they pass ``None`` through rather than coercing.
    """
    if not data:
        return None
    defaults = TankState(deficit_kwh=0.0)

    def _opt_float(key: str, fallback: float | None) -> float | None:
        value = data.get(key, fallback)
        return None if value is None else float(value)

    def _float_tuple(key: str, fallback: tuple[float, ...]) -> tuple[float, ...]:
        value = data.get(key)
        if not value:
            return fallback
        return tuple(float(item) for item in value)

    return TankState(
        deficit_kwh=float(data.get("deficit_kwh", defaults.deficit_kwh)),
        hot_fraction=float(data.get("hot_fraction", defaults.hot_fraction)),
        standby_w=float(data.get("standby_w", defaults.standby_w)),
        calibrated=bool(data.get("calibrated", defaults.calibrated)),
        anchor_latched=bool(data.get("anchor_latched", defaults.anchor_latched)),
        energy_baseline_kwh=_opt_float("energy_baseline_kwh", defaults.energy_baseline_kwh),
        water_baseline_l=_opt_float("water_baseline_l", defaults.water_baseline_l),
        water_baseline_iso=str(data.get("water_baseline_iso", defaults.water_baseline_iso)),
        pending_fallback_kwh=float(data.get("pending_fallback_kwh", defaults.pending_fallback_kwh)),
        last_tick_iso=str(data.get("last_tick_iso", defaults.last_tick_iso)),
        last_anchor_iso=str(data.get("last_anchor_iso", defaults.last_anchor_iso)),
        last_boost_iso=str(data.get("last_boost_iso", defaults.last_boost_iso)),
        boost_armed=bool(data.get("boost_armed", defaults.boost_armed)),
        cycle_start_iso=str(data.get("cycle_start_iso", defaults.cycle_start_iso)),
        cycle_energy_in_kwh=float(data.get("cycle_energy_in_kwh", defaults.cycle_energy_in_kwh)),
        cycle_liters=float(data.get("cycle_liters", defaults.cycle_liters)),
        cycle_clean=bool(data.get("cycle_clean", defaults.cycle_clean)),
        hot_fraction_profile=_float_tuple("hot_fraction_profile", defaults.hot_fraction_profile),
        hysteresis_kwh=float(data.get("hysteresis_kwh", defaults.hysteresis_kwh)),
        residual_ratio=float(data.get("residual_ratio", defaults.residual_ratio)),
        cycle_unclamped_kwh=_opt_float("cycle_unclamped_kwh", defaults.cycle_unclamped_kwh),
        cycle_gross_kwh=float(data.get("cycle_gross_kwh", defaults.cycle_gross_kwh)),
        cycle_relax_kwh=float(data.get("cycle_relax_kwh", defaults.cycle_relax_kwh)),
        cycle_hot_liters_by_bucket=_float_tuple(
            "cycle_hot_liters_by_bucket", defaults.cycle_hot_liters_by_bucket
        ),
        led_kwh_since_counter=float(
            data.get("led_kwh_since_counter", defaults.led_kwh_since_counter)
        ),
        version=str(data.get("version", defaults.version)),
    )
