"""Diagnostics dump for support and debugging.

Surfaces the live model state, the recent training rows and the evaluation ring
per load — exactly what's needed to reason about a prediction in the field —
without exposing anything sensitive (entity ids are config, not secrets, but we
redact them anyway to keep shared dumps clean).
"""

from __future__ import annotations

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.core import HomeAssistant

from .persistence import model_to_dict, tank_to_dict
from .runtime import LoadNeedPredictorConfigEntry

_REDACT = {
    "target_number_entity",
    "delivered_energy_entity",
    "delivered_runtime_entity",
    "controlled_switch_entity",
    "heating_active_entity",
    "person_entities",
    "guests_calendar_entity",
    "supply_temp_entity",
    "outdoor_temp_entity",
    "water_total_entity",
    "price_entity",
    "wind_entity",
    "weather_entity",
    "temp_history_entity",
}
_RECENT_ROWS = 14  # last two weeks of rows is plenty to diagnose


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: LoadNeedPredictorConfigEntry
) -> dict[str, Any]:
    """Return a redacted snapshot of both coordinators' state."""
    runtime = entry.runtime_data
    load = runtime.load
    loads: dict[str, Any] = {}
    for subentry_id, cfg in load.load_configs().items():
        result = (load.data or {}).get(subentry_id)
        tr = (runtime.tank.data or {}).get(subentry_id) if runtime.tank else None
        loads[subentry_id] = {
            "config": async_redact_data(vars(cfg), _REDACT),
            "model": model_to_dict(load.model_for(subentry_id)),
            "result": vars(result) if result else None,
            "tank": tank_to_dict(load.tanks[subentry_id]) if subentry_id in load.tanks else None,
            "tank_result": vars(tr) if tr else None,
            "training_rows": len(load.training.get(subentry_id, [])),
            "recent_training": load.training.get(subentry_id, [])[-_RECENT_ROWS:],
            "eval_errors": load.eval_errors.get(subentry_id, [])[-_RECENT_ROWS:],
        }

    forecasts: dict[str, Any] = {}
    forecast = runtime.forecast
    if forecast is not None:
        for subentry_id, cfg in forecast.forecast_configs().items():
            result = (forecast.data or {}).get(subentry_id)
            model = forecast.models.get(subentry_id)
            forecasts[subentry_id] = {
                "config": async_redact_data(vars(cfg), _REDACT),
                "model": model.to_dict() if model else None,
                "result": vars(result) if result else None,
                "log_rows": len(forecast.log.get(subentry_id, [])),
                "recent_log": forecast.log.get(subentry_id, [])[-_RECENT_ROWS:],
            }

    return {
        "hub": async_redact_data(dict(entry.data), _REDACT),
        "loads": loads,
        "forecasts": forecasts,
    }
