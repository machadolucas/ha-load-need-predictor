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

from .coordinator import LoadNeedPredictorConfigEntry
from .persistence import model_to_dict

_REDACT = {
    "target_number_entity",
    "delivered_energy_entity",
    "delivered_runtime_entity",
    "person_entities",
    "guests_calendar_entity",
    "supply_temp_entity",
    "outdoor_temp_entity",
    "water_total_entity",
}
_RECENT_ROWS = 14  # last two weeks of training rows is plenty to diagnose


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: LoadNeedPredictorConfigEntry
) -> dict[str, Any]:
    """Return a redacted snapshot of the predictor's state."""
    coordinator = entry.runtime_data
    loads: dict[str, Any] = {}
    for subentry_id, cfg in coordinator.load_configs().items():
        result = (coordinator.data or {}).get(subentry_id)
        loads[subentry_id] = {
            "config": async_redact_data(vars(cfg), _REDACT),
            "model": model_to_dict(coordinator.model_for(subentry_id)),
            "result": vars(result) if result else None,
            "training_rows": len(coordinator.training.get(subentry_id, [])),
            "recent_training": coordinator.training.get(subentry_id, [])[-_RECENT_ROWS:],
            "eval_errors": coordinator.eval_errors.get(subentry_id, [])[-_RECENT_ROWS:],
        }
    return {"hub": async_redact_data(dict(entry.data), _REDACT), "loads": loads}
