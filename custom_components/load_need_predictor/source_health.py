"""Health of configured tank sources, without changing the control model.

Repairs name the configured source rather than guessing replacement entities.
Only an entity-registry rename event proves that two IDs belong to the same
record; a newly created status sensor is not interchangeable by name alone.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import (
    EnergyConverter,
    TemperatureConverter,
    VolumeConverter,
)

from .const import DOMAIN, SUBENTRY_TYPE_LOAD
from .models import LoadConfig, load_config_from_data
from .runtime import LoadNeedPredictorConfigEntry

SOURCE_ISSUE_GRACE = timedelta(minutes=15)
_ISSUE_PREFIX = "source_unhealthy_"
_ACTIVE = frozenset(
    {
        "on",
        "running",
        "run",
        "detecting",
        "washing",
        "rinsing",
        "spinning",
        "drying",
        "cool_down",
        "rinse_hold",
        "refreshing",
        "steam_softening",
    }
)
_IDLE = frozenset(
    {
        "off",
        "idle",
        "standby",
        "power_off",
        "stop",
        "stopped",
        "end",
        "finished",
        "complete",
        "completed",
    }
)
_FIELDS = {
    "delivered_energy_entity": "energy",
    "water_total_entity": "volume",
    "heating_active_entity": "binary",
    "controlled_switch_entity": "binary",
    "shower_motion_entity": "binary",
    "shower_humidity_entity": "humidity",
    "shower_ventilation_entity": "binary",
    "supply_temp_entity": "temperature",
    "cold_appliance_entities": "appliance",
    "appliance_household_activity_entities": "binary",
}


def appliance_state(value: str) -> bool | None:
    """Decode run status; unknown enums must never become evidence of idleness."""
    normalized = value.strip().lower().replace(" ", "_")
    if normalized in _ACTIVE:
        return True
    if normalized in _IDLE:
        return False
    return None


@dataclass(frozen=True)
class _Source:
    field: str
    entity_id: str
    kind: str


def _sources(cfg: LoadConfig) -> list[_Source]:
    result = []
    for field, kind in _FIELDS.items():
        value = getattr(cfg, field, None)
        for entity_id in (value,) if isinstance(value, str) else (value or ()):
            if entity_id:
                result.append(_Source(field, entity_id, kind))
    return result


def _problem(hass: HomeAssistant, source: _Source) -> str | None:
    state = hass.states.get(source.entity_id)
    if state is None:
        return "entity is missing"
    value = state.state
    if value.lower() in {"unknown", "unavailable"}:
        return f"state is {value.lower()}"
    if source.kind == "appliance":
        return "run state is unsupported" if appliance_state(value) is None else None
    if source.kind == "binary":
        return "expected an on/off state" if value not in {"on", "off"} else None
    try:
        number = float(value)
        if not math.isfinite(number):
            return "numeric reading is not finite"
    except (ValueError, TypeError):
        return "expected a numeric reading"
    unit = state.attributes.get("unit_of_measurement")
    if source.kind == "humidity" and (unit != "%" or not 0 <= number <= 100):
        return "expected relative humidity in % between 0 and 100"
    if source.kind == "energy" and unit not in EnergyConverter.VALID_UNITS:
        return "energy unit is missing or unsupported"
    if source.kind == "volume":
        # Match the tracker aliases so diagnostics cannot reject a usable source.
        if unit in {"l", "liter", "liters", "Liter", "litre", "litres"}:
            unit = "L"
        if unit not in VolumeConverter.VALID_UNITS:
            return "volume unit is missing or unsupported"
    if source.kind == "temperature":
        if unit not in TemperatureConverter.VALID_UNITS:
            return "temperature unit is missing or unsupported"
    return None


def source_health_warnings(hass: HomeAssistant, cfg: LoadConfig) -> list[str]:
    """Immediate per-source warnings, independent of event classification."""
    return [
        f"Configured source {source.entity_id}: {problem}"
        for source in _sources(cfg)
        if (problem := _problem(hass, source)) is not None
    ]


def _issue_id(entry_id: str, sid: str, source: _Source) -> str:
    identity = hashlib.sha256(f"{source.field}/{source.entity_id}".encode()).hexdigest()[:16]
    return f"{_ISSUE_PREFIX}{entry_id}_{sid}_{identity}"


@callback
def async_delete_stale_source_issues(
    hass: HomeAssistant, *, exclude_entry_id: str | None = None
) -> None:
    """Drop issues for removed/reconfigured sources, subentries, or hubs."""
    keep = {
        _issue_id(entry.entry_id, sid, source)
        for entry in hass.config_entries.async_entries(DOMAIN)
        if entry.entry_id != exclude_entry_id
        for sid, subentry in entry.subentries.items()
        if subentry.subentry_type == SUBENTRY_TYPE_LOAD
        for source in _sources(load_config_from_data(dict(subentry.data)))
    }
    for domain, issue_id in list(ir.async_get(hass).issues):
        if domain == DOMAIN and issue_id.startswith(_ISSUE_PREFIX) and issue_id not in keep:
            ir.async_delete_issue(hass, DOMAIN, issue_id)


class SourceHealthMonitor:
    """Cheap source checks with a grace period for startup and transient outages."""

    def __init__(self, hass: HomeAssistant, entry: LoadNeedPredictorConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self._failing_since: dict[str, datetime] = {}
        self._unsubs: list = []

    @callback
    def async_start(self) -> None:
        if self._unsubs:
            return
        async_delete_stale_source_issues(self.hass)
        self._unsubs = [
            async_track_time_interval(self.hass, self._check, timedelta(minutes=1)),
            self.hass.bus.async_listen(er.EVENT_ENTITY_REGISTRY_UPDATED, self._registry_updated),
        ]
        self._check(dt_util.utcnow())

    @callback
    def async_stop(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs.clear()

    @callback
    def _check(self, now: datetime) -> None:
        current = set()
        for sid, subentry in self.entry.subentries.items():
            if subentry.subentry_type != SUBENTRY_TYPE_LOAD:
                continue
            cfg = load_config_from_data(dict(subentry.data))
            for source in _sources(cfg):
                issue_id = _issue_id(self.entry.entry_id, sid, source)
                current.add(issue_id)
                problem = _problem(self.hass, source)
                if problem is None:
                    self._failing_since.pop(issue_id, None)
                    ir.async_delete_issue(self.hass, DOMAIN, issue_id)
                    continue
                started = self._failing_since.setdefault(issue_id, now)
                existing = (DOMAIN, issue_id) in ir.async_get(self.hass).issues
                if now - started >= SOURCE_ISSUE_GRACE or existing:
                    ir.async_create_issue(
                        self.hass,
                        DOMAIN,
                        issue_id,
                        is_fixable=False,
                        is_persistent=True,
                        severity=ir.IssueSeverity.WARNING,
                        translation_key="configured_source_unhealthy",
                        translation_placeholders={
                            "load": subentry.title,
                            "source": source.entity_id,
                            "problem": problem,
                        },
                    )
        self._failing_since = {
            key: value for key, value in self._failing_since.items() if key in current
        }
        async_delete_stale_source_issues(self.hass)

    @callback
    def _registry_updated(self, event) -> None:
        data = event.data
        old = data.get("old_entity_id")
        new = data.get("entity_id")
        if data.get("action") != "update" or not old or not new or old == new:
            return
        # HA emits old_entity_id only when updating the same registry record.
        # Creation/removal events cannot prove a replacement is equivalent.
        if er.async_get(self.hass).async_get(new) is None:
            return
        for subentry in tuple(self.entry.subentries.values()):
            updated = dict(subentry.data)
            changed = False
            for key, value in updated.items():
                if key.endswith("_entity") and value == old:
                    updated[key] = new
                    changed = True
                elif (
                    key.endswith("_entities") and isinstance(value, (list, tuple)) and old in value
                ):
                    updated[key] = [new if entity_id == old else entity_id for entity_id in value]
                    changed = True
            if changed:
                self.hass.config_entries.async_update_subentry(self.entry, subentry, data=updated)
        async_delete_stale_source_issues(self.hass)
