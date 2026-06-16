"""Constants for the Load Need Predictor integration.

This module is the *Home Assistant–side* constant table (it imports ``Platform``).
The pure prediction model keeps its own numeric seeds and tuning constants in
``predictor.py`` so that module stays import-clean for the HA-free unit tests.
Anything here is either a config key, a UI default, or a wiring constant.
"""

from __future__ import annotations

from homeassistant.const import Platform

DOMAIN = "load_need_predictor"

# Only sensors in v1: the predictor publishes its forecast + evaluation metrics
# and drives the scheduler through a service call, so it owns no controllable
# entities of its own.
PLATFORMS: list[Platform] = [Platform.SENSOR]

# ── Hub config-entry keys ────────────────────────────────────────────────────
CONF_NAME = "name"
CONF_PREDICT_TIME = "predict_time"  # when to forecast + push the target
CONF_CAPTURE_TIME = "capture_time"  # when to capture actuals + log the day

DEFAULT_NAME = "Load Need Predictor"
# Predict in the afternoon, after Nord Pool publishes tomorrow's prices, so the
# scheduler has the day's target before it plans the overnight window.
DEFAULT_PREDICT_TIME = "14:00:00"
# Capture just before midnight: late enough to catch the day's delivery, early
# enough to avoid day-boundary ambiguity in the recorder's daily statistics.
DEFAULT_CAPTURE_TIME = "23:55:00"

# ── Subentry (per-load) ──────────────────────────────────────────────────────
SUBENTRY_TYPE_LOAD = "load"

# Output: the Load Scheduler number entity whose value we set (minutes/day).
CONF_TARGET_NUMBER_ENTITY = "target_number_entity"
# Delivery feedback (training target): a total_increasing energy sensor whose
# daily ``change`` statistic is the kWh delivered that day.
CONF_DELIVERED_ENERGY_ENTITY = "delivered_energy_entity"
# Optional cross-check: a rolling/total runtime sensor (hours).
CONF_DELIVERED_RUNTIME_ENTITY = "delivered_runtime_entity"
CONF_RATED_POWER_KW = "rated_power_kw"  # kWh -> minutes conversion
# Occupancy drivers (the only feature with real day-ahead leverage).
CONF_PERSON_ENTITIES = "person_entities"
CONF_GUESTS_CALENDAR_ENTITY = "guests_calendar_entity"
# Log-only context sensors (recorded for a future v2, not used in the v1 model
# because they backtested as non-predictive day-ahead — see CLAUDE.md).
CONF_SUPPLY_TEMP_ENTITY = "supply_temp_entity"
CONF_OUTDOOR_TEMP_ENTITY = "outdoor_temp_entity"
CONF_WATER_TOTAL_ENTITY = "water_total_entity"
# Output clamp (minutes/day).
CONF_MIN_MINUTES = "min_minutes"
CONF_MAX_MINUTES = "max_minutes"

# Per-load UI defaults (tuned to the author's ~3 kW LVV; see CLAUDE.md data notes).
DEFAULT_RATED_POWER_KW = 3.0
DEFAULT_MIN_MINUTES = 40  # ~2 kWh safety floor: never starve the tank
DEFAULT_MAX_MINUTES = 240  # ~12 kWh cap: below the meter-reset outliers

# The scheduler's ``number.<load>_target`` accepts whole 15-minute steps.
TARGET_STEP_MINUTES = 15

# ── Persistence: a Store under .storage/ (included in HA backups) ─────────────
STORAGE_VERSION = 1
SAVE_DELAY = 10  # seconds — debounce writes
MAX_TRAINING_ROWS = 400  # cap the self-logged history (~13 months of daily rows)
EVAL_WINDOW_DAYS = 30  # rolling window for the MAE / bias evaluation metrics

# ── Load Scheduler interop ───────────────────────────────────────────────────
# Service used to push the predicted target (the only output path implemented in
# load_scheduler v0.1.0; an external "target source" is planned but not yet live).
SCHEDULER_SET_SERVICE_DOMAIN = "number"
SCHEDULER_SET_SERVICE = "set_value"
# Per-run events the scheduler fires (subscribed to later for finer delivery
# accounting; v1 measures delivery from the energy sensor's daily change).
SCHEDULER_EVENT_RUN_STARTED = "load_scheduler_run_started"
SCHEDULER_EVENT_RUN_ENDED = "load_scheduler_run_ended"
