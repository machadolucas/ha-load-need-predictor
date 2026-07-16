"""Constants for the Load Need Predictor integration.

This module is the *Home Assistant–side* constant table (it imports ``Platform``).
The pure prediction model keeps its own numeric seeds and tuning constants in
``predictor.py`` so that module stays import-clean for the HA-free unit tests.
Anything here is either a config key, a UI default, or a wiring constant.
"""

from __future__ import annotations

from homeassistant.const import Platform

DOMAIN = "load_need_predictor"

# Sensors publish the forecast + evaluation metrics; the button is a manual
# "predict / update now" trigger. The predictor drives the scheduler through a
# service call, so it owns no controllable load entities of its own.
PLATFORMS: list[Platform] = [Platform.SENSOR, Platform.BUTTON]

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
# Optional: the load's controlled switch/contactor (the relay the scheduler
# drives). Its recorded on-time — any source — is the runtime actually delivered,
# which drives deficit carryover. Leave unset to disable carryover (plain daily
# predictor). Must be the real contactor, not a thermostat-gated power sensor.
CONF_CONTROLLED_SWITCH_ENTITY = "controlled_switch_entity"
# Optional cap on the carried backlog (minutes). Defaults to a multiple of
# max_minutes so a multi-day skip can be recovered, bounded against runaway.
CONF_DEFICIT_CAP_MINUTES = "deficit_cap_minutes"
CONF_RATED_POWER_KW = "rated_power_kw"  # kWh -> minutes conversion
# Occupancy drivers (the only feature with real day-ahead leverage).
CONF_PERSON_ENTITIES = "person_entities"
CONF_GUESTS_CALENDAR_ENTITY = "guests_calendar_entity"
# Log-only context sensors (recorded for a future v2, not used in the v1 model
# because they backtested as non-predictive day-ahead — see CLAUDE.md).
CONF_SUPPLY_TEMP_ENTITY = "supply_temp_entity"
CONF_OUTDOOR_TEMP_ENTITY = "outdoor_temp_entity"
CONF_WATER_TOTAL_ENTITY = "water_total_entity"

# ── Tank state-of-charge (opt-in per load; see tank_model.py) ────────────────
# Binary sensor proving the element actually draws power (contactor AND
# internal thermostat, e.g. an LED/current detector); setting it enables the
# tank charge sensor + tracker.
CONF_HEATING_ACTIVE_ENTITY = "heating_active_entity"
CONF_TANK_VOLUME_L = "tank_volume_l"
CONF_TANK_SETPOINT_C = "tank_setpoint_c"
# Cold-inlet temperature at the house — an annual average constant, NOT the
# supply-water source sensor, which measures the lake and runs ~10 °C high in
# summer.
CONF_TANK_COLD_IN_C = "tank_cold_in_c"
# Optional low-charge boost threshold; below this calibrated SoC the tracker
# re-runs predict+push so the scheduler plans more heating. Empty disables.
CONF_TANK_BOOST_SOC_PCT = "tank_boost_soc_pct"

# Output clamp (minutes/day).
CONF_MIN_MINUTES = "min_minutes"
CONF_MAX_MINUTES = "max_minutes"

# ── Price-forecast subentry (beyond-horizon price estimate for the scheduler) ─
SUBENTRY_TYPE_PRICE_FORECAST = "price_forecast"

CONF_PRICE_ENTITY = "price_entity"  # actual buy price (€/kWh) — fit + evaluate
CONF_WIND_ENTITY = "wind_entity"  # Finland wind production forecast (series attr)
CONF_WEATHER_ENTITY = "weather_entity"  # daily temperature forecast source
CONF_TEMP_HISTORY_ENTITY = "temp_history_entity"  # actual outdoor temp — for fitting
CONF_FORECAST_DAYS = "forecast_days"  # how many future days to publish
CONF_FIT_DAYS = "fit_days"  # LTS lookback used to fit the model

DEFAULT_FORECAST_DAYS = 3
DEFAULT_FIT_DAYS = 365

# Per-load UI defaults (tuned to the author's ~3 kW LVV; see CLAUDE.md data notes).
DEFAULT_RATED_POWER_KW = 3.0
DEFAULT_MIN_MINUTES = 40  # ~2 kWh safety floor: never starve the tank
DEFAULT_MAX_MINUTES = 240  # ~12 kWh cap: below the meter-reset outliers
DEFAULT_TANK_VOLUME_L = 300.0
DEFAULT_TANK_SETPOINT_C = 75.0
DEFAULT_TANK_COLD_IN_C = 12.0
DEFAULT_TANK_BOOST_SOC_PCT = 20.0
TANK_TICK_SECONDS = 60  # tank tracker tick cadence

# ── Deficit carryover ────────────────────────────────────────────────────────
# Cap the backlog at this multiple of max_minutes when no explicit cap is set —
# enough to recover a ~2-day skip while bounding runaway after a long outage.
DEFAULT_DEFICIT_CAP_FACTOR = 2.0
# Slack (minutes) for the "clean cycle" gate on the demand learner: a cycle
# is clean — safe to learn the gain from — only when no backlog is being worked
# off and the scheduler ran roughly the full ask, so the meter reflects true
# demand rather than a price-driven skip/defer. One step plus a margin.
CLEAN_CYCLE_TOL_MINUTES = 30

# The scheduler's ``number.<load>_target`` accepts whole 15-minute steps.
TARGET_STEP_MINUTES = 15

# ── Dashboard card (bundled + auto-registered as a frontend module) ───────────
# Served from this integration's ``www/`` dir; the URL is registered on setup so
# ``type: custom:load-need-predictor-card`` works without a manual Lovelace
# resource. See ``frontend.py``.
CARD_FILENAME = "load-need-predictor-card.js"
CARD_URL = f"/{DOMAIN}/{CARD_FILENAME}"

# ── Persistence: a Store under .storage/ (included in HA backups) ─────────────
STORAGE_VERSION = 1
SAVE_DELAY = 10  # seconds — debounce writes
MAX_TRAINING_ROWS = 400  # cap the self-logged history (~13 months of daily rows)
EVAL_WINDOW_DAYS = 30  # rolling window for the MAE / bias evaluation metrics

# ── Structural refit (the prior → learned transition) ────────────────────────
# Below this many clean, occupancy-labelled rows the seeds + online gain carry
# the model; above it, blend in an empirical fit of E_base / E_draw_per_person.
MIN_REFIT_SAMPLES = 14

# ── Load Scheduler interop ───────────────────────────────────────────────────
# Service used to push the predicted target (the only output path implemented in
# load_scheduler v0.1.0; an external "target source" is planned but not yet live).
SCHEDULER_SET_SERVICE_DOMAIN = "number"
SCHEDULER_SET_SERVICE = "set_value"
# Per-run events the scheduler fires (subscribed to later for finer delivery
# accounting; v1 measures delivery from the energy sensor's daily change).
SCHEDULER_EVENT_RUN_STARTED = "load_scheduler_run_started"
SCHEDULER_EVENT_RUN_ENDED = "load_scheduler_run_ended"
