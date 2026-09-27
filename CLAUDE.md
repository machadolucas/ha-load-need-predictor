# CLAUDE.md — Load Need Predictor

Working notes for AI agents and future-me. Read this before changing code.

**Status (alpha):** built incrementally in milestones (see the approved plan at
`~/.claude/plans/i-want-to-build-compressed-backus.md` — the *why*; this file is
the *how*). Tests run under Python **3.13** (Home Assistant doesn't support 3.14).
The CI workflow files exist locally but are git-ignored (the push token lacks
`workflow` scope) — mirrors the `ha-load-scheduler` repo.

## What this is

A Home Assistant custom integration (`load_need_predictor`) that predicts *how
much* a flexible load needs to run each day and pushes that to the
[`load_scheduler`](https://github.com/machadolucas/ha-load-scheduler) integration,
which decides *when*. First (only) load: the hot-water heater (LVV). It replaces
the `input_number.water_heater_hours` "set the runtime by hunch" knob in the
`macserver` repo.

This integration consumes only the scheduler's **public surface** — it writes the
scheduler's target `number` and (later) listens to its run events. It never
imports scheduler internals; the dependency runs one way.

Both `recorder` and `load_scheduler` are **soft** deps (`after_dependencies`):
the integration must load and keep predicting/publishing even if either is
absent. The predict path needs neither; only the evening capture/learning path
reads statistics, and it guards for a missing recorder.

## The data findings (do NOT re-litigate)

Validated on ~3–4 months of long-term statistics for the author's LVV:

- Daily delivered energy (`sensor.leddetector_water_heater_energy` daily `change`)
  ~7.3 kWh mean, **CV ~45%, lag-1 autocorrelation ~0.07** — stochastic draw.
- **Temperature/season has ~zero day-ahead power.** Supply-water-temp regression
  R² ≈ 0.006; forward-CV *worse than a constant* (+3.5 kWh bias). A flat constant
  (MAE ≈ 2.65 kWh) beats every temperature/season model.
- **Total household water**: same-day r ≈ 0.29, but lag-1 / trailing-3d (the only
  causal versions) lose to the constant; the OCR meter also had multi-week
  dropouts. Fragile, not predictive ahead of time.
- **Occupancy is the only feature with real leverage** and has **no LTS** (only
  ~10 days raw). → We must self-log occupancy + outcomes daily.

**Tank (v0.9.0 replay of 2026-08-28→09-04):**

- Anchors fire roughly daily; true trip residual rms ≈ 1 kWh (± ~4.5 SoC points).
- Bias ≈ +0.1–0.3 kWh against the replay-fitted params (`hot_fraction` 0.248,
  `standby_w` 114 W) — small enough that the old hard floor was solving a
  problem the physics didn't have.
- The old heating-active floor (v0.8.1) pinned the display at 95.4 % for 146 of
  997 heating minutes across the week and caused 6 non-anchor jumps/week — the
  soft-saturation redesign (below) replaces it.
- Post-trip re-heat delivers ≈ 0.81 kWh roughly 45–60 min after a trip.
- powercalc's delivered-energy counter tracks LED × 3 kW exactly but steps
  0.5 kWh every 10 min rather than continuously — the source of the
  display-only LED smoothing.
- Hot/cold attribution noise (garden hose vs. shower) is the remaining error
  floor; a hot-outlet pipe sensor is the only real fix, deferred.

Consequence: v1 = calibrated, occupancy-gated constant + online gain + safety
floor. Temperature/water are **logged only**, not used in the prediction.

## Architecture

- **Hub config entry** — the global predict/capture schedule + both coordinators
  (held in `runtime.RuntimeData` on `entry.runtime_data`).
- **`load` subentries** — each one's sensors, scheduler-target link,
  delivery/occupancy sources, clamp.
- **A `price_forecast` subentry** — the beyond-horizon price forecaster (its
  inputs, the published `data_today` sensor, accuracy metrics).

The two capabilities share only the hub's two daily times; their state and tests
are otherwise independent. The `ConfigSubentry` API is relatively new; the
`homeassistant` floor in `hacs.json` tracks it.

### Modules (`custom_components/load_need_predictor/`)

| File | Role | HA? |
|---|---|---|
| `predictor.py` | **Pure** load model: features → kWh → minutes, seeds, gain EWMA, prior↔empirical blend, `build_features`, rolling MAE, deficit carryover (`close_cycle`/`open_cycle`) | no |
| `price_model.py` | **Pure** price model: ridge regression of price on wind+temp with a cold interaction; seed fallback; fit/predict/serialize | no |
| `spot_forecast.py` | **Pure** Wattcast parse + compact cache (`WattcastSeries`), spot→retail mapping (`fit_retail_mapping`), local intraday shape, DST-safe 15-min `build_slots`, per-hour scoring + `select_primary` | no |
| `tank_model.py` | **Pure** tank state-of-charge: energy-deficit integration (`apply_tick`), 100 %-anchor + EWMA calibration of `hot_fraction`/`standby_w`, hot-flow cap, meter-misread/fallback guards, boost gating (`should_boost`), liters/showers helpers | no |
| `models.py` | Subentry config → frozen `LoadConfig` / `PriceForecastConfig`; `LoadConfig.tank_tracking_enabled` is the one "is the tank live?" predicate | no |
| `statistics_source.py` | Load delivery from the recorder (`change` over an arbitrary window via the singular `statistic_during_period`, kWh-normalised) + commanded switch on-time over a window (`async_commanded_minutes`, for deficit carryover); every helper returns `None` on a recorder error, never raises | yes |
| `forecast_source.py` | Wind series + daily temp forecast + LTS fit rows / realised price (daily + hourly); the Wattcast HTTP fetch (never raises → `WattcastFetch`); real-price slot/pair helpers | yes |
| `occupancy.py` | Duration-based occupancy: residents home ≥12 h over the trailing 24 h (from history) + guests weighted by visit length (next-24 h calendar); instantaneous fallbacks | yes |
| `persistence.py` | `Store` (load: model+training+eval; forecast uses a `.forecast` file) | yes |
| `actuation.py` | Resilient `number.set_value` push to the scheduler target | yes |
| `jobs.py` | The two daily jobs; drives both coordinators (each capability step isolated — one failing never skips the other) | yes |
| `coordinator.py` | Load `DataUpdateCoordinator`; per-load `LoadResult` + the push-time `PublishedTarget` cache; predict/capture serialised by one lock, per-load work isolated | yes |
| `forecast_coordinator.py` | Price-forecast coordinator: 5-min due-check tick → hourly Wattcast fetch (persisted cache, backoff, repair issue) → mapping refit → build slots → per-source snapshot; daily local refit; nightly per-source scoring; `ForecastResult` | yes |
| `tank_tracker.py` | 60 s-tick coordinator: reads counters/switch/detector states, drives `tank_model.apply_tick`, publishes per-load `TankResult`, fires the low-charge boost | yes |
| `runtime.py` | `RuntimeData` (both coordinators) + the `ConfigEntry` type alias | yes |
| `config_flow.py` | Hub flow + `load` and `price_forecast` subentry wizards | yes |
| `entity.py` | `PredictorEntity` (load) + `ForecastEntity` bases | yes |
| `sensor.py` / `button.py` | Per-load + price-forecast sensors; "predict/forecast now" buttons | yes |
| `diagnostics.py` | Redacted diagnostics dump (loads + forecasts) | yes |
| `frontend.py` | Serves + auto-registers the dashboard card (long-cached static path registered first, then an extra JS module with a guarded `?v=<content-hash>` cache-bust that falls back to the bare URL); best-effort, never breaks setup | yes |
| `www/load-need-predictor-card.js` | The Lovelace diagnostic card (vanilla JS, no build) + its `ha-form` editor | no |

## The price forecast (read before touching `price_model.py` / `spot_forecast.py`)

- **Two sources, one series.** Wattcast (`wattcast.eu`, free/key-less, server-side
  gradient-boosted trees + Open-Meteo weather + LLM outage adjustments; ≈ 7 days,
  15-min, p10/p50/p90 spot €/MWh ex-VAT) is primary; the local ridge (below) is the
  fallback beyond Wattcast's coverage / before a first fetch. `build_slots` picks the
  source **per slot**: Wattcast's cached *settled* spot first (where the real-price
  entity lags it — mapped to retail, `p10 = p90 = buy`, `src: "wattcast_known"`,
  never snapshotted/scored), then the primary, then the other forecast source
  (`prefer_local` when local is primary, so its short weather reach doesn't drop the
  Wattcast days beyond). The HA repo
  `SectorTll/wattcast-homeassistant` is just an API client — there is no model to
  "replicate" locally; don't try to rebuild their GBT here.
- **`price_model.py` and `spot_forecast.py` must stay Home-Assistant-free**
  (importlib-tested like `predictor.py`). Output contract for the scheduler: a
  `data_today` attribute of 15-min `{start, end, buy[, p10, p90], src}` slots —
  **tz-aware ISO** starts, `buy` in **€/kWh all-in** — from the first slot without a
  real price to local midnight after `forecast_days` days; the scheduler ignores
  overlap/extra keys and adds its own `forecast_price_margin`.
- **Polling discipline (API terms + user requirement):** one request per hourly issue
  (the first HH:32 after the last fetch — Wattcast re-issues ≈ :25), only from
  the 5-minute due-check tick (own `async_track_time_interval`, not `update_interval`).
  The raw series is **persisted** and only replaced by a newer successful fetch — a
  failure keeps serving the cache (`stale` once > 2 h). Every 200 is judged against the
  cache (`_judge`): issue order (`made_at`) is enforced whatever the cache's coverage —
  an **older** issue is *degraded*; the **same** issue is *unchanged* unless it strictly
  extends coverage (forecast or settled); a newer issue without forecast still ahead is
  *degraded* while the cache is usable. *Degraded* keeps the cache, sets `fetch_error`,
  starts the failure clock and waits for the **next issue**, not the retry ladder (the
  same issue would come back — no extra requests). *Unchanged* keeps the cache, mapping
  and failure bookkeeping, only moving `fetched_at` + the next-issue deadline. Startup
  skips the fetch while the cache is from the current issue window. A known-only series
  stays in use while its settled spot reaches past the slot start. Failures back off
  5 → 15 → 30 → 60 min (≥ `Retry-After`); the whole fetch state (`next_fetch`,
  `failures`, `failing_since`, `last_error`) persists, so a reload keeps a backoff (a
  restored deadline > 7 days out is treated as corruption, not a multi-day
  `Retry-After`).
  6 h of consecutive failures raises repair issue `wattcast_unreachable_<sid>`,
  deleted on the next success, when Wattcast is turned off, and by a setup-time sweep
  once its subentry is gone (`async_delete_stale_wattcast_issues`). The button forces a
  fetch only if the cache is ≥ 15 min old *and* no failure backoff or degraded
  deadline is in force (`failures == 0` and no `last_error`). The
  Store is flushed on unload — under the lock, then closed to further saves, so a
  fetch still awaiting HTTP can't overwrite the reloaded cache — so a reload reads the fresh cache. The tick also rebuilds
  (no fetch) once the first published slot is over or at local midnight, so an outage
  ages expired slots out instead of publishing them as `ok`.
- **Config fingerprint:** each subentry's state is stamped with the config it was
  learned under (`zone`, `price_entity`, `price_series_entity`, captured at load — not
  at the unload flush, which already sees the new config). On a mismatch at load the
  dependent state is dropped: any key → pairs + mapping; zone/price entity → the
  scored log; zone → the Wattcast cache + fetch state (fetch now); price entity → the
  local model + shape. A payload without a fingerprint is adopted, not wiped.
- **Spot → retail mapping** (`fit_retail_mapping`): `buy = a⁺·max(s,0)/1000 +
  a⁻·min(s,0)/1000 + b[daytype(wd/sat/sun), local hour]`, hierarchically shrunk, on a
  14-day buffer of (settled spot, real buy) pairs from the optional
  `price_series_entity` (Nord Pool-shaped slot lists), else the buy-price sensor's
  current value. Refit on every successful fetch *and* on every build/reload (from the
  cached settled spot), so a reconfigure takes effect immediately. A level whose
  groups hold one pair each (day 1 of an hourly series) can't separate noise from
  signal, so it is shrunk with the full `SHRINK_LAMBDA` (not treated as noise-free). Validated 2026-09-24 on the author's contract: recovers ×1.255 VAT +
  4.70 c night (22–07) / 6.77 c day (07–22) with ~1e-6 residual from 2 days of pairs.
- **Local fallback:** features `[temp, wind, cold_hinge, wind×cold_hinge]`,
  `cold_hinge = max(0,−temp)`; wind in **GW** (sensor series GW, state/LTS MW —
  normalise). Fit daily on LTS; seed formula until enough history. Daily price ×
  learned intraday shape (per daytype × hour offsets from 28 days of hourly LTS).
  Days past the wind feed use climatological wind (`days[].wind_src`). Temperatures are
  °C throughout: the weather forecast is converted from the entity's
  `temperature_unit`, and LTS is requested with `units={"temperature": "°C"}`. Recorder
  errors in `forecast_source` are caught (→ `{}`/`[]`/None), never abort a build.
- **Scoring:** each rebuild upserts, per still-forecast day (≥ 20 forecast hours),
  per-hour vectors for `wattcast`, `wattcast_raw` (`p50Raw`, i.e. without their LLM
  adjustments) and `local` → the log keeps the *last pre-publication* forecast. The
  capture job scores them vs hourly LTS (`scores`, `daily_err`) — only once the day's
  **next local midnight** has passed (a 25-h DST day isn't final at bucket + 24 h) and
  ≥ `EVAL_MIN_ACTUAL_HOURS` (20) realised hours exist; a day with no hourly LTS falls
  back to that exact day's daily mean (LTS `start` is epoch **seconds**; never a
  neighbouring day). Still unscorable `EVAL_GIVE_UP_DAYS` (7) after it ended → marked
  `unscorable` and skipped, so nights don't re-query it forever. `select_primary` picks
  the lowest trailing-14-day hourly MAE once each source has ≥ 7 scored days, else
  Wattcast. Wattcast's own published live MAE (FI, 2026-09): ~29 €/MWh D+1 → ~41 D+6.
- Big attributes are `_unrecorded_attributes` (≫ the recorder's 16 KB cap).
- Forecast *price/opportunity, not demand*: treat the tank as a buffer and let
  the scheduler shift discretionary heating into the forecast-cheap window; the
  minimum-service floor is the safety net.

## The model contract (read before touching `predictor.py`)

- **`predictor.py` must stay Home-Assistant-free** — the pure unit tests load it
  via `importlib` (see `tests/test_predictor.py`). No `homeassistant` imports.
- Energy is in **kWh**, runtime in **minutes**; convert at the boundary:
  `minutes = kWh / rated_kW × 60`, then round to 15 and clamp to `[min, max]`.
- **Seeds** (cold start, day 1): `E_base = 3.0` kWh, `E_draw_per_person = 2.2`,
  `guest_bonus = 2.5`, `gain = 1.0`, `empty_house_factor = 0.4`.
- **Online gain**: `r = clamp(actual/predicted, 0.5, 2.0)`,
  `target = g_pred · r`, `gain = clamp((1−β)·gain + β·target, 0.7, 1.5)`,
  `β = 0.15` (~6-day half-life). `g_pred` is the gain the prediction was made
  with (the row's `gain`; the current gain for legacy rows): `predicted_kwh`
  already includes it, so EWMA-ing the raw `r` has the fixed point `g = true/g`
  (→ √true). The clamps are the anti-drift guardrail — the gain corrects ±50%
  but can't run away. A closed-loop test pins convergence to the true ratio.
- **Prior→empirical blend**: `θ = (n_prior·θ_prior + n·θ_emp)/(n_prior+n)`,
  `n_prior = 10`. The structural refit (`refit_occupancy_params`) must invert
  `predict_kwh`'s own equation: `y = kwh/row_gain − guest_bonus·guests`, and
  `÷ empty_house_factor` for `p == 0` rows, then OLS `y ~ p` — fitting raw
  actuals biased `E_base`/`E_draw` and made the gain double-apply. Rows must be
  `data_quality` **and** `clean_cycle` (missing key = legacy = clean); the
  target is `demand_kwh` when the row has one. Needs ≥ `MIN_REFIT_SAMPLES` rows
  incl. ≥1 zero-person **and** ≥1 multi-person (`p ≥ 2`) day, else `None`.
- **Safety floor** (`min_minutes`, default 40 ≈ 2 kWh) always wins: even a
  "nobody home" prediction keeps standby + one shower's worth. The one
  exception is a degenerate band with no 15-min step inside `[min, max]`
  (e.g. min = max = 40): `clamp_minutes` then returns the floor-rounded
  **max** (never above `max_minutes`, never < 0). The config flow now rejects
  such bands — and a heating detector without the controlled switch (the
  tank could never anchor) — so only legacy entries hit this.
- **Data-quality gate**: ignore days with delivered energy ≤ 0.2 or > 18 kWh
  (meter resets/outliers) when calibrating.
- **Deficit carryover** (opt-in via a load's `controlled_switch_entity`): a
  bounded backlog (a one-number proxy for tank state-of-charge) so a day the
  scheduler skips/under-runs on price/solar is made up the next day. Each predict
  closes the previous predict→predict cycle — `deficit = close_cycle(pending_owed,
  commanded, cap)` where `commanded` is the switch's recorded **on-time** (any
  source: scheduler, manual, automation — *not* the thermostat-gated energy
  meter, so an over-ask self-heals) — then `open_cycle` pushes `need + deficit`
  clamped to `[min,max]`; the uncapped `pending_owed` persists a deficit the daily
  `max` can't satisfy. Cap defaults to `2 × max_minutes`. Both functions are pure
  + tested. The row keeps `predicted_minutes` as the occupancy *need* (so
  error/eval/gain measure demand), and `pushed_minutes`/`deficit_minutes` for what
  was actually sent. The gain learns only on **clean** cycles (no backlog in play
  *and* the switch ran ≈ the full ask — `CLEAN_CYCLE_TOL_MINUTES`), so a skip/defer
  no longer drags it down. No switch / no recorder → backlog stays 0 = the plain
  daily predictor.
- **Cycles are daily.** Only the first predict of a local day closes/opens a
  cycle. A same-day re-predict (the "Predict now" button, the tank's
  low-charge boost) *re-plans*: no `close_cycle`, keeps `cycle_start_iso` and
  `deficit_minutes`, recomputes `pending_owed` via `open_cycle(need, deficit)`
  and pushes. (Closing a few-hours-old cycle found little on-time and rolled
  ~the whole ask into backlog on top of a fresh need.)
- **Capture window** = `statistics_source.capture_window(now)`, read with
  `statistic_during_period` (blends 5-min short-term stats) — a calendar-day
  read at 23:55 never counted 23:00–24:00. `end = floor₅ₘᵢₙ(utcnow − 1 min)`
  (23:55:00 → 23:50: the last bucket is compiled ~10 s *after* its period, and
  the API silently returns what exists; flooring also drops seconds/µs);
  `start` = the same local wall-clock time on the previous day, so daily
  captures **tile exactly** — including DST days, where that is 23/25 elapsed
  hours (a fixed 24 h would leave a 1-hour hole/overlap). Endpoints are UTC.
  Commanded on-time for the clean gate uses the same window; the row stays
  keyed by today's date and records `capture_window_start`/`_end`.
- **Energy-balance demand** (tank-tracked + calibrated loads): each capture
  snapshots the control ledger `tank.deficit_kwh` onto the row as
  `tank_deficit_end_kwh` + `tank_deficit_end_at` (when sampled). When the
  previous date's row has one too **and** its timestamp is within
  `ENERGY_BALANCE_TOL` (15 min) of this window's start (and this snapshot of
  its end) — a same-date row alone doesn't prove the spans meet —
  `demand_kwh = max(0, actual + end_today − end_prev)` (energy in − ΔSoC over the
  same 24 h; pure `energy_balance_demand`) is stored and is what the gain, eval
  errors and refit learn from, with the row counted **clean** — the backlog is
  accounted for, so the heavy-draw refill days (~40 % on a calibrated tank)
  stop being thrown away and biasing the gain low. `is_valid_delivery` still
  gates on the raw actual. Without both snapshots the clean-cycle gate above
  applies unchanged. **A captured row is immutable**: a re-predict after the
  capture on the same date still pushes and publishes, but leaves the row
  (gain, features, prediction) alone — the observation was judged against it,
  and a refit divides it by the row's gain.
- **Tank deficit only when tracked**: the tank override (predict, the live
  result, the balance snapshot) applies only when
  `LoadConfig.tank_tracking_enabled` (the tracker's own predicate —
  `heating_active_entity` set); clearing the detector no longer leaves a
  restored calibrated state overriding the backlog forever.
- **Published runtime = what was pushed.** A predict caches a
  `PublishedTarget` (pushed minutes, kWh, deficit, `explain_load` rationale from
  the same features/deficit — so `target_minutes` equals the pushed value) only
  when the push **succeeded** (or the load is publish-only); a failed push keeps
  the previous value (`last_push_ok` shows the failure). It is persisted as a
  `"published"` key in the per-subentry Store payload (with target entity +
  date; additive, `STORAGE_VERSION` stays 1) and restored only while the load
  still targets the same entity. `_build_results` serves it (metrics always
  fresh) and builds a live snapshot only for a load with no cache. So a
  capture's gain step, a restart or an unrelated reload can't show a target
  the scheduler never got, and a predict doesn't query history/calendar/stats
  twice.
- **Coordinator contract**: `async_predict_and_push` returns True iff every
  attempted push succeeded (vacuously True for publish-only loads; a load that
  raised counts as failed). Predict and capture share one `asyncio.Lock`
  (read-state → await → write-back would otherwise drop a concurrent capture's
  gain step); the trailing refresh runs outside it. Per-load work is isolated
  (logged, others proceed, persist still runs); a failing `_build_results` load
  keeps its previous result instead of failing the coordinator.
- **Unload**: the tank tick is stopped and in-flight boost tasks are drained
  (`tank.async_drain()`, bounded — a boost finishing after the flush would lose
  its cooldown and re-fire on reload), the forecast tick is stopped, then
  `async_flush` takes the lock
  (draining an in-flight predict/capture), sets `_closing` — after which
  predict/capture/`async_persist` on this coordinator are no-ops, so no
  delayed write lands behind the reloaded one — and saves the final snapshot
  (`Store.async_save` cancels any pending delayed write). Flush errors are
  logged, never abort the unload; if the platform unload fails,
  `async_reopen()` + `tank.async_start()` (and the forecast's
  `async_reopen()` + `async_start()`) resume writes and the ticks.

## The tank model (read before touching `tank_model.py` / `tank_tracker.py`)

- **`tank_model.py` must stay Home-Assistant-free** (importlib-tested like
  `predictor.py`). `SoC = 1 − display_deficit/E_cap`; `E_cap` uses the
  configured `tank_cold_in_c` (default 12 °C), **not** the supply-temp sensor
  (that one measures the lake source and runs ~10 °C high in summer; the
  user's 7–8 h full-heat observation validates 300 L × ΔT63 ≈ 22 kWh at 3 kW).
- **Inputs are cumulative counters** (energy kWh, water litres) — deltas are
  lossless across restarts/downtime; negative deltas mean resets → re-baseline,
  never negative energy. The tracker converts each counter from its
  `unit_of_measurement` with HA's `EnergyConverter`/`VolumeConverter` (Wh, MWh,
  m³, gal, ft³, mL, … — plus legacy `liters`-style spellings); a missing or
  unsupported unit reads as **unavailable** (warned once), never guessed. Water
  litres pass a rate-based misread guard (`MAX_PLAUSIBLE_FLOW_LPM`) and a
  **hot-flow cap** (`MAX_HOT_FLOW_LPM`, taps/showers only — garden/appliance
  cold draws beyond it are attributed cold) before `hot_fraction` applies.
- **Rate spans + slow meters**: both water guards rate a delta over the time
  since the last *read* (a dropout/restart gap counts in full), stretched back
  over the meter's own change-to-change interval (its `last_changed`, passed as
  `TickInputs.water_changed_iso`) only as far as the meter's learned publish
  cadence `water_cadence_min`, capped at `WATER_RATE_WINDOW_MIN` (15 min). The
  cadence starts at one tick (0 = unknown) and is **raised only on evidence of
  batching**: a step impossible over the current span (> `MAX_PLAUSIBLE_FLOW_LPM`
  — no plumbing does that, so it's an OCR spike or a batch) yet plausible over
  the meter's own change interval. That evidencing step is still rejected
  (fallback, dirty) — it can't yet be told from a spike — and the next step
  settles it. It is *lowered* by a valid step arriving faster than it. The
  8 L/min hot-flow cap is deliberately not the trigger (a > 8 L/min tick is a
  real hose/appliance, not batching). Rollbacks/resets (negative deltas),
  adoptions, steps implausible even over their own interval, and ordinary valid
  steps slower than the cadence (isolated small draws minutes apart) never raise
  it. An unchanged re-report advances the read stamp but **not** the change
  stamp. So a meter publishing a 50 L step every 10 min is rated at 5 L/min from
  its second step on, while a responsive meter keeps one-tick caps. **Don't
  stretch to the raw change interval, or creep the cadence on slow intervals**:
  on the replayed week (a fast meter) stretching over-attributed ~0.5 kWh/week of
  evening flow and worsened the trip residuals (rms 1.17 → 1.24);
  `test_tank_replay` asserts the fixture's cadence stays ≤ 1 min on *every* tick
  (it stays 0 — the replay is identical to the pre-cadence model). The first
  water reading (no baseline) is adopted with zero draw, source `none`, cycle
  still clean — not treated as a misread.
- **Counter gaps across an anchor**: a counter that is unavailable *on* the
  anchor tick has an unknown delta straddling the trip, so that cycle doesn't
  learn, its baseline (if any) is dropped (the returning reading is adopted with
  no delta — otherwise pre-anchor kWh/litres are credited again to the new
  cycle), the new cycle opens dirty, and `pending_fallback_kwh` is always
  cleared at an anchor (it belongs to the closed ledger). This is **regardless
  of whether a baseline exists**: an outage spanning several anchors has none by
  the second, and that cycle must still open dirty or a later long cycle learns
  from draws + standby with zero delivered energy. Cycles stay dirty until an
  anchor sees the counter back. With no counter configured nothing learns
  (that balance is missing a side anyway).
- **Counter source changes**: `TankState.energy_source`/`water_source` record
  the entity each baseline came from; `rebind_sources` (called by the tracker
  before each tick) drops the baseline of a counter whose configured entity
  changed and dirties the cycle, so e.g. a powercalc `_2` re-add reading higher
  can't dump its whole difference into one tick. An empty stored source (state
  saved before this existed) is adopted without re-baselining. There is
  deliberately **no "implausibly fast" energy-delta guard**: after a restart a
  stale intermediate reading followed by the authoritative one looks exactly
  like a jump, and it would silently discard delivered energy — the failure the
  replay exists to catch. Entity swaps are `rebind_sources`' job, units the
  tracker's.
- **Three ledgers, three jobs** (v0.9.0 — a real-week replay showed the old
  single clamped `deficit_kwh` was discarding information, not adding safety):
  - `deficit_kwh` — clamped to `[0, E_cap]`, drives *control* (the
    SoC→prediction feedback below; over-asking is physically safe, the
    thermostat just trips).
  - `cycle_unclamped_kwh` — the identical arithmetic with no clamp and no
    post-trip relaxation, reset to 0 at each anchor; its value **at the next
    trip is the residual** — the model's true error over that cycle — and is
    what `learn_from_cycle` trains on.
  - `cycle_relax_kwh` — post-trip relaxation accrued since the last trip and
    not yet paid back; **paid down first** by any energy in, so a re-heat
    clears it before it can bleed into the next cycle's draw.
- **Soft saturation instead of a floor**: the *displayed* deficit
  (`display_deficit_kwh` — what the % and the `deficit_kwh` attribute show)
  runs `unclamped + relax` through a C¹ curve: identity above the model's own
  uncertainty `σ`, and `σ²/(2σ − u)` below it (a slow `1/|u|` approach that
  never quite reaches 0). `σ = max(0.05, residual_ratio × cycle_gross_kwh)` —
  ~0 right at an anchor (no post-anchor cliff), growing with the flow
  attributed since. **100 % shows only on the anchor transition tick**; every
  tick after that the % declines again, smoothly, whether or not the element
  is heating.
- **Post-trip relaxation** models the mixing loss the thermostat feels once the
  element idles: `cycle_relax_kwh` relaxes toward a learned `hysteresis_kwh`
  (seed 0.8 kWh, τ = 45 min — the LVV's re-engage is seen ~45–60 min after a
  trip) and feeds only the *display*. It is deliberately **excluded from the
  learning ledger**: trip-to-trip energy conservation has no mixing loss in
  it, and the 2026-08-28 replay showed including it biased the learner
  +0.9 kWh.
- **Daypart `hot_fraction_profile`** — 4 buckets by local hour (night 0–6,
  morning 6–11, day 11–17, evening 17–24). Each qualifying anchor takes a
  normalised-LMS step on the residual across whichever buckets carried litres
  that cycle, then every bucket is pulled 5 % toward the profile mean (a
  sparse bucket can't run away) and hard-clamped; `hot_fraction` is published
  as the profile's mean. The replay separated evening ≈ 0.27 (showers) from
  the rest of the day ≈ 0.20 (toilets/dishwasher/washer — cold-only) on the
  author's house.
- **Learner regimes** (`learn_from_cycle`, gated on `calibrated` + a clean
  cycle — no fallback/misread ticks): cycles < 4 h are post-trip re-heats and
  teach `hysteresis_kwh` only; ≥ 50 L metered in a longer cycle → the
  hot-fraction profile learns; < 10 L over ≥ 12 h → `standby_w` absorbs the
  residual instead. Long cycles also EWMA-update `residual_ratio` (the
  saturation curve's σ scale). Step size is `LEARN_BETA = 0.1` (hysteresis and
  residual_ratio use a faster 0.2) — the replay showed a faster learner chases
  ±1 kWh of cycle-to-cycle noise and gets worse. Meter dropouts fall back to
  an occupancy-based draw and reconcile via `pending_fallback_kwh` when the
  meter returns (no double-count).
- **The anchor + latch**: contactor commanded on + heating-active detector off,
  both sustained (≥ 120 s / ≥ 60 s via `last_changed` age — `unknown`/
  `unavailable` map to `None` = "don't anchor", never "off"). The *transition*
  needs those sustained thresholds, but once latched (`anchor_latched`) the
  latch itself only dedupes: it survives `unknown` blips and the scheduler's
  off→on re-toggle at a slot boundary, and releases only on a *definite*
  "element heating" or "contactor off". `TickResult.anchored` is True on the
  transition tick only (that's when learning happens); `latched` stays True
  for the whole trip, during which the tank keeps accruing standby/
  relaxation/draws — the % drifts below 100 until the element re-engages or
  the contactor drops.
- **Counter rollback guard** (`COUNTER_ROLLBACK_TOL_KWH = 1.0`): a cumulative
  energy counter that steps *down* by less than this is a restore-after-restart
  (powercalc republished an older value after an HA restart on 2026-09-03),
  not a reset — keep the old baseline and let the counter catch up; a larger
  drop re-baselines as a genuine reset/meter swap.
- **Display-only LED smoothing**: the powercalc counter steps 0.5 kWh every
  10 min, so between steps `rated_power_kw × on-time` fills the gap for
  display (`led_kwh_since_counter`, capped at 1 kWh so a stalled counter can't
  run it away), cleared the instant the authoritative counter actually moves.
- **Sensor attributes**: `deficit_kwh` (the shown/saturated value the % is
  derived from), `deficit_raw_kwh` (the clamped control ledger — what the
  SoC→prediction feedback actually uses), `uncertainty_kwh` (σ),
  `hysteresis_kwh`, `hot_fraction_profile`, `latched`, plus the pre-existing
  `capacity_kwh`, `hot_fraction`, `standby_w`, `calibrated`, `last_full`,
  `draw_source`, `liters_40c`, `showers_left`. The per-tick-volatile ones
  (`deficit_kwh`, `deficit_raw_kwh`, `uncertainty_kwh`, `latched`,
  `draw_source`, `liters_40c`, `showers_left`) are `_unrecorded_attributes` —
  otherwise the recorder writes a new attributes row every minute; nothing
  reads their history (the card reads them live, the replay only the state).
  The slow learned params stay recorded.
- **SoC → prediction feedback**, gated on `calibrated`, still reads the raw
  **control ledger** (`deficit_raw_kwh`/`state.deficit_kwh`), never the
  saturated display value — the curve is for the human-facing %, not for what
  gets pushed to the scheduler. At predict time the measured deficit
  (kWh → minutes, clamped to the deficit cap) **replaces** the
  commanded-minutes `close_cycle` backlog (which still runs as the fallback;
  the training row records `deficit_source`), and the tracker's low-charge
  boost (`tank_boost_soc_pct`) re-runs `async_predict_and_push` — hysteresis +
  ≥ 6 h rate limit live in `should_boost` (the re-arm level is
  `min(threshold + 15, 100)`, so a high threshold still re-arms at an anchor,
  where the control ledger reads exactly 100). The push runs as its own entry
  task (`_async_boost` — predict/recorder/scheduler can outlast a 60 s tick);
  while it's in flight that load's boost isn't re-evaluated, so no duplicate
  predict queues behind the coordinator lock. Only on success
  (`async_predict_and_push` returns a bool; `None` counts as success, a raise
  as failure) are the **boost fields alone** (disarm + `last_boost_iso`)
  applied — onto the *latest* tank state, never a snapshot from before the
  await (that would undo intervening anchors/learning/counter moves). A failed
  push keeps the trigger armed and retries no sooner than 15 min after it
  *completed* (`BOOST_RETRY_INTERVAL`, in-memory). Over-ask is physically
  safe: the tank thermostat trips and the element idles.
- Persistence: a `"tank"` key inside the load's existing per-subentry Store dict
  (`tank_to_dict`/`tank_from_dict`, defaults-tolerant, `STORAGE_VERSION` still
  1) — the v2 fields (`hot_fraction_profile`, `hysteresis_kwh`,
  `residual_ratio`, `cycle_unclamped_kwh`, `cycle_relax_kwh`,
  `led_kwh_since_counter`, `water_changed_iso`,
  `water_cadence_min`, `energy_source`/`water_source`, …) all default so a
  pre-v2 payload loads as a flat profile with no learning history (and the
  counter timing/sources as "unknown" = the old behaviour). `TankState` lives in
  `LoadNeedPredictorCoordinator.tanks` because `_runtime_snapshot` rebuilds the
  whole dict on every save — state owned elsewhere would be dropped. Saves:
  immediately on anchor/learn/boost, else every ~15 ticks (the cumulative
  counters make the lost tail harmless).
- The tracker ticks via its own `async_track_time_interval` (60 s), NOT a
  coordinator `update_interval` — a polling coordinator stops while no entity
  listens, which would silently freeze anchoring/learning. Per-load work is
  wrapped so one broken entity never kills the loop.
- **Regression-test the physics before touching a constant**:
  `tests/fixtures/tank_week_2026-08-28.csv` (exported from HA history JSON via
  `tools/replay_tank.py`) + `tests/test_tank_replay.py` replay a real week
  through `apply_tick` and report/assert on the trip-residual distribution.
  **Re-run the replay before changing any tank constant** — it's what caught
  the old hard-floor design silently discarding delivered energy in the first
  place. Future work: room-occupancy draw attribution (bathroom motion ⇒ hot)
  if garden-heavy days still skew the estimate; a hot-outlet pipe sensor
  remains the real fix for hot/cold attribution noise, deferred.

## The dashboard card (read before touching `www/…card.js` or the attrs)

A Lovelace card can only read **entity state + attributes**, so the rationale it
shows is published as sensor attributes — the card itself holds no model logic:

- Load: `sensor.<load>_predicted_runtime` carries `breakdown` (the
  `predictor.explain_load` dict — every formula term + the in-force params, incl.
  `deficit_minutes` backlog and the `target_minutes` actually pushed) and
  `metrics` (the published actual-vs-predicted summary + `deficit_minutes`). The
  sensor *state* is the pushed target (need + backlog); `breakdown.predicted_minutes`
  stays the need alone. Only the *runtime* sensor gets these (via `attr_fn`); the
  other load sensors stay attribute-free. `explain_load` is **pure** and a test
  asserts it agrees with `predict_kwh`/`predict_minutes`/`open_cycle` so the
  explanation can't drift.
- Forecast: `sensor.<name>_price_forecast` adds `coefficients`
  (`price_model.describe` — feature names + betas + intercept, sign = effect
  direction) and `fitted`, alongside `days` (per-day `mean/min/max/src` + local
  inputs), `source`, `retail_mapping`, `mae_by_source`, `stale`/`cache_age_h`,
  `wattcast_made_at`, `fetch_error` and `data_today`. Dashboard plotly cards overlay
  `data_today` (faded bars + p10–p90 band) where real prices are missing.
- The card **auto-discovers** devices from the entity registry
  (`platform == "load_need_predictor"`) and classifies each by attribute content
  (`breakdown` ⇒ load, `data_today` ⇒ forecast) — rename-proof, no entity-id
  parsing. It is **vanilla JS, no build step** (ships + runs as-is); keep it that
  way. `frontend.py` registration is best-effort and must never break setup.
- Loads may also own a `tank_soc` sensor (translation-key matched, opt-in): the
  card renders a charge bar and must tolerate its absence, and the sensor's
  state **must be part of the render fingerprint** — it updates every minute
  while the runtime sensor changes daily, so leaving it out freezes the bar.

## Dev workflow

```bash
uv venv --python 3.13 .venv313
uv pip install --python .venv313/bin/python -r requirements_test.txt
.venv313/bin/python -m pytest
.venv313/bin/ruff check . && .venv313/bin/ruff format --check .
```

- Pure tests (`test_predictor`/`test_features`/`test_models`) load their module
  via `importlib`, so the logic needs no HA — **keep those modules HA-free**. The
  `tests/ha/` tests use `pytest-homeassistant-custom-component`.
- New model behaviour goes into a pure module as a tested function first, then is
  wired into the coordinator/jobs.

## Releasing a version

A release is the **version bump + commit on `main` + push + a GitHub release** —
all four. HACS reads the version from `manifest.json` **and** picks up GitHub
releases, so both must move together.

1. Bump `"version"` in `custom_components/load_need_predictor/manifest.json`
   (SemVer: **minor** for a feature — the 0.3.0/0.4.0/0.5.0/0.6.0 line — **patch**
   for a fix/tweak — 0.2.2/0.5.1/0.5.2).
2. Commit **directly to `main`** (every prior release is a direct main commit, not
   a PR/branch — don't branch for a release). Message convention:
   `vX.Y.Z: short description`, e.g. `v0.6.0: deficit carryover — make up loads…`.
3. `git push origin main`.
4. `gh release create vX.Y.Z --title "vX.Y.Z — short human description" --notes "…"`
   The **GitHub release creates the tag** — there is no separate `git tag` step.

Gotcha: local `git tag` lags far behind (it showed `v0.4.0` while shipped was
`v0.6.0`) because release tags are created server-side by `gh release create` and
never fetched. **Use `gh release list` to see the real latest version, not
`git tag`.** Match the latest release's title style when writing the new one.

## Conventions

- Comment the *why*, not the *what*; match the density in `predictor.py`.
- Don't commit secrets; config lives in the config entry, runtime state in
  `.storage/` (both in HA backups).
- The integration must never raise if the scheduler is absent — degrade to
  publish-only and log.

## Branding / icon

`brands/` holds the icon: `icon.svg` (editable source) + `icon.png` (256) +
`icon@2x.png` (512), rendered with `rsvg-convert`. It's a full-bleed app tile
(blue energy gradient, white price bars with the cheapest in green) and is
**derived from the Load Scheduler icon** — same tile/bars/green/amber, but the
scheduler's bolt becomes a **forecast line projecting (dashed) beyond the bars**.
Re-render after editing the SVG:

```bash
cd brands && rsvg-convert -w 512 -h 512 icon.svg -o icon@2x.png \
                        && rsvg-convert -w 256 -h 256 icon.svg -o icon.png
```

**TODO — make it show in HA + HACS (not done yet):** HA/HACS load integration
icons only from the `home-assistant/brands` repo (no repo-local or manifest
override). Open a PR there adding `custom_integrations/load_need_predictor/icon.png`
+ `icon@2x.png` (the files in `brands/`). Keep it full-bleed so brands' trim
check passes. After merge, an HA restart may be needed to clear the brand cache.
