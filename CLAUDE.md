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
| `tank_model.py` | **Pure** tank state-of-charge: energy-deficit integration (`apply_tick`), 100 %-anchor + EWMA calibration of `hot_fraction`/`standby_w`, hot-flow cap, meter-misread/fallback guards, boost gating (`should_boost`), liters/showers helpers | no |
| `models.py` | Subentry config → frozen `LoadConfig` / `PriceForecastConfig` | no |
| `statistics_source.py` | Load delivery from the recorder (daily `change`) + commanded switch on-time over a window (`async_commanded_minutes`, for deficit carryover) | yes |
| `forecast_source.py` | Wind series + daily temp forecast + LTS fit rows / realised price | yes |
| `occupancy.py` | Duration-based occupancy: residents home ≥12 h over the trailing 24 h (from history) + guests weighted by visit length (next-24 h calendar); instantaneous fallbacks | yes |
| `persistence.py` | `Store` (load: model+training+eval; forecast uses a `.forecast` file) | yes |
| `actuation.py` | Resilient `number.set_value` push to the scheduler target | yes |
| `jobs.py` | The two daily jobs; drives both coordinators | yes |
| `coordinator.py` | Load `DataUpdateCoordinator`; per-load `LoadResult` | yes |
| `forecast_coordinator.py` | Price-forecast coordinator: fit → build slots → evaluate; `ForecastResult` | yes |
| `tank_tracker.py` | 60 s-tick coordinator: reads counters/switch/detector states, drives `tank_model.apply_tick`, publishes per-load `TankResult`, fires the low-charge boost | yes |
| `runtime.py` | `RuntimeData` (both coordinators) + the `ConfigEntry` type alias | yes |
| `config_flow.py` | Hub flow + `load` and `price_forecast` subentry wizards | yes |
| `entity.py` | `PredictorEntity` (load) + `ForecastEntity` bases | yes |
| `sensor.py` / `button.py` | Per-load + price-forecast sensors; "predict/forecast now" buttons | yes |
| `diagnostics.py` | Redacted diagnostics dump (loads + forecasts) | yes |
| `frontend.py` | Serves + auto-registers the dashboard card (long-cached static path registered first, then an extra JS module with a guarded `?v=<content-hash>` cache-bust that falls back to the bare URL); best-effort, never breaks setup | yes |
| `www/load-need-predictor-card.js` | The Lovelace diagnostic card (vanilla JS, no build) + its `ha-form` editor | no |

## The price forecast (read before touching `price_model.py`)

- **`price_model.py` must stay Home-Assistant-free** (importlib-tested like
  `predictor.py`). Output contract for the scheduler: a `data_today` attribute of
  `{start, end, buy}` slots — **tz-aware ISO** starts, `buy` in **€/kWh** — for
  times beyond the real horizon; the scheduler ignores overlap and adds its own
  `forecast_price_margin`.
- Features `[temp, wind, cold_hinge, wind×cold_hinge]`, `cold_hinge = max(0,−temp)`;
  wind in **GW** (the sensor's series is GW but its state/LTS is MW — normalise).
  Fit daily on LTS (price/temp/wind), seed formula until enough history.
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
  `gain = clamp((1−β)·gain + β·r, 0.7, 1.5)`, `β = 0.15` (~6-day half-life). The
  clamps are the anti-drift guardrail — the gain corrects ±50% but can't run away.
- **Prior→empirical blend**: `θ = (n_prior·θ_prior + n·θ_emp)/(n_prior+n)`,
  `n_prior = 10`. Structural refit needs ≥1 zero-person and ≥1 multi-person day.
- **Safety floor** (`min_minutes`, default 40 ≈ 2 kWh) always wins: even a
  "nobody home" prediction keeps standby + one shower's worth.
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

## The tank model (read before touching `tank_model.py` / `tank_tracker.py`)

- **`tank_model.py` must stay Home-Assistant-free** (importlib-tested like
  `predictor.py`). `SoC = 1 − display_deficit/E_cap`; `E_cap` uses the
  configured `tank_cold_in_c` (default 12 °C), **not** the supply-temp sensor
  (that one measures the lake source and runs ~10 °C high in summer; the
  user's 7–8 h full-heat observation validates 300 L × ΔT63 ≈ 22 kWh at 3 kW).
- **Inputs are cumulative counters** (energy kWh, water litres) — deltas are
  lossless across restarts/downtime; negative deltas mean resets → re-baseline,
  never negative energy. Water litres pass a rate-based misread guard
  (`MAX_PLAUSIBLE_FLOW_LPM` over the span since baseline) and a **hot-flow cap**
  (`MAX_HOT_FLOW_LPM`, taps/showers only — garden/appliance cold draws beyond it
  are attributed cold) before `hot_fraction` applies.
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
  `draw_source`, `liters_40c`, `showers_left`.
- **SoC → prediction feedback**, gated on `calibrated`, still reads the raw
  **control ledger** (`deficit_raw_kwh`/`state.deficit_kwh`), never the
  saturated display value — the curve is for the human-facing %, not for what
  gets pushed to the scheduler. At predict time the measured deficit
  (kWh → minutes, clamped to the deficit cap) **replaces** the
  commanded-minutes `close_cycle` backlog (which still runs as the fallback;
  the training row records `deficit_source`), and the tracker's low-charge
  boost (`tank_boost_soc_pct`) re-runs `async_predict_and_push` — hysteresis +
  ≥ 6 h rate limit live in `should_boost`. Over-ask is physically safe: the
  tank thermostat trips and the element idles.
- Persistence: a `"tank"` key inside the load's existing per-subentry Store dict
  (`tank_to_dict`/`tank_from_dict`, defaults-tolerant, `STORAGE_VERSION` still
  1) — the v2 fields (`hot_fraction_profile`, `hysteresis_kwh`,
  `residual_ratio`, `cycle_unclamped_kwh`, `cycle_relax_kwh`,
  `led_kwh_since_counter`, …) all default so a pre-v2 payload loads as a flat
  profile with no learning history. `TankState` lives in
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
  direction) and `fitted`, alongside the existing `days` / `data_today`.
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
