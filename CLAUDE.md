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
| `predictor.py` | **Pure** load model: features → kWh → minutes, seeds, gain EWMA, prior↔empirical blend, `build_features`, rolling MAE | no |
| `price_model.py` | **Pure** price model: ridge regression of price on wind+temp with a cold interaction; seed fallback; fit/predict/serialize | no |
| `models.py` | Subentry config → frozen `LoadConfig` / `PriceForecastConfig` | no |
| `statistics_source.py` | Load delivery from the recorder (daily `change`) | yes |
| `forecast_source.py` | Wind series + daily temp forecast + LTS fit rows / realised price | yes |
| `occupancy.py` | Sample `person.*` + guests calendar (no LTS to mine) | yes |
| `persistence.py` | `Store` (load: model+training+eval; forecast uses a `.forecast` file) | yes |
| `actuation.py` | Resilient `number.set_value` push to the scheduler target | yes |
| `jobs.py` | The two daily jobs; drives both coordinators | yes |
| `coordinator.py` | Load `DataUpdateCoordinator`; per-load `LoadResult` | yes |
| `forecast_coordinator.py` | Price-forecast coordinator: fit → build slots → evaluate; `ForecastResult` | yes |
| `runtime.py` | `RuntimeData` (both coordinators) + the `ConfigEntry` type alias | yes |
| `config_flow.py` | Hub flow + `load` and `price_forecast` subentry wizards | yes |
| `entity.py` / `sensor.py` | Entity bases + per-load and price-forecast sensors | yes |
| `diagnostics.py` | Redacted diagnostics dump (loads + forecasts) | yes |

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
