# Load Need Predictor

[![hacs](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Home Assistant custom integration with two forecasting capabilities that feed
the [Load Scheduler](https://github.com/machadolucas/ha-load-scheduler):

1. **Load-need prediction** — predicts *how much* a flexible load (the hot-water
   heater first) needs to run each day, and pushes that target to the scheduler,
   which decides *when* (cheapest / greenest slots). Replaces "set the runtime by
   hunch" with a small, explainable, self-improving model.
2. **Beyond-horizon price forecast** — estimates electricity prices for the day
   *after* tomorrow (past Nord Pool's day-ahead horizon) from Finland's wind +
   temperature forecasts, published as the scheduler's `forecast_price_entity` so
   a load with a multi-day horizon can defer an expensive 24 h to a
   forecast-cheaper one.

Both forecast the *forecastable* thing — never day-to-day hot-water demand.

> **Status: alpha.** Install as a HACS custom repository (below). Backed by an
> extensive test suite.

## Why a predictor — and why this design

Daily hot-water energy was analysed against ~3–4 months of Home Assistant
long-term statistics. The result shaped the whole design:

- **Daily energy is dominated by stochastic hot-water draw**, not weather. Day to
  day it varies ~45% with almost no autocorrelation.
- **Temperature / season has ~zero day-ahead predictive power.** A regression on
  incoming-water and outdoor temperature backtested *worse than a constant* (it
  systematically over-predicted late season). A flat "predict the average" model
  beat every temperature/season model.
- **Total household water consumption** correlates only *contemporaneously*; the
  causal (lag/trailing) versions you could actually use ahead of time also lose
  to the constant. (And the meter had multi-week dropouts.)
- **Occupancy is the one feature with real leverage** — how many people are home,
  plus guests — yet it has *no* long-term statistics in this setup.

So v1 is deliberately simple: a **calibrated baseline gated by occupancy**, with
an **online calibration gain** that tracks slow drift, and a **safety floor** so
the tank never starves. Temperature and water are *logged* for a future model but
not used in the prediction. The single highest-value thing the integration does
is **log occupancy + the actual outcome every day** — that's the signal Home
Assistant's statistics threw away, and it's what lets the model improve.

## How it works

```
predicted_kWh = occupancy_factor × [ E_base + E_draw_per_person × people_home ]
              + guest_bonus × guests_present
predicted_kWh × = gain            # online EWMA correction (actual ÷ predicted)
predicted_min  = clamp( predicted_kWh / rated_kW × 60 , min , max )
```

- **Predict + push** runs each afternoon (after tomorrow's prices publish): it
  builds the features, computes minutes (today's need plus any carried-over
  backlog — see *Catching up after a skipped day*), writes the Load Scheduler
  target, and publishes its own `sensor.<load>_predicted_runtime`.
- **Capture + log** runs late each evening: it reads the day's *actual* delivered
  energy from the recorder, completes the training row, updates the calibration
  gain (only from *clean* days — see below), and recomputes the evaluation
  metrics.

### How the runtime is calculated (worked example)

Occupancy is measured by **duration**, not a point-in-time snapshot — so being out
at a meeting when the prediction runs doesn't drop you from the count.

- **Residents** (`people_home`): the number of configured `person.*` entities that
  were `home` for **at least 12 h over the trailing 24 h**. A regular occupant who
  slept home and is out for a daytime meeting still clears 12 h and counts; someone
  genuinely travelling (away overnight) does not. Only `home` counts (`not_home`
  or a zone name like `Tampere` → away); `unknown`/`unavailable` or no recorder →
  treated as present (conservative — never under-serve).
- **Guests**: weighted by **visit length**, from the guests calendar's events in
  the next 24 h. No visit → 0; a **short visit (< 6 h)** → **0.5** guest-equivalents
  (a dinner — little hot water); a **long visit (≥ 6 h, or all-day)** → **2.0**
  guest-equivalents (likely a sauna + showers, possibly several guests — we
  deliberately over-provision so guests never run the tank cold). The longest event
  in the window decides the weight.

These feed the formula above. With the **seed** parameters and a 3 kW heater (so
kWh → minutes is `kWh ÷ 3 × 60`), each factor contributes:

| Factor | Energy | ≈ Runtime at 3 kW |
|---|---|---|
| Base `E_base` (standby + minimal use) | 3.0 kWh | ~60 min |
| **Each resident home ≥ 12 h of the last 24 h** | +2.2 kWh each | **+44 min each** |
| Short guest visit (< 6 h → 0.5 ×) | +1.25 kWh | +25 min |
| Long guest visit (≥ 6 h → 2.0 ×) | +5.0 kWh | +100 min |
| Nobody home (base scaled by 0.4) | 1.2 kWh | ~24 min → floored |

Putting it together (seed values, `gain = 1.0`, clamp 40–240 min, rounded to 15):

| Occupancy over the trailing/upcoming day | Predicted kWh | Pushed runtime |
|---|---|---|
| Nobody home ≥ 12 h | 1.2 | **45 min** (safety floor) |
| 1 resident | 5.2 | **105 min** |
| 2 residents | 7.4 | **150 min** |
| 2 residents + short guest (< 6 h) | 8.65 | **180 min** |
| 2 residents + long guest (≥ 6 h) | 12.4 | **240 min** (hits the cap) |

(The last row shows the cap doing its job — raise *Maximum minutes/day* on the load
if you want long-guest days to reheat even more.)

Two things then adjust these numbers over time:

- **Calibration gain** (`× gain`; starts 1.0, clamped 0.7–1.5, ~6-day EWMA
  half-life) nudges the whole prediction toward your *actual* delivered energy —
  if the model runs ~10 % hot, the gain drifts toward ~0.9 and every figure above
  scales down ~10 %.
- **Refit** — once ≥14 clean days are logged, `E_base` and `E_draw_per_person`
  are re-fit from *your own* `actual_kWh` vs `people_home` history (blended toward
  the seeds by sample count). So "+44 min per person" is only the starting point;
  it becomes whatever your household's data says.

> Residents look at the **trailing 24 h** (history); guests look at the **next
> 24 h** (calendar). The **pushed** target — and the row logged for training — is
> computed at the predict time; `sensor.<load>_predicted_runtime` is recomputed on
> every coordinator refresh (predict, capture, restart). Thresholds (12 h, 6 h) and
> the guest weights (0.5 / 2.0) live in `occupancy.py`.

### Catching up after a skipped day

The scheduler is free to **skip** a day — defer an expensive, cloudy 24 h expecting
a cheaper or sunnier tomorrow. By default that unmet runtime would simply be lost:
the next day's target is sized for that day's occupancy alone. **Deficit carryover**
fixes that. Point the load at its **controlled switch** (the relay the scheduler
drives) and the predictor measures how long the load actually ran each
predict-to-predict cycle and carries any shortfall forward — so a skipped day is
*added* to the next day's target and made up.

- The runtime actually delivered is read from the **switch's on-time**, from *any*
  source — the scheduler, a manual boost, a comfort automation — so anything that
  heats the tank correctly shrinks the backlog. It must be the real contactor,
  **not** a thermostat-gated power sensor: asking for too much is harmless (the
  tank's own thermostat trips early), and the backlog then clears on its own.
- The backlog is **bounded** (default: twice the daily maximum) so a long outage
  can't make it run away, and the per-day cap still limits one day's catch-up — a
  deep deficit is recovered over several days, not in one giant run.
- The calibration gain only learns from **clean** days: ones with no backlog being
  worked off *and* where the load ran roughly the full target. So a price-driven
  skip or defer can no longer trick the model into predicting less.

Leave the controlled switch empty to disable catch-up — the predictor then behaves
exactly as the plain daily model above. When enabled,
`sensor.<load>_predicted_runtime` shows the **target** (need + backlog), and its
`breakdown` attribute (and the dashboard card) splits out the need, the backlog,
and the final target.

## Entities (per load)

| Entity | Meaning |
|---|---|
| `sensor.<load>_predicted_runtime` | The pushed target in minutes — today's need plus any carried-over backlog (also the forward-compatible "target source"). |
| `sensor.<load>_predicted_energy` | The forecast in kWh. |
| `sensor.<load>_last_delivered` | Actual delivered energy captured for the previous day (kWh). |
| `sensor.<load>_prediction_error` | Yesterday's \|predicted − actual\| in minutes. |
| `sensor.<load>_rolling_mae` | Rolling mean absolute error (minutes) over the evaluation window. |
| `sensor.<load>_sample_count` | How many self-logged days the model has learned from. |
| `button.<load>_predict_now` | Recompute the prediction and push it to the scheduler **now** (no need to wait for the daily predict time). |

### When does it run? Publish time & DST

The daily **predict + push** fires at the hub's `predict_time` (default 14:00) and
**capture + log** at `capture_time`. These are tracked on the **local wall clock**,
so they're DST-correct automatically — 14:00 stays 14:00 local across the spring/
autumn transitions, never drifting by an hour.

About Nord Pool's variable publish hour: the prediction **doesn't depend on the
spot prices** — it answers *how much* (from occupancy) and writes the scheduler's
target; the **scheduler** consumes the prices and re-plans *when* automatically
whenever the prices or the target change. So the exact `predict_time` isn't
critical: 14:00 local sits comfortably after the usual day-ahead publish, and if
prices land late (or you change occupancy/guests), just hit **Predict now**.

To react the moment prices publish — whatever hour that lands on — point an
automation at the button using your Nord Pool "tomorrow prices available" sensor:

```yaml
automation:
  - alias: "Re-predict LVV when tomorrow's prices publish"
    triggers:
      - trigger: state
        entity_id: binary_sensor.nordpool_tomorrow_prices_availability
        to: "on"
    actions:
      - action: button.press
        target:
          entity_id: button.lvv_predict_now
```

## Beyond-horizon price forecast

Nord Pool only publishes prices through tomorrow. This capability estimates the
*day after* (and beyond) so the scheduler can plan further out.

Add a **price forecast** subentry and point it at: your actual buy-price sensor
(€/kWh, for fitting + evaluation), the Finland wind-production forecast sensor,
a `weather` entity (daily temperature forecast), and an outdoor-temperature
sensor (history for fitting). It publishes one `sensor.<name>_price_forecast`
whose attributes carry a `data_today` list of `{start, end, buy}` slots — the
exact shape Nord Pool sensors use — covering roughly the next few days. Set that
sensor as the scheduler's `forecast_price_entity` and tune its confidence margin.
A `button.<name>_forecast_now` rebuilds the forecast on demand.

**Model.** A small, explainable regression of daily price on wind + temperature
with a **cold-weather interaction**, fit on long-term statistics. On ~356 days
(incl. a full winter): more wind ⇒ lower price (r ≈ −0.23, −0.49 below −5 °C),
colder ⇒ higher price (r ≈ −0.45), jointly R² ≈ 0.37 — so the model leans on
temperature most in the cold and on wind most in the cold band. Daily price is
expanded into flat hourly slots; the scheduler's margin absorbs the coarseness.
`sensor.<name>_forecast_mae` / `_forecast_error` log forecast-vs-actual price.

## Dashboard card

A bundled Lovelace card explains, in plain language, **how** each number was
calculated — the occupancy/baseline/gain breakdown behind every load's runtime
(plus any carried-over backlog), and the wind/temperature regression behind the
price forecast — with a "Predict now" / "Update forecast now" button on each.

The integration **auto-registers** the card on startup — it adds itself to the
Lovelace **resource registry** (the same mechanism HACS uses), so no manual
resource is needed and a single managed entry appears under *Settings →
Dashboards → Resources*. After a restart you can just add it:

```yaml
type: custom:load-need-predictor-card
# Everything below is optional:
title: Load Need Predictor          # card heading
devices: []                         # leave empty to auto-discover all loads + forecasts
default_expanded: false             # start with the detail section open
show_context: false                 # also show the logged temps / water context
```

It is also available from the dashboard's **“Add card”** picker (with a
graphical editor), is responsive across grid/section sizes, and is compact —
each load/forecast is one collapsible block. The rationale is read from entity
attributes the integration publishes: `breakdown` + `metrics` on
`sensor.<load>_predicted_runtime`, and `coefficients` + `fitted` on
`sensor.<name>_price_forecast`.

## Installation

### HACS (custom repository)

1. HACS → ⋮ → *Custom repositories* → add
   `https://github.com/machadolucas/ha-load-need-predictor`, category
   **Integration**.
2. Install **Load Need Predictor**, then restart Home Assistant.
3. *Settings → Devices & Services → Add Integration → Load Need Predictor.*

### Manual

Copy `custom_components/load_need_predictor` into your Home Assistant
`config/custom_components/` directory and restart.

## Configuration

1. **Add the hub** and (optionally) adjust the predict/capture times.
2. **Add a load** and follow the wizard: the Load Scheduler target `number` to
   drive, the delivered-energy sensor, the load's rated power, the people and
   guests-calendar entities, optional log-only context sensors, the minute clamp,
   and — to enable catch-up after skipped days — the load's **controlled switch**
   (the relay the scheduler drives) plus an optional backlog cap.

The predictor is resilient if the Load Scheduler isn't installed yet: it keeps
predicting and publishing its sensor, and simply skips the push.

## Development

```bash
# HA-side tests need Python 3.13 (HA doesn't support 3.14 yet):
uv venv --python 3.13 .venv313
uv pip install --python .venv313/bin/python -r requirements_test.txt
.venv313/bin/python -m pytest
.venv313/bin/ruff check . && .venv313/bin/ruff format --check .
```

The prediction model
([`predictor.py`](custom_components/load_need_predictor/predictor.py)) has **no
Home Assistant dependency** and is tested in isolation.

See [`CLAUDE.md`](CLAUDE.md) for the architecture, the model contract, and the
data findings behind the design.

## License

MIT © Lucas Machado
