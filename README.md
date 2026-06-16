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
  builds the features, computes minutes, writes the Load Scheduler target, and
  publishes its own `sensor.<load>_predicted_runtime`.
- **Capture + log** runs late each evening: it reads the day's *actual* delivered
  energy from the recorder, completes the training row, updates the calibration
  gain, and recomputes the evaluation metrics.

## Entities (per load)

| Entity | Meaning |
|---|---|
| `sensor.<load>_predicted_runtime` | The forecast in minutes (also the forward-compatible "target source"). |
| `sensor.<load>_predicted_energy` | The forecast in kWh. |
| `sensor.<load>_last_delivered` | Actual delivered energy captured for the previous day (kWh). |
| `sensor.<load>_prediction_error` | Yesterday's \|predicted − actual\| in minutes. |
| `sensor.<load>_rolling_mae` | Rolling mean absolute error (minutes) over the evaluation window. |
| `sensor.<load>_sample_count` | How many self-logged days the model has learned from. |

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

**Model.** A small, explainable regression of daily price on wind + temperature
with a **cold-weather interaction**, fit on long-term statistics. On ~356 days
(incl. a full winter): more wind ⇒ lower price (r ≈ −0.23, −0.49 below −5 °C),
colder ⇒ higher price (r ≈ −0.45), jointly R² ≈ 0.37 — so the model leans on
temperature most in the cold and on wind most in the cold band. Daily price is
expanded into flat hourly slots; the scheduler's margin absorbs the coarseness.
`sensor.<name>_forecast_mae` / `_forecast_error` log forecast-vs-actual price.

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
   guests-calendar entities, optional log-only context sensors, and the
   minute clamp.

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
