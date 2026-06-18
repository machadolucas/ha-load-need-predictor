/*
 * Load Need Predictor — diagnostic dashboard card.
 *
 * A compact, responsive Lovelace card that explains, in plain language, *how*
 * each configured load's runtime and each price forecast was calculated. It is
 * read-only diagnostics plus the two "run now" buttons; all the rationale comes
 * from entity attributes the integration publishes:
 *   - load    → the `*_predicted_runtime` sensor's `breakdown` + `metrics` attrs
 *   - forecast → the `*_price_forecast` sensor's `days` + `coefficients` attrs
 *
 * No build step: a self-contained custom element (vanilla JS + shadow DOM) so it
 * works offline and ships inside the integration. Devices are auto-discovered
 * from the entity registry (`platform === DOMAIN`), classified by which
 * translation_key entities they own — so renaming entities never breaks it.
 */

const DOMAIN = "load_need_predictor";
const CARD_VERSION = "0.5.1";
const DOC_URL = "https://github.com/machadolucas/ha-load-need-predictor";

// translation_key values the integration assigns to its entities.
const TK = {
  loadPrimary: "predicted_runtime",
  forecastPrimary: "price_forecast",
  predictNow: "predict_now",
  forecastNow: "forecast_now",
};

// ── small formatting helpers ────────────────────────────────────────────────
const isNum = (x) => typeof x === "number" && !Number.isNaN(x);
const kwh = (x) => (isNum(x) ? x.toFixed(1) : "—");
const eur = (x) => (isNum(x) ? x.toFixed(3) : "—");
const gain = (x) => (isNum(x) ? "×" + x.toFixed(2) : "—");
const mins = (x) => (isNum(x) ? Math.round(x) + " min" : "—");
const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const bold = (x) => `<b>${x}</b>`;
const esc = (s) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

class LoadNeedPredictorCard extends HTMLElement {
  static getConfigElement() {
    return document.createElement("load-need-predictor-card-editor");
  }
  static getStubConfig() {
    return {}; // empty config = auto-discover every load + forecast
  }

  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._config = {};
    this._hass = null;
    this._toggled = new Map(); // deviceId → explicit open/closed (overrides default)
    this._fingerprint = null;
    // One delegated listener survives innerHTML re-renders (it's on the root).
    this.shadowRoot.addEventListener("click", (ev) => this._onClick(ev));
  }

  setConfig(config) {
    this._config = config || {};
    this._fingerprint = null; // force a rebuild on next render
    if (this._hass) this._render();
  }

  set hass(hass) {
    this._hass = hass;
    this._render();
  }

  getCardSize() {
    return Math.max(2, this._devices().length * 3);
  }

  // Sections view: never narrower than half a row, auto height.
  getGridOptions() {
    return { min_columns: 6, min_rows: 2 };
  }

  // ── discovery ─────────────────────────────────────────────────────────────
  _devices() {
    const hass = this._hass;
    if (!hass || !hass.entities) return [];
    const filter = this._config.devices && this._config.devices.length
      ? new Set(this._config.devices)
      : null;

    // Group this integration's entities by device, keyed by translation_key.
    const byDevice = {};
    for (const ent of Object.values(hass.entities)) {
      if (ent.platform !== DOMAIN || !ent.device_id) continue;
      if (filter && !filter.has(ent.device_id)) continue;
      const tk = ent.translation_key;
      if (!tk) continue;
      (byDevice[ent.device_id] ||= {})[tk] = ent.entity_id;
    }

    const out = [];
    for (const [deviceId, keys] of Object.entries(byDevice)) {
      let type = null;
      if (keys[TK.loadPrimary]) type = "load";
      else if (keys[TK.forecastPrimary]) type = "forecast";
      if (!type) continue;
      const dev = (hass.devices && hass.devices[deviceId]) || {};
      out.push({
        deviceId,
        type,
        name: dev.name_by_user || dev.name || deviceId,
        keys,
      });
    }
    // Loads first, then forecasts; alphabetical within each.
    out.sort((a, b) =>
      a.type === b.type ? a.name.localeCompare(b.name) : a.type === "load" ? -1 : 1
    );
    return out;
  }

  // ── render ──────────────────────────────────────────────────────────────--
  _render() {
    if (!this._hass) return;
    const devices = this._devices();

    // Only rebuild when something the card shows actually changed — avoids
    // thrashing the DOM on every unrelated state update HA pushes.
    const fp = JSON.stringify({
      cfg: this._config,
      exp: [...this._toggled.entries()].sort(),
      st: devices.map((d) => {
        const eid = d.keys[d.type === "load" ? TK.loadPrimary : TK.forecastPrimary];
        const s = this._hass.states[eid];
        return [eid, s ? s.state + "@" + s.last_updated : "none"];
      }),
    });
    if (fp === this._fingerprint) return;
    this._fingerprint = fp;

    const title = this._config.title;
    let body;
    if (!devices.length) {
      body = `<div class="empty">No Load Need Predictor loads or forecasts found.</div>`;
    } else {
      body = devices.map((d) => this._renderDevice(d)).join("");
    }

    this.shadowRoot.innerHTML = `
      <style>${STYLES}</style>
      <ha-card>
        ${title ? `<h1 class="card-header">${esc(title)}</h1>` : ""}
        <div class="content">${body}</div>
      </ha-card>`;
  }

  _renderDevice(d) {
    return d.type === "load" ? this._renderLoad(d) : this._renderForecast(d);
  }

  _isOpen(id) {
    return this._toggled.has(id) ? this._toggled.get(id) : !!this._config.default_expanded;
  }

  _renderLoad(d) {
    const hass = this._hass;
    const primary = hass.states[d.keys[TK.loadPrimary]];
    const bd = primary && primary.attributes && primary.attributes.breakdown;
    const m = (primary && primary.attributes && primary.attributes.metrics) || {};
    const open = this._isOpen(d.deviceId);

    const dot = pushDot(m.last_push_ok);
    const headline = primary
      ? `${bold(mins(Number(primary.state)))} <span class="muted">·</span> ~${bold(
          kwh(bd && bd.predicted_kwh)
        )} kWh`
      : `<span class="muted">unavailable</span>`;

    const btn = d.keys[TK.predictNow]
      ? actionBtn("mdi-play", "Predict now", "button.press", d.keys[TK.predictNow])
      : "";

    return `
      <div class="block">
        <div class="head">
          <ha-icon icon="mdi:water-boiler"></ha-icon>
          <span class="name">${esc(d.name)}</span>
          ${dot}
          <span class="spacer"></span>
          ${btn}
        </div>
        <div class="big">${headline}</div>
        ${loadExplanation(bd)}
        <div class="toggle" data-action="toggle" data-device="${d.deviceId}">
          ${open ? "Hide detail ▴" : "Show detail ▾"}
        </div>
        <div class="detail" style="${open ? "" : "display:none"}">
          ${loadDetail(bd, m, this._config.show_context)}
        </div>
      </div>`;
  }

  _renderForecast(d) {
    const hass = this._hass;
    const primary = hass.states[d.keys[TK.forecastPrimary]];
    const a = (primary && primary.attributes) || {};
    const open = this._isOpen(d.deviceId);

    const headline = primary
      ? `mean ${bold(eur(Number(primary.state)))} <span class="muted">€/kWh</span>`
      : `<span class="muted">unavailable</span>`;
    const status =
      a.status && a.status !== "ok" ? `<span class="warn">${esc(a.status)}</span>` : "";

    const btn = d.keys[TK.forecastNow]
      ? actionBtn("mdi-refresh", "Update forecast now", "button.press", d.keys[TK.forecastNow])
      : "";

    return `
      <div class="block">
        <div class="head">
          <ha-icon icon="mdi:cash-clock"></ha-icon>
          <span class="name">${esc(d.name)}</span>
          ${status}
          <span class="spacer"></span>
          ${btn}
        </div>
        <div class="big">${headline}</div>
        ${forecastExplanation(a)}
        <div class="toggle" data-action="toggle" data-device="${d.deviceId}">
          ${open ? "Hide detail ▴" : "Show detail ▾"}
        </div>
        <div class="detail" style="${open ? "" : "display:none"}">
          ${forecastDetail(a)}
        </div>
      </div>`;
  }

  // ── interactivity ─────────────────────────────────────────────────────────
  _onClick(ev) {
    const el = ev.target.closest && ev.target.closest("[data-action]");
    if (!el) return;
    if (el.dataset.action === "toggle") {
      const id = el.dataset.device;
      this._toggled.set(id, !this._isOpen(id));
      this._render();
    } else if (el.dataset.action === "service" && this._hass) {
      const [domain, service] = el.dataset.service.split(".");
      this._hass.callService(domain, service, {}, { entity_id: el.dataset.entity });
    }
  }
}

// ── natural-language composition (load) ─────────────────────────────────────
function loadExplanation(bd) {
  if (!bd) {
    return `<div class="explain muted">Rationale unavailable — update the integration
      to v0.5.0+ to see how this is calculated.</div>`;
  }
  const people = bd.people_home || 0;
  let s1;
  if (people > 0) {
    const who = `${bold(people)} ${people === 1 ? "person" : "people"} home`;
    s1 = `${cap(who)} → ${bold(kwh(bd.e_base))} kWh baseline + ${people}×${kwh(
      bd.e_draw_per_person
    )} kWh draw = ${bold(kwh(bd.occupied_kwh))} kWh`;
  } else {
    s1 = `${bold("Nobody home")} → baseline scaled to ${bold(
      Math.round((bd.empty_house_factor || 0) * 100) + "%"
    )} = ${bold(kwh(bd.occupied_kwh))} kWh`;
  }
  if (bd.guest_kwh > 0) s1 += ` + ${bold(kwh(bd.guest_kwh))} kWh for guests`;

  const s2 = `Calibration gain ${bold(gain(bd.gain))} → ${bold(
    kwh(bd.predicted_kwh)
  )} kWh ≈ ${bold(mins(bd.raw_minutes))} at ${kwh(bd.rated_power_kw)} kW`;
  const s3 = bd.clamped
    ? `clamped to ${bold(mins(bd.predicted_minutes))} (limits ${Math.round(
        bd.min_minutes
      )}–${Math.round(bd.max_minutes)} min)`
    : `rounded to ${bold(mins(bd.predicted_minutes))}`;

  return `<div class="explain">${s1}. ${s2}, ${s3}.</div>`;
}

function loadDetail(bd, m, showContext) {
  if (!bd) return "";
  const learned = (bd.sample_count || 0) > 0;
  const rows = [
    chip("Model", learned ? `learned (${bd.sample_count} days)` : "seed (no data yet)"),
    chip("Baseline", `${kwh(bd.e_base)} kWh`),
    chip("Per person", `${kwh(bd.e_draw_per_person)} kWh`),
    chip("Guest bonus", `${kwh(bd.guest_bonus)} kWh`),
    chip("Empty-house", `${Math.round((bd.empty_house_factor || 0) * 100)}%`),
    chip("Gain", gain(bd.gain)),
  ];
  const acc = [
    chip("Rolling MAE", m.rolling_mae_minutes != null ? mins(m.rolling_mae_minutes) : "—"),
    chip("Last delivered", m.last_delivered_kwh != null ? `${kwh(m.last_delivered_kwh)} kWh` : "—"),
    chip("Last error", m.prediction_error_minutes != null ? mins(m.prediction_error_minutes) : "—"),
  ];
  let ctx = "";
  if (showContext && bd.context) {
    const c = bd.context;
    const items = [];
    if (c.supply_temp != null) items.push(chip("Supply temp", `${c.supply_temp}°C`));
    if (c.outdoor_temp != null) items.push(chip("Outdoor temp", `${c.outdoor_temp}°C`));
    if (c.water_total_delta != null) items.push(chip("Water Δ", `${c.water_total_delta}`));
    if (items.length) ctx = `<div class="sub">Logged context (not used by v1)</div><div class="chips">${items.join("")}</div>`;
  }
  return `
    <div class="sub">Model parameters</div>
    <div class="chips">${rows.join("")}</div>
    <div class="sub">Accuracy</div>
    <div class="chips">${acc.join("")}</div>
    ${ctx}`;
}

// ── natural-language composition (forecast) ─────────────────────────────────
function forecastExplanation(a) {
  const days = a.days || [];
  const parts = [];
  if (days.length) {
    const first = days[0].buy;
    const last = days[days.length - 1].buy;
    const trend =
      last > first + 0.001 ? "rising" : last < first - 0.001 ? "falling" : "roughly steady";
    parts.push(
      `Next ${bold(days.length)} day${days.length === 1 ? "" : "s"}: ${bold(eur(first))}→${bold(
        eur(last)
      )} €/kWh (${trend})`
    );
  } else {
    parts.push("No forecast days available yet");
  }

  let model;
  if (a.fitted) {
    model = `Fitted on ${bold(a.model_samples)} days`;
    if (a.fit_mae_eur_kwh != null) model += ` (±${bold(eur(a.fit_mae_eur_kwh))} €/kWh in-sample)`;
  } else {
    model = "Using the seed formula (not enough history yet)";
  }

  let dir = "";
  const c = a.coefficients;
  if (c && c.betas) {
    const phr = [];
    if (Math.abs(c.betas[1]) > 1e-9) phr.push(`more wind ${c.betas[1] < 0 ? "lowers" : "raises"} price`);
    if (Math.abs(c.betas[2]) > 1e-9) phr.push(`cold ${c.betas[2] > 0 ? "raises" : "lowers"} price`);
    if (phr.length) dir = " " + cap(phr.join("; ")) + ".";
  }
  return `<div class="explain">${parts.join(". ")}. ${model}.${dir}</div>`;
}

function forecastDetail(a) {
  const days = a.days || [];
  let dayList = "";
  if (days.length) {
    const cheapest = Math.min(...days.map((d) => d.buy));
    dayList = days
      .map((d) => {
        const hl = d.buy === cheapest ? ' class="cheap"' : "";
        return `<div class="day"><span>${esc(d.date)}</span>
          <span class="muted">${d.temp}°C · ${d.wind_gw} GW</span>
          <span${hl}>${eur(d.buy)} €/kWh</span></div>`;
      })
      .join("");
    dayList = `<div class="sub">Forecast days</div><div class="days">${dayList}</div>`;
  }

  let coeffs = "";
  const c = a.coefficients;
  if (c && c.betas && c.features) {
    const rows = c.features
      .map((f, i) => `<div class="day"><span>${esc(f)}</span><span>${c.betas[i].toFixed(4)}</span></div>`)
      .join("");
    coeffs = `<div class="sub">Regression coefficients (standardised)</div>
      <div class="days">${rows}
      <div class="day"><span>intercept</span><span>${eur(c.intercept)}</span></div></div>`;
  }

  const acc = [
    chip("Forecast MAE", a.forecast_mae_eur_kwh != null ? `${eur(a.forecast_mae_eur_kwh)} €/kWh` : "—"),
    chip("Eval samples", a.coefficients ? a.coefficients.n : "—"),
  ];

  return `${dayList}${coeffs}
    <div class="sub">Accuracy</div><div class="chips">${acc.join("")}</div>`;
}

// ── shared snippets ─────────────────────────────────────────────────────────
function chip(label, value) {
  return `<span class="chip"><span class="cl">${esc(label)}</span>${esc(value)}</span>`;
}
function pushDot(ok) {
  const cls = ok === true ? "ok" : ok === false ? "bad" : "unk";
  const title = ok === true ? "Pushed OK" : ok === false ? "Push failed" : "Not pushed yet";
  return `<span class="dot ${cls}" title="${title}"></span>`;
}
function actionBtn(icon, label, service, entity) {
  return `<button class="action" title="${esc(label)}"
    data-action="service" data-service="${service}" data-entity="${entity}">
    <ha-icon icon="mdi:${icon.replace("mdi-", "")}"></ha-icon></button>`;
}

const STYLES = `
  :host { display: block; }
  ha-card { padding: 12px 16px 16px; }
  .card-header { font-size: 1.1rem; margin: 0 0 4px; padding: 0; }
  .content { container-type: inline-size; }
  .block { padding: 10px 0; border-top: 1px solid var(--divider-color); }
  .block:first-child { border-top: none; }
  .head { display: flex; align-items: center; gap: 8px; }
  .head ha-icon { color: var(--state-icon-color, var(--primary-color)); --mdc-icon-size: 20px; }
  .name { font-weight: 600; }
  .spacer { flex: 1; }
  .big { font-size: 1.25rem; margin: 4px 0 2px; }
  .muted { color: var(--secondary-text-color); font-weight: 400; }
  .warn { color: var(--warning-color, #ffa600); font-size: 0.8rem; }
  .explain { font-size: 0.92rem; line-height: 1.45; color: var(--primary-text-color); }
  .explain b { font-weight: 600; }
  .toggle { margin-top: 6px; font-size: 0.82rem; color: var(--primary-color); cursor: pointer; user-select: none; width: fit-content; }
  .detail { margin-top: 6px; }
  .sub { font-size: 0.74rem; text-transform: uppercase; letter-spacing: .04em; color: var(--secondary-text-color); margin: 8px 0 4px; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip { background: var(--secondary-background-color); border-radius: 10px; padding: 2px 8px; font-size: 0.82rem; white-space: nowrap; }
  .chip .cl { color: var(--secondary-text-color); margin-right: 4px; }
  .days { display: flex; flex-direction: column; gap: 2px; }
  .day { display: flex; justify-content: space-between; gap: 10px; font-size: 0.85rem; padding: 1px 0; }
  .day .cheap { color: var(--success-color, #43a047); font-weight: 600; }
  .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
  .dot.ok { background: var(--success-color, #43a047); }
  .dot.bad { background: var(--error-color, #db4437); }
  .dot.unk { background: var(--disabled-text-color, #9e9e9e); }
  .action { border: none; background: var(--secondary-background-color); color: var(--primary-color);
            border-radius: 8px; padding: 4px 6px; cursor: pointer; display: inline-flex; align-items: center; }
  .action:hover { background: var(--primary-color); color: var(--text-primary-color, #fff); }
  .action ha-icon { --mdc-icon-size: 18px; }
  .empty { color: var(--secondary-text-color); padding: 8px 0; }
  @container (min-width: 420px) {
    .big { font-size: 1.35rem; }
  }
`;

// ── graphical config editor (ha-form + selectors) ───────────────────────────
const EDITOR_SCHEMA = [
  { name: "title", selector: { text: {} } },
  { name: "devices", selector: { device: { integration: DOMAIN, multiple: true } } },
  { name: "default_expanded", selector: { boolean: {} } },
  { name: "show_context", selector: { boolean: {} } },
];
const EDITOR_LABELS = {
  title: "Title (optional)",
  devices: "Loads / forecasts (leave empty for all)",
  default_expanded: "Expand detail by default",
  show_context: "Show logged context (temps, water)",
};

class LoadNeedPredictorCardEditor extends HTMLElement {
  setConfig(config) {
    this._config = { ...config };
    this._update();
  }
  set hass(hass) {
    this._hass = hass;
    this._update();
  }
  _update() {
    if (!this._hass) return;
    if (!this._form) {
      this._form = document.createElement("ha-form");
      this._form.computeLabel = (s) => EDITOR_LABELS[s.name] || s.name;
      this._form.addEventListener("value-changed", (ev) => {
        ev.stopPropagation();
        this.dispatchEvent(
          new CustomEvent("config-changed", {
            detail: { config: ev.detail.value },
            bubbles: true,
            composed: true,
          })
        );
      });
      this.appendChild(this._form);
    }
    this._form.hass = this._hass;
    this._form.schema = EDITOR_SCHEMA;
    this._form.data = this._config || {};
  }
}

// ── registration ────────────────────────────────────────────────────────────
if (!customElements.get("load-need-predictor-card")) {
  customElements.define("load-need-predictor-card", LoadNeedPredictorCard);
}
if (!customElements.get("load-need-predictor-card-editor")) {
  customElements.define("load-need-predictor-card-editor", LoadNeedPredictorCardEditor);
}

window.customCards = window.customCards || [];
if (!window.customCards.some((c) => c.type === "load-need-predictor-card")) {
  window.customCards.push({
    type: "load-need-predictor-card",
    name: "Load Need Predictor",
    description: "Explains how each load's runtime and the price forecast are calculated.",
    preview: true,
    documentationURL: DOC_URL,
  });
}

console.info(
  `%c LOAD-NEED-PREDICTOR-CARD %c v${CARD_VERSION} `,
  "color:#fff;background:#0277bd;border-radius:3px 0 0 3px;padding:2px 4px",
  "color:#0277bd;background:#e1f5fe;border-radius:0 3px 3px 0;padding:2px 4px"
);
