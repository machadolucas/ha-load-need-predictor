// Pure card contract tests: no browser or Home Assistant instance needed.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function loadCard() {
  const registry = new Map();
  const context = vm.createContext({
    HTMLElement: class {
      attachShadow() {
        this.shadowRoot = { innerHTML: "", addEventListener() {} };
      }
    },
    customElements: { get: (key) => registry.get(key), define: (key, value) => registry.set(key, value) },
    window: {},
    console: { info() {} },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../custom_components/load_need_predictor/www/load-need-predictor-card.js"), "utf8"), context);
  const Card = registry.get("load-need-predictor-card");
  return { card: new Card(), run: (expression) => vm.runInContext(expression, context), context };
}
function makeHass(attributes, callService = async () => {}) {
  return {
    entities: {
      runtime: { platform: "load_need_predictor", device_id: "tank", translation_key: "predicted_runtime", entity_id: "sensor.runtime" },
      tank: { platform: "load_need_predictor", device_id: "tank", translation_key: "tank_soc", entity_id: "sensor.tank" },
    },
    devices: { tank: { name: "Water heater" } },
    states: {
      "sensor.runtime": { state: "120", last_updated: "same", attributes: {} },
      "sensor.tank": { state: "55", last_updated: "same", attributes },
    },
    callService,
  };
}
const event = { id: "ev-1", start: "2026-10-13T09:00:00+03:00", end: "2026-10-13T09:08:00+03:00", water_l: 55, confidence: "medium" };

test("stored heat never implies remaining showers on old or new backends", () => {
  const { card } = loadCard();
  card.hass = makeHass({ showers_left: 6, liters_40c: 400, equivalent_liters_40c: 400 });
  assert.match(card.shadowRoot.innerHTML, /Estimated stored heat/);
  assert.match(card.shadowRoot.innerHTML, /Hot-water availability unknown/);
  assert.doesNotMatch(card.shadowRoot.innerHTML, /6 showers|400 L|data-action="comfort"/);
});

test("conditional liters and warnings escape backend strings", () => {
  const { card } = loadCard();
  card.hass = makeHass({ equivalent_liters_40c: 200, equivalent_liters_assumption: '<img src="bad">', quality_warnings: ['<script>bad</script>'], anchor_age_hours: 4.5 });
  const html = card.shadowRoot.innerHTML;
  assert.match(html, /Energy equivalent: ~200 L at 40 °C/);
  assert.match(html, /not a measured volume/);
  assert.match(html, /4\.5 h ago/);
  assert.match(html, /&lt;script&gt;bad&lt;\/script&gt;/);
  assert.doesNotMatch(html, /<img src="bad">|<script>bad<\/script>/);
});

test("attribute changes redraw even when state and timestamp are unchanged", () => {
  const { card } = loadCard();
  const hass = makeHass({ comfort_feedback_supported: true, recent_showers: [event] });
  card.hass = hass;
  const before = card.shadowRoot.innerHTML;
  hass.states["sensor.tank"] = { ...hass.states["sensor.tank"], attributes: { comfort_feedback_supported: true, recent_showers: [{ ...event, feedback: "comfortable" }] } };
  card.hass = hass;
  assert.notEqual(card.shadowRoot.innerHTML, before);
  assert.match(card.shadowRoot.innerHTML, /medium · Comfortable/);
});

test("selected event sends one exact comfort report and shows async progress", async () => {
  const { card } = loadCard();
  let resolve;
  const calls = [];
  card.hass = makeHass({ comfort_feedback_supported: true, recent_showers: [event, { ...event, id: 'ev-2"<unsafe>' }] }, (...args) => {
    calls.push(args);
    return new Promise((r) => { resolve = r; });
  });
  card._onChange({ target: { dataset: { action: "shower-select", entity: "sensor.tank" }, value: 'ev-2"<unsafe>' } });
  const pending = card._reportComfort("sensor.tank", "cooler_than_usual");
  assert.match(card.shadowRoot.innerHTML, /Saving feedback…/);
  assert.match(card.shadowRoot.innerHTML, /data-comfort="comfortable" disabled/);
  await card._reportComfort("sensor.tank", "not_warm_enough");
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0], "load_need_predictor");
  assert.equal(calls[0][1], "report_shower_comfort");
  assert.deepEqual(JSON.parse(JSON.stringify(calls[0][2])), { entity_id: "sensor.tank", event_id: 'ev-2"<unsafe>', comfort: "cooler_than_usual" });
  resolve();
  await pending;
  assert.match(card.shadowRoot.innerHTML, /Saved: Cooler than usual/);
  assert.doesNotMatch(card.shadowRoot.innerHTML, /data-comfort="comfortable" disabled/);
});

test("service failure displays escaped error and supports retry", async () => {
  const { card } = loadCard();
  let calls = 0;
  card.hass = makeHass({ comfort_feedback_supported: true, recent_showers: [event] }, async () => {
    if (++calls === 1) throw new Error('<img src="bad">');
  });
  await card._reportComfort("sensor.tank", "not_warm_enough");
  assert.match(card.shadowRoot.innerHTML, /Could not save feedback: &lt;img/);
  assert.doesNotMatch(card.shadowRoot.innerHTML, /<img src="bad">/);
  await card._reportComfort("sensor.tank", "comfortable");
  assert.match(card.shadowRoot.innerHTML, /Saved: Comfortable/);
});

test("unsupported backend or stale event cannot submit feedback", async () => {
  const { card } = loadCard();
  let calls = 0;
  const hass = makeHass({ recent_showers: [event] }, async () => { calls++; });
  card.hass = hass;
  await card._reportComfort("sensor.tank", "comfortable");
  hass.states["sensor.tank"].attributes = { comfort_feedback_supported: true, recent_showers: [] };
  await card._reportComfort("sensor.tank", "comfortable");
  assert.equal(calls, 0);
});


test("expired selection never silently reports comfort against a different event", async () => {
  const { card } = loadCard();
  let calls = 0;
  const hass = makeHass({ comfort_feedback_supported: true, recent_showers: [event] }, async () => { calls++; });
  card.hass = hass;
  card._onChange({ target: { dataset: { action: "shower-select", entity: "sensor.tank" }, value: event.id } });
  hass.states["sensor.tank"].attributes = { comfort_feedback_supported: true, recent_showers: [{ ...event, id: "new-event" }] };
  await card._reportComfort("sensor.tank", "comfortable");
  assert.equal(calls, 0);
  assert.match(card.shadowRoot.innerHTML, /no longer available/);
});

test("appliance shadow activity never promises cold-water subtraction", () => {
  const { card } = loadCard();
  card.hass = makeHass({ appliance_profiles: { status: "observational_only", active_run_count: 1 } });
  assert.match(card.shadowRoot.innerHTML, /1 runs in progress/);
  assert.match(card.shadowRoot.innerHTML, /Appliance water attribution is unvalidated/);
  assert.match(card.shadowRoot.innerHTML, /stored heat is unchanged by this analysis/);
});
