"""Diagnostics dump: includes model state, redacts entity ids."""

from __future__ import annotations

import json
import pathlib

from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant
from homeassistant.helpers.json import JSONEncoder
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.load_need_predictor.const import (
    DOMAIN,
    SUBENTRY_TYPE_LOAD,
    SUBENTRY_TYPE_PRICE_FORECAST,
)
from custom_components.load_need_predictor.diagnostics import (
    async_get_config_entry_diagnostics,
)
from custom_components.load_need_predictor.forecast_source import WattcastFetch
from custom_components.load_need_predictor.spot_forecast import parse_wattcast

_FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "wattcast_fi_15min.json"

_LOAD_DATA = {
    "name": "LVV",
    "delivered_energy_entity": "sensor.lvv_energy",
    "person_entities": ["person.a"],
    "rated_power_kw": 3.0,
}


async def _setup(hass: HomeAssistant):
    hass.states.async_set("person.a", "home")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"name": "Predictor", "predict_time": "14:00:00"},
        subentries_data=[
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_LOAD, title="LVV", unique_id=None, data=_LOAD_DATA
            )
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


async def test_diagnostics_structure_and_redaction(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    diag = await async_get_config_entry_diagnostics(hass, entry)

    assert set(diag) == {"hub", "loads", "forecasts"}
    sid = next(iter(diag["loads"]))
    load = diag["loads"][sid]

    # Model + result are present for debugging.
    assert "gain" in load["model"]
    # One person home → 3.0 + 2.2 = 5.2 kWh → /3 kW ×60 = 104 → rounded to 105 min.
    assert load["result"]["predicted_minutes"] == 105

    # Entity ids are redacted; the friendly name is not.
    assert load["config"]["delivered_energy_entity"] != "sensor.lvv_energy"
    assert load["config"]["name"] == "LVV"


async def test_diagnostics_forecast_wattcast_section(
    hass: HomeAssistant, freezer, no_wattcast_network
) -> None:
    series = parse_wattcast(json.loads(_FIXTURE.read_text()))
    no_wattcast_network.return_value = WattcastFetch(series)
    freezer.move_to("2026-09-24T08:42:00+00:00")
    hass.states.async_set("sensor.price", "0.1")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"name": "Predictor", "predict_time": "14:00:00"},
        subentries_data=[
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_PRICE_FORECAST,
                title="LVV",
                unique_id=None,
                data={
                    "name": "LVV",
                    "price_entity": "sensor.price",
                    "price_series_entity": "sensor.nordpool",
                },
            )
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    diag = await async_get_config_entry_diagnostics(hass, entry)
    # Must survive HA's own serialiser (datetimes in the fetch state, dataclass
    # dicts, the mapping) — that's what the download button runs.
    json.dumps(diag, cls=JSONEncoder)

    fc = next(iter(diag["forecasts"].values()))
    assert fc["config"]["price_series_entity"] != "sensor.nordpool"  # redacted
    assert fc["config"]["price_entity"] != "sensor.price"
    assert fc["config"]["use_wattcast"] is True
    wc = fc["wattcast"]
    assert wc["made_at"] == "2026-09-24T08:25:00+00:00"
    assert wc["fetched_at"].startswith("2026-09-24T08:42:00")
    assert wc["forecast_points"] == 24
    assert wc["fetch"]["failures"] == 0
    assert fc["mapping"]["slope_pos"] > 1.0  # seed (no series pairs) = 1 + VAT
    assert fc["retail_pairs"] >= 0
    assert "shape" in fc
    # The published series is trimmed to a preview.
    assert len(fc["result"]["slots"]) <= 8
    assert fc["result"]["source"] == "wattcast"
