"""Structural refit: blending E_base / E_draw toward an empirical fit."""

from __future__ import annotations

import pytest
from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.load_need_predictor.const import DOMAIN, SUBENTRY_TYPE_LOAD
from custom_components.load_need_predictor.predictor import (
    N_PRIOR,
    SEED_E_BASE,
    SEED_E_DRAW_PER_PERSON,
)

_LOAD_DATA = {
    "name": "LVV",
    "delivered_energy_entity": "sensor.lvv_energy",
    "rated_power_kw": 3.0,
    "person_entities": ["person.a"],
}


async def _coordinator(hass: HomeAssistant):
    hass.states.async_set("person.a", "home")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"name": "Predictor"},
        subentries_data=[
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_LOAD, title="LVV", unique_id=None, data=_LOAD_DATA
            )
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry.runtime_data


def _rows(n: int, *, base: float, slope: float, people_cycle=(0, 1, 2)) -> list[dict]:
    rows = []
    for i in range(n):
        people = people_cycle[i % len(people_cycle)]
        rows.append(
            {
                "date": f"d{i}",
                "people_home": people,
                "actual_kwh": base + slope * people,
                "data_quality": True,
                "predicted_kwh": 5.0,
                "predicted_minutes": 100,
            }
        )
    return rows


async def test_refit_blends_toward_empirical(hass: HomeAssistant) -> None:
    coordinator = await _coordinator(hass)
    sid = next(iter(coordinator.load_configs()))
    n = 15
    coordinator.training[sid] = _rows(n, base=4.0, slope=1.5)

    coordinator._maybe_refit(sid)

    model = coordinator.models[sid]
    # blend(prior, empirical, n) = (N_PRIOR*prior + n*emp)/(N_PRIOR+n)
    assert model.e_base == pytest.approx((N_PRIOR * SEED_E_BASE + n * 4.0) / (N_PRIOR + n))
    assert model.e_draw_per_person == pytest.approx(
        (N_PRIOR * SEED_E_DRAW_PER_PERSON + n * 1.5) / (N_PRIOR + n)
    )


async def test_refit_skips_below_threshold(hass: HomeAssistant) -> None:
    coordinator = await _coordinator(hass)
    sid = next(iter(coordinator.load_configs()))
    coordinator.training[sid] = _rows(10, base=4.0, slope=1.5)  # < MIN_REFIT_SAMPLES

    coordinator._maybe_refit(sid)

    assert coordinator.models[sid].e_base == SEED_E_BASE  # seeds untouched


async def test_refit_skips_without_occupancy_variation(hass: HomeAssistant) -> None:
    coordinator = await _coordinator(hass)
    sid = next(iter(coordinator.load_configs()))
    coordinator.training[sid] = _rows(20, base=4.0, slope=1.5, people_cycle=(2,))  # all 2 people

    coordinator._maybe_refit(sid)

    assert coordinator.models[sid].e_base == SEED_E_BASE  # not identifiable → unchanged
