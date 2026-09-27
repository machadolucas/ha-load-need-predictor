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
    SEED_EMPTY_HOUSE_FACTOR,
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
    return entry.runtime_data.load


def _rows(
    n: int, *, base: float, slope: float, people_cycle=(0, 1, 2), gain: float = 1.0
) -> list[dict]:
    """Rows whose actual follows ``predict_kwh``'s own equation exactly.

    ``gain × (factor × (base + slope × p))`` with the seeded empty-house factor
    — so a correct refit recovers ``(base, slope)`` whatever gain was in force.
    """
    rows = []
    for i in range(n):
        people = people_cycle[i % len(people_cycle)]
        factor = 1.0 if people > 0 else SEED_EMPTY_HOUSE_FACTOR
        rows.append(
            {
                "date": f"d{i}",
                "people_home": people,
                "guests": 0.0,
                "gain": gain,
                "actual_kwh": gain * factor * (base + slope * people),
                "data_quality": True,
                "predicted_kwh": 5.0,
                "predicted_minutes": 100,
            }
        )
    return rows


def _blended(prior: float, emp: float, n: int) -> float:
    return (N_PRIOR * prior + n * emp) / (N_PRIOR + n)


async def test_refit_blends_toward_empirical(hass: HomeAssistant) -> None:
    coordinator = await _coordinator(hass)
    sid = next(iter(coordinator.load_configs()))
    n = 15
    coordinator.training[sid] = _rows(n, base=4.0, slope=1.5)

    coordinator._maybe_refit(sid)

    model = coordinator.models[sid]
    # blend(prior, empirical, n) = (N_PRIOR*prior + n*emp)/(N_PRIOR+n)
    assert model.e_base == pytest.approx(_blended(SEED_E_BASE, 4.0, n))
    assert model.e_draw_per_person == pytest.approx(_blended(SEED_E_DRAW_PER_PERSON, 1.5, n))


async def test_refit_divides_out_the_row_gain(hass: HomeAssistant) -> None:
    """Rows predicted/observed under gain 1.3 still fit the seeds' structure.

    Regression: fitting raw actuals baked the gain into E_base/E_draw, and the
    online gain then applied on top a second time.
    """
    coordinator = await _coordinator(hass)
    sid = next(iter(coordinator.load_configs()))
    n = 15
    coordinator.training[sid] = _rows(n, base=SEED_E_BASE, slope=SEED_E_DRAW_PER_PERSON, gain=1.3)

    coordinator._maybe_refit(sid)

    model = coordinator.models[sid]
    assert model.e_base == pytest.approx(SEED_E_BASE)
    assert model.e_draw_per_person == pytest.approx(SEED_E_DRAW_PER_PERSON)


async def test_refit_skips_unclean_rows_but_keeps_legacy(hass: HomeAssistant) -> None:
    coordinator = await _coordinator(hass)
    sid = next(iter(coordinator.load_configs()))
    good = _rows(15, base=4.0, slope=1.5)  # no clean_cycle key → legacy → counts
    skipped = _rows(15, base=0.5, slope=0.1)
    for row in skipped:
        row["clean_cycle"] = False  # a skip/defer day: meter ≠ demand
    coordinator.training[sid] = good + skipped

    coordinator._maybe_refit(sid)

    model = coordinator.models[sid]
    assert model.e_base == pytest.approx(_blended(SEED_E_BASE, 4.0, 15))
    assert model.e_draw_per_person == pytest.approx(_blended(SEED_E_DRAW_PER_PERSON, 1.5, 15))


async def test_refit_prefers_energy_balance_demand(hass: HomeAssistant) -> None:
    coordinator = await _coordinator(hass)
    sid = next(iter(coordinator.load_configs()))
    rows = _rows(15, base=4.0, slope=1.5)
    for row in rows:
        row["demand_kwh"] = row["actual_kwh"]
        row["actual_kwh"] += 3.0  # a refill on top of demand — must be ignored
    coordinator.training[sid] = rows

    coordinator._maybe_refit(sid)

    assert coordinator.models[sid].e_base == pytest.approx(_blended(SEED_E_BASE, 4.0, 15))


async def test_refit_needs_a_multi_person_day(hass: HomeAssistant) -> None:
    coordinator = await _coordinator(hass)
    sid = next(iter(coordinator.load_configs()))
    coordinator.training[sid] = _rows(20, base=4.0, slope=1.5, people_cycle=(0, 1))

    coordinator._maybe_refit(sid)

    assert coordinator.models[sid].e_base == SEED_E_BASE  # contract: ≥1 day with p ≥ 2


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
