"""Deficit carryover: a skipped/under-run cycle is made up the next day, and the
gain only learns from clean (fully-delivered, no-backlog) cycles.

``_commanded_since`` (predict path) and ``async_commanded_minutes`` (capture
path) are patched so a test controls how much "actually ran" without a recorder.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_mock_service

from custom_components.load_need_predictor.const import DOMAIN, SUBENTRY_TYPE_LOAD
from custom_components.load_need_predictor.statistics_source import capture_window
from custom_components.load_need_predictor.tank_model import initial_state

_STATS = "custom_components.load_need_predictor.coordinator.async_daily_delivered_kwh"
_CMD = "custom_components.load_need_predictor.coordinator.async_commanded_minutes"

_LOAD_DATA = {
    "name": "LVV",
    "target_number_entity": "number.lvv_target",
    "delivered_energy_entity": "sensor.lvv_energy",
    "controlled_switch_entity": "switch.lvv",
    "rated_power_kw": 3.0,
    "person_entities": ["person.a", "person.b"],
    "min_minutes": 40,
    "max_minutes": 480,
}


async def _setup(hass: HomeAssistant, load_data: dict | None = None):
    hass.states.async_set("person.a", "home")
    hass.states.async_set("person.b", "home")
    hass.states.async_set("number.lvv_target", "0")
    hass.states.async_set("switch.lvv", "off")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"name": "Predictor", "predict_time": "14:00:00", "capture_time": "23:55:00"},
        subentries_data=[
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_LOAD,
                title="LVV",
                unique_id=None,
                data=load_data or _LOAD_DATA,
            )
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry, entry.runtime_data.load


def _age_cycle(coord, sid: str, days: int = 1) -> None:
    """Pretend the open cycle began ``days`` ago — cycles close once per local day."""
    state = coord.models[sid]
    start = dt_util.parse_datetime(state.cycle_start_iso) - timedelta(days=days)
    coord.models[sid] = replace(state, cycle_start_iso=start.isoformat())


async def test_skip_accumulates_into_next_target(hass: HomeAssistant) -> None:
    entry, coord = await _setup(hass)
    calls = async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))

    # Two daily predicts with nothing running between them: the first opens a
    # cycle (no prior cycle to close), the next day's closes the skipped one.
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=0.0)):
        await coord.async_predict_and_push()
        assert calls[-1].data["value"] == 150  # plain need (2 people → 7.4 kWh)
        _age_cycle(coord, sid)
        await coord.async_predict_and_push()

    # Need (150) + carried backlog (~148) → a clearly larger pushed target.
    assert calls[-1].data["value"] > 150
    assert coord.models[sid].deficit_minutes > 100
    assert coord.data[sid].deficit_minutes is not None

    # Run the full ask this cycle → the backlog clears.
    ran = float(calls[-1].data["value"])
    _age_cycle(coord, sid)
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=ran)):
        await coord.async_predict_and_push()
    assert coord.models[sid].deficit_minutes == 0.0


async def test_same_day_repredict_replans_without_double_counting(hass: HomeAssistant) -> None:
    """The button / a boost mid-cycle must not roll the whole ask into backlog.

    Regression: each re-predict closed the partial cycle (little on-time so
    far) → ~the whole ask became backlog *and* a fresh need was added on top.
    """
    entry, coord = await _setup(hass)
    calls = async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))

    commanded = AsyncMock(return_value=0.0)
    with patch.object(coord, "_commanded_since", new=commanded):
        await coord.async_predict_and_push()
        first = calls[-1].data["value"]
        opened = coord.models[sid].cycle_start_iso
        await coord.async_predict_and_push()  # e.g. "Predict now" an hour later

    assert calls[-1].data["value"] == first == 150
    assert coord.models[sid].deficit_minutes == 0.0
    assert coord.models[sid].cycle_start_iso == opened  # the day's cycle stays open
    assert coord.models[sid].pending_owed_minutes == pytest.approx(148.0)  # need only
    assert commanded.await_count == 1  # only the opening predict; a re-plan never closes


async def test_same_day_repredict_keeps_carried_backlog(hass: HomeAssistant) -> None:
    entry, coord = await _setup(hass)
    calls = async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))

    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=0.0)):
        await coord.async_predict_and_push()
        _age_cycle(coord, sid)
        await coord.async_predict_and_push()  # new day: closes the skip → backlog
        backlog = coord.models[sid].deficit_minutes
        pushed = calls[-1].data["value"]
        await coord.async_predict_and_push()  # same day: re-plan only

    assert coord.models[sid].deficit_minutes == backlog  # neither grown nor lost
    assert calls[-1].data["value"] == pushed


async def test_predict_returns_push_success(hass: HomeAssistant) -> None:
    entry, coord = await _setup(hass)
    async_mock_service(hass, "number", "set_value")
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        assert await coord.async_predict_and_push() is True
        hass.states.async_remove("number.lvv_target")  # scheduler gone → push fails
        assert await coord.async_predict_and_push() is False


async def test_missing_recorder_keeps_backlog_zero(hass: HomeAssistant) -> None:
    entry, coord = await _setup(hass)
    calls = async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))

    # No commanded reading available → never close → backlog stays 0, behaviour
    # is the plain daily predictor.
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        await coord.async_predict_and_push()
        await coord.async_predict_and_push()
    assert coord.models[sid].deficit_minutes == 0.0
    assert calls[-1].data["value"] == 150


async def test_gain_learns_on_clean_cycle(hass: HomeAssistant) -> None:
    entry, coord = await _setup(hass)
    async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))

    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        await coord.async_predict_and_push()  # pushes 150, opens a cycle, no backlog

    # Full delivery (commanded ≈ ask) with no backlog → clean → the gain learns.
    with (
        patch(_STATS, new=AsyncMock(return_value=6.9)),
        patch(_CMD, new=AsyncMock(return_value=150.0)),
    ):
        await coord.async_capture_and_log()

    row = coord.training[sid][-1]
    assert row["clean_cycle"] is True
    assert coord.models[sid].sample_count == 1
    assert coord.models[sid].gain < 1.0  # 6.9 < 7.4 predicted


async def test_gain_not_learned_when_scheduler_underran(hass: HomeAssistant) -> None:
    entry, coord = await _setup(hass)
    async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))

    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        await coord.async_predict_and_push()  # pushes 150

    # The scheduler ran far less than asked (a price/solar defer): the meter
    # under-reads demand, so the day is not clean and the gain must not move.
    before = coord.models[sid].gain
    with (
        patch(_STATS, new=AsyncMock(return_value=3.0)),
        patch(_CMD, new=AsyncMock(return_value=20.0)),
    ):
        await coord.async_capture_and_log()

    row = coord.training[sid][-1]
    assert row["clean_cycle"] is False
    assert coord.models[sid].sample_count == 0
    assert coord.models[sid].gain == before


async def test_gain_not_learned_while_backlog_active(hass: HomeAssistant) -> None:
    entry, coord = await _setup(hass)
    async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))

    # Build a backlog by skipping a (daily) cycle.
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=0.0)):
        await coord.async_predict_and_push()
        _age_cycle(coord, sid)
        await coord.async_predict_and_push()
    assert coord.models[sid].deficit_minutes > 30  # backlog in play

    # Even a full, valid delivery isn't a clean demand sample while recovering.
    before = coord.models[sid].gain
    with (
        patch(_STATS, new=AsyncMock(return_value=6.0)),
        patch(_CMD, new=AsyncMock(return_value=300.0)),
    ):
        await coord.async_capture_and_log()

    row = coord.training[sid][-1]
    assert row["clean_cycle"] is False
    assert coord.models[sid].sample_count == 0
    assert coord.models[sid].gain == before


# ── tank feedback: energy-balance learning + tracking-enabled predicate ────────

_TANK_DATA = {**_LOAD_DATA, "heating_active_entity": "binary_sensor.lvv_heating"}
_CAP = 22.0


def _calibrated(deficit_kwh: float):
    return replace(initial_state(_CAP), deficit_kwh=deficit_kwh, calibrated=True)


def _yesterday() -> str:
    return (dt_util.now().date() - timedelta(days=1)).isoformat()


def _prev_snapshot(deficit_kwh: float, offset: timedelta = timedelta(0)) -> dict:
    start, _ = capture_window(dt_util.utcnow())
    return {
        "date": _yesterday(),
        "tank_deficit_end_kwh": deficit_kwh,
        "tank_deficit_end_at": (start + offset).isoformat(),
    }


async def test_energy_balance_learns_on_refill_day(hass: HomeAssistant) -> None:
    """A tank-refill day is a fair demand sample once ΔSoC is accounted for.

    The old gate threw these away (row backlog ≥ 30 min) — ~40 % of days on a
    calibrated tank, mostly the heavy-draw ones, biasing the gain low.
    """
    hass.states.async_set("binary_sensor.lvv_heating", "off")
    entry, coord = await _setup(hass, _TANK_DATA)
    async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))
    # Yesterday's capture left the tank 3 kWh short — snapshotted right at
    # the start of today's delivery window.
    coord.training[sid] = [_prev_snapshot(3.0)]

    coord.tanks[sid] = _calibrated(3.0)  # 60 min folded into today's push
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        await coord.async_predict_and_push()
    assert coord.training[sid][-1]["deficit_minutes"] >= 30  # old gate would reject

    # Refilled to full by capture: delivered 8.2 = 5.2 demand + 3.0 backlog.
    coord.tanks[sid] = _calibrated(0.0)
    with (
        patch(_STATS, new=AsyncMock(return_value=8.2)),
        patch(_CMD, new=AsyncMock(return_value=210.0)),
    ):
        await coord.async_capture_and_log()

    row = coord.training[sid][-1]
    assert row["tank_deficit_end_kwh"] == 0.0
    assert row["demand_kwh"] == pytest.approx(5.2)
    assert row["clean_cycle"] is True
    assert row["abs_error_minutes"] == 150 - 104  # need vs demand, not vs meter
    assert coord.models[sid].sample_count == 1
    assert coord.models[sid].gain < 1.0  # learned 5.2 vs 7.4, not 8.2


async def test_energy_balance_needs_both_snapshots(hass: HomeAssistant) -> None:
    hass.states.async_set("binary_sensor.lvv_heating", "off")
    entry, coord = await _setup(hass, _TANK_DATA)
    async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))

    coord.tanks[sid] = _calibrated(3.0)
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        await coord.async_predict_and_push()
    with (
        patch(_STATS, new=AsyncMock(return_value=8.2)),
        patch(_CMD, new=AsyncMock(return_value=210.0)),
    ):
        await coord.async_capture_and_log()

    # No previous snapshot → the old clean-cycle gate applies unchanged…
    row = coord.training[sid][-1]
    assert "demand_kwh" not in row
    assert row["clean_cycle"] is False
    assert coord.models[sid].sample_count == 0
    # …but today's snapshot is taken (with its time), so tomorrow can balance.
    assert row["tank_deficit_end_kwh"] == 3.0
    assert dt_util.parse_datetime(row["tank_deficit_end_at"]) is not None
    assert row["capture_window_end"]


@pytest.mark.parametrize(
    "prev",
    [
        {"tank_deficit_end_kwh": 3.0},  # legacy: no timestamp → proves nothing
        "shifted",  # e.g. capture time moved 23:55 → 22:00
    ],
)
async def test_energy_balance_rejects_mismatched_spans(hass: HomeAssistant, prev) -> None:
    """Delivery window and ΔSoC must cover the same span, else the old gate."""
    hass.states.async_set("binary_sensor.lvv_heating", "off")
    entry, coord = await _setup(hass, _TANK_DATA)
    async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))
    if prev == "shifted":
        prev = _prev_snapshot(3.0, offset=-timedelta(hours=2))
    else:
        prev = {"date": _yesterday(), **prev}
    coord.training[sid] = [prev]

    coord.tanks[sid] = _calibrated(3.0)
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        await coord.async_predict_and_push()
    coord.tanks[sid] = _calibrated(0.0)
    with (
        patch(_STATS, new=AsyncMock(return_value=8.2)),
        patch(_CMD, new=AsyncMock(return_value=210.0)),
    ):
        await coord.async_capture_and_log()

    row = coord.training[sid][-1]
    assert "demand_kwh" not in row
    assert row["clean_cycle"] is False  # the refill day, judged by the old gate
    assert coord.models[sid].sample_count == 0


async def test_restored_tank_ignored_when_tracking_disabled(hass: HomeAssistant) -> None:
    """Clearing the heating detector turns the tank feedback off for good."""
    entry, coord = await _setup(hass)  # no heating_active_entity
    calls = async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))
    coord.tanks[sid] = _calibrated(6.0)  # a stale, restored calibrated state

    await coord.async_refresh()  # live (pre-push) result
    assert coord.data[sid].predicted_minutes == 150
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        await coord.async_predict_and_push()
    assert calls[-1].data["value"] == 150  # need only, no 120-min tank override
    assert coord.training[sid][-1]["deficit_source"] == "commanded"

    with (
        patch(_STATS, new=AsyncMock(return_value=7.0)),
        patch(_CMD, new=AsyncMock(return_value=150.0)),
    ):
        await coord.async_capture_and_log()
    assert "tank_deficit_end_kwh" not in coord.training[sid][-1]


# ── concurrency ────────────────────────────────────────────────────────────────


async def test_capture_during_predict_is_not_lost(hass: HomeAssistant) -> None:
    """A capture interleaving a slow predict must not have its gain step dropped."""
    import asyncio

    entry, coord = await _setup(hass)
    async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        await coord.async_predict_and_push()  # today's row exists
    _age_cycle(coord, sid)

    release = asyncio.Event()

    async def slow_commanded(*_args):
        await release.wait()  # predict is mid-flight, holding a copy of state
        return 150.0

    with (
        patch.object(coord, "_commanded_since", new=slow_commanded),
        patch(_STATS, new=AsyncMock(return_value=6.0)),
        patch(_CMD, new=AsyncMock(return_value=150.0)),
    ):
        predict = hass.async_create_task(coord.async_predict_and_push())
        await asyncio.sleep(0)
        capture = hass.async_create_task(coord.async_capture_and_log())
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(predict, capture)

    assert coord.models[sid].sample_count == 1
    assert coord.models[sid].gain < 1.0


# ── the published runtime is what was pushed ──────────────────────────────────


async def test_published_runtime_is_the_pushed_target(hass: HomeAssistant) -> None:
    entry, coord = await _setup(hass)
    calls = async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))

    real = coord.async_build_snapshot
    build = AsyncMock(side_effect=real)
    with (
        patch.object(coord, "async_build_snapshot", new=build),
        patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)),
    ):
        await coord.async_predict_and_push()
    assert build.await_count == 1  # predict + refresh share one snapshot
    pushed = calls[-1].data["value"]
    assert coord.data[sid].predicted_minutes == pushed
    assert coord.data[sid].rationale["target_minutes"] == pushed

    # A capture that moves the gain a lot must not change what the sensor says
    # the scheduler holds (it'd be a hypothetical target nobody pushed).
    with (
        patch(_STATS, new=AsyncMock(return_value=14.0)),
        patch(_CMD, new=AsyncMock(return_value=150.0)),
    ):
        await coord.async_capture_and_log()
    assert coord.models[sid].gain > 1.0
    assert coord.data[sid].predicted_minutes == pushed
    assert coord.data[sid].sample_count == 1  # metrics stay fresh
    assert coord.data[sid].last_delivered_kwh == 14.0


# ── post-capture re-predict, failed pushes, unload flush ───────────────────────


async def test_repredict_after_capture_keeps_the_completed_row(hass: HomeAssistant) -> None:
    """The observation was judged against this row's prediction — keep it."""
    entry, coord = await _setup(hass)
    calls = async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        await coord.async_predict_and_push()
    with (
        patch(_STATS, new=AsyncMock(return_value=6.9)),
        patch(_CMD, new=AsyncMock(return_value=150.0)),
    ):
        await coord.async_capture_and_log()
    completed = dict(coord.training[sid][-1])
    assert completed["gain"] == 1.0 and coord.models[sid].gain < 1.0

    hass.states.async_set("person.b", "not_home")  # occupancy changed too
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        await coord.async_predict_and_push()  # "Predict now" at 23:58

    assert coord.training[sid][-1] == completed  # gain/features/prediction intact
    assert calls[-1].data["value"] != 150  # …but the new plan was still pushed
    assert coord.data[sid].predicted_minutes == calls[-1].data["value"]


async def test_failed_push_keeps_previous_published_target(hass: HomeAssistant) -> None:
    entry, coord = await _setup(hass)
    calls = async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        await coord.async_predict_and_push()
        assert coord.data[sid].predicted_minutes == 150
        hass.states.async_remove("number.lvv_target")  # scheduler gone
        hass.states.async_set("person.b", "not_home")  # a different plan now
        assert await coord.async_predict_and_push() is False

    assert len(calls) == 1
    assert coord.data[sid].predicted_minutes == 150  # still what the scheduler holds
    assert coord.data[sid].last_push_ok is False


async def test_publish_only_load_publishes_its_plan(hass: HomeAssistant) -> None:
    data = {k: v for k, v in _LOAD_DATA.items() if k != "target_number_entity"}
    entry, coord = await _setup(hass, data)
    sid = next(iter(coord.load_configs()))
    hass.states.async_set("person.b", "not_home")
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        assert await coord.async_predict_and_push() is True  # nothing attempted
    assert coord.data[sid].predicted_minutes == 105  # 1 person, no scheduler


async def test_flush_drains_inflight_predict_then_blocks_writes(hass: HomeAssistant) -> None:
    import asyncio

    entry, coord = await _setup(hass)
    calls = async_mock_service(hass, "number", "set_value")
    sid = next(iter(coord.load_configs()))
    with patch.object(coord, "_commanded_since", new=AsyncMock(return_value=None)):
        await coord.async_predict_and_push()
    _age_cycle(coord, sid)

    release = asyncio.Event()

    async def slow_commanded(*_args):
        await release.wait()
        return 0.0

    saved: list[dict] = []

    async def save_now(data):
        saved.append(data)

    with (
        patch.object(coord, "_commanded_since", new=slow_commanded),
        patch.object(coord._store, "async_save_now", new=save_now),
        patch.object(coord._store, "async_schedule_save") as schedule,
    ):
        predict = hass.async_create_task(coord.async_predict_and_push())
        await asyncio.sleep(0)
        flush = hass.async_create_task(coord.async_flush())
        await asyncio.sleep(0)
        assert not flush.done()  # waits for the in-flight predict
        release.set()
        await asyncio.gather(predict, flush)
        # The drained predict's backlog made it into the final snapshot.
        assert saved[-1][sid]["model"]["deficit_minutes"] > 100
        schedule.reset_mock()

        # After the flush this coordinator is dead: no push, no row, no write.
        pushes = len(calls)
        assert await coord.async_predict_and_push() is False
        await coord.async_capture_and_log()
        coord.async_persist()
        assert len(calls) == pushes
        schedule.assert_not_called()

        coord.async_reopen()  # a failed unload resumes writes
        coord.async_persist()
        schedule.assert_called_once()
