"""Occupancy sampling — instantaneous fallbacks + duration-based measurement."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.core import HomeAssistant, State, SupportsResponse
from homeassistant.util import dt as dt_util

from custom_components.load_need_predictor import occupancy as occ

_GET_INSTANCE = "homeassistant.components.recorder.get_instance"


# ── instantaneous fallback helpers ───────────────────────────────────────────


async def test_count_people_home_maps_states(hass: HomeAssistant) -> None:
    hass.states.async_set("person.a", "home")
    hass.states.async_set("person.b", "not_home")
    hass.states.async_set("person.c", "Tampere")  # a zone name = away
    hass.states.async_set("person.d", "unavailable")  # conservative → present
    # person.e is missing entirely → conservative → present
    count = occ.count_people_home(
        hass, ["person.a", "person.b", "person.c", "person.d", "person.e"]
    )
    assert count == 3  # a (home) + d (unavailable) + e (missing)


async def test_guests_active(hass: HomeAssistant) -> None:
    assert occ.guests_active(hass, None) is False
    hass.states.async_set("calendar.guests", "off")
    assert occ.guests_active(hass, "calendar.guests") is False
    hass.states.async_set("calendar.guests", "on")
    assert occ.guests_active(hass, "calendar.guests") is True


# ── pure duration math ───────────────────────────────────────────────────────


def test_home_seconds():
    start = dt_util.utcnow() - timedelta(hours=24)
    end = start + timedelta(hours=24)
    states = [
        # state as of window start (last_changed before start → clamped)
        State("person.a", "home", last_changed=start - timedelta(hours=2)),
        State("person.a", "not_home", last_changed=start + timedelta(hours=10)),
    ]
    assert occ.home_seconds(states, start, end) == 10 * 3600  # home for the first 10 h


def test_home_seconds_empty():
    now = dt_util.utcnow()
    assert occ.home_seconds([], now - timedelta(hours=24), now) == 0.0


def test_event_hours():
    base = dt_util.now()
    ev = {"start": base.isoformat(), "end": (base + timedelta(hours=4)).isoformat()}
    assert occ.event_hours(ev) == 4.0
    # all-day (date-only) events count as a full day
    assert occ.event_hours({"start": "2026-06-18", "end": "2026-06-19"}) == 24.0


def test_guest_factor():
    assert occ.guest_factor(0) == 0.0
    assert occ.guest_factor(3) == occ.GUEST_SHORT_FACTOR  # < 6 h → short
    assert occ.guest_factor(6) == occ.GUEST_LONG_FACTOR  # ≥ 6 h → long
    assert occ.guest_factor(10) == occ.GUEST_LONG_FACTOR


# ── duration-based residents (history) ───────────────────────────────────────


async def test_async_count_residents_home_uses_duration(hass: HomeAssistant) -> None:
    now = dt_util.utcnow()
    wstart = now - timedelta(hours=24)
    history = {
        # home across the whole window → ≥ 12 h → counts
        "person.a": [State("person.a", "home", last_changed=wstart - timedelta(hours=1))],
        # home only the first 6 h → < 12 h → does NOT count, even if home now
        "person.b": [
            State("person.b", "home", last_changed=wstart - timedelta(hours=1)),
            State("person.b", "not_home", last_changed=wstart + timedelta(hours=6)),
        ],
    }
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock(return_value=history)
    with patch(_GET_INSTANCE, return_value=instance):
        count = await occ.async_count_residents_home(hass, ["person.a", "person.b"])
    assert count == 1


async def test_async_count_residents_home_fallback_without_recorder(hass: HomeAssistant) -> None:
    # No recorder in this test → falls back to the instantaneous count.
    hass.states.async_set("person.a", "home")
    hass.states.async_set("person.b", "not_home")
    count = await occ.async_count_residents_home(hass, ["person.a", "person.b"])
    assert count == 1


# ── duration-based guests (calendar events) ──────────────────────────────────


def _register_events(hass: HomeAssistant, events: list[dict]) -> None:
    async def _get_events(call):
        return {"calendar.guests": {"events": events}}

    hass.services.async_register(
        "calendar", "get_events", _get_events, supports_response=SupportsResponse.ONLY
    )


async def test_async_guest_equivalents_long_visit(hass: HomeAssistant) -> None:
    base = dt_util.now()
    _register_events(
        hass, [{"start": base.isoformat(), "end": (base + timedelta(hours=7)).isoformat()}]
    )
    assert await occ.async_guest_equivalents(hass, "calendar.guests") == occ.GUEST_LONG_FACTOR


async def test_async_guest_equivalents_short_visit(hass: HomeAssistant) -> None:
    base = dt_util.now()
    _register_events(
        hass, [{"start": base.isoformat(), "end": (base + timedelta(hours=4)).isoformat()}]
    )
    assert await occ.async_guest_equivalents(hass, "calendar.guests") == occ.GUEST_SHORT_FACTOR


async def test_async_guest_equivalents_no_events(hass: HomeAssistant) -> None:
    _register_events(hass, [])
    assert await occ.async_guest_equivalents(hass, "calendar.guests") == 0.0
    assert await occ.async_guest_equivalents(hass, None) == 0.0


async def test_async_guest_equivalents_fallback_without_service(hass: HomeAssistant) -> None:
    # calendar.get_events not registered → falls back to the on/off state.
    hass.states.async_set("calendar.guests", "on")
    assert await occ.async_guest_equivalents(hass, "calendar.guests") == occ.GUEST_FALLBACK_FACTOR
    hass.states.async_set("calendar.guests", "off")
    assert await occ.async_guest_equivalents(hass, "calendar.guests") == 0.0
