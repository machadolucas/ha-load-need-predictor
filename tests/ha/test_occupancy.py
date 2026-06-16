"""Live occupancy sampling."""

from __future__ import annotations

from homeassistant.core import HomeAssistant

from custom_components.load_need_predictor.occupancy import count_people_home, guests_active


async def test_count_people_home_maps_states(hass: HomeAssistant) -> None:
    hass.states.async_set("person.a", "home")
    hass.states.async_set("person.b", "not_home")
    hass.states.async_set("person.c", "Tampere")  # a zone name = away
    hass.states.async_set("person.d", "unavailable")  # conservative → present
    # person.e is missing entirely → conservative → present
    count = count_people_home(hass, ["person.a", "person.b", "person.c", "person.d", "person.e"])
    assert count == 3  # a (home) + d (unavailable) + e (missing)


async def test_count_people_home_empty_list(hass: HomeAssistant) -> None:
    assert count_people_home(hass, []) == 0


async def test_guests_active(hass: HomeAssistant) -> None:
    assert guests_active(hass, None) is False
    hass.states.async_set("calendar.guests", "off")
    assert guests_active(hass, "calendar.guests") is False
    hass.states.async_set("calendar.guests", "on")
    assert guests_active(hass, "calendar.guests") is True
    assert guests_active(hass, "calendar.missing") is False
