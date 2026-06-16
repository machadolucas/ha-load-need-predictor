"""Pushing the target into the scheduler — including resilience when it's absent."""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from pytest_homeassistant_custom_component.common import async_mock_service

from custom_components.load_need_predictor.actuation import async_push_target


async def test_push_sets_value_when_target_present(hass: HomeAssistant) -> None:
    hass.states.async_set("number.lvv_target", "0")
    calls = async_mock_service(hass, "number", "set_value")

    assert await async_push_target(hass, "number.lvv_target", 150) is True
    assert len(calls) == 1
    assert calls[0].data["entity_id"] == "number.lvv_target"
    assert calls[0].data["value"] == 150


async def test_push_skipped_when_target_missing(hass: HomeAssistant) -> None:
    # Scheduler not installed → entity absent → no raise, returns False.
    calls = async_mock_service(hass, "number", "set_value")
    assert await async_push_target(hass, "number.lvv_target", 150) is False
    assert len(calls) == 0


async def test_push_noop_without_entity_or_value(hass: HomeAssistant) -> None:
    assert await async_push_target(hass, None, 150) is False
    hass.states.async_set("number.lvv_target", "0")
    assert await async_push_target(hass, "number.lvv_target", None) is False


async def test_push_handles_service_error(hass: HomeAssistant) -> None:
    hass.states.async_set("number.lvv_target", "0")

    async def _boom(call):
        raise HomeAssistantError("boom")

    hass.services.async_register("number", "set_value", _boom)
    # The service raises, but the push swallows it and reports failure.
    assert await async_push_target(hass, "number.lvv_target", 150) is False
