"""Fixtures for the Home-Assistant-backed tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from custom_components.load_need_predictor.forecast_source import WattcastFetch


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Load the custom integration in every HA test."""
    yield


@pytest.fixture(autouse=True)
def no_wattcast_network():
    """Safety net: no test ever reaches wattcast.eu.

    Any price-forecast subentry set up by a test (including the reload after a
    config flow) would otherwise try a real HTTP fetch. The default outcome is a
    plain failure, so the coordinator falls back to the local model; tests that
    exercise the Wattcast path patch the same name with their own mock (the
    inner patch wins). ``forecast_source.async_fetch_wattcast`` itself is left
    alone — its own tests drive it through ``aioclient_mock``.
    """
    with patch(
        "custom_components.load_need_predictor.forecast_coordinator.async_fetch_wattcast",
        new=AsyncMock(return_value=WattcastFetch(None, "network disabled in tests")),
    ) as mock:
        yield mock
