"""Serve and register the dashboard card as a frontend module.

HA (and HACS) only load a Lovelace card when its JS is registered as a frontend
resource. Rather than make every user add a resource by hand, the integration
serves the bundled ``www/<card>.js`` and registers it on first setup, so
``type: custom:load-need-predictor-card`` is available on all dashboards after a
restart. Registration runs once per Home Assistant process — config-entry
reloads must not re-add it.

**Why the resource registry, not ``add_extra_js_url``.** ``add_extra_js_url``
injects a ``<script>`` into the *server-rendered index HTML*. But that HTML / app
shell is cached upstream (a Cloudflare edge, the frontend service worker, the
Companion-app WebView), and a cached shell does **not** contain the injected tag —
so the browser never requests the card module and the dashboard breaks with
"Custom element doesn't exist". A URL cache-buster can't help because the stale
artifact is the *HTML*, not the JS. So we register the card in the **Lovelace
resource registry** (exactly what HACS does): the frontend fetches that list at
runtime over WebSocket, independent of the cached HTML. ``add_extra_js_url``
remains only as a fallback for YAML resource mode / when the registry isn't ready.

**Cache-busting.** The file is still served with a long immutable cache and the
registered URL carries a ``?v=<content-hash>`` suffix, so the JS itself refetches
exactly when it changes. The registry write dedupes by URL path and updates the
version in place, so the resource list never accumulates duplicates across
updates.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from homeassistant.components.frontend import add_extra_js_url
from homeassistant.components.http import StaticPathConfig
from homeassistant.core import HomeAssistant

from .const import CARD_FILENAME, CARD_URL, DOMAIN

_LOGGER = logging.getLogger(__name__)

_REGISTERED_KEY = f"{DOMAIN}_card_registered"


def _card_path() -> Path:
    return Path(__file__).parent / "www" / CARD_FILENAME


def _card_version(path: Path) -> str:
    """Short content hash of the card file — the cache-bust key. Blocking I/O."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


async def _async_register_resource(hass: HomeAssistant, url: str) -> bool:
    """Add/refresh the card in the Lovelace resource registry (storage mode).

    Returns ``True`` if handled, ``False`` to fall back to ``add_extra_js_url``
    (YAML resource mode, or Lovelace not ready). Idempotent across restarts: it
    dedupes by URL *path* and updates the ``?v=`` version in place, so the
    resource list never grows duplicates.
    """
    try:
        from homeassistant.components.lovelace.const import LOVELACE_DATA
        from homeassistant.components.lovelace.resources import (
            ResourceStorageCollection,
        )
    except ImportError:
        return False

    data = hass.data.get(LOVELACE_DATA)
    if data is None:
        return False
    resources = data.resources
    if not isinstance(resources, ResourceStorageCollection):
        return False  # YAML resource mode — can't edit programmatically

    if not resources.loaded:
        await resources.async_load()
        resources.loaded = True

    base = url.split("?", 1)[0]
    existing = [
        r for r in resources.async_items() if str(r.get("url", "")).split("?", 1)[0] == base
    ]
    if existing:
        keep, *dupes = existing
        if keep.get("url") != url:
            await resources.async_update_item(keep["id"], {"url": url})
        for dupe in dupes:  # collapse any duplicates
            await resources.async_delete_item(dupe["id"])
    else:
        await resources.async_create_item({"res_type": "module", "url": url})
    return True


async def async_register_card(hass: HomeAssistant) -> None:
    """Serve the card JS and register it for the frontend (once per process).

    Best-effort: the card is a UI nicety, so a missing ``http`` component (some
    minimal test setups) or any registration error must never block the
    integration from loading. The once-flag is only set on success so a setup
    retry can register the card later.
    """
    if hass.data.get(_REGISTERED_KEY):
        return
    if hass.http is None:  # e.g. a stripped-down test environment
        return

    try:
        card_path = _card_path()
        # Register the route FIRST so serving the card never depends on the
        # cache-buster. cache_headers=True → long immutable cache; the hashed URL
        # below is what forces clients to refetch when the file changes.
        await hass.http.async_register_static_paths(
            [StaticPathConfig(CARD_URL, str(card_path), True)]
        )
        # Cache-bust on a content hash, best-effort: if hashing fails we still
        # register the bare URL so the card loads (just without the buster).
        url = CARD_URL
        try:
            version = await hass.async_add_executor_job(_card_version, card_path)
            url = f"{CARD_URL}?v={version}"
        except Exception as err:  # noqa: BLE001 - cache-buster is best-effort
            _LOGGER.debug("Card cache-buster skipped: %s", err)

        # Prefer the resource registry (survives a cached HTML shell); fall back
        # to add_extra_js_url for YAML mode or if the registry write fails.
        registered = False
        try:
            registered = await _async_register_resource(hass, url)
        except Exception as err:  # noqa: BLE001 - fall back, never break setup
            _LOGGER.debug("Lovelace resource registration failed, falling back: %s", err)
        if not registered:
            add_extra_js_url(hass, url)

        hass.data[_REGISTERED_KEY] = True
        _LOGGER.debug(
            "Registered Load Need Predictor dashboard card at %s (%s)",
            url,
            "resource registry" if registered else "extra_js_url",
        )
    except Exception as err:  # noqa: BLE001 - a UI nicety must never break setup
        _LOGGER.debug("Dashboard card not registered: %s", err)
