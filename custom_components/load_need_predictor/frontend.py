"""Serve and register the dashboard card as a frontend module.

HA (and HACS) only load a Lovelace card when its JS is registered as a frontend
resource. Rather than make every user add a resource by hand, the integration
serves the bundled ``www/<card>.js`` and registers it as an *extra module URL*
on first setup, so ``type: custom:load-need-predictor-card`` is available on all
dashboards after a restart. Registration runs once per Home Assistant process —
config-entry reloads must not re-add the static path or the module URL.

**Cache-busting.** The file is served with a long (~31-day) immutable cache, and
the injected URL carries a ``?v=<content-hash>`` suffix — the same pattern HACS
uses with ``?hacs=<version>``. A *content* hash (not the release version) means
the URL changes exactly when the JS changes, so the browser HTTP cache, the PWA
service worker and the Companion-app WebView all refetch a stale copy while
unchanged files stay cached. HA restarts alone don't fix this — the stale copy
is client-side — which is why the URL itself must change.

The order matters: the static route is registered **first**, then the hash is
computed in its own guarded step that falls back to the bare URL. Serving the
card must never depend on the cache-buster — if hashing threw before the route
existed, the card would 404 and every dashboard using it would break with
"Custom element doesn't exist".
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


async def async_register_card(hass: HomeAssistant) -> None:
    """Serve the card JS and add it as a frontend module (once per process).

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
        # inject the bare URL so the card loads (just without the buster).
        url = CARD_URL
        try:
            version = await hass.async_add_executor_job(_card_version, card_path)
            url = f"{CARD_URL}?v={version}"
        except Exception as err:  # noqa: BLE001 - cache-buster is best-effort
            _LOGGER.debug("Card cache-buster skipped: %s", err)
        add_extra_js_url(hass, url)
        hass.data[_REGISTERED_KEY] = True
        _LOGGER.debug("Registered Load Need Predictor dashboard card at %s", url)
    except Exception as err:  # noqa: BLE001 - a UI nicety must never break setup
        _LOGGER.debug("Dashboard card not registered: %s", err)
