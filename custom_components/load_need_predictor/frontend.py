"""Serve and register the dashboard card as a frontend module.

HA (and HACS) only load a Lovelace card when its JS is registered as a frontend
resource. Rather than make every user add a resource by hand, the integration
serves the bundled ``www/<card>.js`` and registers it as an *extra module URL*
on first setup, so ``type: custom:load-need-predictor-card`` is available on all
dashboards after a restart. Registration runs once per Home Assistant process —
config-entry reloads must not re-add the static path or the module URL.

**Cache-busting.** The file is served with a long browser cache, and the module
URL carries a ``?hash=<content-hash>`` suffix — the same pattern HACS uses with
``?hacs=<version>``. A *content* hash (not the release version) means the URL
changes whenever the JS actually changes, so every cache layer (browser, proxy,
service worker) refetches a stale copy while unchanged files stay cached. The
hash is recomputed on each startup, and editing a custom component requires a
restart, so the URL is always in step with the served bytes.
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


def _content_hash(path: Path) -> str | None:
    """Short hash of the card file's bytes, the cache-bust key. Blocking I/O."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    except OSError:
        return None


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
        # cache_headers=True → long immutable cache; the hashed URL below is what
        # forces a refetch when the file changes.
        await hass.http.async_register_static_paths(
            [StaticPathConfig(CARD_URL, str(card_path), True)]
        )
        cache_key = await hass.async_add_executor_job(_content_hash, card_path)
        url = f"{CARD_URL}?hash={cache_key}" if cache_key else CARD_URL
        add_extra_js_url(hass, url)
    except Exception as err:  # never let a UI nicety break setup
        _LOGGER.warning("Could not register the dashboard card: %s", err)
        return

    hass.data[_REGISTERED_KEY] = True
    _LOGGER.debug("Registered Load Need Predictor dashboard card at %s", url)
