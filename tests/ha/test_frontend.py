"""Card registration + content-hash cache-busting."""

from __future__ import annotations

from custom_components.load_need_predictor import frontend
from custom_components.load_need_predictor.const import CARD_FILENAME, CARD_URL

# ── content hash ─────────────────────────────────────────────────────────────


def test_card_version_is_stable_and_sensitive(tmp_path):
    f = tmp_path / "card.js"
    f.write_text("console.log('alpha');")
    first = frontend._card_version(f)
    assert len(first) == 8
    assert all(c in "0123456789abcdef" for c in first)
    assert frontend._card_version(f) == first  # same bytes → same hash

    f.write_text("console.log('beta');")
    assert frontend._card_version(f) != first  # changed bytes → changed hash


def test_shipped_card_hashes():
    # The bundled card file hashes to an 8-char key (the real cache-bust source).
    assert len(frontend._card_version(frontend._card_path())) == 8


# ── async_register_card ──────────────────────────────────────────────────────


class _FakeHttp:
    def __init__(self) -> None:
        self.registered: list = []

    async def async_register_static_paths(self, configs) -> None:
        self.registered.extend(configs)


class _FakeHass:
    """Just the surface async_register_card touches."""

    def __init__(self, http) -> None:
        self.http = http
        self.data: dict = {}

    async def async_add_executor_job(self, func, *args):
        return func(*args)


async def test_register_appends_content_version(monkeypatch):
    urls: list[str] = []
    monkeypatch.setattr(frontend, "add_extra_js_url", lambda hass, url: urls.append(url))
    hass = _FakeHass(_FakeHttp())

    await frontend.async_register_card(hass)

    expected = frontend._card_version(frontend._card_path())
    assert urls == [f"{CARD_URL}?v={expected}"]
    assert hass.http.registered  # the static path was registered
    assert CARD_FILENAME in str(hass.http.registered[0].path)
    assert hass.data[frontend._REGISTERED_KEY] is True

    # Once per process: a second call (e.g. a reload) must not re-add the URL.
    await frontend.async_register_card(hass)
    assert len(urls) == 1


async def test_register_falls_back_to_bare_url_when_hash_fails(monkeypatch):
    # The CRITICAL guarantee: if the cache-buster throws, the card is still
    # served and injected (route registered first, bare URL added).
    urls: list[str] = []
    monkeypatch.setattr(frontend, "add_extra_js_url", lambda hass, url: urls.append(url))

    def _boom(_path):
        raise RuntimeError("hash exploded")

    monkeypatch.setattr(frontend, "_card_version", _boom)
    hass = _FakeHass(_FakeHttp())

    await frontend.async_register_card(hass)

    assert urls == [CARD_URL]  # bare URL, no ?v= suffix
    assert hass.http.registered  # route still registered → card serves
    assert hass.data[frontend._REGISTERED_KEY] is True


async def test_register_skips_without_http(monkeypatch):
    urls: list[str] = []
    monkeypatch.setattr(frontend, "add_extra_js_url", lambda hass, url: urls.append(url))
    hass = _FakeHass(None)  # minimal environment without the http component

    await frontend.async_register_card(hass)

    assert urls == []
    assert frontend._REGISTERED_KEY not in hass.data
