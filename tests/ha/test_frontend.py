"""Card registration + content-hash cache-busting."""

from __future__ import annotations

from custom_components.load_need_predictor import frontend
from custom_components.load_need_predictor.const import CARD_FILENAME, CARD_URL

# ── content hash ─────────────────────────────────────────────────────────────


def test_content_hash_is_stable_and_sensitive(tmp_path):
    f = tmp_path / "card.js"
    f.write_text("console.log('alpha');")
    first = frontend._content_hash(f)
    assert first is not None
    assert len(first) == 12
    assert all(c in "0123456789abcdef" for c in first)
    assert frontend._content_hash(f) == first  # same bytes → same hash

    f.write_text("console.log('beta');")
    assert frontend._content_hash(f) != first  # changed bytes → changed hash


def test_content_hash_missing_file_is_none(tmp_path):
    assert frontend._content_hash(tmp_path / "absent.js") is None


def test_shipped_card_hashes(tmp_path):
    # The bundled card file hashes to a 12-char key (the real cache-bust source).
    assert frontend._content_hash(frontend._card_path()) is not None


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


async def test_register_appends_content_hash(monkeypatch):
    urls: list[str] = []
    monkeypatch.setattr(frontend, "add_extra_js_url", lambda hass, url: urls.append(url))
    hass = _FakeHass(_FakeHttp())

    await frontend.async_register_card(hass)

    assert len(urls) == 1
    assert urls[0].startswith(f"{CARD_URL}?hash=")
    assert hass.http.registered  # the static path was registered
    assert hass.data[frontend._REGISTERED_KEY] is True

    # Once per process: a second call (e.g. a reload) must not re-add the URL.
    await frontend.async_register_card(hass)
    assert len(urls) == 1


async def test_register_url_matches_file_hash(monkeypatch):
    urls: list[str] = []
    monkeypatch.setattr(frontend, "add_extra_js_url", lambda hass, url: urls.append(url))
    hass = _FakeHass(_FakeHttp())

    await frontend.async_register_card(hass)

    expected = frontend._content_hash(frontend._card_path())
    assert urls[0] == f"{CARD_URL}?hash={expected}"
    assert CARD_FILENAME in str(hass.http.registered[0].path)


async def test_register_skips_without_http(monkeypatch):
    urls: list[str] = []
    monkeypatch.setattr(frontend, "add_extra_js_url", lambda hass, url: urls.append(url))
    hass = _FakeHass(None)  # minimal environment without the http component

    await frontend.async_register_card(hass)

    assert urls == []
    assert frontend._REGISTERED_KEY not in hass.data
