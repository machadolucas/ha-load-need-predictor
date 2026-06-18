"""Card registration: content-hash cache-busting + Lovelace resource registry."""

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


# ── test doubles ─────────────────────────────────────────────────────────────


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


class _FakeResources:
    """Stand-in for Lovelace's ResourceStorageCollection (storage mode)."""

    def __init__(self, items, *, loaded=True) -> None:
        self._items = list(items)
        self.loaded = loaded
        self.created: list = []
        self.updated: list = []
        self.deleted: list = []

    async def async_load(self) -> None:
        self.loaded = True

    def async_items(self):
        return list(self._items)

    async def async_create_item(self, item):
        self.created.append(item)
        self._items.append({**item, "id": f"id{len(self._items)}"})

    async def async_update_item(self, item_id, changes):
        self.updated.append((item_id, changes))

    async def async_delete_item(self, item_id):
        self.deleted.append(item_id)


class _FakeLovelaceData:
    def __init__(self, resources) -> None:
        self.resources = resources


def _use_fake_lovelace(monkeypatch, hass, items, *, loaded=True):
    """Wire a fake Lovelace storage collection into hass + make isinstance pass."""
    from homeassistant.components.lovelace import resources as lr
    from homeassistant.components.lovelace.const import LOVELACE_DATA

    res = _FakeResources(items, loaded=loaded)
    monkeypatch.setattr(lr, "ResourceStorageCollection", _FakeResources)
    hass.data[LOVELACE_DATA] = _FakeLovelaceData(res)
    return res


# ── _async_register_resource ─────────────────────────────────────────────────


async def test_resource_registry_creates_new(monkeypatch):
    hass = _FakeHass(_FakeHttp())
    res = _use_fake_lovelace(monkeypatch, hass, [])

    handled = await frontend._async_register_resource(hass, f"{CARD_URL}?v=aaa")

    assert handled is True
    assert res.created == [{"res_type": "module", "url": f"{CARD_URL}?v=aaa"}]
    assert res.updated == []
    assert res.deleted == []


async def test_resource_registry_updates_version_in_place(monkeypatch):
    hass = _FakeHass(_FakeHttp())
    res = _use_fake_lovelace(monkeypatch, hass, [{"id": "1", "url": f"{CARD_URL}?v=old"}])

    handled = await frontend._async_register_resource(hass, f"{CARD_URL}?v=new")

    assert handled is True
    assert res.updated == [("1", {"url": f"{CARD_URL}?v=new"})]
    assert res.created == []
    assert res.deleted == []


async def test_resource_registry_dedupes(monkeypatch):
    hass = _FakeHass(_FakeHttp())
    res = _use_fake_lovelace(
        monkeypatch,
        hass,
        [
            {"id": "1", "url": f"{CARD_URL}?v=new"},  # already current → kept, no update
            {"id": "2", "url": f"{CARD_URL}?v=stale"},
            {"id": "3", "url": CARD_URL},
        ],
    )

    handled = await frontend._async_register_resource(hass, f"{CARD_URL}?v=new")

    assert handled is True
    assert res.deleted == ["2", "3"]
    assert res.updated == []
    assert res.created == []


async def test_resource_registry_loads_when_unloaded(monkeypatch):
    hass = _FakeHass(_FakeHttp())
    res = _use_fake_lovelace(monkeypatch, hass, [], loaded=False)

    await frontend._async_register_resource(hass, f"{CARD_URL}?v=aaa")

    assert res.loaded is True


async def test_resource_registry_falls_back_without_lovelace_data():
    hass = _FakeHass(_FakeHttp())  # no LOVELACE_DATA in hass.data

    assert await frontend._async_register_resource(hass, f"{CARD_URL}?v=x") is False


# ── async_register_card ──────────────────────────────────────────────────────


async def test_register_uses_resource_registry(monkeypatch):
    urls: list[str] = []
    monkeypatch.setattr(frontend, "add_extra_js_url", lambda hass, url: urls.append(url))
    hass = _FakeHass(_FakeHttp())
    res = _use_fake_lovelace(monkeypatch, hass, [])

    await frontend.async_register_card(hass)

    expected = frontend._card_version(frontend._card_path())
    assert res.created == [{"res_type": "module", "url": f"{CARD_URL}?v={expected}"}]
    assert urls == []  # registry handled it → no HTML-shell injection
    assert hass.http.registered  # static path still served
    assert CARD_FILENAME in str(hass.http.registered[0].path)
    assert hass.data[frontend._REGISTERED_KEY] is True

    # Once per process: a reload must not re-register.
    await frontend.async_register_card(hass)
    assert res.created == [{"res_type": "module", "url": f"{CARD_URL}?v={expected}"}]


async def test_register_falls_back_to_extra_js_without_lovelace(monkeypatch):
    # No Lovelace storage collection → fall back to add_extra_js_url (YAML mode).
    urls: list[str] = []
    monkeypatch.setattr(frontend, "add_extra_js_url", lambda hass, url: urls.append(url))
    hass = _FakeHass(_FakeHttp())

    await frontend.async_register_card(hass)

    expected = frontend._card_version(frontend._card_path())
    assert urls == [f"{CARD_URL}?v={expected}"]
    assert hass.http.registered
    assert hass.data[frontend._REGISTERED_KEY] is True


async def test_register_falls_back_when_registry_raises(monkeypatch):
    # A registry write blowing up must still load the card (via add_extra_js_url).
    urls: list[str] = []
    monkeypatch.setattr(frontend, "add_extra_js_url", lambda hass, url: urls.append(url))

    async def _boom(hass, url):
        raise RuntimeError("registry down")

    monkeypatch.setattr(frontend, "_async_register_resource", _boom)
    hass = _FakeHass(_FakeHttp())

    await frontend.async_register_card(hass)

    expected = frontend._card_version(frontend._card_path())
    assert urls == [f"{CARD_URL}?v={expected}"]
    assert hass.data[frontend._REGISTERED_KEY] is True


async def test_register_falls_back_to_bare_url_when_hash_fails(monkeypatch):
    # If the cache-buster throws, the card is still registered with the bare URL.
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
