"""Manifest / HACS metadata sanity checks (pure — no Home Assistant import).

These guard the packaging contract HACS and hassfest rely on, and give the M0
scaffold a green test run before any model code exists.
"""

from __future__ import annotations

import json
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_MANIFEST = _ROOT / "custom_components" / "load_need_predictor" / "manifest.json"
_HACS = _ROOT / "hacs.json"


def _manifest() -> dict:
    return json.loads(_MANIFEST.read_text())


def test_manifest_domain_and_version():
    data = _manifest()
    assert data["domain"] == "load_need_predictor"
    assert data["version"] == "0.3.0"  # HACS requires a version string
    assert data["config_flow"] is True


def test_manifest_dependencies():
    data = _manifest()
    # Both recorder and the scheduler are SOFT deps: the predictor must still load
    # without them. It degrades to publish-only / no-learning rather than failing,
    # and the statistics read guards for a missing recorder at job time.
    assert "recorder" in data["after_dependencies"]
    assert "load_scheduler" in data["after_dependencies"]
    assert data["dependencies"] == []


def test_hacs_json():
    data = json.loads(_HACS.read_text())
    assert data["name"]
    assert data["render_readme"] is True
    assert "homeassistant" in data  # minimum HA version floor for ConfigSubentry
