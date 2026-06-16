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
    assert data["version"] == "0.1.0"  # HACS requires a version string
    assert data["config_flow"] is True


def test_manifest_dependencies():
    data = _manifest()
    # We read long-term statistics, so recorder is a hard dependency.
    assert "recorder" in data["dependencies"]
    # The scheduler is a soft dependency: the predictor must still load without it.
    assert "load_scheduler" in data["after_dependencies"]
    assert data["dependencies"].count("load_scheduler") == 0


def test_hacs_json():
    data = json.loads(_HACS.read_text())
    assert data["name"]
    assert data["render_readme"] is True
    assert "homeassistant" in data  # minimum HA version floor for ConfigSubentry
