"""Shared pytest configuration.

Puts the repo root on ``sys.path`` so ``custom_components.load_need_predictor``
is importable, and registers the Home Assistant test plugin. The HA-only autouse
fixture lives in ``tests/ha/conftest.py`` so the pure tests stay HA-free.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

pytest_plugins = ["pytest_homeassistant_custom_component"]
