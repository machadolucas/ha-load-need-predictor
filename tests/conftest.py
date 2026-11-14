"""Shared pytest configuration.

Puts the repo root on ``sys.path`` so ``custom_components.load_need_predictor``
is importable, and conditionally loads the Home Assistant test plugin after its compatibility
aliases. The HA-only autouse fixture remains in ``tests/ha/conftest.py``; pure
tests do not instantiate HA.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Keep pure tests runnable with pytest alone. Only import HA/plugin when both
# are installed; import HA first so its aliases are present before respx loads.
if (
    importlib.util.find_spec("homeassistant") is not None
    and importlib.util.find_spec("pytest_homeassistant_custom_component") is not None
):
    importlib.import_module("homeassistant")
    logger = importlib.import_module("homeassistant.components.logger")

    # Plugin 0.13.375 expects this HA 2026.10 helper. It is unused by this
    # suite, so fail loudly if a future fixture starts depending on it.
    def _unused_old_logger_helper(*args: object) -> None:
        raise AssertionError(
            "pytest-homeassistant-custom-component 0.13.375 legacy logger helper "
            "is unsupported on HA 2026.11.1"
        )

    if not hasattr(logger, "_clear_logger_overwrites"):
        logger._clear_logger_overwrites = _unused_old_logger_helper

    pytest_plugins = ["pytest_homeassistant_custom_component.plugins"]
else:
    pytest_plugins = []
