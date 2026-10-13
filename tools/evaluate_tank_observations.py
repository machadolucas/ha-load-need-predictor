#!/usr/bin/env python3
"""Evaluate exported shadow observations without Home Assistant or network IO.

Usage: python tools/evaluate_tank_observations.py diagnostics.json

Accepts a standalone shadow state, a per-load persistence export, or an entire
diagnostic export. It recursively locates separate observation stores and emits
one chronological holdout report per store. No data or parameters are changed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "load_need_predictor"
    / "tank_observation.py"
)


def load_model() -> Any:
    spec = importlib.util.spec_from_file_location("lnp_observation_evaluation", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def find_observations(value: Any, path: str = "root") -> dict[str, dict[str, Any]]:
    """Find stores only, not event rows or the old control tank's persistence."""
    if isinstance(value, dict):
        if value.get("mode") == "shadow" and isinstance(value.get("cycles"), list):
            return {path: value}
        found = {}
        for key, child in value.items():
            found.update(find_observations(child, f"{path}.{key}"))
        return found
    if isinstance(value, list):
        found = {}
        for index, child in enumerate(value):
            found.update(find_observations(child, f"{path}[{index}]"))
        return found
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON persistence or diagnostics export")
    args = parser.parse_args()
    try:
        payload = json.loads(args.input.read_text())
        stores = find_observations(payload)
        if not stores:
            parser.error("No shadow observation stores found in the JSON export")
        model = load_model()
        reports = {name: model.evaluate_observations(state) for name, state in stores.items()}
        print(json.dumps(reports, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError) as err:
        parser.error(str(err))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
