#!/usr/bin/env python3
"""Replay :mod:`tank_model` over a recorded week of real telemetry.

**Home-Assistant-free, stdlib only.** ``tank_model`` is loaded by path via
``importlib`` (exactly like ``tests/test_predictor.py`` does for ``predictor``),
so this tool — and the regression test that imports it — runs without the
integration, the recorder, or a live HA.

Why a replay tool at all: the tank state-of-charge estimator is a *continuous
integrator with self-correcting anchors*. Unit tests can pin single ticks, but
only a real week catches the failure modes that matter — the clamp discarding
delivered energy and turning it into a 13-point jump, a saturation curve that
pins the display at one value for hours, a learner that chases cycle noise. So
the week is committed as a fixture (``tests/fixtures/tank_week_2026-08-28.csv``)
and replayed tick-by-tick.

Two commands:

``export``
    Rebuild the fixture from the raw Home Assistant history JSON files (the
    tool-result dumps of ``/api/history``). Consecutive rows carrying an
    unchanged state string are dropped — those are attribute-only updates and
    HA's own ``last_changed`` already ignores them — so every retained row is a
    genuine state change and the fixture needs no ``last_changed`` column.

``replay``
    Step the fixture through :func:`tank_model.apply_tick` at 60 s and print the
    acceptance report. :func:`replay` returns the same numbers as a dict so
    ``tests/test_tank_replay.py`` can assert on them.

Usage::

    python tools/replay_tank.py export --window 2026-08-28T11:00+03:00 \\
        2026-09-04T11:30+03:00 -o tests/fixtures/tank_week_2026-08-28.csv a.json b.json
    python tools/replay_tank.py replay tests/fixtures/tank_week_2026-08-28.csv
    python tools/replay_tank.py replay <csv> --hot-fraction 0.248 --standby-w 114 --no-learn
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "custom_components" / "load_need_predictor" / "tank_model.py"

# ── Fixture schema ────────────────────────────────────────────────────────────
# The five series the model needs, and the entity ids they came from on the
# author's install. Exported under generic names so the fixture is readable and
# an entity rename can't invalidate it.
CONTACTOR = "contactor"
HEATING = "heating"
ENERGY = "energy_kwh"
WATER = "water_m3"
LIVE_SOC = "live_soc_pct"
SERIES = (CONTACTOR, HEATING, ENERGY, WATER, LIVE_SOC)
# The four the tick actually consumes; ``live_soc_pct`` only seeds the starting
# deficit and provides the informational replay-vs-live comparison.
REQUIRED_SERIES = (CONTACTOR, HEATING, ENERGY, WATER)

ENTITY_TO_SERIES = {
    "switch.shellypro1_30c6f78b0f24_switch_0": CONTACTOR,
    "binary_sensor.leddetector_water_heater": HEATING,
    "sensor.leddetector_water_heater_energy": ENERGY,
    "sensor.water_meter_mac_ocr_water_total": WATER,
    "sensor.lvv_water_heater_tank_charge": LIVE_SOC,
}

# ── Replay setup (the author's LVV; see CLAUDE.md "The tank model") ───────────
TANK_VOLUME_L = 300.0
TANK_SETPOINT_C = 75.0
TANK_COLD_IN_C = 12.0
RATED_POWER_KW = 3.0
TICK_S = 60
# Occupancy params only feed the meter-dropout fallback draw; these are the
# load model's fitted values at the time the week was recorded.
PEOPLE_HOME = 1
E_BASE = 3.96
E_DRAW_PER_PERSON = 2.12
EMPTY_HOUSE_FACTOR = 0.4
# Params the live v0.8 sensor was running with — the replay's starting point.
LIVE_HOT_FRACTION = 0.248
LIVE_STANDBY_W = 114.0

JUMP_PTS = 3.0  # a non-anchor SoC step this large is a discontinuity, not physics
LONG_CYCLE_H = 4.0  # below this a cycle is a post-trip re-heat, not a real cycle
PLATEAU_MIN_SOC = 95.0  # only a *high* frozen reading is the pathology we hunt


def load_tank_model(path: Path = MODEL_PATH) -> Any:
    """Import ``tank_model`` standalone, by path — no package, no HA."""
    spec = importlib.util.spec_from_file_location("lnp_tank_model_replay", path)
    if spec is None or spec.loader is None:  # pragma: no cover - unreachable in repo
        raise RuntimeError(f"cannot load tank_model from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ── Fixture I/O ───────────────────────────────────────────────────────────────


def _dedupe(rows: list[tuple[datetime, str]]) -> list[tuple[datetime, str]]:
    """Keep only rows whose state string differs from the previous row's.

    Home Assistant records a new row for attribute-only updates too (and on a
    restart, when the restored state re-publishes unchanged). Those carry no
    information for a state machine, so they go — which is also what makes the
    retained rows' timestamps double as ``last_changed``.
    """
    out: list[tuple[datetime, str]] = []
    for ts, state in rows:
        if out and out[-1][1] == state:
            continue
        out.append((ts, state))
    return out


def export_fixture(
    json_paths: list[Path], out_path: Path, start: datetime, end: datetime
) -> dict[str, int]:
    """Build the CSV fixture from raw HA history JSON dumps.

    Rows are windowed to ``[start, end]`` *plus*, per series, the last row at or
    before ``start`` — the state in force when the window opens (a cumulative
    counter whose last update was hours earlier still has to seed its baseline).
    That carry-in row keeps its own timestamp, so "held for" stays truthful.
    """
    raw: dict[str, list[tuple[datetime, str]]] = {}
    for path in json_paths:
        payload = json.loads(path.read_text())
        for entity in payload["data"]["entities"]:
            entity_id = entity["entity_id"]
            series = ENTITY_TO_SERIES.get(entity_id)
            if series is None:
                raise SystemExit(f"unmapped entity {entity_id} in {path}")
            bucket = raw.setdefault(series, [])
            for state in entity["states"]:
                bucket.append((datetime.fromisoformat(state["last_updated"]), state["state"]))

    counts: dict[str, int] = {}
    rows: list[tuple[datetime, str, str]] = []
    for series, unsorted in raw.items():
        unsorted.sort(key=lambda row: row[0])
        deduped = _dedupe(unsorted)
        carry = [row for row in deduped if row[0] < start]
        windowed = [row for row in deduped if start <= row[0] <= end]
        selected = ([carry[-1]] if carry else []) + windowed
        counts[series] = len(selected)
        rows.extend((ts, series, state) for ts, state in selected)

    rows.sort(key=lambda row: (row[0], row[1]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        # Unix line endings explicitly: csv defaults to CRLF, which has no place
        # in a committed text fixture. Timestamps keep their microseconds — the
        # anchor thresholds are sustained-state *durations*, and truncating to
        # whole seconds moves the 10-minute powercalc rows (which land at
        # :00.15) a tick earlier and visibly perturbs the trip residuals.
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["ts", "series", "state"])
        for ts, series, state in rows:
            writer.writerow([ts.isoformat(), series, state])
    counts["total"] = len(rows)
    return counts


def load_fixture(csv_path: Path) -> dict[str, list[tuple[datetime, str, datetime]]]:
    """Read the fixture → ``series -> [(ts, state, changed_at)]``, sorted.

    ``changed_at`` is the timestamp of the first row of the current run of equal
    states — i.e. HA's ``last_changed``. The export already collapses runs, so
    this is normally just ``ts``; computing it anyway makes the loader correct
    for a hand-edited fixture too.
    """
    series: dict[str, list[tuple[datetime, str]]] = {name: [] for name in SERIES}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["series"]
            if name not in series:
                raise SystemExit(f"unknown series {name!r} in {csv_path}")
            series[name].append((datetime.fromisoformat(row["ts"]), row["state"]))

    out: dict[str, list[tuple[datetime, str, datetime]]] = {}
    for name, rows in series.items():
        rows.sort(key=lambda row: row[0])
        withchange: list[tuple[datetime, str, datetime]] = []
        for ts, state in rows:
            changed_at = ts if not withchange or withchange[-1][1] != state else withchange[-1][2]
            withchange.append((ts, state, changed_at))
        out[name] = withchange
    return out


class Cursor:
    """Forward-only walk over one series: the row in force at time ``t``.

    Forward-only because the replay never steps backwards, which keeps a week of
    ticks linear instead of a binary search per tick per series.
    """

    def __init__(self, rows: list[tuple[datetime, str, datetime]]) -> None:
        self._rows = rows
        self._index = -1

    def at(self, when: datetime) -> tuple[str | None, datetime | None]:
        """``(state, changed_at)`` in force at ``when``; ``(None, None)`` before the first row."""
        while self._index + 1 < len(self._rows) and self._rows[self._index + 1][0] <= when:
            self._index += 1
        if self._index < 0:
            return None, None
        _, state, changed_at = self._rows[self._index]
        return state, changed_at


def _as_float(state: str | None) -> float | None:
    """Numeric state, or ``None`` for missing/``unknown``/``unavailable``/garbage."""
    if state is None or state in ("unknown", "unavailable"):
        return None
    try:
        return float(state)
    except ValueError:
        return None


def _as_tristate(
    state: str | None, changed_at: datetime | None, when: datetime
) -> tuple[bool | None, float | None]:
    """``(on/off/None, seconds_held)`` — anything but ``on``/``off`` is unknown.

    Mirrors what the tracker reads off ``hass.states``: ``unavailable`` maps to
    ``None`` (never to "off"), so a dropout can't fabricate an anchor, and its
    "held for" is ``None`` too.
    """
    if state == "on":
        return True, (when - changed_at).total_seconds() if changed_at else None
    if state == "off":
        return False, (when - changed_at).total_seconds() if changed_at else None
    return None, None


def _ceil_minute(when: datetime) -> datetime:
    """``when`` rounded up to the next whole minute (the tick grid)."""
    floored = when.replace(second=0, microsecond=0)
    return floored if floored == when else floored + timedelta(minutes=1)


# ── The replay ────────────────────────────────────────────────────────────────


def replay(
    csv_path: Path,
    *,
    hot_fraction: float = LIVE_HOT_FRACTION,
    standby_w: float = LIVE_STANDBY_W,
    learn: bool = True,
    model: Any | None = None,
) -> dict[str, Any]:
    """Step the fixture through ``apply_tick`` and collect the acceptance metrics.

    The tank starts *calibrated* at the live sensor's SoC: the point of the
    replay is the steady-state behaviour between anchors, not the cold start
    (which ``tests/test_tank_model.py`` covers). With ``learn=False`` the
    learnable params are written back after every tick, which isolates the
    integrator from the learner.
    """
    tank = model if model is not None else load_tank_model()
    fixture = load_fixture(csv_path)
    for name in REQUIRED_SERIES:
        if not fixture[name]:
            raise SystemExit(f"fixture {csv_path} has no {name} rows")

    params = tank.TankParams(TANK_VOLUME_L, TANK_SETPOINT_C, TANK_COLD_IN_C)
    capacity = tank.capacity_kwh(TANK_VOLUME_L, TANK_SETPOINT_C, TANK_COLD_IN_C)

    # Start at the first tick where every consumed series has a value — before
    # that the counters have no baseline and the balance would be fiction.
    start = _ceil_minute(max(fixture[name][0][0] for name in REQUIRED_SERIES))
    end = max(rows[-1][0] for rows in fixture.values() if rows).replace(second=0, microsecond=0)
    cursors = {name: Cursor(rows) for name, rows in fixture.items()}

    live_start = _as_float(cursors[LIVE_SOC].at(start)[0])
    start_deficit = (
        capacity * tank.FIRST_INSTALL_DEFICIT_FRACTION
        if live_start is None
        else (1.0 - live_start / 100.0) * capacity
    )
    energy_start = _as_float(cursors[ENERGY].at(start)[0])
    water_start = _as_float(cursors[WATER].at(start)[0])
    state = tank.TankState(
        deficit_kwh=start_deficit,
        hot_fraction=hot_fraction,
        standby_w=standby_w,
        calibrated=True,
        cycle_start_iso=start.isoformat(),
        energy_baseline_kwh=energy_start,
        water_baseline_l=None if water_start is None else water_start * 1000.0,
        water_baseline_iso=start.isoformat(),
        last_tick_iso=start.isoformat(),
    )

    ticks = 0
    heating_minutes = 0
    prev_soc: float | None = None
    jumps: list[tuple[str, float]] = []
    corrections: list[float] = []
    anchor_socs: list[float] = []
    residuals: list[tuple[str, float]] = []
    trajectory: list[dict[str, Any]] = []
    live_diffs: list[float] = []
    post_trip: list[tuple[float, float]] = []
    heating_socs: list[float] = []
    plateau_run = 0
    max_plateau_run = 0
    plateau_key: tuple[float, ...] | None = None
    last_trip: datetime | None = None

    now = start
    while now <= end:
        now += timedelta(seconds=TICK_S)
        heating_on, heating_held = _as_tristate(*cursors[HEATING].at(now), now)
        contactor_on, contactor_held = _as_tristate(*cursors[CONTACTOR].at(now), now)
        energy = _as_float(cursors[ENERGY].at(now)[0])
        water = _as_float(cursors[WATER].at(now)[0])
        live = _as_float(cursors[LIVE_SOC].at(now)[0])

        inputs = tank.TickInputs(
            now_iso=now.isoformat(),
            elapsed_s=float(TICK_S),
            energy_counter_kwh=energy,
            water_counter_l=None if water is None else water * 1000.0,
            contactor_on=contactor_on,
            heating_on=heating_on,
            # The model's field names read the anchor's way round ("contactor on
            # for", "heating off for"); both are just "held its current state for".
            contactor_on_for_s=contactor_held,
            heating_off_for_s=heating_held,
            people_home=PEOPLE_HOME,
            e_base=E_BASE,
            e_draw_per_person=E_DRAW_PER_PERSON,
            empty_house_factor=EMPTY_HOUSE_FACTOR,
            rated_power_kw=RATED_POWER_KW,
        )
        before = state
        result = tank.apply_tick(state, params, inputs)
        state = result.state
        if not learn:
            state = replace(
                state,
                hot_fraction=hot_fraction,
                hot_fraction_profile=(hot_fraction,) * tank.N_DAYPARTS,
                standby_w=standby_w,
                hysteresis_kwh=tank.SEED_HYSTERESIS_KWH,
                residual_ratio=tank.SEED_RESIDUAL_RATIO,
            )
        soc_pct = result.soc * 100.0
        ticks += 1
        if heating_on:
            heating_minutes += 1

        if result.anchored:
            anchor_socs.append(soc_pct)
            corrections.append(0.0 if prev_soc is None else soc_pct - prev_soc)
            cycle_h = tank._hours_between(before.cycle_start_iso, inputs.now_iso)
            # The learning ledger just before the anchor re-zeroed it: the model's
            # own error over the cycle. ``apply_tick`` doesn't publish it, so
            # re-apply this tick's step to the pre-tick value.
            ledger = (
                before.deficit_kwh
                if before.cycle_unclamped_kwh is None
                else before.cycle_unclamped_kwh
            )
            residual = ledger + result.draw_kwh + result.standby_kwh - result.energy_in_kwh
            if cycle_h >= LONG_CYCLE_H:
                residuals.append((now.isoformat(), residual))
            trajectory.append(
                {
                    "at": now.isoformat(),
                    "cycle_hours": cycle_h,
                    "profile": tuple(tank.profile_of(state)),
                    "standby_w": state.standby_w,
                    "hysteresis_kwh": state.hysteresis_kwh,
                    "residual_ratio": state.residual_ratio,
                }
            )
            last_trip = now
        elif prev_soc is not None and abs(soc_pct - prev_soc) >= JUMP_PTS:
            jumps.append((now.isoformat(), soc_pct - prev_soc))

        # The pathologies the v2 rework had to kill: a % pinned at 100 while the
        # element is still drawing, and a long run frozen at one high value.
        if heating_on and not result.latched:
            heating_socs.append(soc_pct)
            key = (round(soc_pct, 1),)
            if key == plateau_key and soc_pct >= PLATEAU_MIN_SOC:
                plateau_run += 1
            else:
                plateau_run = 1
                plateau_key = key
            max_plateau_run = max(max_plateau_run, plateau_run)
        else:
            plateau_run = 0
            plateau_key = None

        # Post-trip decline: the element idles while mixing pulls the tank back
        # below the thermostat band, so the % must drift down, not sit at 100.
        if last_trip is not None and result.latched and not heating_on:
            post_trip.append(((now - last_trip).total_seconds() / 60.0, soc_pct))
        if live is not None:
            live_diffs.append(abs(soc_pct - live))
        prev_soc = soc_pct

    residual_values = [value for _, value in residuals]
    post_trip_by_minute = {
        minute: [soc for elapsed, soc in post_trip if abs(elapsed - minute) < 1.0]
        for minute in (5, 30, 60)
    }
    return {
        "learn": learn,
        "hot_fraction": hot_fraction,
        "standby_w": standby_w,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "ticks": ticks,
        "capacity_kwh": capacity,
        "start_soc_live": live_start,
        "heating_minutes": heating_minutes,
        "jumps": jumps,
        "anchors": len(anchor_socs),
        "anchor_socs": anchor_socs,
        "anchor_corrections": corrections,
        "long_cycle_anchors": len(residuals),
        "residuals": residuals,
        "residual_rms": (
            math.sqrt(sum(v * v for v in residual_values) / len(residual_values))
            if residual_values
            else 0.0
        ),
        "residual_bias": (sum(residual_values) / len(residual_values) if residual_values else 0.0),
        "trajectory": trajectory,
        "post_trip_soc": post_trip_by_minute,
        "post_trip_median": {
            minute: (statistics.median(values) if values else None)
            for minute, values in post_trip_by_minute.items()
        },
        "heating_soc_max": max(heating_socs) if heating_socs else 0.0,
        "heating_ticks": len(heating_socs),
        "max_plateau_run": max_plateau_run,
        "live_mean_abs_diff": (sum(live_diffs) / len(live_diffs) if live_diffs else None),
        "final_state": state,
    }


def print_report(metrics: dict[str, Any]) -> None:
    """Print the acceptance report for one replay run."""
    print(
        f"=== replay learn={metrics['learn']} hot_fraction={metrics['hot_fraction']:.3f} "
        f"standby={metrics['standby_w']:.0f} W ==="
    )
    print(
        f"  window {metrics['start']} → {metrics['end']}  ({metrics['ticks']} ticks of 60 s), "
        f"capacity {metrics['capacity_kwh']:.2f} kWh, start SoC (live) {metrics['start_soc_live']}%"
    )
    print(
        f"  minutes heating {metrics['heating_minutes']}; anchors {metrics['anchors']} "
        f"({metrics['long_cycle_anchors']} closing a ≥{LONG_CYCLE_H:.0f} h cycle)"
    )
    print(f"  non-anchor jumps ≥{JUMP_PTS:.0f} pts: {len(metrics['jumps'])}")
    for at, delta in metrics["jumps"][:8]:
        print(f"    {at} {delta:+.1f} pts")
    print(
        "  anchor corrections (pts): "
        + ", ".join(f"{value:+.1f}" for value in metrics["anchor_corrections"])
    )
    print(
        f"  trip residuals (≥{LONG_CYCLE_H:.0f} h cycles, kWh): "
        + ", ".join(f"{value:+.2f}" for _, value in metrics["residuals"])
    )
    print(
        f"    rms {metrics['residual_rms']:.2f}  bias {metrics['residual_bias']:+.2f} kWh "
        "(+ = model over-estimated the draw)"
    )
    print("  learner trajectory (per anchor):")
    for row in metrics["trajectory"]:
        profile = ", ".join(f"{value:.3f}" for value in row["profile"])
        print(
            f"    {row['at'][5:16]}  cycle {row['cycle_hours']:5.1f} h  profile ({profile})  "
            f"standby {row['standby_w']:5.1f} W  hyst {row['hysteresis_kwh']:.2f}  "
            f"ratio {row['residual_ratio']:.3f}"
        )
    for minute in (5, 30, 60):
        values = metrics["post_trip_soc"][minute]
        if values:
            median = metrics["post_trip_median"][minute]
            print(
                f"  SoC ~{minute:2d} min after a trip (idle, latched): median {median:.1f} — "
                + ", ".join(f"{value:.1f}" for value in values)
            )
    print(
        f"  max SoC while heating and not latched: {metrics['heating_soc_max']:.3f} "
        f"over {metrics['heating_ticks']} ticks; longest identical-SoC run "
        f"{metrics['max_plateau_run']} ticks"
    )
    live = metrics["live_mean_abs_diff"]
    if live is not None:
        print(
            f"  mean |replay − live v0.8 sensor|: {live:.2f} pts (information only — v2 semantics)"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    exporter = sub.add_parser("export", help="rebuild the CSV fixture from HA history JSON")
    exporter.add_argument("json_paths", nargs="+", type=Path, help="HA history JSON files")
    exporter.add_argument("-o", "--output", required=True, type=Path, help="CSV fixture to write")
    exporter.add_argument(
        "--window",
        nargs=2,
        required=True,
        metavar=("START", "END"),
        help="inclusive ISO-8601 window with offset, e.g. 2026-08-28T11:00+03:00",
    )

    runner = sub.add_parser("replay", help="replay a CSV fixture through tank_model.apply_tick")
    runner.add_argument("csv_path", type=Path, help="CSV fixture to replay")
    runner.add_argument("--hot-fraction", type=float, default=LIVE_HOT_FRACTION)
    runner.add_argument("--standby-w", type=float, default=LIVE_STANDBY_W)
    runner.add_argument(
        "--no-learn",
        action="store_true",
        help="freeze the learnable params (isolate the integrator)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "export":
        counts = export_fixture(
            args.json_paths,
            args.output,
            datetime.fromisoformat(args.window[0]),
            datetime.fromisoformat(args.window[1]),
        )
        total = counts.pop("total")
        print(f"wrote {args.output} — {total} rows")
        for series, count in sorted(counts.items()):
            print(f"  {series}: {count}")
        return 0
    print_report(
        replay(
            args.csv_path,
            hot_fraction=args.hot_fraction,
            standby_w=args.standby_w,
            learn=not args.no_learn,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
