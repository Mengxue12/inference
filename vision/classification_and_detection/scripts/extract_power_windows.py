#!/usr/bin/env python3
"""
Extract power_begin / power_end windows from MLPerf detail logs.

Logs from run_lite_infer_sweep.sh live under OUTPUT_DIR/logs/mlperf_log_detail.txt
(each run_* or accuracy/ directory is an OUTPUT_DIR). This script reads that file, not
run_* metadata directly: for a directory argument it first checks <path>/logs/mlperf_log_detail.txt;
with --recursive it also finds mlperf_log_detail.txt anywhere under <path> (e.g. run_*/logs/).
Each consecutive power_begin / power_end pair in a file is one window (group).

With -o, writes OUTPUT_DIR/power/time.json (run_1, run_2, accuracy/, etc.) with Unix
timestamps and manifest node_name/node_arch, ready for fetch_prometheus_metrics.py --time-json.
By default an existing time.json is left unchanged; use --overwrite with -o to replace it.

Examples:
  python3 scripts/extract_power_windows.py /path/to/output/run_1
  python3 scripts/extract_power_windows.py /path/to/experiment_root --recursive
  python3 scripts/extract_power_windows.py /path/to/output --recursive -o
  python3 scripts/extract_power_windows.py /path/to/run_1 -o --overwrite
  python3 scripts/fetch_prometheus_metrics.py --url http://localhost:9090 \\
    --time-json /path/to/run_1/power/time.json --name-prefix kepler_ --format csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO
from zoneinfo import ZoneInfo

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from extract_common import (
    DEFAULT_PROM_STEP,
    DETAIL_LOG_NAME,
    MANIFEST_NAME,
    MLLOG_PREFIX,
    POWER_DATETIME_FMT,
    POWER_SUBDIR,
    TIME_JSON_NAME,
    expected_detail_log,
    iter_detail_log_paths,
    iter_run_dirs,
    print_no_detail_log,
    read_node_arch_from_run_dir,
    read_node_name_from_run_dir,
    resolve_tz,
    roots_label,
    run_dir_from_log_path,
    time_json_path,
)

# Re-export for scripts that still import from extract_power_windows.
_expected_detail_log = expected_detail_log
_print_no_detail_log = print_no_detail_log
_resolve_tz = resolve_tz
_roots_label = roots_label


@dataclass(frozen=True)
class PowerEvent:
    key: str
    value: str
    time_ms: float | None


@dataclass(frozen=True)
class PowerWindow:
    log_path: str
    group_index: int
    power_begin: PowerEvent
    power_end: PowerEvent
    duration_s: float | None = None


@dataclass
class RunCollect:
    windows: list[PowerWindow]
    detail_logs: list[str]


def _parse_power_datetime(value: str) -> datetime | None:
    value = value.strip()
    for fmt in (POWER_DATETIME_FMT, "%m-%d-%Y %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _window_duration_s(begin: PowerEvent, end: PowerEvent) -> float | None:
    begin_dt = _parse_power_datetime(begin.value)
    end_dt = _parse_power_datetime(end.value)
    if begin_dt is None or end_dt is None:
        return None
    return (end_dt - begin_dt).total_seconds()


def _to_unix_seconds(dt: datetime, tz: timezone | ZoneInfo) -> int | None:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)
    return int(dt.timestamp())


def _to_unix_seconds_ceil(dt: datetime, tz: timezone | ZoneInfo) -> int | None:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)
    return math.ceil(dt.timestamp())


def _to_rfc3339(dt: datetime, tz: timezone | ZoneInfo) -> str | None:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)
    return dt.isoformat(timespec="milliseconds")


def _parse_mllog_messages(log_path: Path) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    with log_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.startswith(MLLOG_PREFIX):
                continue
            payload = line[len(MLLOG_PREFIX) :].lstrip()
            try:
                messages.append(json.loads(payload))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{log_path}:{line_no}: invalid MLLOG JSON: {exc}"
                ) from exc
    return messages


def extract_windows_from_log(log_path: Path) -> list[PowerWindow]:
    messages = _parse_mllog_messages(log_path)
    windows: list[PowerWindow] = []
    pending_begin: PowerEvent | None = None
    group_index = 0

    for msg in messages:
        key = msg.get("key")
        if key not in ("power_begin", "power_end"):
            continue
        event = PowerEvent(
            key=key,
            value=str(msg.get("value", "")),
            time_ms=msg.get("time_ms"),
        )
        if key == "power_begin":
            if pending_begin is not None:
                print(
                    f"warning: {log_path}: group {group_index + 1} has "
                    f"power_begin without matching power_end; starting new group",
                    file=sys.stderr,
                )
            pending_begin = event
            continue

        if pending_begin is None:
            print(
                f"warning: {log_path}: power_end without power_begin "
                f"({event.value}); skipped",
                file=sys.stderr,
            )
            continue

        group_index += 1
        windows.append(
            PowerWindow(
                log_path=str(log_path),
                group_index=group_index,
                power_begin=pending_begin,
                power_end=event,
                duration_s=_window_duration_s(pending_begin, event),
            )
        )
        pending_begin = None

    if pending_begin is not None:
        print(
            f"warning: {log_path}: trailing power_begin without power_end "
            f"({pending_begin.value})",
            file=sys.stderr,
        )

    return windows


def _print_no_power_pairs(
    roots: list[Path], by_run: dict[Path, RunCollect], out: TextIO, *, suffix: str = ""
) -> None:
    under = roots_label(roots)
    locations: list[str] = []
    for run_dir in sorted(by_run):
        rc = by_run[run_dir]
        for log in rc.detail_logs:
            locations.append(f"{run_dir} ({log})")
    in_what = "; ".join(locations) if locations else under
    msg = f"No power_begin / power_end pairs found under {under} in {in_what}."
    if suffix:
        msg = f"{msg} {suffix}"
    print(msg, file=out)


def collect_by_run_dir(
    paths: list[Path], recursive: bool
) -> dict[Path, RunCollect]:
    by_run: dict[Path, RunCollect] = {}
    seen_logs: set[Path] = set()

    for root in paths:
        root = root.resolve()
        for log_path in iter_detail_log_paths(root, recursive):
            log_path = log_path.resolve()
            if log_path in seen_logs:
                continue
            seen_logs.add(log_path)
            run_dir = run_dir_from_log_path(log_path)
            rc = by_run.setdefault(run_dir, RunCollect(windows=[], detail_logs=[]))
            log_str = str(log_path)
            if log_str not in rc.detail_logs:
                rc.detail_logs.append(log_str)
            rc.windows.extend(extract_windows_from_log(log_path))

    return by_run


def _window_entry(w: PowerWindow, tz: timezone | ZoneInfo) -> dict[str, Any]:
    begin_dt = _parse_power_datetime(w.power_begin.value)
    end_dt = _parse_power_datetime(w.power_end.value)
    entry: dict[str, Any] = {
        "group_index": w.group_index,
        "power_begin": w.power_begin.value,
        "power_end": w.power_end.value,
        "power_begin_time_ms": w.power_begin.time_ms,
        "power_end_time_ms": w.power_end.time_ms,
    }
    if w.duration_s is not None:
        entry["duration_s"] = round(w.duration_s, 6)
    if begin_dt is not None:
        entry["start_unix"] = _to_unix_seconds(begin_dt, tz)
        entry["start_rfc3339"] = _to_rfc3339(begin_dt, tz)
    if end_dt is not None:
        entry["end_unix"] = _to_unix_seconds_ceil(end_dt, tz)
        entry["end_rfc3339"] = _to_rfc3339(end_dt, tz)
    return entry


def build_time_json(
    run_dir: Path,
    windows: list[PowerWindow],
    *,
    tz: timezone | ZoneInfo,
    step: str,
) -> dict[str, Any]:
    if not windows:
        raise ValueError("no windows")

    log_path = windows[0].log_path
    entries = [_window_entry(w, tz) for w in windows]

    starts = [e["start_unix"] for e in entries if "start_unix" in e]
    ends = [e["end_unix"] for e in entries if "end_unix" in e]
    if not starts or not ends:
        raise ValueError(f"could not parse power timestamps in {log_path}")

    prom_start = min(starts)
    prom_end = max(ends)

    payload: dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "source_log": log_path,
        "timezone": str(getattr(tz, "key", tz)),
        "prometheus": {
            "start": str(prom_start),
            "end": str(prom_end),
            "step": step,
        },
        "windows": entries,
    }
    node_name = read_node_name_from_run_dir(run_dir)
    if node_name is not None:
        payload["node_name"] = node_name
    else:
        print(
            f"warning: {run_dir / MANIFEST_NAME}: no node_name; "
            f"time.json written without node_name",
            file=sys.stderr,
        )
    node_arch = read_node_arch_from_run_dir(run_dir)
    if node_arch is not None:
        payload["node_arch"] = node_arch
    return payload


def write_time_json(
    run_dir: Path,
    windows: list[PowerWindow],
    *,
    tz: timezone | ZoneInfo,
    step: str,
) -> Path:
    out_path = time_json_path(run_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_time_json(run_dir, windows, tz=tz, step=step)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return out_path


def _format_text(
    by_run: dict[Path, RunCollect],
    roots: list[Path],
    recursive: bool,
    out: TextIO,
) -> None:
    if not by_run:
        print_no_detail_log(roots, recursive, out)
        return
    if not any(rc.windows for rc in by_run.values()):
        _print_no_power_pairs(roots, by_run, out)
        return

    for run_dir in sorted(by_run):
        rc = by_run[run_dir]
        windows = rc.windows
        if not windows:
            continue
        log_path = windows[0].log_path
        print(f"\n{run_dir}  ({log_path})", file=out)
        for w in windows:
            print(f"  group {w.group_index}:", file=out)
            print(
                f"    power_begin: {w.power_begin.value}"
                f"  (time_ms={w.power_begin.time_ms})",
                file=out,
            )
            print(
                f"    power_end:   {w.power_end.value}"
                f"  (time_ms={w.power_end.time_ms})",
                file=out,
            )
            if w.duration_s is not None:
                print(f"    duration_s:  {w.duration_s:.3f}", file=out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract power_begin / power_end groups from mlperf_log_detail.txt"
    )
    parser.add_argument(
        "path",
        nargs="+",
        help="OUTPUT_DIR (run_*), logs directory, detail log file, or experiment root",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="search recursively for mlperf_log_detail.txt under each path",
    )
    parser.add_argument(
        "-o",
        "--write-time-json",
        action="store_true",
        help=(
            f"write {POWER_SUBDIR}/{TIME_JSON_NAME} under each run directory "
            f"(skip if that file already exists unless --overwrite)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            f"with -o, replace existing {POWER_SUBDIR}/{TIME_JSON_NAME} "
            f"(default: skip existing files)"
        ),
    )
    parser.add_argument(
        "--step",
        default=DEFAULT_PROM_STEP,
        help=f"Prometheus range step written to time.json (default: {DEFAULT_PROM_STEP})",
    )
    parser.add_argument(
        "--tz",
        default="UTC",
        help="IANA timezone for MLPerf wall-clock strings (default: UTC; use 'local')",
    )
    args = parser.parse_args()

    if args.overwrite and not args.write_time_json:
        print("error: --overwrite requires -o / --write-time-json", file=sys.stderr)
        return 1

    roots = [Path(p) for p in args.path]
    for root in roots:
        if not root.exists():
            print(f"error: path does not exist: {root}", file=sys.stderr)
            return 1

    try:
        tz = resolve_tz(args.tz)
    except Exception as exc:
        print(f"error: invalid --tz {args.tz!r}: {exc}", file=sys.stderr)
        return 1

    by_run = collect_by_run_dir(roots, args.recursive)
    total_windows = sum(len(rc.windows) for rc in by_run.values())

    if args.write_time_json:
        written: list[Path] = []
        if not by_run:
            print_no_detail_log(
                roots,
                args.recursive,
                sys.stderr,
                suffix="No time.json written.",
            )
        for run_dir in sorted(by_run):
            rc = by_run[run_dir]
            windows = rc.windows
            out_path = time_json_path(run_dir)
            if not windows:
                in_log = ", ".join(rc.detail_logs) or str(expected_detail_log(run_dir))
                print(
                    f"warning: no power_begin / power_end pairs found under "
                    f"{run_dir} in {in_log}; skipping {POWER_SUBDIR}/{TIME_JSON_NAME}",
                    file=sys.stderr,
                )
                continue
            if out_path.is_file() and not args.overwrite:
                print(
                    f"warning: {out_path} already exists; skipping (not overwritten)",
                    file=sys.stderr,
                )
                continue
            try:
                out_path = write_time_json(
                    run_dir, windows, tz=tz, step=args.step
                )
            except ValueError as exc:
                print(f"warning: {run_dir}: {exc}", file=sys.stderr)
                continue
            written.append(out_path)
            print(f"Wrote {out_path}", file=sys.stderr)
        if not written:
            if not by_run:
                pass
            elif total_windows == 0:
                _print_no_power_pairs(
                    roots,
                    by_run,
                    sys.stderr,
                    suffix="No time.json written.",
                )
            else:
                skip_reason = (
                    "timestamp parse failed"
                    if args.overwrite
                    else "existing files skipped or timestamp parse failed"
                )
                print(
                    f"warning: no time.json written under {roots_label(roots)} "
                    f"({skip_reason})",
                    file=sys.stderr,
                )
            return 2
        return 0

    _format_text(by_run, roots, args.recursive, sys.stdout)
    return 0 if total_windows else 2


if __name__ == "__main__":
    raise SystemExit(main())
