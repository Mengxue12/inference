#!/usr/bin/env python3
"""
Extract power_begin / power_end windows from MLPerf detail logs.

Logs from run_in_container_mounted.sh live under OUTPUT_DIR/logs/mlperf_log_detail.txt
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
from typing import Any, Iterator, TextIO
from zoneinfo import ZoneInfo

MLLOG_PREFIX = ":::MLLOG"
POWER_DATETIME_FMT = "%m-%d-%Y %H:%M:%S.%f"
DETAIL_LOG_NAME = "mlperf_log_detail.txt"
MANIFEST_NAME = "manifest.json"
TIME_JSON_NAME = "time.json"
POWER_SUBDIR = "power"
DEFAULT_PROM_STEP = "1s"


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


def _resolve_tz(tz_name: str) -> timezone | ZoneInfo:
    if tz_name.upper() == "UTC":
        return timezone.utc
    if tz_name == "local":
        return datetime.now().astimezone().tzinfo or timezone.utc
    return ZoneInfo(tz_name)


def _to_unix_seconds(dt: datetime, tz: timezone | ZoneInfo) -> int | None:
    """Unix seconds for start times (truncate sub-second fraction)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)
    return int(dt.timestamp())


def _to_unix_seconds_ceil(dt: datetime, tz: timezone | ZoneInfo) -> int | None:
    """Unix seconds for end times (round up sub-second fraction)."""
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


def _manifest_str_field(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def read_node_name_from_run_dir(run_dir: Path) -> str | None:
    manifest_path = run_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: {manifest_path}: {exc}", file=sys.stderr)
        return None
    return _manifest_str_field(data, "node_name")


def read_node_arch_from_run_dir(run_dir: Path) -> str | None:
    manifest_path = run_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: {manifest_path}: {exc}", file=sys.stderr)
        return None
    return _manifest_str_field(data, "node_arch")


def _run_dir_from_log_path(log_path: Path) -> Path:
    log_path = log_path.resolve()
    if log_path.name == DETAIL_LOG_NAME and log_path.parent.name == "logs":
        return log_path.parent.parent
    return log_path.parent


def _expected_detail_log(root: Path) -> Path:
    return root / "logs" / DETAIL_LOG_NAME


def _is_logs_detail_log(path: Path) -> bool:
    return path.name == DETAIL_LOG_NAME and path.parent.name == "logs"


def _iter_detail_log_paths(path: Path, recursive: bool) -> Iterator[Path]:
    if path.is_file():
        if _is_logs_detail_log(path):
            yield path
        return

    direct = _expected_detail_log(path)
    if direct.is_file():
        yield direct

    if not recursive:
        return

    for candidate in sorted(path.rglob(DETAIL_LOG_NAME)):
        if candidate == direct:
            continue
        if not _is_logs_detail_log(candidate):
            continue
        yield candidate


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


def _roots_label(roots: list[Path]) -> str:
    resolved = [str(r.resolve()) for r in roots]
    return resolved[0] if len(resolved) == 1 else ", ".join(resolved)


def _missing_detail_log_hint(root: Path, recursive: bool) -> str:
    root = root.resolve()
    if root.is_file():
        if root.name == DETAIL_LOG_NAME:
            return str(root)
        return f"not {DETAIL_LOG_NAME}: {root}"

    expected = _expected_detail_log(root)
    hints: list[str] = [f"looked for {expected}"]
    logs_dir = root / "logs"
    if not logs_dir.exists():
        hints.append("no logs/ directory")
    elif not logs_dir.is_dir():
        hints.append("logs exists but is not a directory")
    elif not expected.is_file():
        hints.append("logs/ present but mlperf_log_detail.txt missing")
    if recursive:
        hints.append(f"also searched recursively under {root}")
    elif not root.name.startswith("run_"):
        hints.append(
            "hint: pass a run_* directory, or use --recursive for experiment roots"
        )
    return "; ".join(hints)


def _print_no_detail_log(
    roots: list[Path], recursive: bool, out: TextIO, *, suffix: str = ""
) -> None:
    under = _roots_label(roots)
    hints = " | ".join(_missing_detail_log_hint(r, recursive) for r in roots)
    msg = f"No mlperf_log_detail.txt found under {under} ({hints})."
    if suffix:
        msg = f"{msg} {suffix}"
    print(msg, file=out)


def _print_no_power_pairs(
    roots: list[Path], by_run: dict[Path, RunCollect], out: TextIO, *, suffix: str = ""
) -> None:
    under = _roots_label(roots)
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
        for log_path in _iter_detail_log_paths(root, recursive):
            log_path = log_path.resolve()
            if log_path in seen_logs:
                continue
            seen_logs.add(log_path)
            run_dir = _run_dir_from_log_path(log_path)
            rc = by_run.setdefault(run_dir, RunCollect(windows=[], detail_logs=[]))
            log_str = str(log_path)
            if log_str not in rc.detail_logs:
                rc.detail_logs.append(log_str)
            rc.windows.extend(extract_windows_from_log(log_path))

    return by_run


def iter_run_dirs(paths: list[Path], recursive: bool) -> list[Path]:
    """Each OUTPUT_DIR that has (or is parent of) mlperf_log_detail.txt under paths.

    Uses the same discovery rules as collect_by_run_dir / _iter_detail_log_paths.
    """
    seen: set[Path] = set()
    run_dirs: list[Path] = []

    for root in paths:
        root = root.resolve()
        for log_path in _iter_detail_log_paths(root, recursive):
            run_dir = _run_dir_from_log_path(log_path).resolve()
            if run_dir in seen:
                continue
            seen.add(run_dir)
            run_dirs.append(run_dir)

    return sorted(run_dirs)


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


def time_json_path(run_dir: Path) -> Path:
    return run_dir / POWER_SUBDIR / TIME_JSON_NAME


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
        _print_no_detail_log(roots, recursive, out)
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
        tz = _resolve_tz(args.tz)
    except Exception as exc:
        print(f"error: invalid --tz {args.tz!r}: {exc}", file=sys.stderr)
        return 1

    by_run = collect_by_run_dir(roots, args.recursive)
    total_windows = sum(len(rc.windows) for rc in by_run.values())

    if args.write_time_json:
        written: list[Path] = []
        if not by_run:
            _print_no_detail_log(
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
                in_log = ", ".join(rc.detail_logs) or str(
                    _expected_detail_log(run_dir)
                )
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
                pass  # message already printed above
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
                    f"warning: no time.json written under {_roots_label(roots)} "
                    f"({skip_reason})",
                    file=sys.stderr,
                )
            return 2
        return 0

    _format_text(by_run, roots, args.recursive, sys.stdout)
    return 0 if total_windows else 2


if __name__ == "__main__":
    raise SystemExit(main())
