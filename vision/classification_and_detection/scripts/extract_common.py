"""Shared constants and helpers for power-window extraction scripts."""

from __future__ import annotations

import json
import sys
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


def resolve_tz(tz_name: str) -> timezone | ZoneInfo:
    if tz_name.upper() == "UTC":
        return timezone.utc
    if tz_name == "local":
        return datetime.now().astimezone().tzinfo or timezone.utc
    return ZoneInfo(tz_name)


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


def run_dir_from_log_path(log_path: Path) -> Path:
    log_path = log_path.resolve()
    if log_path.name == DETAIL_LOG_NAME and log_path.parent.name == "logs":
        return log_path.parent.parent
    return log_path.parent


def expected_detail_log(root: Path) -> Path:
    return root / "logs" / DETAIL_LOG_NAME


def is_logs_detail_log(path: Path) -> bool:
    return path.name == DETAIL_LOG_NAME and path.parent.name == "logs"


def iter_detail_log_paths(path: Path, recursive: bool) -> Iterator[Path]:
    if path.is_file():
        if is_logs_detail_log(path):
            yield path
        return

    direct = expected_detail_log(path)
    if direct.is_file():
        yield direct

    if not recursive:
        return

    for candidate in sorted(path.rglob(DETAIL_LOG_NAME)):
        if candidate == direct:
            continue
        if not is_logs_detail_log(candidate):
            continue
        yield candidate


def iter_run_dirs(paths: list[Path], recursive: bool) -> list[Path]:
    """Each OUTPUT_DIR that has mlperf_log_detail.txt under paths."""
    seen: set[Path] = set()
    run_dirs: list[Path] = []

    for root in paths:
        root = root.resolve()
        for log_path in iter_detail_log_paths(root, recursive):
            run_dir = run_dir_from_log_path(log_path).resolve()
            if run_dir in seen:
                continue
            seen.add(run_dir)
            run_dirs.append(run_dir)

    return sorted(run_dirs)


def time_json_path(run_dir: Path) -> Path:
    return run_dir / POWER_SUBDIR / TIME_JSON_NAME


def roots_label(roots: list[Path]) -> str:
    resolved = [str(r.resolve()) for r in roots]
    return resolved[0] if len(resolved) == 1 else ", ".join(resolved)


def missing_detail_log_hint(root: Path, recursive: bool) -> str:
    root = root.resolve()
    if root.is_file():
        if root.name == DETAIL_LOG_NAME:
            return str(root)
        return f"not {DETAIL_LOG_NAME}: {root}"

    expected = expected_detail_log(root)
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


def print_no_detail_log(
    roots: list[Path], recursive: bool, out: TextIO, *, suffix: str = ""
) -> None:
    under = roots_label(roots)
    hints = " | ".join(missing_detail_log_hint(r, recursive) for r in roots)
    msg = f"No mlperf_log_detail.txt found under {under} ({hints})."
    if suffix:
        msg = f"{msg} {suffix}"
    print(msg, file=out)
