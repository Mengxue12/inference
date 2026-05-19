"""Shared constants and helpers for metric fetch scripts (Prometheus, meter)."""

from __future__ import annotations

import argparse
import contextlib
import csv
import fnmatch
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from extract_common import (
    POWER_SUBDIR,
    TIME_JSON_NAME,
    iter_run_dirs,
    read_node_name_from_run_dir,
    time_json_path,
)

TIME_JSON_PROM_KEY = "prometheus"
DEFAULT_METRIC_STEM = "metric"
_STEP_RE = re.compile(r"^(\d+(?:\.\d+)?)(ms|[smhdwy])?$", re.IGNORECASE)


def sanitize_output_stem(text: str, *, max_len: int = 120) -> str:
    stem = re.sub(r"[^\w.\-]+", "_", text.strip())
    stem = stem.strip("._")
    if not stem:
        return DEFAULT_METRIC_STEM
    return stem[:max_len]


def metric_output_extension(args: argparse.Namespace) -> str:
    if args.raw:
        return ".json"
    if args.format == "csv":
        return ".csv"
    return ".jsonl"


def metric_output_stem(args: argparse.Namespace) -> str:
    if args.query:
        if len(args.query) == 1:
            return sanitize_output_stem(args.query[0])
        return sanitize_output_stem("_".join(args.query))
    return DEFAULT_METRIC_STEM


def metric_output_basename(args: argparse.Namespace) -> str:
    return metric_output_stem(args) + metric_output_extension(args)


def resolve_output_path(args: argparse.Namespace) -> Path | None:
    time_dir = Path(args.time_json).resolve().parent if args.time_json else None

    if args.output is not None:
        if args.output == "":
            if time_dir is None:
                raise ValueError("-o without PATH requires --time-json")
            return time_dir / metric_output_basename(args)
        path = Path(args.output)
        if time_dir is not None and not path.is_absolute():
            return time_dir / path
        return path

    return None


def resolve_csv_output_path(
    *,
    time_json: Path | None,
    output: str | None,
    default_stem: str,
) -> Path | None:
    time_dir = time_json.resolve().parent if time_json else None

    if output is not None:
        if output == "":
            if time_dir is None:
                raise ValueError("-o without PATH requires --time-json or run paths")
            return time_dir / f"{default_stem}.csv"
        path = Path(output)
        if time_dir is not None and not path.is_absolute():
            return time_dir / path
        return path

    return None


@contextlib.contextmanager
def open_output_stream(path: Path | None):
    if path is None:
        yield sys.stdout
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        yield f


def load_time_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"time.json not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"time.json root must be an object: {p}")
    return data


def parse_prom_step(step: str) -> float:
    """Prometheus duration (e.g. 1s, 15s, 1m) to seconds."""
    step = step.strip()
    if not step:
        raise ValueError("empty step")
    m = _STEP_RE.match(step)
    if not m:
        return float(step)
    num = float(m.group(1))
    unit = (m.group(2) or "s").lower()
    mult = {
        "ms": 1e-3,
        "s": 1.0,
        "m": 60.0,
        "h": 3600.0,
        "d": 86400.0,
        "w": 604800.0,
        "y": 31536000.0,
    }
    if unit not in mult:
        raise ValueError(f"unsupported step unit in {step!r}")
    return num * mult[unit]


def prom_bounds_from_time_json(data: dict[str, Any]) -> tuple[float, float, float]:
    prom = data.get(TIME_JSON_PROM_KEY)
    if not isinstance(prom, dict):
        raise ValueError(f"time.json missing {TIME_JSON_PROM_KEY!r} object")
    start = prom.get("start")
    end = prom.get("end")
    step = prom.get("step")
    if start is None:
        raise ValueError("time.json prometheus.start is required")
    if end is None:
        raise ValueError("time.json prometheus.end is required")
    if step is None:
        raise ValueError("time.json prometheus.step is required")
    return float(start), float(end), parse_prom_step(str(step))


def iter_range_timestamps(start: float, end: float, step_s: float) -> list[float]:
    if step_s <= 0:
        raise ValueError(f"step must be positive, got {step_s}")
    out: list[float] = []
    ts = float(start)
    end_f = float(end)
    while ts <= end_f + 1e-9:
        out.append(ts)
        ts += step_s
    return out


def discover_time_json_paths(roots: list[Path], recursive: bool) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for run_dir in iter_run_dirs(roots, recursive):
        candidate = time_json_path(run_dir)
        if not candidate.is_file():
            print(
                f"warning: {run_dir}: no {POWER_SUBDIR}/{TIME_JSON_NAME}; skipping",
                file=sys.stderr,
            )
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(resolved)
    return sorted(paths)


def resolve_node_name_from_time_json(data: dict[str, Any]) -> str | None:
    node_name = data.get("node_name")
    if node_name is not None:
        text = str(node_name).strip()
        if text:
            return text
    run_dir = data.get("run_dir")
    if run_dir:
        return read_node_name_from_run_dir(Path(run_dir))
    return None


@dataclass(frozen=True)
class LabelMatch:
    """Glob pattern matched against one label value (KEY=pattern) or any value (pattern only)."""

    key: str | None
    pattern: str


def parse_label_match(spec: str) -> LabelMatch:
    spec = spec.strip()
    if not spec:
        raise ValueError("empty label match")
    if "=" in spec:
        key, pattern = spec.split("=", 1)
        key = key.strip()
        pattern = pattern.strip()
        if not key or not pattern:
            raise ValueError(f"invalid label match {spec!r}: need KEY=PATTERN")
        return LabelMatch(key, pattern)
    return LabelMatch(None, spec)


def metric_matches_label_filters(
    metric: dict[str, Any], filters: list[LabelMatch]
) -> bool:
    if not filters:
        return True
    for f in filters:
        if f.key is None:
            if not any(
                fnmatch.fnmatchcase(str(v), f.pattern) for v in metric.values()
            ):
                return False
        else:
            if not fnmatch.fnmatchcase(str(metric.get(f.key, "")), f.pattern):
                return False
    return True


def filter_rows_by_labels(
    rows: list[dict[str, Any]], filters: list[LabelMatch]
) -> list[dict[str, Any]]:
    if not filters:
        return rows
    return [r for r in rows if metric_matches_label_filters(r["metric"], filters)]


def merge_label_filters(
    base: list[LabelMatch], extra: list[LabelMatch]
) -> list[LabelMatch]:
    if not extra:
        return base
    return base + extra


def label_filters_for_node_name(node_name: str | None) -> list[LabelMatch]:
    filters: list[LabelMatch] = [LabelMatch("zone", "psys")]
    if not node_name:
        return filters
    if "master" in node_name.casefold():
        filters.append(LabelMatch("node_name", f"*{node_name}*"))
    else:
        filters.append(LabelMatch(None, f"*{node_name}*"))
    return filters


def apply_time_json_to_args(
    args: argparse.Namespace, time_json_path: str
) -> list[LabelMatch]:
    data = load_time_json(time_json_path)
    prom = data.get(TIME_JSON_PROM_KEY)
    if not isinstance(prom, dict):
        raise ValueError(f"time.json missing {TIME_JSON_PROM_KEY!r} object")

    if args.start is None:
        start = prom.get("start")
        if start is None:
            raise ValueError("time.json prometheus.start is required")
        args.start = str(start)
    if args.end is None:
        end = prom.get("end")
        if end is None:
            raise ValueError("time.json prometheus.end is required")
        args.end = str(end)
    if prom.get("step") is not None:
        args.step = str(prom["step"])

    node_name = resolve_node_name_from_time_json(data)
    if node_name is None:
        print(
            f"warning: {time_json_path}: no node_name in time.json or manifest; "
            f"using zone=psys filter only",
            file=sys.stderr,
        )
    return label_filters_for_node_name(node_name)


def csv_column_label_keys(
    metrics: list[dict[str, Any]],
    *,
    include_keys: frozenset[str] | None,
    all_labels: bool,
) -> frozenset[str] | None:
    if all_labels:
        return None
    diff = distinguishing_label_keys(metrics)
    if include_keys is None:
        return diff
    allowed = {k for k in include_keys if k != "__name__"}
    if not diff:
        return frozenset(k for k in allowed if any(k in m for m in metrics))
    return frozenset(k for k in diff if k in allowed)


def distinguishing_label_keys(metrics: list[dict[str, Any]]) -> frozenset[str]:
    if len(metrics) < 2:
        return frozenset()
    keys: set[str] = set()
    for key in metrics[0]:
        if key == "__name__":
            continue
        values = {str(m.get(key, "")) for m in metrics}
        if len(values) > 1:
            keys.add(key)
    return frozenset(keys)


def _escape_label_value(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def series_column_name(
    labels: dict[str, Any],
    *,
    label_keys: frozenset[str] | None = None,
) -> str:
    name = str(labels.get("__name__", ""))
    if label_keys is None:
        rest = [(str(k), str(v)) for k, v in sorted(labels.items()) if k != "__name__"]
    else:
        rest = [
            (str(k), str(labels[k]))
            for k in sorted(label_keys)
            if k in labels and k != "__name__"
        ]
    if not rest:
        return name or json.dumps(labels, sort_keys=True, ensure_ascii=False)
    inner = ",".join(f'{k}="{_escape_label_value(v)}"' for k, v in rest)
    return f"{name}{{{inner}}}"


def format_timestamp_cell(ts: float, mode: str) -> str:
    if mode == "unix":
        if ts == int(ts):
            return str(int(ts))
        return str(ts)
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def write_wide_csv(
    records: list[tuple[float, str, str]],
    timestamp_mode: str,
    delimiter: str,
    out: TextIO,
) -> None:
    matrix: dict[float, dict[str, str]] = defaultdict(dict)
    col_order: list[str] = []
    seen: set[str] = set()
    for ts, col, val in records:
        matrix[ts][col] = val
        if col not in seen:
            seen.add(col)
            col_order.append(col)
    col_order.sort()
    w = csv.writer(out, delimiter=delimiter)
    w.writerow(["timestamp", *col_order])
    for ts in sorted(matrix.keys()):
        rowm = matrix[ts]
        w.writerow(
            [format_timestamp_cell(ts, timestamp_mode)]
            + [rowm.get(c, "") for c in col_order]
        )
