#!/usr/bin/env python3
"""
Align meter power samples to power/time.json windows and emit wide CSV.

Reads meter-platform_power from extract_meter_power.py output (*-timestamp.csv),
uses prometheus.start / end / step from time.json (same grid as
fetch_prometheus_metrics.py --time-json), and prints a CSV with timestamp as the
first column.

Examples:
  python3 scripts/fetch_meter_power_metrics.py \\
    --meter-power /path/to/2025-01-01T12_00_00Z \\
    --time-json run_1/power/time.json

  python3 scripts/fetch_meter_power_metrics.py \\
    --meter-power /path/to/2025-01-01T12_00_00Z-timestamp.csv \\
    /path/to/experiment_root --recursive -o

  python3 scripts/fetch_meter_power_metrics.py \\
    --meter-power /path/to/meter -r /path/to/output --format csv -o
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from fetch_common import (  # noqa: E402
    discover_time_json_paths,
    iter_range_timestamps,
    load_time_json,
    open_output_stream,
    prom_bounds_from_time_json,
    resolve_csv_output_path,
    sanitize_output_stem,
    write_wide_csv,
)

METER_COLUMN = "meter-platform_power"
METER_CSV_SUFFIX = "-timestamp.csv"
DEFAULT_OUTPUT_STEM = sanitize_output_stem(METER_COLUMN)


def resolve_meter_csv_path(meter_path: str) -> Path:
    p = Path(meter_path).expanduser()
    if p.is_file():
        return p.resolve()
    base = p.resolve()
    if base.name.endswith(METER_CSV_SUFFIX) and base.is_file():
        return base
    candidate = Path(f"{base}{METER_CSV_SUFFIX}")
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(
        f"meter power file not found: {base} or {candidate} "
        f"(expected extract_meter_power.py *-timestamp.csv output)"
    )


@dataclass(frozen=True)
class MeterSeries:
    timestamps: np.ndarray
    values: np.ndarray


def load_meter_series(path: Path) -> MeterSeries:
    timestamps: list[float] = []
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path}: empty file")
        try:
            col_idx = header.index(METER_COLUMN)
        except ValueError as exc:
            raise ValueError(f"{path}: missing column {METER_COLUMN!r}") from exc
        for row in reader:
            if not row or len(row) <= col_idx:
                continue
            try:
                ts = float(row[0])
                val = float(row[col_idx])
            except ValueError:
                continue
            timestamps.append(ts)
            values.append(val)
    if not timestamps:
        raise ValueError(f"{path}: no meter samples")
    order = np.argsort(timestamps)
    ts_arr = np.asarray(timestamps, dtype=float)[order]
    val_arr = np.asarray(values, dtype=float)[order]
    return MeterSeries(timestamps=ts_arr, values=val_arr)


def sample_meter_at_timestamps(
    meter: MeterSeries,
    targets: list[float],
    *,
    max_offset_s: float | None,
) -> list[tuple[float, str, str]]:
    records: list[tuple[float, str, str]] = []
    if meter.timestamps.size == 0 or not targets:
        return records

    idx = meter.timestamps
    vals = meter.values

    for t in targets:
        pos = int(np.searchsorted(idx, t))
        best_pos: int | None = None
        best_dist = float("inf")
        for candidate in (pos - 1, pos):
            if 0 <= candidate < len(idx):
                dist = abs(idx[candidate] - t)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = candidate
        if best_pos is None:
            records.append((t, METER_COLUMN, ""))
            continue
        if max_offset_s is not None and best_dist > max_offset_s:
            records.append((t, METER_COLUMN, ""))
            continue
        records.append((t, METER_COLUMN, str(vals[best_pos])))
    return records


def process_one_time_json(
    *,
    time_json: Path,
    meter: MeterSeries,
    csv_timestamp: str,
    delimiter: str,
    max_offset_s: float | None,
    out_path: Path | None,
) -> int:
    data = load_time_json(str(time_json))
    start, end, step_s = prom_bounds_from_time_json(data)
    targets = iter_range_timestamps(start, end, step_s)
    records = sample_meter_at_timestamps(
        meter, targets, max_offset_s=max_offset_s
    )

    with open_output_stream(out_path) as out:
        write_wide_csv(records, csv_timestamp, delimiter, out)

    if out_path is not None:
        print(f"Wrote {out_path}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Align meter-platform_power to power/time.json prometheus grid "
            "and write wide CSV (same layout as fetch_prometheus_metrics.py)."
        )
    )
    parser.add_argument(
        "--meter-power",
        required=True,
        metavar="PATH",
        help=(
            "Path to extract_meter_power.py output (*-timestamp.csv) or saved_path "
            "without suffix (script appends -timestamp.csv)"
        ),
    )
    parser.add_argument(
        "--time-json",
        metavar="PATH",
        help="Single power/time.json (omit to discover runs under positional paths)",
    )
    parser.add_argument(
        "path",
        nargs="*",
        help="OUTPUT_DIR or experiment root when discovering time.json (use -r)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="search recursively for mlperf_log_detail.txt (same as extract_power_windows)",
    )
    parser.add_argument(
        "--format",
        choices=("csv",),
        default="csv",
        help="output format (only csv supported)",
    )
    parser.add_argument(
        "--csv-timestamp",
        choices=("rfc3339", "unix"),
        default="rfc3339",
        help="First column format (default: rfc3339 UTC)",
    )
    parser.add_argument(
        "--csv-delimiter",
        default=",",
        metavar="CHAR",
        help="Field separator for CSV (use $'\\t' for TSV)",
    )
    parser.add_argument(
        "--max-offset",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Max |meter_ts - grid_ts| for a match; omit for nearest at any offset",
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Write CSV to file instead of stdout. With one --time-json, -o alone "
            f"writes {DEFAULT_OUTPUT_STEM}.csv beside time.json."
        ),
    )
    args = parser.parse_args()

    try:
        meter_path = resolve_meter_csv_path(args.meter_power)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        meter = load_meter_series(meter_path)
    except (OSError, ValueError) as exc:
        print(f"error: {meter_path}: {exc}", file=sys.stderr)
        return 1

    delim = args.csv_delimiter.encode("utf-8").decode("unicode_escape")

    if args.time_json:
        time_json = Path(args.time_json).resolve()
        if not time_json.is_file():
            print(f"error: time.json not found: {time_json}", file=sys.stderr)
            return 1
        try:
            out_path = resolve_csv_output_path(
                time_json=time_json,
                output=args.output,
                default_stem=DEFAULT_OUTPUT_STEM,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        try:
            return process_one_time_json(
                time_json=time_json,
                meter=meter,
                csv_timestamp=args.csv_timestamp,
                delimiter=delim,
                max_offset_s=args.max_offset,
                out_path=out_path,
            )
        except ValueError as exc:
            print(f"error: {time_json}: {exc}", file=sys.stderr)
            return 1

    if not args.path:
        parser.error("provide --time-json or one or more paths to discover time.json")

    roots = [Path(p) for p in args.path]
    for root in roots:
        if not root.exists():
            print(f"error: path does not exist: {root}", file=sys.stderr)
            return 1

    time_jsons = discover_time_json_paths(roots, args.recursive)
    if not time_jsons:
        print("error: no power/time.json found under given paths", file=sys.stderr)
        return 2

    if args.output not in (None, "") and len(time_jsons) > 1:
        print(
            "error: -o PATH with multiple time.json files is ambiguous; "
            "use --time-json for a single run or use -o without PATH",
            file=sys.stderr,
        )
        return 1

    exit_code = 0
    for time_json in time_jsons:
        try:
            if len(time_jsons) == 1:
                out_path = resolve_csv_output_path(
                    time_json=time_json,
                    output=args.output,
                    default_stem=DEFAULT_OUTPUT_STEM,
                )
            else:
                out_path = time_json.parent / f"{DEFAULT_OUTPUT_STEM}.csv"
            code = process_one_time_json(
                time_json=time_json,
                meter=meter,
                csv_timestamp=args.csv_timestamp,
                delimiter=delim,
                max_offset_s=args.max_offset,
                out_path=out_path,
            )
            exit_code = max(exit_code, code)
        except ValueError as exc:
            print(f"error: {time_json}: {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
