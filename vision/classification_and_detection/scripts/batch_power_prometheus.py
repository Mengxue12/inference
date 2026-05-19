#!/usr/bin/env python3
"""
Extract power windows (-o) then fetch Prometheus metrics for each run.

Input paths match extract_power_windows.py (run_*, accuracy/, experiment root with -r).
Run directories are discovered via the same _iter_detail_log_paths rules as extract.
Always writes power/time.json when power pairs exist (same skip/overwrite rules as -o).
Then, for each discovered OUTPUT_DIR, fetches kepler_node_cpu_joules_total
(cumulative counter) into power/<metric>.csv beside time.json.

Skips Prometheus fetch when the CSV exists and is newer than time.json.
Prometheus requests run serially. Failures list the failing time.json path.

Examples:
  python3 scripts/batch_power_prometheus.py /path/to/experiment --recursive \\
    --url http://localhost:9090
  python3 scripts/batch_power_prometheus.py /path/to/output/run_1 --url http://prom:9090
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from extract_power_windows import (
    DEFAULT_PROM_STEP,
    POWER_SUBDIR,
    TIME_JSON_NAME,
    _expected_detail_log,
    _print_no_detail_log,
    _print_no_power_pairs,
    _resolve_tz,
    _roots_label,
    collect_by_run_dir,
    iter_run_dirs,
    time_json_path,
    write_time_json,
)
from fetch_prometheus_metrics import sanitize_output_stem

DEFAULT_METRIC = "kepler_node_cpu_joules_total"
FETCH_SCRIPT = Path(__file__).resolve().parent / "fetch_prometheus_metrics.py"


def run_extract_phase(
    *,
    roots: list[Path],
    recursive: bool,
    by_run: dict,
    tz,
    step: str,
    overwrite: bool,
) -> tuple[list[Path], int]:
    """Write power/time.json (-o). Returns (written paths, extract exit hint)."""
    written: list[Path] = []
    total_windows = sum(len(rc.windows) for rc in by_run.values())

    if not by_run:
        _print_no_detail_log(
            roots, recursive, sys.stderr, suffix="No time.json written."
        )
        return written, 2

    for run_dir in sorted(by_run):
        rc = by_run[run_dir]
        windows = rc.windows
        out_path = time_json_path(run_dir)
        if not windows:
            in_log = ", ".join(rc.detail_logs) or str(_expected_detail_log(run_dir))
            print(
                f"warning: no power_begin / power_end pairs found under "
                f"{run_dir} in {in_log}; skipping {POWER_SUBDIR}/{TIME_JSON_NAME}",
                file=sys.stderr,
            )
            continue
        if out_path.is_file() and not overwrite:
            print(
                f"warning: {out_path} already exists; skipping (not overwritten)",
                file=sys.stderr,
            )
            continue
        try:
            out_path = write_time_json(run_dir, windows, tz=tz, step=step)
        except ValueError as exc:
            print(f"warning: {run_dir}: {exc}", file=sys.stderr)
            continue
        written.append(out_path)
        print(f"Wrote {out_path}", file=sys.stderr)

    if not written:
        if total_windows == 0:
            _print_no_power_pairs(
                roots, by_run, sys.stderr, suffix="No time.json written."
            )
        else:
            skip_reason = (
                "timestamp parse failed"
                if overwrite
                else "existing files skipped or timestamp parse failed"
            )
            print(
                f"warning: no time.json written under {_roots_label(roots)} "
                f"({skip_reason})",
                file=sys.stderr,
            )
        return written, 2
    return written, 0


def csv_path_for_metric(time_json: Path, metric: str) -> Path:
    return time_json.parent / f"{sanitize_output_stem(metric)}.csv"


def fetch_prometheus_csv(
    *,
    url: str,
    time_json: Path,
    metric: str,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(FETCH_SCRIPT),
        "--url",
        url,
        "--time-json",
        str(time_json),
        "-q",
        metric,
        "--format",
        "csv",
        "-o",
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )


def run_prometheus_phase(
    *,
    output_dirs: list[Path],
    url: str,
    metric: str,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Returns (ok_csv_paths, skipped_csv_paths, failed_time_json_paths)."""
    ok: list[Path] = []
    skipped: list[Path] = []
    failed: list[Path] = []

    for run_dir in output_dirs:
        power_dir = run_dir / POWER_SUBDIR
        time_json = power_dir / TIME_JSON_NAME
        if not power_dir.is_dir():
            print(f"warning: {run_dir}: no {POWER_SUBDIR}/ directory", file=sys.stderr)
            continue
        if not time_json.is_file():
            print(
                f"warning: {run_dir}: {POWER_SUBDIR}/ exists but no {TIME_JSON_NAME}",
                file=sys.stderr,
            )
            continue

        csv_path = csv_path_for_metric(time_json, metric)
        if csv_path.is_file():
            if csv_path.stat().st_mtime >= time_json.stat().st_mtime:
                print(f"skip (csv newer): {csv_path}", file=sys.stderr)
                skipped.append(csv_path)
                continue

        result = fetch_prometheus_csv(url=url, time_json=time_json, metric=metric)
        if result.returncode != 0:
            print(
                f"error: Prometheus fetch failed for {time_json} (exit {result.returncode})",
                file=sys.stderr,
            )
            if result.stderr:
                print(result.stderr.rstrip(), file=sys.stderr)
            failed.append(time_json)
            continue

        if not csv_path.is_file():
            print(
                f"error: Prometheus fetch succeeded but missing output {csv_path} "
                f"(time.json: {time_json})",
                file=sys.stderr,
            )
            failed.append(time_json)
            continue

        print(f"Wrote {csv_path}", file=sys.stderr)
        ok.append(csv_path)

    return ok, skipped, failed


def _print_fetch_summary(
    ok: list[Path],
    skipped: list[Path],
    failed: list[Path],
) -> None:
    print("\nPrometheus fetch summary:", file=sys.stderr)
    print(f"  wrote: {len(ok)}", file=sys.stderr)
    print(f"  skipped (csv newer than time.json): {len(skipped)}", file=sys.stderr)
    print(f"  failed: {len(failed)}", file=sys.stderr)
    for path in failed:
        print(f"    {path}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract power/time.json (-o) then fetch kepler_node_cpu_joules_total "
            "from Prometheus for each OUTPUT_DIR found via mlperf_log_detail.txt."
        )
    )
    parser.add_argument(
        "path",
        nargs="+",
        help="OUTPUT_DIR (run_*), accuracy/, or experiment root (use -r)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="search recursively for mlperf_log_detail.txt under each path (same as extract)",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:9090",
        help="Prometheus base URL (default: http://localhost:9090)",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"PromQL metric name (default: {DEFAULT_METRIC})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing power/time.json (default: skip existing)",
    )
    parser.add_argument(
        "--step",
        default=DEFAULT_PROM_STEP,
        help=f"Prometheus range step in time.json (default: {DEFAULT_PROM_STEP})",
    )
    parser.add_argument(
        "--tz",
        default="UTC",
        help="IANA timezone for MLPerf wall-clock strings (default: UTC; use 'local')",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="only write time.json, do not query Prometheus",
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="only query Prometheus (skip extract phase)",
    )
    args = parser.parse_args()

    if args.extract_only and args.fetch_only:
        print("error: --extract-only and --fetch-only are mutually exclusive", file=sys.stderr)
        return 1

    if not FETCH_SCRIPT.is_file():
        print(f"error: missing {FETCH_SCRIPT}", file=sys.stderr)
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

    extract_status = 0
    if not args.fetch_only:
        by_run = collect_by_run_dir(roots, args.recursive)
        _, extract_status = run_extract_phase(
            roots=roots,
            recursive=args.recursive,
            by_run=by_run,
            tz=tz,
            step=args.step,
            overwrite=args.overwrite,
        )

    if args.extract_only:
        return extract_status

    output_dirs = iter_run_dirs(roots, args.recursive)
    if not output_dirs:
        _print_no_detail_log(
            roots,
            args.recursive,
            sys.stderr,
            suffix="No Prometheus fetch attempted.",
        )

    ok, skipped, failed = run_prometheus_phase(
        output_dirs=output_dirs,
        url=args.url,
        metric=args.metric,
    )
    _print_fetch_summary(ok, skipped, failed)

    if failed:
        return 1
    if not ok and not skipped:
        if extract_status == 2:
            return 2
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
