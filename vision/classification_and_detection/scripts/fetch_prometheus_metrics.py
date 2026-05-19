#!/usr/bin/env python3
"""
Fetch Prometheus series with timestamps via the HTTP API.

Uses only the standard library (no pip packages).

Examples:
  # Instant query (single timestamp per series)
  python3 scripts/fetch_prometheus_metrics.py \\
    --url http://localhost:9090 --query 'up'

  # Multiple instant queries (repeat -q or use --queries-file)
  python3 scripts/fetch_prometheus_metrics.py \\
    --url http://localhost:9090 -q 'up' -q 'process_resident_memory_bytes'

  # All metrics whose __name__ starts with the same prefix (one PromQL regex query)
  python3 scripts/fetch_prometheus_metrics.py \\
    --url http://localhost:9090 --name-prefix kepler_

  # Range query (many [unix_ts, value] pairs per series)
  python3 scripts/fetch_prometheus_metrics.py \\
    --url http://localhost:9090 --query 'rate(http_requests_total[5m])' \\
    --start $(date -d '10 minutes ago' +%s) --end $(date +%s) --step 15s

  # Wide table: first column timestamp, one column per time series (CSV)
  python3 scripts/fetch_prometheus_metrics.py \\
    --url http://localhost:9090 --name-prefix kepler_ \\
    --start ... --end ... --step 15s --format csv

  # Keep series whose label values match a glob (* and ?); repeat for AND
  python3 scripts/fetch_prometheus_metrics.py \\
    --url http://localhost:9090 --name-prefix kepler_ \\
    --label-match 'pod=*mlperf*' --label-match 'container=*' \\
    --start ... --end ... --step 15s --format csv

  # CSV columns: only these label keys (plus minimal differing labels if omitted)
  python3 scripts/fetch_prometheus_metrics.py \\
    ... --format csv --label-key pod --label-key instance

  # Range window + node_name filters (prints to terminal by default)
  python3 scripts/fetch_prometheus_metrics.py \\
    --url http://localhost:9090 --time-json run_1/power/time.json \\
    --name-prefix kepler_

  # Write beside time.json: metric.csv, or {query}.csv when -q is given
  python3 scripts/fetch_prometheus_metrics.py \\
    --time-json run_1/power/time.json -q 'up' -o

  # Custom path (relative → time.json directory)
  python3 scripts/fetch_prometheus_metrics.py \\
    --time-json run_1/power/time.json ... -o custom.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from fetch_common import (
    apply_time_json_to_args,
    csv_column_label_keys,
    filter_rows_by_labels,
    merge_label_filters,
    open_output_stream,
    parse_label_match,
    resolve_output_path,
    series_column_name,
    write_wide_csv,
)


def _get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {e.code} {e.reason}: {detail}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"Request failed: {e}") from e
    return json.loads(body)


def instant_query(base: str, query: str, time_s: str | None) -> dict[str, Any]:
    q = urllib.parse.urlencode({"query": query, **({"time": time_s} if time_s else {})})
    return _get_json(f"{base.rstrip('/')}/api/v1/query?{q}")


def range_query(base: str, query: str, start: str, end: str, step: str) -> dict[str, Any]:
    q = urllib.parse.urlencode({"query": query, "start": start, "end": end, "step": step})
    return _get_json(f"{base.rstrip('/')}/api/v1/query_range?{q}")


def flatten_instant(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if payload.get("status") != "success":
        return out
    for r in payload.get("data", {}).get("result", []):
        metric = r.get("metric", {})
        val = r.get("value")
        if not val or len(val) < 2:
            continue
        ts, v = val[0], val[1]
        out.append({"timestamp_unix": float(ts), "value": v, "metric": metric})
    return out


def flatten_range(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if payload.get("status") != "success":
        return out
    for r in payload.get("data", {}).get("result", []):
        metric = r.get("metric", {})
        for pair in r.get("values", []):
            if not pair or len(pair) < 2:
                continue
            ts, v = pair[0], pair[1]
            out.append({"timestamp_unix": float(ts), "value": v, "metric": metric})
    return out


def load_queries_from_file(path: str) -> list[str]:
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


def promql_for_metric_name_prefix(prefix: str) -> str:
    """Match any series whose metric name (__name__) starts with this literal prefix."""
    escaped = re.escape(prefix)
    return f'{{__name__=~"^{escaped}.*"}}'


def collect_queries(args: argparse.Namespace) -> list[str]:
    queries: list[str] = []
    if args.query:
        queries.extend(args.query)
    if args.queries_file:
        queries.extend(load_queries_from_file(args.queries_file))
    if args.name_prefix:
        for raw in args.name_prefix:
            pfx = raw.strip()
            if not pfx:
                continue
            queries.append(promql_for_metric_name_prefix(pfx))
    return queries


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch Prometheus metrics with timestamps.")
    p.add_argument("--url", default="http://localhost:9090", help="Prometheus base URL")
    p.add_argument(
        "-q",
        "--query",
        action="append",
        default=None,
        metavar="PROMQL",
        help="PromQL expression (repeat for multiple metrics)",
    )
    p.add_argument(
        "--queries-file",
        metavar="PATH",
        help="File with one PromQL per line; empty lines and # comments ignored",
    )
    p.add_argument(
        "--name-prefix",
        action="append",
        default=None,
        metavar="PREFIX",
        help=(
            "Literal metric-name prefix: adds query {__name__=~\"^PREFIX.*\"} "
            "(repeat for multiple prefixes). Escapes regex metacharacters in PREFIX."
        ),
    )
    p.add_argument(
        "--time-json",
        metavar="PATH",
        help=(
            "Use power/time.json from extract_power_windows.py: sets --start/--end/--step "
            "when omitted, and adds label filters (zone=psys; node_name=*node_name* if "
            "node_name contains 'master', else any-label *node_name*)"
        ),
    )
    p.add_argument("--time", help="Evaluation time for instant query (Unix seconds or RFC3339)")
    p.add_argument("--start", help="Range start (Unix seconds or RFC3339)")
    p.add_argument("--end", help="Range end (Unix seconds or RFC3339)")
    p.add_argument("--step", default="15s", help="Range resolution (e.g. 15s, 1m)")
    p.add_argument(
        "--raw",
        action="store_true",
        help="Print full Prometheus JSON (one query: object; multiple: array of {query, response})",
    )
    p.add_argument(
        "--format",
        choices=("jsonl", "csv"),
        default="csv",
        help="csv (default): wide table; jsonl: one JSON object per sample",
    )
    p.add_argument(
        "--csv-timestamp",
        choices=("rfc3339", "unix"),
        default="rfc3339",
        help="First column format when --format csv (default: rfc3339 UTC)",
    )
    p.add_argument(
        "--csv-delimiter",
        default=",",
        metavar="CHAR",
        help="Field separator for CSV (use $'\\t' for TSV)",
    )
    p.add_argument(
        "--csv-all-labels",
        action="store_true",
        help="CSV column names include every label (default: only labels that differ within each query)",
    )
    p.add_argument(
        "--label-match",
        action="append",
        default=None,
        metavar="KEY=PATTERN",
        help=(
            "Keep only series whose label value matches PATTERN (shell glob: * and ?). "
            "KEY=PATTERN matches one label; PATTERN alone matches if any label value fits. "
            "Repeat for AND."
        ),
    )
    p.add_argument(
        "--label-key",
        action="append",
        default=None,
        metavar="KEY",
        help=(
            "CSV/jsonl: only include these label keys in output column names "
            "(intersected with differing labels unless --csv-all-labels)"
        ),
    )
    p.add_argument(
        "-o",
        "--output",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Write to a file instead of stdout (default is terminal). "
            "With --time-json -o (no PATH): {stem}.{csv,jsonl,json} beside time.json; "
            "stem is metric, or a sanitized --query when -q/--query is given. "
            "With --time-json -o PATH: PATH, or time.json directory + PATH if PATH is relative. "
            "-o without PATH requires --time-json."
        ),
    )
    args = p.parse_args()

    try:
        out_path = resolve_output_path(args)
    except ValueError as exc:
        p.error(str(exc))

    label_filters: list[LabelMatch] = []
    if args.time_json:
        try:
            time_json_filters = apply_time_json_to_args(args, args.time_json)
        except ValueError as exc:
            p.error(str(exc))
        label_filters = merge_label_filters(label_filters, time_json_filters)
    if args.label_match:
        for spec in args.label_match:
            try:
                label_filters.append(parse_label_match(spec))
            except ValueError as exc:
                p.error(str(exc))
    include_label_keys: frozenset[str] | None = None
    if args.label_key:
        include_label_keys = frozenset(k.strip() for k in args.label_key if k.strip())
        if not include_label_keys:
            p.error("--label-key: provide at least one non-empty key")
    queries = collect_queries(args)
    if not queries:
        p.error("provide at least one --query / -q, --queries-file, or --name-prefix")
    base = args.url

    is_range = args.start is not None and args.end is not None
    if args.start is not None and args.end is None:
        p.error("--start and --end must be given together for a range query")
    if args.end is not None and args.start is None:
        p.error("--start and --end must be given together for a range query")

    raw_results: list[dict[str, Any]] = []
    exit_code = 0
    csv_records: list[tuple[float, str, str]] = []
    use_csv = not args.raw and args.format == "csv"
    delim = args.csv_delimiter.encode("utf-8").decode("unicode_escape")

    with open_output_stream(out_path) as out:
        for promql in queries:
            if is_range:
                payload = range_query(base, promql, args.start, args.end, args.step)
            else:
                payload = instant_query(base, promql, args.time)

            if args.raw:
                raw_results.append({"query": promql, "response": payload})
                continue

            if payload.get("status") != "success":
                err_obj = {"query": promql, "error": payload}
                if use_csv:
                    print(json.dumps(err_obj, ensure_ascii=False), file=sys.stderr)
                else:
                    json.dump(err_obj, out, indent=2, ensure_ascii=False)
                    out.write("\n")
                exit_code = 1
                continue

            rows = flatten_range(payload) if is_range else flatten_instant(payload)
            rows = filter_rows_by_labels(rows, label_filters)
            if use_csv:
                metrics = [row["metric"] for row in rows]
                col_keys = csv_column_label_keys(
                    metrics,
                    include_keys=include_label_keys,
                    all_labels=args.csv_all_labels,
                )
                for row in rows:
                    col = series_column_name(row["metric"], label_keys=col_keys)
                    csv_records.append((row["timestamp_unix"], col, str(row["value"])))
                continue

            for row in rows:
                line = {
                    "promql": promql,
                    "timestamp_unix": row["timestamp_unix"],
                    "value": row["value"],
                    "labels": row["metric"],
                }
                json.dump(line, out, ensure_ascii=False)
                out.write("\n")

        if use_csv:
            write_wide_csv(csv_records, args.csv_timestamp, delim, out)

        if args.raw:
            if len(raw_results) == 1:
                json.dump(raw_results[0]["response"], out, indent=2, ensure_ascii=False)
            else:
                json.dump(raw_results, out, indent=2, ensure_ascii=False)
            out.write("\n")
            for item in raw_results:
                if item["response"].get("status") != "success":
                    exit_code = 1
                    break

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
