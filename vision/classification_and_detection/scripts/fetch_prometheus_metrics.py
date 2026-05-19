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
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, TextIO

MANIFEST_NAME = "manifest.json"
TIME_JSON_PROM_KEY = "prometheus"
DEFAULT_METRIC_STEM = "metric"


def metric_output_extension(args: argparse.Namespace) -> str:
    if args.raw:
        return ".json"
    if args.format == "csv":
        return ".csv"
    return ".jsonl"


def sanitize_output_stem(text: str, *, max_len: int = 120) -> str:
    stem = re.sub(r"[^\w.\-]+", "_", text.strip())
    stem = stem.strip("._")
    if not stem:
        return DEFAULT_METRIC_STEM
    return stem[:max_len]


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


@contextlib.contextmanager
def open_output_stream(path: Path | None):
    if path is None:
        yield sys.stdout
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        yield f


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


def _escape_label_value(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


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


def read_node_name_from_run_dir(run_dir: Path) -> str | None:
    manifest_path = run_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    node_name = data.get("node_name")
    if node_name is None:
        return None
    text = str(node_name).strip()
    return text or None


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


def label_filters_for_node_name(node_name: str | None) -> list[LabelMatch]:
    """Filters applied when using --time-json (zone=psys; master nodes match node_name label)."""
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
    """Which label keys to embed in CSV column names (None = all non-__name__ labels)."""
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
    """Label keys whose values differ across series in one query result."""
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


def series_column_name(
    labels: dict[str, Any],
    *,
    label_keys: frozenset[str] | None = None,
) -> str:
    """Human-readable column id: metric or metric{label=\"value\",...}."""
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
    """Pivot to wide table: column 0 = timestamp, then one column per series name."""
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
        w.writerow([format_timestamp_cell(ts, timestamp_mode)] + [rowm.get(c, "") for c in col_order])


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
