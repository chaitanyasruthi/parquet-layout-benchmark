#!/usr/bin/env python3
"""Phase 3 - run the query suite against every layout, cold and warm.

    python src/run_queries.py --layouts_dir data/layouts --raw_dir data/raw \
                              --results_dir results

For every (layout, query):
  * 5 COLD runs - each on a brand-new DuckDB connection (empty DuckDB caches);
    before each one the OS page cache for the layout's files is evicted on a
    best-effort basis (posix_fadvise DONTNEED).
  * 5 WARM runs - one shared connection, preceded by one un-recorded priming
    execution, so DuckDB's and the OS caches are populated.

Each run executes with DuckDB JSON profiling enabled (the machine-readable form
of EXPLAIN ANALYZE).  The operator tree is walked recursively by
``profiling.extract_profile_metrics``.

Outputs
-------
results/layout.csv          one row per run, exactly the 11 required columns
results/layout_detail.csv   same runs + extra profile fields (files read, pushdown, ...)
results/run_info.json       environment / parameters of this benchmark run
results/profiles/...        raw DuckDB profile JSON of every run
results/explain/...         (optional, --save_explain) EXPLAIN ANALYZE text
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import duckdb

from common import (
    LAYOUT_CSV_COLUMNS, RssSampler, env_int, env_str, evict_page_cache,
    list_parquet_files, read_layout_meta, read_manifest, setup_logging, write_json,
)
from profiling import extract_profile_metrics, load_profile
from zonemap import FooterIndex, Predicate

log = setup_logging()

DETAIL_COLUMNS = [
    "layout", "query", "cache", "run", "latency_ms", "profile_latency_ms",
    "bytes_read", "rows_scanned", "rows_after_pushdown", "result_rows",
    "result_digest", "files_read", "files_total", "files_scanned_hint",
    "scan_operators", "filter_pushdown", "projection_columns", "filters",
    "row_groups_total", "row_groups_pruned", "row_groups_source",
    "files_pruned_by_path", "peak_rss_mb", "peak_buffer_mb",
    "pages_evicted_files", "layout_truncated", "error",
]


# --------------------------------------------------------------------------
# Query definitions
# --------------------------------------------------------------------------
@dataclass
class Query:
    id: str
    sql: str
    predicates: list = field(default_factory=list)
    required_partition: Optional[str] = None     # only run on layouts with this hive key
    supplementary: bool = False


def _ts(d: dt.datetime) -> str:
    return d.isoformat(sep=" ", timespec="microseconds")


def make_queries(manifest: dict, user_id: Optional[int] = None,
                 day_index: Optional[int] = None) -> list[Query]:
    start = dt.datetime.fromisoformat(manifest["start_ts"])
    days = manifest["days"]
    day_index = days // 2 if day_index is None else day_index
    if not 0 <= day_index < days:
        raise ValueError(f"day_index must be in [0, {days})")
    x = start + dt.timedelta(days=day_index)                       # X
    y = x + dt.timedelta(days=1) - dt.timedelta(microseconds=1)    # Y
    uid = int(manifest["sample_user_id"] if user_id is None else user_id)
    between = f"event_ts BETWEEN TIMESTAMP '{_ts(x)}' AND TIMESTAMP '{_ts(y)}'"
    range_pred = [Predicate("event_ts", "between", x, y)]
    return [
        Query("q1_point_lookup", f"SELECT * FROM events WHERE user_id = {uid}",
              [Predicate("user_id", "eq", uid)]),
        Query("q2_one_day_range", f"SELECT COUNT(*) FROM events WHERE {between}",
              range_pred),
        Query("q3_country_filter",
              "SELECT AVG(value) FROM events WHERE country = 'US'",
              [Predicate("country", "eq", "US")]),
        Query("q4_full_aggregation",
              "SELECT event_type, COUNT(*) FROM events GROUP BY event_type", []),
        Query("q5_complex",
              f"SELECT user_id, SUM(value) FROM events WHERE {between} "
              "GROUP BY user_id ORDER BY SUM(value) DESC LIMIT 10", range_pred),
        # Supplementary: the same one-day count but ALSO filtering the Hive
        # partition key, which is how date partitioning is meant to be used.
        Query("q2b_range_plus_partition_key",
              f"SELECT COUNT(*) FROM events WHERE {between} AND date = DATE '{x:%Y-%m-%d}'",
              range_pred + [Predicate("date", "eq", x.date())],
              required_partition="date", supplementary=True),
    ]


# --------------------------------------------------------------------------
# DuckDB plumbing
# --------------------------------------------------------------------------
def open_connection(layout_dir: Path, meta: dict, threads: Optional[int]
                    ) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    if threads:
        con.execute(f"SET threads = {int(threads)}")
    pattern = str(Path(layout_dir) / meta["glob"]).replace("'", "''")
    hive = "true" if meta["hive_partitioning"] else "false"
    con.execute(f"CREATE VIEW events AS SELECT * FROM "
                f"read_parquet('{pattern}', hive_partitioning = {hive})")
    return con


def result_digest(con, rows) -> str:
    """Order-insensitive digest of a result set (floats rounded to 6 dp)."""
    names = [d[0] for d in con.description]
    order = ["event_ts", "user_id", "country", "event_type", "device", "value"]
    if all(c in names for c in order):  # SELECT * - drop Hive extras, fix order
        idx = [names.index(c) for c in order]
        rows = [tuple(r[i] for i in idx) for r in rows]
    norm = sorted(
        tuple(round(v, 6) if isinstance(v, float) else v for v in r) for r in rows
    )
    return hashlib.sha1(repr(norm).encode()).hexdigest()[:12]


def execute_profiled(con, sql: str, profile_path: Path) -> dict:
    """Run one query with JSON profiling; return wall time, RSS, profile."""
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.unlink(missing_ok=True)
    con.execute("PRAGMA enable_profiling = 'json'")
    con.execute(f"PRAGMA profiling_output = '{profile_path}'")
    with RssSampler() as rss:
        t0 = time.perf_counter()
        rows = con.execute(sql).fetchall()
        wall = time.perf_counter() - t0
    digest = result_digest(con, rows)
    metrics = None
    err = None
    try:
        prof = load_profile(profile_path)
        metrics = extract_profile_metrics(prof)
    except Exception as exc:  # profile missing / malformed: keep the timing
        err = f"profile unavailable: {exc}"
        log.warning(err)
    return {"wall_s": wall, "rss_mb": rss.peak_mb, "rows": len(rows),
            "digest": digest, "metrics": metrics, "error": err}


# --------------------------------------------------------------------------
# Benchmark of one layout
# --------------------------------------------------------------------------
def fmt(v, nd=3):
    return "" if v is None else (round(v, nd) if isinstance(v, float) else v)


def benchmark_layout(name: str, layouts_dir: Path, queries: list[Query],
                     results_dir: Path, cold_runs: int, warm_runs: int,
                     threads: Optional[int], evict: bool, save_profiles: bool,
                     save_explain: bool, want_rg_stats: bool
                     ) -> tuple[list[dict], list[dict], int]:
    layout_dir = Path(layouts_dir) / name
    meta = read_layout_meta(layout_dir)
    files = list_parquet_files(layout_dir)
    if not files:
        raise RuntimeError(f"layout {name} has no parquet files")
    prof_dir = Path(results_dir) / "profiles" / name
    if not save_profiles:
        prof_dir = Path("/tmp") / f"plb_profiles_{os.getpid()}" / name

    index = None
    if want_rg_stats:
        t0 = time.perf_counter()
        index = FooterIndex(layout_dir, files)
        log.info("  footer index for %s: %d files, %d row groups (%.1fs)",
                 name, len(files), index.total_row_groups, time.perf_counter() - t0)

    layout_rows, detail_rows, failures = [], [], 0
    hive_keys = set(meta.get("partition_columns", []))
    for q in queries:
        if q.required_partition and q.required_partition not in hive_keys:
            continue
        rg = index.evaluate(q.predicates) if index else {}
        plan = [("cold", i) for i in range(1, cold_runs + 1)] + \
               [("warm", i) for i in range(1, warm_runs + 1)]
        warm_con = None
        for cache, run in plan:
            evicted = ""
            try:
                if cache == "cold":
                    if evict:
                        evicted = evict_page_cache(files)
                    con = open_connection(layout_dir, meta, threads)   # fresh connection
                else:
                    if warm_con is None:
                        warm_con = open_connection(layout_dir, meta, threads)
                        warm_con.execute(q.sql).fetchall()              # un-recorded priming run
                    con = warm_con
                out = execute_profiled(
                    con, q.sql, prof_dir / f"{q.id}__{cache}{run}.json")
                if cache == "cold":
                    con.close()
            except Exception as exc:
                failures += 1
                log.error("FAILED %s/%s/%s#%d: %s", name, q.id, cache, run, exc)
                detail_rows.append({"layout": name, "query": q.id, "cache": cache,
                                    "run": run, "error": str(exc)})
                continue
            m = out["metrics"] or {}
            row = {
                "layout": name, "row_group_rows": meta["row_group_rows"],
                "files": meta["files"], "total_bytes": meta["total_bytes"],
                "query": q.id, "cache": cache,
                "latency_ms": fmt(out["wall_s"] * 1000, 3),
                "bytes_read": fmt(m.get("bytes_read")),
                "row_groups_total": fmt(rg.get("row_groups_total")),
                "row_groups_pruned": fmt(rg.get("row_groups_pruned")),
                "peak_rss_mb": fmt(out["rss_mb"], 1),
            }
            layout_rows.append(row)
            detail_rows.append({
                "layout": name, "query": q.id, "cache": cache, "run": run,
                "latency_ms": row["latency_ms"],
                "profile_latency_ms": fmt(None if m.get("latency_s") is None
                                          else m["latency_s"] * 1000, 3),
                "bytes_read": row["bytes_read"],
                "rows_scanned": fmt(m.get("rows_scanned")),
                "rows_after_pushdown": fmt(m.get("rows_after_pushdown")),
                "result_rows": out["rows"], "result_digest": out["digest"],
                "files_read": fmt(m.get("files_read")),
                "files_total": fmt(m.get("files_total")),
                "files_scanned_hint": fmt(m.get("files_scanned_hint")),
                "scan_operators": fmt(m.get("scan_operators")),
                "filter_pushdown": fmt(m.get("filter_pushdown")),
                "projection_columns": fmt(m.get("projection_columns")),
                "filters": " | ".join(m.get("filters", [])),
                "row_groups_total": row["row_groups_total"],
                "row_groups_pruned": row["row_groups_pruned"],
                "row_groups_source": "footer_min_max_derived" if rg else "",
                "files_pruned_by_path": fmt(rg.get("files_pruned_by_path")),
                "peak_rss_mb": row["peak_rss_mb"],
                "peak_buffer_mb": fmt(m.get("peak_buffer_mb"), 2),
                "pages_evicted_files": evicted,
                "layout_truncated": meta.get("truncated", False),
                "error": out["error"] or "",
            })
        if warm_con is not None:
            warm_con.close()
        med = {c: statistics.median(float(r["latency_ms"]) for r in layout_rows
                                    if r["query"] == q.id and r["cache"] == c)
               for c in ("cold", "warm")
               if any(r["query"] == q.id and r["cache"] == c for r in layout_rows)}
        log.info("  %-28s %-30s median ms %s", name, q.id,
                 {k: round(v, 1) for k, v in med.items()})
        if save_explain:
            _save_explain(layout_dir, meta, q, Path(results_dir) / "explain" / name, threads)
    return layout_rows, detail_rows, failures


def _save_explain(layout_dir, meta, q, out_dir, threads):
    try:
        con = open_connection(layout_dir, meta, threads)
        rows = con.execute("EXPLAIN ANALYZE " + q.sql).fetchall()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{q.id}.txt").write_text(
            q.sql + "\n\n" + "\n".join(str(c) for r in rows for c in r[1:]))
        con.close()
    except Exception as exc:
        log.warning("EXPLAIN ANALYZE failed for %s: %s", q.id, exc)


# --------------------------------------------------------------------------
# CSV + CLI
# --------------------------------------------------------------------------
def write_csv(path: Path, columns: list[str], rows: list[dict], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not (append and path.exists())
    with open(path, "w" if new else "a", newline="") as fh:  # noqa
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        if new:
            w.writeheader()
        w.writerows(rows)


def run_info(args, manifest, queries) -> dict:
    mem = None
    try:
        import psutil
        mem = round(psutil.virtual_memory().total / 2**30, 2)
    except Exception:
        pass
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "duckdb_version": duckdb.__version__,
        "pyarrow_version": __import__("pyarrow").__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "ram_gib": mem,
        "threads_setting": args.threads or "duckdb default",
        "cold_runs": args.cold_runs, "warm_runs": args.warm_runs,
        "page_cache_eviction": "posix_fadvise(DONTNEED), best effort" if not args.no_evict_cache else "disabled",
        "dataset_rows": manifest["rows"], "seed": manifest["seed"],
        "queries": {q.id: q.sql for q in queries},
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--layouts_dir", default=env_str("LAYOUTS_DIR", "data/layouts"))
    p.add_argument("--raw_dir", default=env_str("RAW_DIR", "data/raw"))
    p.add_argument("--results_dir", default=env_str("RESULTS_DIR", "results"))
    p.add_argument("--layouts", default="auto",
                   help="comma list, or 'auto' = every directory with layout_meta.json")
    p.add_argument("--queries", default="all", help="comma list of query ids or 'all'")
    p.add_argument("--cold_runs", type=int, default=env_int("COLD_RUNS", 5))
    p.add_argument("--warm_runs", type=int, default=env_int("WARM_RUNS", 5))
    p.add_argument("--threads", type=int, default=env_int("DUCKDB_THREADS", 0) or None)
    p.add_argument("--user_id", type=int, default=None, help="override Q1 user_id")
    p.add_argument("--day_index", type=int, default=None, help="day used by Q2/Q5 (0-based)")
    p.add_argument("--no_evict_cache", action="store_true")
    p.add_argument("--no_save_profiles", action="store_true")
    p.add_argument("--save_explain", action="store_true",
                   help="also store EXPLAIN ANALYZE text per layout/query")
    p.add_argument("--skip_row_group_stats", action="store_true",
                   help="do not read Parquet footers (row_groups_* columns stay empty)")
    p.add_argument("--append", action="store_true", help="append to existing CSVs")
    return p.parse_args(argv)


def discover_layouts(layouts_dir: Path) -> list[str]:
    from build_layouts import LAYOUTS
    found = [d.name for d in Path(layouts_dir).iterdir()
             if (d / "layout_meta.json").exists()]
    order = {n: i for i, n in enumerate(LAYOUTS)}
    return sorted(found, key=lambda n: order.get(n, 999))


def run(args) -> int:
    manifest = read_manifest(Path(args.raw_dir))
    queries = make_queries(manifest, args.user_id, args.day_index)
    if args.queries != "all":
        wanted = set(args.queries.split(","))
        unknown = wanted - {q.id for q in queries}
        if unknown:
            raise ValueError(f"unknown queries {sorted(unknown)}")
        queries = [q for q in queries if q.id in wanted]
    names = discover_layouts(Path(args.layouts_dir)) if args.layouts == "auto" \
        else args.layouts.split(",")
    if not names:
        raise RuntimeError(f"no layouts found in {args.layouts_dir}; run build_layouts.py")
    results_dir = Path(args.results_dir)
    write_json(results_dir / "run_info.json", run_info(args, manifest, queries))

    total_fail = 0
    first = True
    for name in names:
        log.info("layout %s", name)
        rows, detail, fails = benchmark_layout(
            name, Path(args.layouts_dir), queries, results_dir, args.cold_runs,
            args.warm_runs, args.threads, not args.no_evict_cache,
            not args.no_save_profiles, args.save_explain,
            not args.skip_row_group_stats)
        total_fail += fails
        append = args.append or not first
        write_csv(results_dir / "layout.csv", LAYOUT_CSV_COLUMNS, rows, append)
        write_csv(results_dir / "layout_detail.csv", DETAIL_COLUMNS, detail, append)
        first = False
    if total_fail:
        log.error("%d run(s) failed - see layout_detail.csv 'error' column", total_fail)
        return 2
    log.info("wrote %s", results_dir / "layout.csv")
    return 0


def main(argv=None) -> int:
    try:
        return run(parse_args(argv))
    except Exception:
        log.exception("query benchmark failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
