#!/usr/bin/env python3
"""Phase 4 - compact many tiny Parquet files into sorted daily files.

    python src/compact.py --input_dir data/layouts/tiny_files \
                          --output_dir data/layouts/compacted \
                          --results_dir results

Steps
-----
1. list the tiny files and assign each to a calendar day using the min(event_ts)
   statistic in its footer (falls back to reading the column);
2. for each day (bounded memory: one day at a time) read its files, sort rows by
   (user_id, event_ts) and write one Snappy Parquet file with a sensible row
   group size (default 122,880 = DuckDB's default);
3. measure wall-clock time and peak RSS of steps 1-2 ONLY;
4. verify: row count and SUM(value) before == after, using DuckDB over the input
   and output files.  SUM(value) is checked twice: as an exact integer number of
   cents (authoritative) and as a float (relative tolerance 1e-9).

Writes ``results/compaction.csv`` with the 7 required columns, plus
``results/compaction_detail.json`` with extra context.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import duckdb
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from common import (
    COMPACTION_CSV_COLUMNS, COMPRESSION, DUCKDB_DEFAULT_ROW_GROUP, EVENTS_SCHEMA,
    current_rss_mb, dir_stats, env_int, env_str, list_parquet_files, self_max_rss_mb,
    setup_logging, write_json,
)

log = setup_logging()


def file_day(path: Path) -> dt.date:
    """Calendar day of a file = day of its smallest event_ts."""
    md = pq.read_metadata(path)
    col = [md.schema.column(i).name for i in range(md.num_columns)].index("event_ts")
    mins = []
    for rg in range(md.num_row_groups):
        st = md.row_group(rg).column(col).statistics
        if st is None or not st.has_min_max:
            mins = []
            break
        mins.append(st.min)
    if not mins:  # no statistics: read the column
        t = pq.read_table(path, columns=["event_ts"])
        return t["event_ts"].to_pylist()[0].date() if t.num_rows else dt.date.min
    return min(mins).date()


def group_by_day(files: list[Path]) -> dict[dt.date, list[Path]]:
    groups: dict[dt.date, list[Path]] = defaultdict(list)
    for f in files:
        groups[file_day(f)].append(f)
    return dict(sorted(groups.items()))


def compact_files(files: list[Path], output_dir: Path,
                  row_group_size: int = DUCKDB_DEFAULT_ROW_GROUP) -> tuple[int, int]:
    """Core transform. Returns (output_files, rows_written)."""
    groups = group_by_day(files)
    log.info("%d input files -> %d day groups", len(files), len(groups))
    out_files = rows = 0
    for i, (day, members) in enumerate(groups.items(), 1):
        table = ds.dataset([str(m) for m in members], format="parquet",
                           schema=EVENTS_SCHEMA).to_table()
        table = table.sort_by([("user_id", "ascending"), ("event_ts", "ascending")])
        out = output_dir / f"events-{day:%Y-%m-%d}.parquet"
        pq.write_table(table, out, compression=COMPRESSION,
                       row_group_size=row_group_size, write_statistics=True)
        rows += table.num_rows
        out_files += 1
        del table
        if i % 10 == 0 or i == len(groups):
            log.info("  compacted %d/%d days (%d rows)", i, len(groups), rows)
    return out_files, rows


def checksum(files_glob: str) -> dict:
    """Row count and value sums over a set of Parquet files."""
    con = duckdb.connect()
    pat = files_glob.replace("'", "''")
    n, s, cents = con.execute(
        "SELECT COUNT(*), COALESCE(SUM(value), 0), "
        "COALESCE(SUM(CAST(ROUND(value * 100) AS HUGEINT)), 0) "
        f"FROM read_parquet('{pat}', hive_partitioning = false)"
    ).fetchone()
    con.close()
    return {"rows": int(n), "sum_value": float(s), "sum_cents": int(cents)}


def sums_match(before: dict, after: dict) -> bool:
    if before["rows"] != after["rows"] or before["sum_cents"] != after["sum_cents"]:
        return False
    scale = max(1.0, abs(before["sum_value"]))
    return abs(before["sum_value"] - after["sum_value"]) <= 1e-9 * scale


def compact(input_dir: Path, output_dir: Path, results_dir: Path,
            row_group_size: int = DUCKDB_DEFAULT_ROW_GROUP,
            append: bool = False) -> dict:
    input_dir, output_dir, results_dir = map(Path, (input_dir, output_dir, results_dir))
    in_res, out_res = input_dir.resolve(), output_dir.resolve()
    if in_res == out_res or in_res in out_res.parents or out_res in in_res.parents:
        raise ValueError("input_dir and output_dir must be separate, non-nested directories")
    files = list_parquet_files(input_dir)
    if not files:
        raise FileNotFoundError(f"no .parquet files under {input_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    baseline_rss = current_rss_mb()
    t0 = time.perf_counter()
    out_files, rows_written = compact_files(files, output_dir, row_group_size)
    wall_s = time.perf_counter() - t0
    peak_rss = self_max_rss_mb()     # process peak up to now = compaction (verification not yet run)

    log.info("verifying row counts and SUM(value) with DuckDB ...")
    before = checksum(str(input_dir / "**" / "*.parquet"))
    after = checksum(str(output_dir / "*.parquet"))
    match = sums_match(before, after)
    in_files, in_bytes = dir_stats(input_dir)
    out_files_n, out_bytes = dir_stats(output_dir)

    row = {
        "input_files": len(files), "output_files": out_files_n,
        "wall_s": round(wall_s, 3), "peak_rss_mb": round(peak_rss, 1),
        "rows_before": before["rows"], "rows_after": after["rows"],
        "checksum_match": match,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "compaction.csv"
    new = not (append and path.exists())
    with open(path, "w" if new else "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COMPACTION_CSV_COLUMNS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(row)
    detail = {
        **row, "row_group_size": row_group_size, "baseline_rss_mb": round(baseline_rss, 1),
        "input_bytes": in_bytes, "output_bytes": out_bytes,
        "sum_value_before": before["sum_value"], "sum_value_after": after["sum_value"],
        "sum_cents_before": before["sum_cents"], "sum_cents_after": after["sum_cents"],
        "rows_per_second": round(rows_written / wall_s, 1) if wall_s else None,
        "sorted_by": ["user_id", "event_ts"],
        "rss_note": "peak_rss_mb is the process peak (ru_maxrss) after compaction, "
                    "before verification; it includes interpreter/library baseline",
    }
    write_json(results_dir / "compaction_detail.json", detail)
    log.info("compaction: %d -> %d files in %.2fs, peak RSS %.0f MiB, checksum_match=%s",
             len(files), out_files_n, wall_s, peak_rss, match)
    return row


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input_dir", default="data/layouts/tiny_files")
    p.add_argument("--output_dir", default="data/layouts/compacted")
    p.add_argument("--results_dir", default=env_str("RESULTS_DIR", "results"))
    p.add_argument("--row_group_size", type=int,
                   default=env_int("COMPACT_ROW_GROUP", DUCKDB_DEFAULT_ROW_GROUP))
    p.add_argument("--append", action="store_true", help="append to compaction.csv")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        row = compact(a.input_dir, a.output_dir, a.results_dir, a.row_group_size, a.append)
    except Exception:
        log.exception("compaction failed")
        return 1
    return 0 if row["checksum_match"] else 3


if __name__ == "__main__":
    sys.exit(main())
