#!/usr/bin/env python3
"""Phase 2 - materialise the physical Parquet layouts from the raw dataset.

    python src/build_layouts.py --raw_dir data/raw --out_dir data/layouts

Required layouts (all are always built unless ``--layouts`` narrows them):

    tiny_files                  ~1 file per minute
    hourly                      1 file per hour
    daily                       1 file per day, row group 122,880 (DuckDB default)
    daily_rg_10k                daily files, row group 10,000
    daily_rg_1m                 daily files, row group 1,000,000
    partitioned_by_date         Hive: date=YYYY-MM-DD/
    partitioned_by_date_country Hive: date=YYYY-MM-DD/country=XX/
    sorted_by_user              daily files sorted by (user_id, event_ts)
    sorted_by_ts                daily files sorted by event_ts
    partitioned_by_user_id      Hive: user_id=N/   (induced HIGH-CARDINALITY FAILURE)

``--with_extras`` adds two supplementary layouts (``daily_rg_2k`` and
``sorted_by_user_rg_2k``) that make row-group effects visible on small
datasets where one day holds fewer rows than one default row group.

Implementation: raw micro-batches cover increasing, disjoint time ranges, so
rows are streamed once and cut into time buckets (minute/hour/day) with
bounded memory (one bucket at a time).  Compression is always Snappy.
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import (
    COMPRESSION, DUCKDB_DEFAULT_ROW_GROUP, LAYOUT_META_NAME, dir_stats,
    env_int, env_str, read_manifest, self_max_rss_mb, setup_logging, write_json,
)

log = setup_logging()


# --------------------------------------------------------------------------
# Layout registry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LayoutSpec:
    name: str
    kind: str                     # time_files | hive_date | hive_date_country | hive_user
    row_group_rows: int
    bucket_seconds: Optional[int] = None
    sort_by: tuple = ()           # ((col, order), ...)
    partition_columns: tuple = ()
    required: bool = True
    description: str = ""

    @property
    def hive(self) -> bool:
        return bool(self.partition_columns)


DAY, HOUR, MINUTE = 86_400, 3_600, 60
RG = DUCKDB_DEFAULT_ROW_GROUP

LAYOUTS: dict[str, LayoutSpec] = {s.name: s for s in [
    LayoutSpec("tiny_files", "time_files", RG, MINUTE,
               description="one file per minute (small-file problem)"),
    LayoutSpec("hourly", "time_files", RG, HOUR, description="one file per hour"),
    LayoutSpec("daily", "time_files", RG, DAY,
               description="one file per day, default row group size"),
    LayoutSpec("daily_rg_10k", "time_files", 10_000, DAY,
               description="daily files, row group = 10,000 rows"),
    LayoutSpec("daily_rg_1m", "time_files", 1_000_000, DAY,
               description="daily files, row group = 1,000,000 rows"),
    LayoutSpec("partitioned_by_date", "hive_date", RG, DAY,
               partition_columns=("date",),
               description="Hive partitions date=YYYY-MM-DD/"),
    LayoutSpec("partitioned_by_date_country", "hive_date_country", RG, DAY,
               partition_columns=("date", "country"),
               description="nested Hive partitions date=.../country=.../"),
    LayoutSpec("sorted_by_user", "time_files", RG, DAY,
               sort_by=(("user_id", "ascending"), ("event_ts", "ascending")),
               description="daily files sorted by user_id"),
    LayoutSpec("sorted_by_ts", "time_files", RG, DAY,
               sort_by=(("event_ts", "ascending"),),
               description="daily files sorted by event_ts"),
    LayoutSpec("partitioned_by_user_id", "hive_user", RG, None,
               partition_columns=("user_id",),
               description="INDUCED FAILURE: Hive partition per user_id (high cardinality)"),
    # ---- supplementary (opt-in) ----
    LayoutSpec("daily_rg_2k", "time_files", 2_000, DAY, required=False,
               description="[extra] daily files, row group = 2,000 rows"),
    LayoutSpec("sorted_by_user_rg_2k", "time_files", 2_000, DAY, required=False,
               sort_by=(("user_id", "ascending"), ("event_ts", "ascending")),
               description="[extra] daily, sorted by user_id, row group = 2,000"),
]}
REQUIRED_LAYOUTS = [n for n, s in LAYOUTS.items() if s.required]
EXTRA_LAYOUTS = [n for n, s in LAYOUTS.items() if not s.required]


# --------------------------------------------------------------------------
# Streaming helpers
# --------------------------------------------------------------------------
def raw_files(raw_dir: Path) -> list[Path]:
    files = sorted(Path(raw_dir).glob("raw_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no raw_*.parquet files in {raw_dir}")
    return files


def iter_raw_tables(raw_dir: Path) -> Iterator[pa.Table]:
    for f in raw_files(raw_dir):
        yield pq.read_table(f)


def iter_buckets(tables: Iterable[pa.Table], start_us: int, bucket_seconds: int
                 ) -> Iterator[tuple[int, pa.Table]]:
    """Cut a time-ordered stream of micro-batches into fixed time buckets.

    Yields (bucket_id, table) with bucket_id = floor((ts - start)/bucket).
    Row order inside a bucket is the arrival order of the raw data.
    Raises if batches are not in non-decreasing time order.
    """
    bucket_us = bucket_seconds * 1_000_000
    pend_id: Optional[int] = None
    pend: list[pa.Table] = []

    def flush():
        return pend[0] if len(pend) == 1 else pa.concat_tables(pend)

    for t in tables:
        if t.num_rows == 0:
            continue
        ts = t["event_ts"].cast(pa.int64()).to_numpy()
        ids = (ts - start_us) // bucket_us
        order = np.argsort(ids, kind="stable")
        t, ids = t.take(pa.array(order)), ids[order]
        cuts = np.flatnonzero(np.diff(ids)) + 1
        starts = np.concatenate([[0], cuts])
        ends = np.concatenate([cuts, [len(ids)]])
        for s, e in zip(starts, ends):
            bid = int(ids[s])
            seg = t.slice(int(s), int(e - s))
            if pend_id is None:
                pend_id, pend = bid, [seg]
            elif bid == pend_id:
                pend.append(seg)
            elif bid > pend_id:
                yield pend_id, flush()
                pend_id, pend = bid, [seg]
            else:
                raise ValueError(
                    "raw data is not ordered by time at batch granularity "
                    f"(bucket {bid} after {pend_id}); regenerate with generate_data.py")
    if pend:
        yield pend_id, flush()


def write_parquet(table: pa.Table, path: Path, row_group_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression=COMPRESSION,
                   row_group_size=row_group_rows, write_statistics=True)


# --------------------------------------------------------------------------
# Builders  (each returns the number of rows written)
# --------------------------------------------------------------------------
def _bucket_start(start: dt.datetime, bucket_id: int, bucket_seconds: int) -> dt.datetime:
    return start + dt.timedelta(seconds=bucket_id * bucket_seconds)


def build_time_files(spec: LayoutSpec, raw_dir: Path, out: Path, manifest: dict) -> int:
    start = dt.datetime.fromisoformat(manifest["start_ts"])
    rows = n_files = 0
    for bid, table in iter_buckets(iter_raw_tables(raw_dir), manifest["start_us"],
                                   spec.bucket_seconds):
        if spec.sort_by:
            table = table.sort_by(list(spec.sort_by))
        b = _bucket_start(start, bid, spec.bucket_seconds)
        if spec.bucket_seconds == DAY:
            path = out / f"events-{b:%Y-%m-%d}.parquet"
        elif spec.bucket_seconds == HOUR:
            path = out / f"{b:%Y-%m-%d}" / f"events-{b:%H}.parquet"
        else:  # minute buckets live in per-day directories (no key=value => not Hive)
            path = out / f"{b:%Y-%m-%d}" / f"events-{b:%H%M}.parquet"
        write_parquet(table, path, spec.row_group_rows)
        rows += table.num_rows
        n_files += 1
        if n_files % 20_000 == 0:
            log.info("  %s: %d files written", spec.name, n_files)
    return rows


def build_hive_date(spec: LayoutSpec, raw_dir: Path, out: Path, manifest: dict) -> int:
    start = dt.datetime.fromisoformat(manifest["start_ts"])
    rows = 0
    for bid, table in iter_buckets(iter_raw_tables(raw_dir), manifest["start_us"], DAY):
        day = _bucket_start(start, bid, DAY)
        write_parquet(table, out / f"date={day:%Y-%m-%d}" / "part-0.parquet",
                      spec.row_group_rows)
        rows += table.num_rows
    return rows


def build_hive_date_country(spec: LayoutSpec, raw_dir: Path, out: Path,
                            manifest: dict) -> int:
    start = dt.datetime.fromisoformat(manifest["start_ts"])
    rows = n_files = 0
    for bid, table in iter_buckets(iter_raw_tables(raw_dir), manifest["start_us"], DAY):
        day = _bucket_start(start, bid, DAY)
        for c in sorted(pc.unique(table["country"]).to_pylist()):
            part = table.filter(pc.equal(table["country"], c)).drop_columns(["country"])
            write_parquet(part, out / f"date={day:%Y-%m-%d}" / f"country={c}" / "part-0.parquet",
                          spec.row_group_rows)
            rows += part.num_rows
            n_files += 1
    log.info("  %s: %d files written", spec.name, n_files)
    return rows


def build_hive_user(spec: LayoutSpec, raw_dir: Path, out: Path, manifest: dict,
                    max_rows: int, info: dict) -> int:
    """One directory + one file per distinct user_id (the induced failure).

    Needs rows grouped by user, which requires holding them in memory, so the
    build is capped at ``max_rows`` source rows (documented in layout meta:
    ``truncated`` / ``source_rows_used``).  Below the cap the layout is complete.
    """
    pieces, used = [], 0
    for t in iter_raw_tables(raw_dir):
        if used + t.num_rows > max_rows:
            t = t.slice(0, max_rows - used)
        pieces.append(t)
        used += t.num_rows
        if used >= max_rows:
            break
    info["source_rows_used"] = used
    info["truncated"] = used < manifest["rows"]
    if info["truncated"]:
        log.warning("partitioned_by_user_id built from the first %d of %d rows "
                    "(cap --user_partition_max_rows); results are NOT comparable "
                    "to other layouts", used, manifest["rows"])
    table = pa.concat_tables(pieces).sort_by([("user_id", "ascending")])
    pieces.clear()
    uid = table["user_id"].to_numpy()
    cuts = np.flatnonzero(np.diff(uid)) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(uid)]])
    info["distinct_users_partitioned"] = int(len(starts))
    data = table.drop_columns(["user_id"])
    for i, (s, e) in enumerate(zip(starts, ends)):
        write_parquet(data.slice(int(s), int(e - s)),
                      out / f"user_id={int(uid[s])}" / "part-0.parquet",
                      spec.row_group_rows)
        if (i + 1) % 20_000 == 0:
            log.info("  %s: %d user partitions written", spec.name, i + 1)
    return used


BUILDERS: dict[str, Callable] = {
    "time_files": build_time_files,
    "hive_date": build_hive_date,
    "hive_date_country": build_hive_date_country,
}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def build_layout(spec: LayoutSpec, raw_dir: Path, layouts_dir: Path,
                 user_partition_max_rows: int = 5_000_000,
                 tiny_bucket_seconds: Optional[int] = None) -> dict:
    manifest = read_manifest(raw_dir)
    if spec.name == "tiny_files" and tiny_bucket_seconds:
        spec = LayoutSpec(**{**spec.__dict__, "bucket_seconds": tiny_bucket_seconds})
    out = Path(layouts_dir) / spec.name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    log.info("building %-28s %s", spec.name, spec.description)
    t0 = time.perf_counter()
    info: dict = {}
    if spec.kind == "hive_user":
        rows = build_hive_user(spec, raw_dir, out, manifest, user_partition_max_rows, info)
    else:
        rows = BUILDERS[spec.kind](spec, raw_dir, out, manifest)
    elapsed = time.perf_counter() - t0

    expected = info.get("source_rows_used", manifest["rows"])
    if rows != expected:
        raise RuntimeError(f"{spec.name}: wrote {rows} rows, expected {expected}")
    files, total_bytes = dir_stats(out)
    meta = {
        "layout": spec.name, "description": spec.description, "kind": spec.kind,
        "required": spec.required,
        "row_group_rows": spec.row_group_rows, "bucket_seconds": spec.bucket_seconds,
        "sort_by": [list(s) for s in spec.sort_by],
        "hive_partitioning": spec.hive, "partition_columns": list(spec.partition_columns),
        "glob": "**/*.parquet", "files": files, "total_bytes": total_bytes,
        "rows": rows, "source_rows_total": manifest["rows"],
        "truncated": bool(info.get("truncated", False)),
        "build_seconds": round(elapsed, 3), "compression": COMPRESSION,
        **{k: v for k, v in info.items() if k != "truncated"},
    }
    write_json(out / LAYOUT_META_NAME, meta)
    log.info("  -> %s: %d files, %.1f MiB, %.1fs", spec.name, files,
             total_bytes / 1048576, elapsed)
    return meta


def build_all(raw_dir: Path, layouts_dir: Path, names: list[str],
              user_partition_max_rows: int = 5_000_000,
              tiny_bucket_seconds: Optional[int] = None) -> list[dict]:
    unknown = [n for n in names if n not in LAYOUTS]
    if unknown:
        raise ValueError(f"unknown layouts {unknown}; known: {list(LAYOUTS)}")
    return [build_layout(LAYOUTS[n], Path(raw_dir), Path(layouts_dir),
                         user_partition_max_rows, tiny_bucket_seconds)
            for n in names]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw_dir", default=env_str("RAW_DIR", "data/raw"))
    p.add_argument("--out_dir", default=env_str("LAYOUTS_DIR", "data/layouts"))
    p.add_argument("--layouts", default="all",
                   help="comma-separated names, or 'all' (= the 10 required layouts)")
    p.add_argument("--with_extras", action="store_true",
                   default=env_str("WITH_EXTRAS", "0") == "1",
                   help="also build the supplementary *_rg_2k layouts")
    p.add_argument("--user_partition_max_rows", type=int,
                   default=env_int("USER_PARTITION_MAX_ROWS", 5_000_000),
                   help="memory guard for partitioned_by_user_id (see docstring)")
    p.add_argument("--tiny_bucket_seconds", type=int, default=None,
                   help="override the 60 s bucket of tiny_files (testing only)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    names = REQUIRED_LAYOUTS if a.layouts == "all" else a.layouts.split(",")
    if a.with_extras and a.layouts == "all":
        names = names + EXTRA_LAYOUTS
    try:
        build_all(Path(a.raw_dir), Path(a.out_dir), names,
                  a.user_partition_max_rows, a.tiny_bucket_seconds)
    except Exception:
        log.exception("layout build failed")
        return 1
    log.info("peak RSS of builder: %.0f MiB", self_max_rss_mb())
    return 0


if __name__ == "__main__":
    sys.exit(main())
