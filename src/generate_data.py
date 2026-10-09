#!/usr/bin/env python3
"""Phase 1 - deterministic, batched generation of the `events` dataset.

    python src/generate_data.py --rows 1000000 --out_dir data/raw --seed 42

Design notes
------------
* Memory is bounded by ``--batch_rows`` (default 100k rows); the full dataset
  is never materialised.
* Batch ``i`` owns a *disjoint, increasing* slice of the 90-day time span, so
  the raw files are ordered by time at batch granularity (like an ingestion
  pipeline writing micro-batches). Inside a batch the rows are shuffled, i.e.
  the raw data is NOT sorted - `sorted_by_ts` / `sorted_by_user` therefore
  differ physically from `daily`.
* Every batch uses its own RNG derived from ``SeedSequence([seed, batch_idx])``
  so output is byte-for-byte reproducible for a given (seed, rows, batch_rows)
  and independent of how many batches are generated before it.
* ``value`` is a whole number of cents / 100. The manifest stores the exact
  integer cent total so later stages can verify checksums without floating
  point ambiguity.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import (
    COMPRESSION, DEFAULT_DAYS, DEFAULT_ROWS, DEFAULT_SEED, DEFAULT_START_DATE,
    DUCKDB_DEFAULT_ROW_GROUP, EVENTS_SCHEMA, FULL_SCALE_ROWS, MANIFEST_NAME,
    env_int, env_str, setup_logging, write_json,
)

log = setup_logging()

COUNTRIES = ["US", "IN", "BR", "GB", "DE", "FR", "JP", "CN", "CA", "AU",
             "MX", "ES", "IT", "NL", "SE", "KR", "RU", "ZA", "NG", "ID"]
COUNTRY_W = np.array([30, 14, 7, 6, 5, 4, 4, 4, 3, 3,
                      3, 3, 3, 2, 2, 2, 2, 1, 1, 1], dtype=float)
COUNTRY_W /= COUNTRY_W.sum()
EVENT_TYPES = ["page_view", "click", "scroll", "search",
               "add_to_cart", "purchase", "login", "logout"]
EVENT_W = np.array([40, 25, 12, 8, 6, 3, 4, 2], dtype=float)
EVENT_W /= EVENT_W.sum()
DEVICES = ["mobile", "desktop", "tablet", "tv"]
DEVICE_W = np.array([55, 35, 8, 2], dtype=float)
DEVICE_W /= DEVICE_W.sum()

_US_PER_DAY = 86_400 * 1_000_000


def default_n_users(rows: int) -> int:
    """~20 events per user on average, clamped to a sane range."""
    return int(min(max(rows // 20, 1_000), 5_000_000))


def generate_batch(seed: int, batch_idx: int, row_offset: int, n: int,
                   total_rows: int, start_us: int, span_us: int,
                   n_users: int) -> tuple[pa.Table, int]:
    """Create one batch. Returns (table, sum_of_value_in_cents)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, batch_idx]))
    # This batch covers the half-open slice [lo, hi) of the time span.
    lo = start_us + (row_offset * span_us) // total_rows
    hi = start_us + ((row_offset + n) * span_us) // total_rows
    hi = max(hi, lo + 1)
    ts = rng.integers(lo, hi, size=n, dtype=np.int64)
    user = rng.integers(0, n_users, size=n, dtype=np.int64)
    country = np.array(COUNTRIES)[rng.choice(len(COUNTRIES), size=n, p=COUNTRY_W)]
    etype = np.array(EVENT_TYPES)[rng.choice(len(EVENT_TYPES), size=n, p=EVENT_W)]
    device = np.array(DEVICES)[rng.choice(len(DEVICES), size=n, p=DEVICE_W)]
    cents = np.rint(rng.lognormal(mean=2.5, sigma=1.0, size=n) * 100).astype(np.int64)
    value = cents / 100.0
    # rows arrive in arbitrary order inside a micro-batch
    perm = rng.permutation(n)
    table = pa.table(
        {
            "event_ts": pa.array(ts[perm], type=pa.timestamp("us")),
            "user_id": pa.array(user[perm]),
            "country": pa.array(country[perm]),
            "event_type": pa.array(etype[perm]),
            "device": pa.array(device[perm]),
            "value": pa.array(value[perm]),
        },
        schema=EVENTS_SCHEMA,
    )
    return table, int(cents.sum())


def generate(rows: int, out_dir: Path, seed: int = DEFAULT_SEED,
             batch_rows: int = 100_000, days: int = DEFAULT_DAYS,
             start_date: str = DEFAULT_START_DATE, n_users: int | None = None,
             row_group_size: int = DUCKDB_DEFAULT_ROW_GROUP) -> dict:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale output so reruns are idempotent.
    for old in out_dir.glob("raw_*.parquet"):
        old.unlink()
    (out_dir / MANIFEST_NAME).unlink(missing_ok=True)

    start = dt.datetime.fromisoformat(start_date)
    start_us = int((start - dt.datetime(1970, 1, 1)).total_seconds()) * 1_000_000
    span_us = days * _US_PER_DAY
    n_users = n_users or default_n_users(rows)

    n_batches = -(-rows // batch_rows)
    total_cents = 0
    sample_user = None
    files = []
    min_ts = max_ts = None
    for b in range(n_batches):
        offset = b * batch_rows
        n = min(batch_rows, rows - offset)
        table, cents = generate_batch(seed, b, offset, n, rows, start_us, span_us, n_users)
        total_cents += cents
        if sample_user is None:
            # a user id that certainly exists in the data (used by Q1)
            sample_user = int(table["user_id"][n // 2].as_py())
        path = out_dir / f"raw_{b:06d}.parquet"
        pq.write_table(table, path, compression=COMPRESSION,
                       row_group_size=row_group_size)
        files.append(path.name)
        ts_np = table["event_ts"].cast(pa.int64()).to_numpy()
        bmin, bmax = int(ts_np.min()), int(ts_np.max())
        min_ts = bmin if min_ts is None else min(min_ts, bmin)
        max_ts = bmax if max_ts is None else max(max_ts, bmax)
        if b % 20 == 0 or b == n_batches - 1:
            log.info("batch %d/%d written (%d rows)", b + 1, n_batches, offset + n)

    def us_to_iso(us: int) -> str:
        return (dt.datetime(1970, 1, 1) + dt.timedelta(microseconds=us)).isoformat(sep=" ")

    manifest = {
        "rows": rows, "seed": seed, "days": days, "start_date": start_date,
        "start_ts": us_to_iso(start_us), "start_us": start_us,
        "end_ts_exclusive": us_to_iso(start_us + span_us),
        "min_event_ts": us_to_iso(min_ts), "max_event_ts": us_to_iso(max_ts),
        "n_users": n_users, "batch_rows": batch_rows, "n_batches": n_batches,
        "files": files, "row_group_size": row_group_size,
        "compression": COMPRESSION, "sample_user_id": sample_user,
        "value_sum_cents": total_cents,
        "schema": {f.name: str(f.type) for f in EVENTS_SCHEMA},
        "full_scale_rows": FULL_SCALE_ROWS,
    }
    write_json(out_dir / MANIFEST_NAME, manifest)
    log.info("done: %d rows, %d files in %s", rows, len(files), out_dir)
    return manifest


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--rows", type=int, default=env_int("ROWS", DEFAULT_ROWS),
                   help=f"number of rows (default {DEFAULT_ROWS:,}; "
                        f"full-scale target is {FULL_SCALE_ROWS:,})")
    p.add_argument("--full_scale", action="store_true",
                   help=f"shortcut for --rows {FULL_SCALE_ROWS}")
    p.add_argument("--out_dir", default=env_str("RAW_DIR", "data/raw"))
    p.add_argument("--seed", type=int, default=env_int("SEED", DEFAULT_SEED))
    p.add_argument("--batch_rows", type=int, default=env_int("BATCH_ROWS", 100_000),
                   help="rows per generated batch / raw file (bounds memory)")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS)
    p.add_argument("--start_date", default=DEFAULT_START_DATE)
    p.add_argument("--users", type=int, default=None,
                   help="distinct user_ids (default rows/20, clamped 1k..5M)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    rows = FULL_SCALE_ROWS if a.full_scale else a.rows
    try:
        generate(rows, Path(a.out_dir), a.seed, a.batch_rows, a.days,
                 a.start_date, a.users)
    except Exception:
        log.exception("data generation failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
