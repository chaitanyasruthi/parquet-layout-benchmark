import json
import os
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import build_layouts as bl
from common import EVENTS_SCHEMA, LAYOUT_META_NAME
from conftest import SMALL_ROWS

REQUIRED = ["tiny_files", "hourly", "daily", "daily_rg_10k", "daily_rg_1m",
            "partitioned_by_date", "partitioned_by_date_country", "sorted_by_user",
            "sorted_by_ts", "partitioned_by_user_id"]


def meta(layouts_dir, name):
    return json.loads((layouts_dir / name / LAYOUT_META_NAME).read_text())


def count_rows(layouts_dir, name, hive=None):
    m = meta(layouts_dir, name)
    pat = str(layouts_dir / name / "**" / "*.parquet")
    return duckdb.sql(f"SELECT COUNT(*) FROM read_parquet('{pat}', "
                      f"hive_partitioning={str(m['hive_partitioning']).lower()})").fetchone()[0]


def ts_of(path):
    return pq.read_table(path, columns=["event_ts"])["event_ts"].cast(pa.int64()).to_numpy()


def test_all_required_layouts_registered():
    assert bl.REQUIRED_LAYOUTS == REQUIRED                  # none silently removed


@pytest.mark.parametrize("name", REQUIRED)
def test_every_required_layout_built_and_preserves_rows(layouts_dir, name):
    assert (layouts_dir / name).is_dir()
    assert meta(layouts_dir, name)["files"] >= 1
    assert count_rows(layouts_dir, name) == SMALL_ROWS


def test_tiny_files_are_one_minute_each(layouts_dir):
    files = sorted((layouts_dir / "tiny_files").rglob("*.parquet"))
    assert len(files) > 1000                        # ~one file per non-empty minute
    assert len(files) <= 90 * 24 * 60
    for f in files[::50]:
        ts = ts_of(f)
        assert ts.min() // 60_000_000 == ts.max() // 60_000_000


def test_hourly_and_daily_bucketing(layouts_dir):
    hourly = sorted((layouts_dir / "hourly").rglob("*.parquet"))
    assert len(hourly) <= 90 * 24
    for f in hourly[::40]:
        ts = ts_of(f)
        assert ts.min() // 3_600_000_000 == ts.max() // 3_600_000_000
    daily = sorted((layouts_dir / "daily").glob("*.parquet"))
    assert len(daily) == 90
    for f in daily[::10]:
        ts = ts_of(f)
        assert ts.min() // 86_400_000_000 == ts.max() // 86_400_000_000


def test_file_count_ordering(layouts_dir):
    n = {k: meta(layouts_dir, k)["files"] for k in ("tiny_files", "hourly", "daily")}
    assert n["tiny_files"] > n["hourly"] > n["daily"] == 90


def test_row_group_sizes_declared(layouts_dir):
    assert meta(layouts_dir, "daily")["row_group_rows"] == 122_880
    assert meta(layouts_dir, "daily_rg_10k")["row_group_rows"] == 10_000
    assert meta(layouts_dir, "daily_rg_1m")["row_group_rows"] == 1_000_000


def test_row_group_size_is_honoured_on_disk(dense_layouts):
    def rg_rows(layout):
        sizes = []
        for f in (dense_layouts / layout).glob("*.parquet"):
            md = pq.ParquetFile(f).metadata
            sizes += [md.row_group(i).num_rows for i in range(md.num_row_groups)]
        return sizes
    assert max(rg_rows("daily_rg_2k")) <= 2_000
    assert len(rg_rows("daily_rg_2k")) > len(rg_rows("daily_rg_10k"))
    assert max(rg_rows("daily_rg_10k")) <= 10_000


def test_hive_date_partitions(layouts_dir):
    root = layouts_dir / "partitioned_by_date"
    dirs = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert len(dirs) == 90 and all(d.startswith("date=2024-") for d in dirs)
    assert dirs[0] == "date=2024-01-01"
    # the partition column is not stored in the file, DuckDB derives it from the path
    f = next(root.rglob("*.parquet"))
    assert "date" not in pq.read_schema(f).names
    got = duckdb.sql(f"SELECT COUNT(DISTINCT date), MIN(date), MAX(date) FROM "
                     f"read_parquet('{root}/**/*.parquet', hive_partitioning=true)").fetchone()
    assert got[0] == 90 and str(got[1]) == "2024-01-01"


def test_nested_hive_partitions(layouts_dir):
    root = layouts_dir / "partitioned_by_date_country"
    files = list(root.rglob("*.parquet"))
    assert all(p.relative_to(root).parts[0].startswith("date=") and
               p.relative_to(root).parts[1].startswith("country=") for p in files)
    assert len(files) > 90                      # more files than plain date partitioning
    assert "country" not in pq.read_schema(files[0]).names
    got = duckdb.sql(f"SELECT COUNT(*) FROM read_parquet('{root}/**/*.parquet', "
                     f"hive_partitioning=true) WHERE country='US'").fetchone()[0]
    assert got > 0


def test_high_cardinality_partition_failure_creates_file_explosion(layouts_dir, raw_dir):
    root = layouts_dir / "partitioned_by_user_id"
    files = list(root.rglob("*.parquet"))
    users = duckdb.sql(f"SELECT COUNT(DISTINCT user_id) FROM "
                       f"read_parquet('{raw_dir}/raw_*.parquet')").fetchone()[0]
    assert len(files) == users                       # exactly one file per distinct user
    assert len(files) > 10 * meta(layouts_dir, "daily")["files"]
    assert meta(layouts_dir, "partitioned_by_user_id")["truncated"] is False
    # each directory holds only its user
    d = sorted(p for p in root.iterdir() if p.is_dir())[5]
    uid = int(d.name.split("=")[1])
    assert duckdb.sql(f"SELECT COUNT(*) FROM read_parquet('{d}/*.parquet') ").fetchone()[0] \
        == duckdb.sql(f"SELECT COUNT(*) FROM read_parquet('{raw_dir}/raw_*.parquet') "
                      f"WHERE user_id={uid}").fetchone()[0]


def test_user_partition_cap_is_reported(tmp_path, raw_dir):
    m = bl.build_layout(bl.LAYOUTS["partitioned_by_user_id"], raw_dir, tmp_path,
                        user_partition_max_rows=1_000)
    assert m["truncated"] is True and m["source_rows_used"] == 1_000 and m["rows"] == 1_000


def test_sorted_by_user_and_ts(layouts_dir):
    for f in sorted((layouts_dir / "sorted_by_user").glob("*.parquet"))[:20]:
        t = pq.read_table(f)
        u = t["user_id"].to_numpy()
        assert np.all(np.diff(u) >= 0)
    for f in sorted((layouts_dir / "sorted_by_ts").glob("*.parquet"))[:20]:
        ts = ts_of(f)
        assert np.all(np.diff(ts) >= 0)


def test_unsorted_daily_is_really_unsorted(dense_layouts):
    f = sorted((dense_layouts / "daily").glob("*.parquet"))[0]
    t = pq.read_table(f)
    assert not np.all(np.diff(t["user_id"].to_numpy()) >= 0)
    assert not np.all(np.diff(ts_of(f)) >= 0)


def test_sorted_user_has_narrow_minmax_ranges(dense_layouts):
    """The mechanism behind zone-map pruning, read straight from the footers."""
    def mean_width(layout):
        widths = []
        for f in (dense_layouts / layout).glob("*.parquet"):
            md = pq.ParquetFile(f).metadata
            ci = md.schema.names.index("user_id")
            for i in range(md.num_row_groups):
                st = md.row_group(i).column(ci).statistics
                widths.append(st.max - st.min)
        return np.mean(widths)
    assert mean_width("sorted_by_user_rg_2k") < 0.7 * mean_width("daily_rg_2k")


def test_schema_preserved_in_all_non_hive_layouts(layouts_dir):
    for name in ("tiny_files", "hourly", "daily", "sorted_by_user"):
        f = next((layouts_dir / name).rglob("*.parquet"))
        assert pq.read_schema(f).equals(EVENTS_SCHEMA)


def test_compression_is_snappy(layouts_dir):
    f = next((layouts_dir / "daily").glob("*.parquet"))
    md = pq.ParquetFile(f).metadata
    assert md.row_group(0).column(0).compression == "SNAPPY"


def test_unknown_layout_rejected(tmp_path, raw_dir):
    with pytest.raises(ValueError):
        bl.build_all(raw_dir, tmp_path, ["no_such_layout"])


def test_unordered_raw_data_is_detected(tmp_path):
    t = pa.table({"event_ts": pa.array([5_000_000_000_000, 1_000_000_000_000], pa.timestamp("us")),
                  "user_id": [1, 2], "country": ["US", "US"], "event_type": ["a", "b"],
                  "device": ["x", "y"], "value": [1.0, 2.0]})
    t2 = t.slice(1, 1)                               # earlier bucket arrives after a later one
    with pytest.raises(ValueError):
        list(bl.iter_buckets([t.slice(0, 1), t2], 0, 86_400))
