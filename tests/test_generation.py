import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import generate_data
from common import EVENTS_SCHEMA, MANIFEST_NAME
from conftest import SMALL_ROWS


def _read(dirpath):
    files = sorted(dirpath.glob("raw_*.parquet"))
    return pa.concat_tables([pq.read_table(f) for f in files]), files


def test_schema_matches_assignment(raw_dir):
    t, files = _read(raw_dir)
    assert t.schema.names == ["event_ts", "user_id", "country", "event_type", "device", "value"]
    assert t.schema.equals(EVENTS_SCHEMA)
    for f in files:
        assert pq.read_schema(f).equals(EVENTS_SCHEMA)


def test_row_count_and_batching(raw_dir):
    t, files = _read(raw_dir)
    assert t.num_rows == SMALL_ROWS
    assert len(files) == 4                                  # 8000 rows / 2000 per batch
    assert all(pq.ParquetFile(f).metadata.num_rows <= 2_000 for f in files)
    m = json.loads((raw_dir / MANIFEST_NAME).read_text())
    assert m["rows"] == SMALL_ROWS and m["n_batches"] == 4


def test_snappy_compression(raw_dir):
    md = pq.ParquetFile(sorted(raw_dir.glob("raw_*.parquet"))[0]).metadata
    codecs = {md.row_group(0).column(i).compression for i in range(md.num_columns)}
    assert codecs == {"SNAPPY"}


def test_spans_90_days(raw_dir):
    t, _ = _read(raw_dir)
    ts = t["event_ts"].cast(pa.int64()).to_numpy()
    span_days = (ts.max() - ts.min()) / 86_400_000_000
    assert 89 <= span_days < 90.0001
    days = np.unique(ts // 86_400_000_000)
    assert len(days) == 90


def test_deterministic_for_same_seed(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    generate_data.generate(3_000, a, seed=7, batch_rows=1_000)
    generate_data.generate(3_000, b, seed=7, batch_rows=1_000)
    generate_data.generate(3_000, c, seed=8, batch_rows=1_000)
    ta, _ = _read(a); tb, _ = _read(b); tc, _ = _read(c)
    assert ta.equals(tb)
    assert not ta.equals(tc)


def test_batches_are_time_ordered_and_unsorted_inside(raw_dir):
    files = sorted(raw_dir.glob("raw_*.parquet"))
    prev_max = None
    for f in files:
        ts = pq.read_table(f, columns=["event_ts"])["event_ts"].cast(pa.int64()).to_numpy()
        if prev_max is not None:
            assert ts.min() >= prev_max
        prev_max = ts.max()
        assert not np.all(np.diff(ts) >= 0)       # rows shuffled inside a batch


def test_manifest_checksum_is_exact(raw_dir):
    t, _ = _read(raw_dir)
    cents = int(np.rint(np.array(t["value"].to_pylist()) * 100).astype(np.int64).sum())
    assert cents == json.loads((raw_dir / MANIFEST_NAME).read_text())["value_sum_cents"]


def test_sample_user_exists(raw_dir):
    t, _ = _read(raw_dir)
    m = json.loads((raw_dir / MANIFEST_NAME).read_text())
    assert m["sample_user_id"] in set(t["user_id"].to_pylist())


def test_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError):
        generate_data.generate(0, tmp_path)
    with pytest.raises(ValueError):
        generate_data.generate(10, tmp_path, batch_rows=0)


def test_cli_entrypoint(tmp_path):
    rc = generate_data.main(["--rows", "1500", "--out_dir", str(tmp_path / "o"),
                             "--seed", "1", "--batch_rows", "1000"])
    assert rc == 0 and (tmp_path / "o" / MANIFEST_NAME).exists()
