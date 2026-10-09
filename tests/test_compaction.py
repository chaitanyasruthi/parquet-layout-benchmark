import csv
import json

import duckdb
import pyarrow.parquet as pq
import pytest

import compact
from common import COMPACTION_CSV_COLUMNS
from conftest import SMALL_ROWS


@pytest.fixture(scope="module")
def compacted(layouts_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("compacted") / "out"
    res = tmp_path_factory.mktemp("cres")
    row = compact.compact(layouts_dir / "tiny_files", out, res, row_group_size=1_000)
    return out, res, row


def test_row_count_preserved(compacted, layouts_dir):
    out, res, row = compacted
    assert row["rows_before"] == row["rows_after"] == SMALL_ROWS


def test_checksum_preserved(compacted):
    out, res, row = compacted
    assert row["checksum_match"] is True
    d = json.loads((res / "compaction_detail.json").read_text())
    assert d["sum_cents_before"] == d["sum_cents_after"]
    assert d["sum_value_before"] == pytest.approx(d["sum_value_after"], rel=1e-9)


def test_files_consolidated(compacted, layouts_dir):
    out, res, row = compacted
    n_in = len(list((layouts_dir / "tiny_files").rglob("*.parquet")))
    assert row["input_files"] == n_in > 1000
    assert row["output_files"] == 90 == len(list(out.glob("*.parquet")))


def test_output_sorted_by_user_with_sensible_row_groups(compacted):
    out, _, _ = compacted
    for f in out.glob("*.parquet"):
        pf = pq.ParquetFile(f)
        users = pf.read(columns=["user_id"])["user_id"].to_numpy()
        assert (users[1:] >= users[:-1]).all()
        assert all(pf.metadata.row_group(i).num_rows <= 1_000
                   for i in range(pf.metadata.num_row_groups))
        assert pf.metadata.row_group(0).column(0).compression == "SNAPPY"


def test_compaction_csv_schema_and_values(compacted):
    _, res, row = compacted
    text = (res / "compaction.csv").read_text().splitlines()
    assert text[0] == "input_files,output_files,wall_s,peak_rss_mb,rows_before,rows_after,checksum_match"
    assert text[0].split(",") == COMPACTION_CSV_COLUMNS
    rec = next(csv.DictReader(text))
    assert float(rec["wall_s"]) > 0 and float(rec["peak_rss_mb"]) > 0
    assert rec["checksum_match"] == "True" and len(text) == 2


def test_actual_data_unchanged_row_for_row(compacted, layouts_dir):
    out, _, _ = compacted
    a = duckdb.sql(f"SELECT * FROM read_parquet('{layouts_dir}/tiny_files/**/*.parquet', "
                   "hive_partitioning=false) ORDER BY ALL").fetchall()
    b = duckdb.sql(f"SELECT * FROM read_parquet('{out}/*.parquet') ORDER BY ALL").fetchall()
    assert a == b


def test_sums_match_detects_corruption():
    base = {"rows": 10, "sum_value": 100.0, "sum_cents": 10_000}
    assert compact.sums_match(base, dict(base))
    assert not compact.sums_match(base, {**base, "rows": 9})
    assert not compact.sums_match(base, {**base, "sum_cents": 10_001, "sum_value": 100.01})
    assert not compact.sums_match(base, {**base, "sum_value": 101.0})


def test_nested_or_identical_dirs_rejected(layouts_dir, tmp_path):
    src = layouts_dir / "daily"
    with pytest.raises(ValueError):
        compact.compact(src, src, tmp_path)
    with pytest.raises(ValueError):
        compact.compact(src, src / "sub", tmp_path)


def test_empty_input_rejected(tmp_path):
    (tmp_path / "in").mkdir()
    with pytest.raises(FileNotFoundError):
        compact.compact(tmp_path / "in", tmp_path / "out", tmp_path / "res")


def test_cli_returns_zero_on_success(layouts_dir, tmp_path):
    rc = compact.main(["--input_dir", str(layouts_dir / "hourly"),
                       "--output_dir", str(tmp_path / "o"), "--results_dir", str(tmp_path / "r")])
    assert rc == 0
