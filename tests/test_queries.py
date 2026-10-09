import csv
import json
from pathlib import Path

import duckdb
import pytest

import run_queries as rq
from common import LAYOUT_CSV_COLUMNS, read_manifest
from profiling import extract_profile_metrics, is_parquet_scan, walk
from zonemap import FooterIndex, Predicate
from conftest import SMALL_ROWS

REQUIRED_QUERIES = ["q1_point_lookup", "q2_one_day_range", "q3_country_filter",
                    "q4_full_aggregation", "q5_complex"]


def load(results_dir, name="layout.csv"):
    with open(results_dir / name, newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- query suite
def test_query_definitions(raw_dir):
    qs = {q.id: q for q in rq.make_queries(read_manifest(raw_dir))}
    for qid in REQUIRED_QUERIES:
        assert qid in qs
    assert qs["q1_point_lookup"].sql.startswith("SELECT * FROM events WHERE user_id = ")
    assert "BETWEEN" in qs["q2_one_day_range"].sql and "COUNT(*)" in qs["q2_one_day_range"].sql
    assert qs["q3_country_filter"].sql == "SELECT AVG(value) FROM events WHERE country = 'US'"
    assert qs["q4_full_aggregation"].sql == \
        "SELECT event_type, COUNT(*) FROM events GROUP BY event_type"
    s5 = qs["q5_complex"].sql
    assert "GROUP BY user_id" in s5 and "ORDER BY SUM(value) DESC LIMIT 10" in s5


def test_one_day_window_is_one_day(raw_dir):
    m = read_manifest(raw_dir)
    con = duckdb.connect()
    con.execute(f"CREATE VIEW events AS SELECT * FROM read_parquet('{raw_dir}/raw_*.parquet')")
    q = {q.id: q for q in rq.make_queries(m)}["q2_one_day_range"]
    n = con.execute(q.sql).fetchone()[0]
    assert 0 < n < SMALL_ROWS / 30          # one day out of 90 holds ~1/90 of the rows


def test_bad_day_index_rejected(raw_dir):
    with pytest.raises(ValueError):
        rq.make_queries(read_manifest(raw_dir), day_index=999)


# ---------------------------------------------------------------- execution
def test_benchmark_produces_exact_layout_csv_header(results_dir):
    header = (results_dir / "layout.csv").read_text().splitlines()[0]
    assert header == ("layout,row_group_rows,files,total_bytes,query,cache,latency_ms,"
                      "bytes_read,row_groups_total,row_groups_pruned,peak_rss_mb")
    assert header.split(",") == LAYOUT_CSV_COLUMNS


def test_every_layout_query_cache_arm_present(results_dir):
    rows = load(results_dir)
    layouts = {r["layout"] for r in rows}
    assert {"tiny_files", "hourly", "daily", "daily_rg_10k", "daily_rg_1m",
            "partitioned_by_date", "partitioned_by_date_country", "sorted_by_user",
            "sorted_by_ts", "partitioned_by_user_id"} <= layouts
    for l in layouts:
        for q in REQUIRED_QUERIES:
            for cache in ("cold", "warm"):
                n = sum(1 for r in rows if (r["layout"], r["query"], r["cache"]) == (l, q, cache))
                assert n == 1, (l, q, cache)          # 1 cold + 1 warm in this test run


def test_measured_fields_are_sane(results_dir):
    for r in load(results_dir):
        assert float(r["latency_ms"]) > 0
        assert float(r["peak_rss_mb"]) > 0
        assert int(r["files"]) >= 1 and int(r["total_bytes"]) > 0
        assert r["cache"] in ("cold", "warm")
        if r["bytes_read"] != "":
            assert int(r["bytes_read"]) >= 0
        if r["row_groups_total"] != "":
            assert 0 <= int(r["row_groups_pruned"]) <= int(r["row_groups_total"])


def test_cold_runs_read_bytes_from_storage(results_dir):
    rows = [r for r in load(results_dir) if r["layout"] == "daily" and
            r["query"] == "q4_full_aggregation" and r["cache"] == "cold"]
    assert rows and all(int(r["bytes_read"]) > 0 for r in rows)


def test_results_identical_across_layouts(results_dir):
    d = {}
    for r in load(results_dir, "layout_detail.csv"):
        if r.get("result_digest"):
            d.setdefault(r["query"], set()).add(r["result_digest"])
    for q in REQUIRED_QUERIES + ["q2b_range_plus_partition_key"]:
        assert len(d[q]) == 1, f"{q}: layouts disagree on the answer"


def test_supplementary_partition_key_query_only_on_date_layouts(results_dir):
    q2b = {r["layout"] for r in load(results_dir) if r["query"].startswith("q2b")}
    assert q2b == {"partitioned_by_date", "partitioned_by_date_country"}


def test_partition_key_query_prunes_files(results_dir):
    d = [r for r in load(results_dir, "layout_detail.csv")
         if r["layout"] == "partitioned_by_date" and r["query"].startswith("q2b")
         and r["cache"] == "cold"]
    assert d and d[0]["files_scanned_hint"].startswith("1/90")
    assert int(d[0]["files_pruned_by_path"]) == 89


def test_point_lookup_on_user_partitions_prunes_all_but_one_file(results_dir):
    d = [r for r in load(results_dir, "layout_detail.csv")
         if r["layout"] == "partitioned_by_user_id" and r["query"] == "q1_point_lookup"
         and r["cache"] == "cold"][0]
    assert int(d["row_groups_total"]) - int(d["row_groups_pruned"]) == 1
    assert d["files_read"] == "1"


def test_profiles_saved(results_dir):
    p = results_dir / "profiles" / "daily" / "q1_point_lookup__cold1.json"
    assert p.exists() and "children" in json.loads(p.read_text())
    assert json.loads((results_dir / "run_info.json").read_text())["cold_runs"] == 1


def test_cold_uses_fresh_connection_warm_reuses(monkeypatch, raw_dir, layouts_dir, tmp_path):
    opened = []
    real = rq.open_connection

    def spy(*a, **k):
        c = real(*a, **k)
        opened.append(id(c))
        return c

    monkeypatch.setattr(rq, "open_connection", spy)
    qs = [q for q in rq.make_queries(read_manifest(raw_dir)) if q.id == "q4_full_aggregation"]
    rq.benchmark_layout("daily", layouts_dir, qs, tmp_path, 5, 5, None, False, False,
                        False, False)
    # 5 cold runs -> 5 fresh connections; 5 warm runs share exactly 1 more
    assert len(opened) == 6


def test_default_is_five_cold_five_warm():
    a = rq.parse_args([])
    assert (a.cold_runs, a.warm_runs) == (5, 5)


def test_failed_run_is_not_silent(raw_dir, layouts_dir, tmp_path):
    bad = rq.Query("q_bad", "SELECT nonexistent FROM events")
    rows, detail, fails = rq.benchmark_layout("daily", layouts_dir, [bad], tmp_path, 1, 1,
                                              None, False, False, False, False)
    assert fails == 2 and rows == [] and all(d["error"] for d in detail)


# ---------------------------------------------------------------- profile parser
def test_parser_on_real_duckdb_profile(tmp_path, layouts_dir):
    con = duckdb.connect()
    out = tmp_path / "p.json"
    con.execute("PRAGMA enable_profiling='json'")
    con.execute(f"PRAGMA profiling_output='{out}'")
    pat = layouts_dir / "daily" / "*.parquet"
    con.execute(f"SELECT COUNT(*) FROM read_parquet('{pat}') WHERE country='US'").fetchall()
    m = extract_profile_metrics(json.loads(out.read_text()))
    assert m["scan_operators"] == 1
    assert m["files_read"] == 90
    assert m["bytes_read"] > 0 and m["latency_s"] > 0
    assert m["filter_pushdown"] is True
    assert any("country" in f for f in m["filters"])


def test_parser_recurses_arbitrarily_deep_and_finds_multiple_scans():
    leaf = lambda extra, card=1, scanned=10: {
        "operator_type": "TABLE_SCAN", "extra_info": extra,
        "operator_cardinality": card, "operator_rows_scanned": scanned, "children": []}
    tree = {"latency": 0.5, "total_bytes_read": 4096, "children": [
        {"operator_type": "PROJECTION", "children": [
            {"operator_type": "HASH_JOIN", "children": [
                {"operator_type": "FILTER", "children": [
                    leaf({"Function": "READ_PARQUET", "Total Files Read": "3",
                          "Filters": "a>1", "Projections": ["a", "b"]}, 5, 100)]},
                leaf({"Function": "PARQUET_SCAN", "Total Files Read": "2",
                      "Projections": ["c"]}, 7, 50),
                leaf({"Function": "SEQ_SCAN"}),               # not parquet -> ignored
            ]}]}]}
    m = extract_profile_metrics(tree)
    assert m["scan_operators"] == 2
    assert m["files_read"] == 5 and m["rows_scanned"] == 150 and m["rows_after_pushdown"] == 12
    assert m["bytes_read"] == 4096 and m["projection_columns"] == 2
    assert m["filter_pushdown"] is True


def test_parser_tolerates_other_schema_variants():
    # old style: 'name' key + extra_info as one string, different latency key
    tree = {"timing": 1.5, "children": [
        {"name": "PARQUET_SCAN", "extra_info":
            "Function: PARQUET_SCAN\nFilters: x=1\nTotal Files Read: 4\nProjections:\nx\ny",
         "cardinality": 3, "children": []}]}
    m = extract_profile_metrics(tree)
    assert m["latency_s"] == 1.5 and m["files_read"] == 4 and m["scan_operators"] == 1
    assert m["bytes_read"] is None                       # absent -> None, never invented


@pytest.mark.parametrize("bad", [{}, {"children": []}, {"children": None}, [], "junk", None])
def test_parser_never_invents_values(bad):
    m = extract_profile_metrics(bad if isinstance(bad, dict) else {})
    assert m["scan_operators"] == 0
    for k in ("bytes_read", "rows_scanned", "files_read", "latency_s"):
        assert m[k] is None


def test_is_parquet_scan_and_walk():
    assert is_parquet_scan({"operator_type": "TABLE_SCAN", "extra_info": {"Function": "READ_PARQUET"}})
    assert not is_parquet_scan({"operator_type": "TABLE_SCAN", "extra_info": {"Function": "SEQ_SCAN"}})
    assert len(list(walk({"children": [{"children": [{}]}, {}]}))) == 4


# ---------------------------------------------------------------- zone maps
def test_footer_index_sorted_prunes_more_than_unsorted(dense_layouts):
    def pruned(layout):
        files = sorted((dense_layouts / layout).rglob("*.parquet"))
        idx = FooterIndex(dense_layouts / layout, files)
        # average over several users so the test is not tied to a lucky id
        tot = p = 0
        for u in range(0, 1000, 50):
            r = idx.evaluate([Predicate("user_id", "eq", u)])
            tot += r["row_groups_total"]; p += r["row_groups_pruned"]
        return p / tot
    assert pruned("sorted_by_user_rg_2k") > pruned("daily_rg_2k") + 0.2


def test_footer_index_matches_duckdb_bytes_ordering(dense_layouts):
    """Sanity: the layout the footers say prunes more also makes DuckDB read fewer bytes."""
    def bytes_read(layout, tmp):
        con = duckdb.connect()
        pf = tmp / f"{layout}.json"
        con.execute("PRAGMA enable_profiling='json'")
        con.execute(f"PRAGMA profiling_output='{pf}'")
        con.execute(f"SELECT * FROM read_parquet('{dense_layouts / layout}/*.parquet') "
                    "WHERE user_id = 123").fetchall()
        return extract_profile_metrics(json.loads(pf.read_text()))["bytes_read"]
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        assert bytes_read("sorted_by_user_rg_2k", t) <= bytes_read("daily_rg_2k", t)
