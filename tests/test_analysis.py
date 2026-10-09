import csv
import statistics

import analyze_results as ar
import compact
from conftest import SMALL_ROWS


def test_analysis_report_is_computed_from_csv(results_dir, layouts_dir, tmp_path):
    compact.compact(layouts_dir / "tiny_files", tmp_path / "c", results_dir)
    text, data = ar.analyze(results_dir)
    for h in ("Tiny-files vs daily", "Row-group pruning", "Best row-group size",
              "Compaction runtime", "File-count explosion", "larger files always better"):
        assert h.lower() in text.lower()
    # recompute the tiny/daily ratio independently from layout.csv
    rows = list(csv.DictReader(open(results_dir / "layout.csv")))
    med = lambda l: statistics.median(float(r["latency_ms"]) for r in rows
                                      if r["layout"] == l and r["query"] == "q2_one_day_range"
                                      and r["cache"] == "warm")
    ratio = data["1. Tiny-files vs daily latency (one-day query)"]["warm"]["tiny_over_daily"]
    assert abs(ratio - med("tiny_files") / med("daily")) < 1e-9
    files = data["6. File-count explosion from `partitioned_by_user_id`"]["files"]
    assert files == next(int(r["files"]) for r in rows if r["layout"] == "partitioned_by_user_id")
    assert "MISMATCH" not in text


def test_analysis_cli_writes_files(results_dir):
    assert ar.main(["--results_dir", str(results_dir)]) == 0
    assert (results_dir / "analysis.md").exists() and (results_dir / "analysis.json").exists()
