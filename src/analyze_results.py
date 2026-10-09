#!/usr/bin/env python3
"""Analysis - derive the assignment's questions from the MEASURED CSV files.

    python src/analyze_results.py --results_dir results

Reads  results/layout.csv, results/layout_detail.csv, results/compaction.csv,
       results/run_info.json (all produced by the benchmark).
Writes results/analysis.md and results/analysis.json and prints the report.

Nothing here is hard-coded: every figure is computed from the input files, and
every verdict sentence is generated from those figures (so it can say "no").
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from common import COMPACTION_CSV_COLUMNS, LAYOUT_CSV_COLUMNS, setup_logging

log = setup_logging()

ROW_GROUP_LAYOUTS = ["daily_rg_2k", "daily_rg_10k", "daily", "daily_rg_1m"]
FILE_SIZE_LAYOUTS = ["tiny_files", "hourly", "daily"]
NOISE = 0.10      # differences below 10% are reported as "within noise"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _num(v: str) -> Optional[float]:
    try:
        return float(v) if v not in ("", None) else None
    except ValueError:
        return None


def read_csv(path: Path, required: list[str]) -> list[dict]:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing - run the benchmark first")
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        missing = [c for c in required if c not in (rd.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} lacks columns {missing}")
        return list(rd)


class Results:
    def __init__(self, results_dir: Path):
        d = Path(results_dir)
        self.rows = read_csv(d / "layout.csv", LAYOUT_CSV_COLUMNS)
        self.detail = read_csv(d / "layout_detail.csv", ["layout", "query", "cache"]) \
            if (d / "layout_detail.csv").exists() else []
        self.compaction = read_csv(d / "compaction.csv", COMPACTION_CSV_COLUMNS) \
            if (d / "compaction.csv").exists() else []
        self.info = json.loads((d / "run_info.json").read_text()) \
            if (d / "run_info.json").exists() else {}
        self.cdetail = json.loads((d / "compaction_detail.json").read_text()) \
            if (d / "compaction_detail.json").exists() else {}
        self._lat = defaultdict(list)
        self._meta = {}
        for r in self.rows:
            k = (r["layout"], r["query"], r["cache"])
            v = _num(r["latency_ms"])
            if v is not None:
                self._lat[k].append(v)
            self._meta.setdefault(r["layout"], r)
        self.layouts = list(dict.fromkeys(r["layout"] for r in self.rows))
        self.queries = list(dict.fromkeys(r["query"] for r in self.rows))
        self.truncated = {r["layout"] for r in self.detail
                          if str(r.get("layout_truncated")) == "True"}

    def median(self, layout, query, cache) -> Optional[float]:
        v = self._lat.get((layout, query, cache))
        return statistics.median(v) if v else None

    def spread(self, layout, query, cache) -> Optional[tuple[float, float]]:
        v = self._lat.get((layout, query, cache))
        return (min(v), max(v)) if v else None

    def field_median(self, layout, query, cache, col) -> Optional[float]:
        vals = [_num(r[col]) for r in self.rows
                if (r["layout"], r["query"], r["cache"]) == (layout, query, cache)]
        vals = [v for v in vals if v is not None]
        return statistics.median(vals) if vals else None

    def meta(self, layout, col) -> Optional[float]:
        r = self._meta.get(layout)
        return _num(r[col]) if r else None


def f(v, nd=1, unit=""):
    return "n/a" if v is None else f"{v:,.{nd}f}{unit}"


# --------------------------------------------------------------------------
# Individual analyses
# --------------------------------------------------------------------------
def a1_tiny_vs_daily(R: Results) -> tuple[list[str], dict]:
    out, data = [], {}
    q = "q2_one_day_range"
    out.append(f"Query `{q}` (one-day COUNT(*)), median of the recorded runs:\n")
    out.append("| cache | tiny_files ms | hourly ms | daily ms | tiny/daily | hourly/daily |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for cache in ("cold", "warm"):
        t, h, d = (R.median(l, q, cache) for l in ("tiny_files", "hourly", "daily"))
        rt = t / d if t and d else None
        rh = h / d if h and d else None
        data[cache] = {"tiny_ms": t, "hourly_ms": h, "daily_ms": d,
                       "tiny_over_daily": rt, "hourly_over_daily": rh}
        out.append(f"| {cache} | {f(t)} | {f(h)} | {f(d)} | {f(rt, 2, 'x')} | {f(rh, 2, 'x')} |")
    return out, data


def a2_pruning(R: Results) -> tuple[list[str], dict]:
    q = "q1_point_lookup"
    pairs = [("sorted_by_user", "daily"), ("sorted_by_user_rg_2k", "daily_rg_2k"),
             ("daily_rg_10k", "daily")]
    out = [f"Query `{q}`. Pruned row groups come from Parquet footer min/max statistics "
           "(`zonemap.py`, derived - DuckDB's profile does not expose it); "
           "bytes_read is DuckDB's own measurement (cold median).\n",
           "| layout | row groups | pruned | pruned % | bytes_read (cold) | cold ms | warm ms |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    data = {}
    names = [n for p in pairs for n in p]
    for name in dict.fromkeys(names + ["sorted_by_ts", "partitioned_by_user_id"]):
        if name not in R.layouts:
            continue
        tot = R.field_median(name, q, "cold", "row_groups_total")
        pr = R.field_median(name, q, "cold", "row_groups_pruned")
        pct = 100 * pr / tot if tot else None
        br = R.field_median(name, q, "cold", "bytes_read")
        data[name] = {"row_groups_total": tot, "row_groups_pruned": pr, "pruned_pct": pct,
                      "bytes_read_cold": br, "cold_ms": R.median(name, q, "cold"),
                      "warm_ms": R.median(name, q, "warm")}
        out.append(f"| {name} | {f(tot, 0)} | {f(pr, 0)} | {f(pct, 1, '%')} | "
                   f"{f(br, 0)} | {f(data[name]['cold_ms'])} | {f(data[name]['warm_ms'])} |")
    out.append("")
    for s, u in pairs[:2]:
        if s in data and u in data and data[s]["pruned_pct"] is not None:
            ps, pu = data[s]["pruned_pct"], data[u]["pruned_pct"]
            note = ""
            if data[s]["row_groups_total"] == data[u]["row_groups_total"] == \
                    R.meta(s, "files"):
                note = (" Both layouts hold exactly one row group per file, so sorting "
                        "cannot create row-group-level pruning here (increase ROWS or use "
                        "the `*_rg_2k` layouts).")
            out.append(f"* **{s} vs {u}**: {f(ps, 1, '%')} vs {f(pu, 1, '%')} of row groups "
                       f"pruned (difference {f(ps - pu, 1, ' pp')}).{note}")
    return out, data


def a3_best_row_group(R: Results) -> tuple[list[str], dict]:
    present = [l for l in ROW_GROUP_LAYOUTS if l in R.layouts]
    out = ["Candidates: " + ", ".join(
        f"`{l}` ({f(R.meta(l, 'row_group_rows'), 0)} rows/row group)" for l in present) + ".\n",
        "| query | cache | best | best ms | runner-up | runner-up ms | verdict |",
        "|---|---|---|---:|---|---:|---|"]
    data = {}
    for q in R.queries:
        if q.startswith("q2b"):
            continue
        for cache in ("cold", "warm"):
            ranked = sorted(((R.median(l, q, cache), l) for l in present
                             if R.median(l, q, cache) is not None))
            if len(ranked) < 2:
                continue
            (b, bl), (s, sl) = ranked[0], ranked[1]
            rel = (s - b) / s if s else 0
            # physically identical layouts => any difference is noise
            tot = {l: R.field_median(l, q, cache, "row_groups_total") for l in present}
            same = tot.get(bl) is not None and tot.get(bl) == tot.get(sl)
            verdict = ("within noise (<%d%%)" % (NOISE * 100) if rel < NOISE else
                       f"{rel * 100:.0f}% faster than runner-up")
            if same:
                verdict += "; identical row-group count => same physical structure"
            data[f"{q}/{cache}"] = {"best": bl, "best_ms": b, "runner_up": sl,
                                    "runner_up_ms": s, "relative_gain": rel,
                                    "same_row_group_count": same}
            out.append(f"| {q} | {cache} | {bl} | {f(b)} | {sl} | {f(s)} | {verdict} |")
    return out, data


def a4_5_compaction(R: Results) -> tuple[list[str], dict]:
    if not R.compaction:
        return ["compaction.csv not found."], {}
    r = R.compaction[-1]
    d = R.cdetail
    data = {"wall_s": _num(r["wall_s"]), "peak_rss_mb": _num(r["peak_rss_mb"]),
            "input_files": int(r["input_files"]), "output_files": int(r["output_files"]),
            "rows_before": int(r["rows_before"]), "rows_after": int(r["rows_after"]),
            "checksum_match": r["checksum_match"]}
    out = [f"* **Runtime (wall clock)**: {f(data['wall_s'], 2)} s "
           f"for {data['input_files']:,} -> {data['output_files']:,} files.",
           f"* **Peak RSS**: {f(data['peak_rss_mb'], 0)} MiB (process peak; "
           f"baseline before work: {f(d.get('baseline_rss_mb'), 0)} MiB).",
           f"* **Rows before/after**: {data['rows_before']:,} / {data['rows_after']:,}; "
           f"checksum_match = **{data['checksum_match']}**."]
    if d.get("rows_per_second"):
        out.append(f"* Throughput: {d['rows_per_second']:,.0f} rows/s.")
    return out, data


def a6_file_explosion(R: Results) -> tuple[list[str], dict]:
    name = "partitioned_by_user_id"
    if name not in R.layouts:
        return [f"`{name}` was not benchmarked."], {}
    files = R.meta(name, "files")
    base = {l: R.meta(l, "files") for l in ("daily", "partitioned_by_date", "hourly",
                                             "tiny_files") if l in R.layouts}
    out = [f"* `{name}` produced **{f(files, 0)} files** "
           + ", ".join(f"({files / n:,.1f}x `{l}`'s {n:,.0f})" for l, n in base.items() if n)
           + "."]
    if name in R.truncated:
        out.append("* WARNING: this layout was built from a capped prefix of the data "
                   "(`--user_partition_max_rows`); its numbers are not comparable.")
    out += ["", "| query | cache | partitioned_by_user_id ms | daily ms | slowdown |",
            "|---|---|---:|---:|---:|"]
    data = {"files": files, "slowdown": {}}
    for q in R.queries:
        for cache in ("cold", "warm"):
            u, d = R.median(name, q, cache), R.median("daily", q, cache)
            if u is None:
                continue
            sd = u / d if d else None
            data["slowdown"][f"{q}/{cache}"] = sd
            out.append(f"| {q} | {cache} | {f(u)} | {f(d)} | {f(sd, 2, 'x')} |")
    return out, data


def a7_larger_better(R: Results) -> tuple[list[str], dict]:
    out, data = [], {}
    qs = [q for q in R.queries if not q.startswith("q2b")]
    out.append("File-size ladder (tiny_files < hourly < daily), median ms:\n")
    out.append("| query | cache | tiny_files | hourly | daily | monotonic (bigger=faster)? |")
    out.append("|---|---|---:|---:|---:|---|")
    exceptions = []
    for q in qs:
        for cache in ("cold", "warm"):
            t, h, d = (R.median(l, q, cache) for l in FILE_SIZE_LAYOUTS)
            if None in (t, h, d):
                continue
            mono = t >= h >= d
            if not mono:
                exceptions.append(f"{q}/{cache}")
            data[f"{q}/{cache}"] = {"tiny": t, "hourly": h, "daily": d, "monotonic": mono}
            out.append(f"| {q} | {cache} | {f(t)} | {f(h)} | {f(d)} | {'yes' if mono else 'NO'} |")
    out.append("")
    # row-group size and "bigger row group" dimension
    rg_present = [l for l in ROW_GROUP_LAYOUTS if l in R.layouts]
    rg_losses = []
    for q in qs:
        for cache in ("cold", "warm"):
            ranked = sorted((R.median(l, q, cache), l) for l in rg_present
                            if R.median(l, q, cache) is not None)
            if len(ranked) >= 2 and ranked[0][1] != "daily_rg_1m" and \
                    (ranked[1][0] - ranked[0][0]) / ranked[1][0] >= NOISE:
                rg_losses.append(f"{q}/{cache} (best: {ranked[0][1]})")
    n = len(data)
    if n:
        out.append(f"* File-size ladder: strictly monotonic in {n - len(exceptions)}/{n} "
                   "query/cache cells."
                   + (f" Exceptions: {', '.join(exceptions)}." if exceptions else ""))
    if rg_present:
        out.append("* Row-group size: the largest row group (`daily_rg_1m`) was NOT clearly "
                   "best (>=10% margin) in: "
                   + (", ".join(rg_losses) if rg_losses else "none of the cells") + ".")
    # partition pruning counter-example
    if "partitioned_by_date" in R.layouts:
        a = R.median("partitioned_by_date", "q2b_range_plus_partition_key", "warm")
        b = R.median("daily", "q2_one_day_range", "warm")
        if a and b:
            out.append(f"* Partition-key predicate (`q2b`, warm): {f(a)} ms vs plain `daily` "
                       f"range query {f(b)} ms ({f(b / a, 2, 'x')}).")
    verdict = ("Larger files were faster in every measured query/cache cell of the ladder."
               if n and not exceptions else
               "Larger files were NOT always faster - see exceptions above."
               if n else "insufficient data")
    out.append(f"\n**Verdict from these measurements:** {verdict}")
    out.append("Caveat: with few rows per day the row-group dimension may be degenerate "
               "(see README 'Interpreting results').")
    data["verdict"] = verdict
    return out, data


def consistency_check(R: Results) -> list[str]:
    if not R.detail:
        return ["layout_detail.csv absent - result-consistency check skipped."]
    digests = defaultdict(lambda: defaultdict(set))
    for r in R.detail:
        if r.get("result_digest") and r["layout"] not in R.truncated:
            digests[r["query"]][r["result_digest"]].add(r["layout"])
    out = []
    for q, d in digests.items():
        if len(d) == 1:
            out.append(f"* `{q}`: identical results on all {len(next(iter(d.values())))} layouts")
        else:
            out.append(f"* `{q}`: **MISMATCH** - {({k: sorted(v) for k, v in d.items()})}")
    return out


def overview_table(R: Results) -> list[str]:
    out = ["| layout | files | MiB | " + " | ".join(
        f"{q.split('_')[0]} cold / warm" for q in R.queries) + " |",
        "|---|---:|---:|" + "---:|" * len(R.queries)]
    for l in R.layouts:
        mib = (R.meta(l, "total_bytes") or 0) / 1048576
        cells = [f"{f(R.median(l, q, 'cold'), 1)} / {f(R.median(l, q, 'warm'), 1)}"
                 if R.median(l, q, "cold") is not None else "-" for q in R.queries]
        out.append(f"| {l} | {f(R.meta(l, 'files'), 0)} | {mib:,.1f} | " + " | ".join(cells) + " |")
    return out


# --------------------------------------------------------------------------
def analyze(results_dir: Path) -> tuple[str, dict]:
    R = Results(results_dir)
    md = ["# Benchmark analysis (computed from measured CSVs)", ""]
    i = R.info
    if i:
        md += [f"*DuckDB {i.get('duckdb_version')} / PyArrow {i.get('pyarrow_version')} / "
               f"{i.get('cpu_count')} CPU / {i.get('ram_gib')} GiB RAM / dataset rows "
               f"{i.get('dataset_rows'):,} / seed {i.get('seed')} / "
               f"{i.get('cold_runs')} cold + {i.get('warm_runs')} warm runs per cell / "
               f"threads: {i.get('threads_setting')}*", ""]
    sections = []
    md += ["## Overview - median latency (ms), cold / warm", "", *overview_table(R), ""]
    for title, fn in [
        ("1. Tiny-files vs daily latency (one-day query)", a1_tiny_vs_daily),
        ("2. Row-group pruning: sorted vs unsorted `user_id` lookup", a2_pruning),
        ("3. Best row-group size per query", a3_best_row_group),
        ("4-5. Compaction runtime and memory", a4_5_compaction),
        ("6. File-count explosion from `partitioned_by_user_id`", a6_file_explosion),
        ("7. Are larger files always better?", a7_larger_better),
    ]:
        lines, data = fn(R)
        md += [f"## {title}", "", *lines, ""]
        sections.append((title, data))
    md += ["## Result-consistency check (all layouts must return the same answer)", "",
           *consistency_check(R), ""]
    return "\n".join(md), {t: d for t, d in sections}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results_dir", default="results")
    a = p.parse_args(argv)
    try:
        text, data = analyze(Path(a.results_dir))
    except Exception:
        log.exception("analysis failed")
        return 1
    out = Path(a.results_dir)
    (out / "analysis.md").write_text(text)
    (out / "analysis.json").write_text(json.dumps(data, indent=2, default=str))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
