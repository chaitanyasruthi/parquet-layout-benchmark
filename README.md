# Parquet Layout Benchmark Engine (DuckDB + PyArrow)

An end-to-end, reproducible harness that measures **how the physical layout of Parquet files
changes DuckDB query performance**: file size, row-group size, partitioning, sort order,
predicate pushdown, zone-map pruning, the small-file problem, compaction cost and a deliberately
induced high-cardinality partitioning failure.

> **How to read this document.** Every section is tagged:
> **[THEORY]** = how Parquet/DuckDB are *supposed* to behave (explanation, not a result),
> **[EXPECTED]** = a hypothesis the benchmark is designed to test (may turn out false),
> **[MEASURED]** = a number produced by running this code. The README contains **no** measured
> numbers except in the clearly marked *Reference run* section, which was copied from
> `results/analysis.md` after an actual run. Everything else you should regenerate yourself.

---------------------------------------------------------------------

## Table of contents
1. [Problem statement](#1-problem-statement)
2. [Architecture](#2-architecture)
3. [Parquet internals](#3-parquet-internals)
4. [Row groups](#4-row-groups)
5. [Column chunks](#5-column-chunks)
6. [Footer statistics](#6-footer-statistics)
7. [Predicate pushdown](#7-predicate-pushdown)
8. [Zone-map pruning](#8-zone-map-pruning)
9. [The small-file problem](#9-the-small-file-problem)
10. [Benchmark methodology](#10-benchmark-methodology)
11. [Layout comparison](#11-layout-comparison)
12. [Query suite](#12-query-suite)
13. [Cold vs warm methodology](#13-cold-vs-warm-methodology)
14. [Compaction](#14-compaction)
15. [High-cardinality partition failure](#15-high-cardinality-partition-failure)
16. [Reproducibility](#16-reproducibility)
17. [How to run](#17-how-to-run)
18. [Expected artifacts](#18-expected-artifacts)
19. [How to interpret results](#19-how-to-interpret-results)
20. [Human verification with DuckDB](#20-human-verification-with-duckdb)
21. [Limitations](#21-limitations-read-before-trusting-a-number)
22. [Verification status of this repository](#22-verification-status-of-this-repository)
23. [Reference run (MEASURED)](#23-reference-run-measured)

---------------------------------------------------------------------

## 1. Problem statement

Two Parquet datasets with *identical rows* can differ by orders of magnitude in query time purely
because of how the bytes are arranged on disk. The questions this project answers **with
measurements, not assumptions**:

1. How much slower is "one file per minute" than "one file per day" for a one-day query?
2. Does sorting by `user_id` let DuckDB skip row groups for a `user_id = X` lookup?
3. Which row-group size is best for each query?
4. What does it cost (time, memory) to compact tiny files, and is the data preserved bit-for-bit
   (row count + `SUM(value)`)?
5. How badly does partitioning by a high-cardinality column (`user_id`) break things?
6. Are bigger files *always* better?

## 2. Architecture

```text
 generate_data.py            build_layouts.py              run_queries.py              analyze_results.py
 ┌───────────────┐  raw_*.parquet  ┌───────────────┐ 10(+2) layouts ┌──────────────────┐ layout.csv  ┌────────────────┐
 │ seeded batches├───────────────▶ │ stream + bucket├──────────────▶│ 5 queries x 10   ├───────────▶ │ ratios, pruning│
 │ (bounded RAM) │  manifest.json  │ minute/hour/day│ layout_meta   │ runs x each layout│ detail.csv │ verdicts (md)  │
 └───────────────┘                 └──────┬─────────┘               └───────┬──────────┘             └────────────────┘
                                          │ tiny_files                      │ profiles/*.json (DuckDB JSON profile)
                                          ▼                                 │ zonemap.py (footer min/max evaluation)
                                   compact.py ──▶ compacted/ + compaction.csv
```

| File | Role |
|---|---|
| `src/generate_data.py` | Phase 1. Deterministic, batched generator (90 days). Writes `raw_*.parquet` + `manifest.json`. |
| `src/build_layouts.py` | Phase 2. Streams raw data once per layout; writes the 10 required layouts (+2 optional extras) and `layout_meta.json`. |
| `src/run_queries.py` | Phase 3. 5 queries × (5 cold + 5 warm) × every layout, with DuckDB JSON profiling. |
| `src/profiling.py` | Version-tolerant, recursive parser of DuckDB profile JSON. |
| `src/zonemap.py` | Evaluates Parquet footer min/max statistics to count prunable row groups (see [limitations](#21-limitations-read-before-trusting-a-number)). |
| `src/compact.py` | Phase 4. Tiny files → sorted daily files; time, RSS, row count and checksum verification. |
| `src/analyze_results.py` | Computes the 7 required analyses from the CSVs and writes `results/analysis.md`. |
| `src/common.py` | Schema, constants, logging, RSS sampler, page-cache eviction. |

## 3. Parquet internals  **[THEORY]**

```text
file.parquet
├── "PAR1" magic
├── Row group 0
│   ├── Column chunk: event_ts   ── dictionary page? + data pages (encoded, Snappy-compressed)
│   ├── Column chunk: user_id
│   └── ...
├── Row group 1 ...
├── Footer (Thrift FileMetaData): schema, row-group list, per-column-chunk offsets,
│            sizes, encodings, **min/max/null-count statistics**
├── footer length (4 bytes)
└── "PAR1" magic
```

A reader first reads the **footer** (tail of the file), then fetches only the byte ranges of the
column chunks / row groups it needs. Every file costs at least one footer read (and an open/list
operation) no matter how few rows it contains - the root of the small-file problem.

## 4. Row groups  **[THEORY]**

A row group is a horizontal slice of rows (e.g. 122,880 rows) stored column-by-column. It is the
unit of (a) parallelism in DuckDB (threads claim row groups), (b) statistics and therefore
pruning, and (c) memory buffering at write time.

* **Too small** (e.g. 10k rows): more footer metadata, less compression/encoding efficiency,
  more scheduling overhead; but finer pruning granularity.
* **Too large** (e.g. 1M rows): coarser pruning (a needle row forces reading a huge chunk), less
  parallelism for a single file, more writer memory.
* DuckDB's default is **122,880** rows (60 × 2,048-row vectors).
* Layout `daily_rg_10k`, `daily`, `daily_rg_1m` isolate this variable. **Caveat:** a row group can
  never hold more rows than the file, so when one day contains fewer rows than the row-group size
  the layouts are *physically identical*. At 1M total rows a day has ≈11k rows, so `daily` and
  `daily_rg_1m` are the same and `daily_rg_10k` is split in two. The row-group variable becomes
  meaningful at roughly ≥ 20M rows (≈ 220k rows/day) - which is why the optional `*_rg_2k`
  layouts exist.

## 5. Column chunks  **[THEORY]**

Within a row group each column is stored contiguously as a *column chunk*. Consequences:

* **Projection pushdown**: `SELECT event_type, COUNT(*)` reads only the `event_type` chunks.
  `SELECT *` must read all six.
* Similar values sit together, so dictionary + run-length + Snappy compress well. Sorting by a
  column improves compression of *that* column (and correlated ones).

## 6. Footer statistics  **[THEORY]**

For every column chunk the footer stores `min`, `max` and `null_count`. They are tiny, written
once, and let a reader decide whether a row group *could* contain matching rows without reading
any data pages.

## 7. Predicate pushdown  **[THEORY]**

Instead of reading all data and filtering afterwards, DuckDB pushes `WHERE` predicates *into* the
Parquet scan. The scan can then: skip whole files (Hive path pruning on `date=`/`user_id=`
directories), skip row groups (footer min/max, also dictionary/bloom filters when present), and
filter rows during decoding. DuckDB's profile shows this as `Filters:` (and `File Filters:` /
`Scanning Files: a/b` for path pruning) on the `READ_PARQUET` operator; the benchmark parses and
stores it (`filters`, `filter_pushdown` in `layout_detail.csv`).

## 8. Zone-map pruning  **[THEORY]**

A *zone map* is the `[min, max]` interval of a block. For `user_id = X` a row group is skipped iff
`X < min` or `X > max`.

* **Narrow ranges prune.** If rows are sorted by `user_id`, row group *k* covers a tight
  interval (e.g. users 514…1017); a lookup intersects at most one or two row groups.
* **Broad ranges prevent pruning.** If rows are in random order every row group's range is
  ≈ `[0, max_user_id]`, so *every* row group "might" contain `X` and none can be skipped.

Section [20](#20-human-verification-with-duckdb) shows how to see this yourself.

## 9. The small-file problem  **[THEORY]**

Per-file fixed costs: directory listing, `open`, footer read & parse, planning, task scheduling,
and (on object stores) one or more HTTP round-trips. With *N* tiny files these costs scale with
*N* while useful work scales with bytes. Also: tiny files compress poorly (every file repeats
dictionaries and footer), so total bytes grow. `tiny_files` (≈ one file per minute → up to
129,600 files for 90 days) is the intentionally bad layout.

## 10. Benchmark methodology

* **Dataset**: `events(event_ts, user_id, country, event_type, device, value)`; 90 days; seeded;
  Snappy Parquet. Rows arrive in time-ordered micro-batches, *shuffled inside a batch*
  (arrival disorder), so `daily` is not secretly sorted and `sorted_by_*` really differ.
* **Layouts** are all built from the *same* raw files, so every layout contains identical rows.
  This is verified: the benchmark stores a digest of every query result and `analyze_results.py`
  fails loudly (`MISMATCH`) if two layouts ever disagree.
* **Latency** = wall-clock of `execute()+fetchall()` of the query on an already-created view
  (glob expansion, footer reads, planning and execution are *inside* the timing; connection
  creation is outside).
* **Profiling**: each run executes with `PRAGMA enable_profiling='json'` (the machine-readable
  output of `EXPLAIN ANALYZE`). The parser walks the operator tree recursively, finds
  `READ_PARQUET` scans, and extracts bytes read, files opened, rows scanned, filters and
  projections. Missing metrics stay **empty**, never invented. `--save_explain` additionally
  stores the textual `EXPLAIN ANALYZE` per layout/query.
* **Peak RSS** = maximum of the process RSS sampled every 5 ms during the run (DuckDB is
  in-process).
* **Statistics reported**: `layout.csv` holds *every individual run*; the analysis uses medians.

## 11. Layout comparison

| Layout | Files (≈) | Row group | Variable it isolates |
|---|---|---|---|
| `tiny_files` | one per non-empty minute (≤ 129,600) | 122,880 (data smaller) | small-file problem |
| `hourly` | ≤ 2,160 | 122,880 | intermediate file size |
| `daily` | 90 | 122,880 | baseline |
| `daily_rg_10k` | 90 | 10,000 | small row groups |
| `daily_rg_1m` | 90 | 1,000,000 | large row groups |
| `partitioned_by_date` | 90 (`date=YYYY-MM-DD/`) | 122,880 | Hive partition pruning |
| `partitioned_by_date_country` | ≤ 1,800 (`date=…/country=…/`) | 122,880 | nested partitions |
| `sorted_by_user` | 90 | 122,880 | sort order → `user_id` zone maps |
| `sorted_by_ts` | 90 | 122,880 | sort order → `event_ts` zone maps |
| `partitioned_by_user_id` | = number of distinct users (`user_id=N/`) | 122,880 | **induced failure** |
| *(extra)* `daily_rg_2k` | 90 | 2,000 | row-group effects at small scale |
| *(extra)* `sorted_by_user_rg_2k` | 90 | 2,000 | sorted + several row groups per file |

The extras are enabled by default in `make bench` (`WITH_EXTRAS=0` to disable); the ten required
layouts are always built and never silently dropped.

## 12. Query suite

| id | SQL |
|---|---|
| `q1_point_lookup` | `SELECT * FROM events WHERE user_id = X` |
| `q2_one_day_range` | `SELECT COUNT(*) FROM events WHERE event_ts BETWEEN X AND Y` (one calendar day) |
| `q3_country_filter` | `SELECT AVG(value) FROM events WHERE country = 'US'` |
| `q4_full_aggregation` | `SELECT event_type, COUNT(*) FROM events GROUP BY event_type` |
| `q5_complex` | `SELECT user_id, SUM(value) FROM events WHERE event_ts BETWEEN X AND Y GROUP BY user_id ORDER BY SUM(value) DESC LIMIT 10` |
| `q2b_range_plus_partition_key` *(supplementary, `date` layouts only)* | `q2` plus `AND date = DATE '…'` |

`X`/`Y` (and the `user_id`) are deterministic: the middle day of the span and a user id that is
guaranteed to exist (stored in `manifest.json`). `events` is a view over
`read_parquet('<layout>/**/*.parquet', hive_partitioning=<per layout>)`.

Why `q2b`? The mandatory `q2` filters `event_ts`, not the partition column `date`; DuckDB does not
derive `date=` from an `event_ts` predicate, so Hive date partitioning cannot prune *files* for
`q2`. `q2b` shows what partitioning is for. It is reported separately and never replaces `q2`.

## 13. Cold vs warm methodology

| | Cold | Warm |
|---|---|---|
| runs | 5 | 5 (after one un-recorded priming run) |
| DuckDB connection | **fresh connection each run** | **one connection reused** |
| OS page cache | evicted before each run with `posix_fadvise(DONTNEED)` (best effort) | populated |

**Cold ≠ guaranteed disk read.** True cold-cache measurement needs `echo 3 > /proc/sys/vm/drop_caches`
(root) or a reboot. `fadvise` only drops *clean* pages and is ignored by some container/overlay/
network filesystems, and storage controllers have their own caches. A "cold" run therefore means
"fresh DuckDB state (empty metadata/file caches) + best-effort OS cache eviction". The columns
`pages_evicted_files` (detail CSV) and `run_info.json` record whether eviction was attempted. In a
warm run DuckDB's external-file cache may serve everything from memory, so `bytes_read` is
legitimately `0` for warm runs of small datasets.

## 14. Compaction

`python src/compact.py --input_dir data/layouts/tiny_files --output_dir data/layouts/compacted --results_dir results`

1. List tiny files; assign each to a day from its footer `min(event_ts)` (no data read).
2. For each day: read that day's files, sort by `(user_id, event_ts)`, write one Snappy file with
   row-group size 122,880 (`--row_group_size`). One day at a time ⇒ memory bounded by the largest day.
3. Time and peak RSS cover steps 1-2 only. Verification happens *after* peak RSS is captured.
4. Verify with DuckDB over input and output: `COUNT(*)`, `SUM(value)` and the exact integer
   `SUM(CAST(ROUND(value*100) AS HUGEINT))` (cents, immune to float summation order).
   `checksum_match` is `True` only if rows are equal **and** cents are equal **and** the float
   sums agree to 1e-9 relative.

Exit code is `3` if verification fails.

## 15. High-cardinality partition failure

`partitioned_by_user_id` writes `user_id=N/part-0.parquet` for every user. Hive partitioning
assumes few, large partitions; with ~`rows/20` distinct users it creates one directory + file per
user holding a handful of rows. **[EXPECTED]** file explosion (files ≈ distinct users, ≫ `daily`),
tiny files with large per-file overhead, and slow full scans. **[THEORY]** a point lookup on
`user_id` can *win* (path pruning → one file) while *every other query* loses; the benchmark
reports both sides. Real consequences beyond latency (not measured here): inode/metadata
pressure, listing cost and request cost on object stores, catalog/metastore size, and slow
compaction.

Memory guard: grouping by user needs the rows in memory, so this layout is built from at most
`--user_partition_max_rows` source rows (default 5M). If the dataset is larger the build **logs a
warning, records `truncated=true` in `layout_meta.json`, and the analysis flags the layout as not
comparable** - nothing is hidden. At the default 1M rows the layout is complete.

## 16. Reproducibility

* Per-batch RNG = `numpy.random.default_rng(SeedSequence([seed, batch_index]))`; same
  `(rows, seed, batch_rows)` ⇒ identical data (tested).
* Dependency versions are pinned in `requirements.txt` and `Dockerfile`; DuckDB/PyArrow versions,
  CPU count, RAM, run counts and the exact SQL are written to `results/run_info.json`.
* Timings are **not** reproducible bit-for-bit (they depend on hardware, other load, filesystem);
  *row counts, checksums, file counts, bytes and pruning counts are*.
* Paths are configurable via CLI flags and environment variables (`.env.example`).

## 17. How to run

```bash
# 0. environment (Python 3.10+; developed on 3.12)
python -m venv .venv && source .venv/bin/activate
make install                       # pip install -r requirements.txt

# 1. unit tests (uses 6-8k rows, ~1 min)
make test

# 2. benchmark pipeline (default ROWS=1000000)
ROWS=1000000 make bench            # generate -> layouts -> queries -> compact -> analyze

# individual stages
make generate layouts queries compact analyze

# full scale target: 200M rows (needs lots of disk/time; tiny_files alone is ~130k files)
ROWS=200000000 make bench          # or: make bench-full

# the same commands without make
python src/generate_data.py --rows 1000000 --out_dir data/raw --seed 42
python src/build_layouts.py --raw_dir data/raw --out_dir data/layouts --with_extras
python src/run_queries.py   --layouts_dir data/layouts --raw_dir data/raw --results_dir results
python src/compact.py --input_dir data/layouts/tiny_files --output_dir data/layouts/compacted --results_dir results
python src/analyze_results.py --results_dir results

# Docker (no cloud services)
docker compose run --rm test
docker compose run --rm bench      # ROWS=5000000 docker compose run --rm bench
```

**Scaling guidance.** Disk ≈ (raw + ~12 layouts) ≈ 12-13× the raw Parquet size, and `tiny_files`
creates up to 129,600 files regardless of `ROWS`. Quick sanity runs: `ROWS=200000 make bench`.
`run_queries.py` spends most of its time on `tiny_files` and `partitioned_by_user_id` - that *is*
the small-file problem. Reduce `COLD_RUNS`/`WARM_RUNS` for a faster (but noisier) pass, or restrict
with `--layouts a,b,c` / `--queries q1_point_lookup,...`.

## 18. Expected artifacts

```text
data/raw/raw_*.parquet, manifest.json        generated dataset
data/layouts/<layout>/…, layout_meta.json    one directory per layout
data/layouts/compacted/events-YYYY-MM-DD.parquet
results/layout.csv            REQUIRED - one row per run, 11 columns (below)
results/compaction.csv        REQUIRED - one row, 7 columns (below)
results/layout_detail.csv     extra profile fields per run (files read, pushdown, digest, ...)
results/compaction_detail.json, run_info.json
results/profiles/<layout>/<query>__<cache><n>.json   raw DuckDB profiles
results/analysis.md, analysis.json   generated by analyze_results.py
```

`layout.csv` columns: `layout,row_group_rows,files,total_bytes,query,cache,latency_ms,bytes_read,row_groups_total,row_groups_pruned,peak_rss_mb`
`compaction.csv` columns: `input_files,output_files,wall_s,peak_rss_mb,rows_before,rows_after,checksum_match`

Column provenance in `layout.csv`:

| column | source |
|---|---|
| `row_group_rows` | **configured** row-group size of the layout (`layout_meta.json`); actual row groups may be smaller when the file has fewer rows |
| `files`, `total_bytes` | counted from disk after the build |
| `latency_ms` | wall clock of execute+fetch |
| `bytes_read` | **DuckDB profile** (`total_bytes_read`) |
| `row_groups_total`, `row_groups_pruned` | **derived from Parquet footers** (`zonemap.py`), *not* reported by DuckDB - see limitations |
| `peak_rss_mb` | sampled process RSS during the run |

## 19. How to interpret results

1. **Check correctness first**: the consistency section of `analysis.md` must say identical results
   on all layouts, and `compaction.csv` must show `checksum_match=True`.
2. **Prefer medians**; look at `latency_ms` spread in `layout.csv` before believing a 10-20%
   difference. The analysis labels gaps < 10% as *within noise*.
3. **Use `bytes_read` as the I/O ground truth** (DuckDB-reported) and `files_read` /
   `files_scanned_hint` in the detail CSV for file-level pruning.
4. **Row-group-size results are only meaningful if layouts differ physically.** Compare
   `row_groups_total` across `daily`, `daily_rg_10k`, `daily_rg_1m`; identical counts mean identical
   structure and any latency difference is noise (the analysis states this).
5. **Sorted vs unsorted pruning needs ≥ 2 row groups per file.** With one row group per file,
   sorting cannot prune at row-group level. Use `*_rg_2k` layouts or a larger `ROWS`.
6. **Small data hides effects.** At ≤ a few million rows everything fits in cache and per-file
   overhead dominates; at 200M rows scan volume dominates. Do not extrapolate a 1M-row ratio to 200M.
7. **Do not conclude "larger files are always better" from one dimension.** The analysis tests this
   claim against the data (file-size ladder, row-group size, partition-key predicates) and only
   reports what was measured.

## 20. Human verification with DuckDB

Inspect footers directly:

```sql
SELECT * FROM parquet_metadata('data/layouts/daily/events-2024-02-15.parquet');
SELECT * FROM parquet_file_metadata('data/layouts/daily/events-2024-02-15.parquet');
```

Compare `user_id` min/max per row group, **sorted vs unsorted** (statistics are stored as text,
so cast for numeric comparison; use the `*_rg_2k` layouts to get several row groups per file):

```sql
-- unsorted: every row group spans (almost) the whole user range  -> cannot prune
SELECT row_group_id, row_group_num_rows,
       CAST(stats_min_value AS BIGINT) AS min_user, CAST(stats_max_value AS BIGINT) AS max_user
FROM parquet_metadata('data/layouts/daily_rg_2k/events-2024-02-15.parquet')
WHERE path_in_schema = 'user_id' ORDER BY row_group_id;

-- sorted: ranges are narrow and non-overlapping                    -> prunes
SELECT row_group_id, row_group_num_rows,
       CAST(stats_min_value AS BIGINT) AS min_user, CAST(stats_max_value AS BIGINT) AS max_user
FROM parquet_metadata('data/layouts/sorted_by_user_rg_2k/events-2024-02-15.parquet')
WHERE path_in_schema = 'user_id' ORDER BY row_group_id;

-- average width of the min/max interval across the whole layout
SELECT AVG(CAST(stats_max_value AS BIGINT) - CAST(stats_min_value AS BIGINT)) AS avg_width
FROM parquet_metadata('data/layouts/daily_rg_2k/*.parquet') WHERE path_in_schema = 'user_id';
SELECT AVG(CAST(stats_max_value AS BIGINT) - CAST(stats_min_value AS BIGINT)) AS avg_width
FROM parquet_metadata('data/layouts/sorted_by_user_rg_2k/*.parquet') WHERE path_in_schema = 'user_id';
```

How many row groups can `user_id = 123` skip?

```sql
SELECT COUNT(*) AS row_groups,
       SUM(CASE WHEN 123 < CAST(stats_min_value AS BIGINT)
                  OR 123 > CAST(stats_max_value AS BIGINT) THEN 1 ELSE 0 END) AS prunable
FROM parquet_metadata('data/layouts/sorted_by_user_rg_2k/*.parquet') WHERE path_in_schema = 'user_id';
```

Inspect what DuckDB actually did:

```sql
EXPLAIN ANALYZE SELECT * FROM read_parquet('data/layouts/sorted_by_user_rg_2k/*.parquet') WHERE user_id = 123;
```

(or read any `results/profiles/<layout>/*.json`).

## 21. Limitations (read before trusting a number)

1. **DuckDB does not report row-group counts in its profile** (verified on DuckDB 1.5.6: the scan
   operator exposes files, filters, projections and `operator_rows_scanned`, but neither "row groups
   total" nor "pruned"). Therefore `row_groups_total/pruned` are **derived from Parquet footers**:
   every row group of every file counts in the total; a row group counts as pruned when Hive path
   predicates exclude its file, or when the predicate interval does not intersect the column's footer
   `[min, max]`. DuckDB may additionally skip row groups via dictionary/bloom filters, so this is a
   *lower bound* of what DuckDB skips. It is a deterministic function of files + predicate, **not a
   measurement of DuckDB**. Use `bytes_read` for what DuckDB really read.
2. `operator_rows_scanned` is **not** a reliable "rows decoded" number: it did not shrink under
   row-group pruning in our probes and can exceed the table's row count (vector-granularity
   rounding per file). It is stored in the detail CSV only as raw information.
3. `bytes_read` is a query-level counter; DuckDB's external file cache makes warm runs read 0 bytes.
4. **Page-cache "cold" is best effort** (section 13).
5. **Peak RSS** is process-wide, sampled, and includes interpreter + libraries (~100+ MiB baseline).
   The allocator may retain memory between runs, so values are upper bounds and not a clean
   per-query footprint. Compaction RSS is `ru_maxrss` of the process after compaction.
6. Profile JSON key names and `extra_info` shape are DuckDB-version dependent. The parser is
   tolerant (tested on several shapes) but unknown future formats may yield empty cells - never
   fabricated values.
7. Results depend on the machine (CPU count, disk, filesystem). Numbers from a 1-core container
   are not representative of a multi-core server: parallel scan of many files/row groups is the
   very thing that differs.
8. Synthetic data: uniform `user_id`, weighted categorical columns, log-normal `value`. Real data
   has skew, which changes pruning and partition behaviour.
9. `partitioned_by_user_id` memory cap (section 15).
10. Only DuckDB is benchmarked; Spark/Trino/Athena behave differently with many small files.

## 22. Verification status of this repository

* Unit tests (`make test`) and the 1M-row pipeline were executed in the authoring environment.
* The `Dockerfile` and `docker-compose.yml` were **written but not executed** there (no Docker
  daemon available); the compose file was only parsed for YAML validity. Treat them as unverified
  until you run `docker compose run --rm test`.

## 23. Reference run (MEASURED)

Everything in this section was copied from `results/analysis.md`, produced by
`ROWS=1000000` (seed 42), DuckDB 1.5.6, PyArrow 25.0.1, **1 CPU core, 3.9 GiB RAM, container
filesystem**, 5 cold + 5 warm runs per cell, medians. `tiny_files` was run one query per
invocation (`--append`) because the sandbox killed long jobs; settings were identical.
Your numbers will differ; regenerate with `make bench && make analyze`.

| layout | files | q1 cold/warm ms | q2 cold/warm ms | q3 cold/warm ms | q4 cold/warm ms | q5 cold/warm ms |
|---|---:|---:|---:|---:|---:|---:|
| daily | 90 | 84.9 / 42.3 | 34.0 / 12.0 | 81.7 / 34.2 | 55.9 / 20.1 | 34.3 / 13.8 |
| daily_rg_10k | 90 | 106.0 / 43.4 | 28.6 / 13.1 | 107.9 / 42.7 | 70.6 / 23.6 | 37.9 / 17.1 |
| daily_rg_1m | 90 | 86.4 / 36.5 | 29.5 / 12.1 | 78.1 / 42.3 | 53.8 / 20.5 | 31.2 / 14.8 |
| sorted_by_user | 90 | 103.2 / 57.6 | 34.8 / 12.6 | 78.2 / 40.9 | 54.8 / 21.7 | 29.8 / 14.7 |
| sorted_by_ts | 90 | 86.2 / 38.0 | 25.5 / 12.1 | 93.1 / 40.4 | 56.5 / 21.0 | 31.2 / 15.2 |
| partitioned_by_date | 90 | 88.8 / 45.3 | 28.4 / 15.5 | 86.2 / 41.5 | 58.8 / 23.0 | 37.1 / 23.3 |
| partitioned_by_date_country | 1,800 | 953.8 / 457.2 | 582.8 / 293.7 | 131.7 / 98.9 | 856.3 / 344.8 | 579.4 / 277.6 |
| hourly | 2,160 | 1,068 / 462 | 622 / 285 | 1,081 / 455 | 963 / 366 | 626 / 346 |
| partitioned_by_user_id | 50,000 | 2,604 / 2,600 | 19,346 / 11,122 | 22,085 / 13,021 | 19,068 / 11,128 | 20,555 / 12,461 |
| tiny_files | 129,551 | 33,140 / 18,085 | 26,690 / 12,385 | 37,227 / 32,904 | 31,267 / 16,176 | 28,298 / 13,773 |
| *daily_rg_2k (extra)* | 90 | 211.0 / 70.6 | 51.7 / 21.0 | 220.0 / 67.7 | 122.8 / 40.0 | 56.4 / 26.1 |
| *sorted_by_user_rg_2k (extra)* | 90 | 81.8 / 36.3 | 42.0 / 22.1 | 212.3 / 67.3 | 122.9 / 37.2 | 49.7 / 25.0 |

| # | Question | Measured answer |
|---|---|---|
| 1 | tiny_files vs daily, one-day query (q2) | cold 26,690 ms vs 34.0 ms = **784x**; warm 12,385 ms vs 12.0 ms = **1,029x** (hourly/daily: 18x cold, 24x warm) |
| 2 | Row groups pruned, `user_id` lookup | `sorted_by_user_rg_2k` **83.3%** (450/540) vs `daily_rg_2k` **0.0%**. With default row groups (1 per file) `sorted_by_user` and `daily` both prune **0%**, so sorting alone did nothing at this scale. Counts are footer-derived (see Limitations). |
| 3 | Best row-group size per query | Inconclusive at 1M rows: `daily` and `daily_rg_1m` are physically identical (same row-group count), so their differences (up to 19%) are noise. `daily_rg_2k` was clearly the slowest (about 2x on q1, q3, q4), and `daily_rg_10k` was slower than `daily` on q1, q3, q4 cold (25-32%). |
| 4 | Compaction runtime | **64.1 s** (129,551 -> 90 files, 15.6k rows/s) |
| 5 | Compaction memory | **316 MiB** process peak RSS (252 MiB baseline before work) |
| 6 | File explosion | `partitioned_by_user_id`: **50,000 files** = 555x `daily`. Point lookup 2.6 s (path pruning: 49,999/50,000 files skipped, 2 KB read) but every other query 270-570x slower cold than `daily` |
| 7 | Larger files always better? | Across the file-size ladder (tiny < hourly < daily): yes, in 10/10 cells at this scale. **Not** across row-group size: larger row groups gave no clear gain (`daily_rg_1m` is physically identical to `daily` at this scale, and the only >=10% gap between them was q3 warm in favour of `daily`), and the supplementary `q2b` shows a partition-key predicate (8.6 ms) beating the plain daily range query (12.0 ms) |

Observations and caveats (MEASURED unless stated):
* Result consistency check passed: every query returned identical results on all layouts.
* `sorted_by_user_rg_2k` pruned 83% of row groups yet DuckDB's `bytes_read` for q1 was 485 KB vs
  207 KB for `daily`: smaller row groups add metadata and per-group overhead. Its q1 latency was
  similar to `daily` (81.8 vs 84.9 ms cold) and about 2.6x faster than `daily_rg_2k` (211 ms).
  Pruning helped relative to the same row-group size, not relative to a single large row group.
* The "within noise" threshold in the analysis is 10%; run-to-run differences of up to ~19% between
  physically identical layouts were observed (e.g. q3 warm, daily vs daily_rg_1m), so treat gaps
  under ~20% as noise on this machine.
* Hive `partitioned_by_date` did not beat `daily` for the mandatory q2 (28.4 vs 34.0 ms cold, similar
  warm) because q2 filters `event_ts`, not `date`; only q2b benefits from partition pruning.
* Single core, 1M rows (~18 MiB per layout): this is the regime where per-file overhead dominates
  and the dataset fits in cache. Do not extrapolate these ratios to 200M rows.
