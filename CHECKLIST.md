# Requirement -> implementation map

| Requirement | Where |
|---|---|
| Project structure (README, Makefile, Dockerfile, compose, .env.example, requirements, src/, tests/, data/, results/) | repo root (extra helpers: `src/common.py`, `src/profiling.py`, `src/zonemap.py`, `tests/test_analysis.py`, `pytest.ini`) |
| Phase 1: CLI `--rows --out_dir --seed`; 200M supported; schema; 90 days; seeded; batched; Snappy | `src/generate_data.py` `generate()`, `generate_batch()`, `--full_scale`; tests `test_generation.py` |
| Phase 2: 10 required layouts, none removable | `src/build_layouts.py` `LAYOUTS`, `REQUIRED_LAYOUTS`; `test_all_required_layouts_registered` |
| tiny/hourly/daily bucketing; row groups 122,880 / 10k / 1M | `build_time_files`, `iter_buckets`; `LayoutSpec` |
| Hive date, nested date+country | `build_hive_date`, `build_hive_date_country` |
| sorted_by_user / sorted_by_ts | `build_time_files` with `sort_by` |
| partitioned_by_user_id (induced failure) | `build_hive_user` (memory cap `--user_partition_max_rows`, reported as `truncated`) |
| Phase 3: 5 queries | `run_queries.py` `make_queries()` (+ supplementary `q2b`) |
| 5 cold (fresh connection) / 5 warm (shared connection) | `benchmark_layout()`; `test_cold_uses_fresh_connection_warm_reuses` |
| Profiling, recursive tolerant parser | `src/profiling.py` `walk`, `extract_profile_metrics`; parser tests in `test_queries.py` |
| bytes read, files, rows scanned, filters/projections, peak RSS | profile fields + `RssSampler`; `results/layout_detail.csv` |
| row groups total/pruned (not in DuckDB profile) | derived from footers: `src/zonemap.py`; limitation documented in README s21 |
| `results/layout.csv` exact 11 columns | `common.LAYOUT_CSV_COLUMNS`; `test_benchmark_produces_exact_layout_csv_header` |
| Phase 4 compaction (1-11) | `src/compact.py` `compact()`, `checksum()`, `sums_match()` |
| `results/compaction.csv` exact 7 columns | `common.COMPACTION_CSV_COLUMNS`; `test_compaction_csv_schema_and_values` |
| Tests (schema, rows, layouts, partitions, sorted, queries, parser, compaction, checksums, CSV schemas) | `tests/` (76 tests, 8k-row datasets) |
| Makefile: generate layouts queries compact test bench, `ROWS=... make bench` | `Makefile` (+ analyze, bench-full, syntax, clean) |
| Docker / compose / healthchecks | `Dockerfile`, `docker-compose.yml` (written, NOT executed in authoring sandbox) |
| README (19 topics) | `README.md` sections 1-19 (+20 human verification, 21 limitations, 22 status, 23 measured run) |
| Analysis items 1-7 | `src/analyze_results.py` -> `results/analysis.md` |
| Human verification (parquet_metadata, min/max) | README section 20 |
| No fake metrics | missing metrics stay empty; `test_parser_never_invents_values` |
