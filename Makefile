# Parquet layout benchmark - see README.md
SHELL := /bin/bash
-include .env
export

PY            ?= python
ROWS          ?= 1000000
SEED          ?= 42
BATCH_ROWS    ?= 100000
RAW_DIR       ?= data/raw
LAYOUTS_DIR   ?= data/layouts
RESULTS_DIR   ?= results
COLD_RUNS     ?= 5
WARM_RUNS     ?= 5
WITH_EXTRAS   ?= 1
FULL_SCALE_ROWS := 200000000

EXTRA_FLAG := $(if $(filter 1,$(WITH_EXTRAS)),--with_extras,)

.PHONY: help install generate layouts queries compact analyze test bench bench-full syntax clean clean-data

help:
	@echo "targets: generate layouts queries compact analyze test bench bench-full syntax clean clean-data"
	@echo "example: ROWS=1000000 make bench      (full-scale target: ROWS=$(FULL_SCALE_ROWS) make bench)"

install:
	$(PY) -m pip install -r requirements.txt

generate:
	$(PY) src/generate_data.py --rows $(ROWS) --out_dir $(RAW_DIR) --seed $(SEED) --batch_rows $(BATCH_ROWS)

layouts:
	$(PY) src/build_layouts.py --raw_dir $(RAW_DIR) --out_dir $(LAYOUTS_DIR) $(EXTRA_FLAG)

queries:
	$(PY) src/run_queries.py --layouts_dir $(LAYOUTS_DIR) --raw_dir $(RAW_DIR) \
	    --results_dir $(RESULTS_DIR) --cold_runs $(COLD_RUNS) --warm_runs $(WARM_RUNS)

compact:
	$(PY) src/compact.py --input_dir $(LAYOUTS_DIR)/tiny_files \
	    --output_dir $(LAYOUTS_DIR)/compacted --results_dir $(RESULTS_DIR)

analyze:
	$(PY) src/analyze_results.py --results_dir $(RESULTS_DIR)

test:
	$(PY) -m pytest

syntax:
	$(PY) -m compileall -q src tests

# Whole pipeline, strictly in order (recursive make keeps ordering even with -j).
bench:
	$(MAKE) generate
	$(MAKE) layouts
	$(MAKE) queries
	$(MAKE) compact
	$(MAKE) analyze

bench-full:
	$(MAKE) bench ROWS=$(FULL_SCALE_ROWS)

clean:
	rm -rf $(RESULTS_DIR)/*.csv $(RESULTS_DIR)/*.json $(RESULTS_DIR)/*.md $(RESULTS_DIR)/profiles $(RESULTS_DIR)/explain

clean-data:
	rm -rf $(RAW_DIR)/* $(LAYOUTS_DIR)/*
	touch $(RAW_DIR)/.gitkeep $(LAYOUTS_DIR)/.gitkeep
