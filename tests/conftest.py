"""Shared, session-scoped fixtures. Everything uses tiny datasets (<= 8,000 rows)."""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import build_layouts  # noqa: E402
import generate_data  # noqa: E402
import run_queries    # noqa: E402

SMALL_ROWS = 8_000
SEED = 42


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    generate_data.generate(SMALL_ROWS, d, seed=SEED, batch_rows=2_000)
    return d


@pytest.fixture(scope="session")
def layouts_dir(tmp_path_factory, raw_dir):
    d = tmp_path_factory.mktemp("layouts")
    names = build_layouts.REQUIRED_LAYOUTS + build_layouts.EXTRA_LAYOUTS
    build_layouts.build_all(raw_dir, d, names)
    return d


@pytest.fixture(scope="session")
def dense_raw(tmp_path_factory):
    """2 days x 3,000 rows/day: enough rows per file for multiple row groups."""
    d = tmp_path_factory.mktemp("dense_raw")
    generate_data.generate(6_000, d, seed=SEED, batch_rows=1_500, days=2, n_users=1_000)
    return d


@pytest.fixture(scope="session")
def dense_layouts(tmp_path_factory, dense_raw):
    d = tmp_path_factory.mktemp("dense_layouts")
    build_layouts.build_all(dense_raw, d, ["daily", "daily_rg_10k", "daily_rg_2k",
                                           "sorted_by_user_rg_2k", "sorted_by_user",
                                           "sorted_by_ts"])
    return d


@pytest.fixture(scope="session")
def results_dir(tmp_path_factory, raw_dir, layouts_dir):
    """A real (tiny) benchmark run over every layout, 1 cold + 1 warm run."""
    out = tmp_path_factory.mktemp("results")
    rc = run_queries.main([
        "--layouts_dir", str(layouts_dir), "--raw_dir", str(raw_dir),
        "--results_dir", str(out), "--cold_runs", "1", "--warm_runs", "1",
        "--no_evict_cache"])
    assert rc == 0
    return out
