"""
Shared constants and helpers used by every stage of the benchmark.

Nothing in here produces benchmark numbers; it only holds the schema, path
conventions, logging setup and small utilities (manifest I/O, RSS sampling).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

# resource exists on Linux/macOS, but NOT on normal Windows Python.
try:
    import resource
except ImportError:
    resource = None

import pyarrow as pa


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

FULL_SCALE_ROWS = 200_000_000
DEFAULT_ROWS = 1_000_000
DEFAULT_SEED = 42
DEFAULT_DAYS = 90

DEFAULT_START_DATE = "2024-01-01"

DUCKDB_DEFAULT_ROW_GROUP = 122_880

MANIFEST_NAME = "manifest.json"
LAYOUT_META_NAME = "layout_meta.json"

COMPRESSION = "snappy"


EVENTS_SCHEMA = pa.schema(
    [
        ("event_ts", pa.timestamp("us")),
        ("user_id", pa.int64()),
        ("country", pa.string()),
        ("event_type", pa.string()),
        ("device", pa.string()),
        ("value", pa.float64()),
    ]
)

EVENTS_COLUMNS = [f.name for f in EVENTS_SCHEMA]


LAYOUT_CSV_COLUMNS = [
    "layout",
    "row_group_rows",
    "files",
    "total_bytes",
    "query",
    "cache",
    "latency_ms",
    "bytes_read",
    "row_groups_total",
    "row_groups_pruned",
    "peak_rss_mb",
]


COMPACTION_CSV_COLUMNS = [
    "input_files",
    "output_files",
    "wall_s",
    "peak_rss_mb",
    "rows_before",
    "rows_after",
    "checksum_match",
]


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def setup_logging(level: str | int = None) -> logging.Logger:
    """
    Configure application logging.
    """

    level = level or os.environ.get("LOG_LEVEL", "INFO")

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    return logging.getLogger("plb")


# --------------------------------------------------------------------------
# Environment-variable friendly helpers
# --------------------------------------------------------------------------

def env_int(name: str, default: int) -> int:
    """
    Read an integer from an environment variable.
    """

    try:
        return int(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


def env_str(name: str, default: str) -> str:
    """
    Read a string from an environment variable.
    """

    return os.environ.get(name) or default


# --------------------------------------------------------------------------
# Manifest + layout metadata
# --------------------------------------------------------------------------

def write_json(path: Path, obj: dict) -> None:
    """
    Write JSON atomically.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    tmp.replace(path)


def read_manifest(raw_dir: Path) -> dict:
    """
    Read the generated dataset manifest.
    """

    p = Path(raw_dir) / MANIFEST_NAME

    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found - run "
            "`python src/generate_data.py` first."
        )

    return json.loads(
        p.read_text(encoding="utf-8")
    )


def read_layout_meta(layout_dir: Path) -> dict:
    """
    Read layout metadata.
    """

    p = Path(layout_dir) / LAYOUT_META_NAME

    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found - run "
            "`python src/build_layouts.py` first."
        )

    return json.loads(
        p.read_text(encoding="utf-8")
    )


# --------------------------------------------------------------------------
# Filesystem helpers
# --------------------------------------------------------------------------

def list_parquet_files(root: Path) -> list[Path]:
    """
    Return all Parquet files below root in deterministic order.
    """

    return sorted(
        Path(root).rglob("*.parquet")
    )


def dir_stats(root: Path) -> tuple[int, int]:
    """
    Return:

        (number_of_parquet_files, total_bytes_of_parquet_files)
    """

    n = 0
    total = 0

    root = Path(root)

    if not root.exists():
        return 0, 0

    for dirpath, _dirs, files in os.walk(root):
        for filename in files:

            if not filename.endswith(".parquet"):
                continue

            filepath = os.path.join(
                dirpath,
                filename,
            )

            try:
                total += os.stat(filepath).st_size
                n += 1
            except OSError:
                pass

    return n, total


# --------------------------------------------------------------------------
# Page cache helper
# --------------------------------------------------------------------------

def evict_page_cache(
    files: Iterable[os.PathLike],
) -> int:
    """
    Best-effort page-cache eviction.

    Linux supports os.posix_fadvise().
    Windows normally does not.

    Therefore this function simply returns 0 on Windows.
    """

    if not hasattr(os, "posix_fadvise"):
        return 0

    ok = 0

    for f in files:

        try:

            fd = os.open(
                f,
                os.O_RDONLY,
            )

            try:

                os.posix_fadvise(
                    fd,
                    0,
                    0,
                    os.POSIX_FADV_DONTNEED,
                )

                ok += 1

            finally:
                os.close(fd)

        except (OSError, AttributeError):
            pass

    return ok


# --------------------------------------------------------------------------
# Memory helpers
# --------------------------------------------------------------------------

def self_max_rss_mb() -> float:
    """
    Return peak resident set size in MiB.

    Linux/macOS:
        Uses resource.getrusage().

    Windows:
        Uses psutil.

    If neither method is available, returns 0.0.
    """

    # --------------------------------------------------------------
    # Linux / macOS
    # --------------------------------------------------------------

    if resource is not None:

        try:

            usage = resource.getrusage(
                resource.RUSAGE_SELF
            )

            ru = usage.ru_maxrss

            # macOS reports bytes.
            if sys.platform == "darwin":
                return ru / (1024 * 1024)

            # Linux reports KiB.
            return ru / 1024

        except Exception:
            pass

    # --------------------------------------------------------------
    # Windows fallback
    # --------------------------------------------------------------

    try:

        import psutil

        process = psutil.Process(
            os.getpid()
        )

        rss = process.memory_info().rss

        return rss / (1024 * 1024)

    except Exception:
        return 0.0


def current_rss_mb() -> float:
    """
    Return current process RSS in MiB.
    """

    try:

        import psutil

        process = psutil.Process(
            os.getpid()
        )

        return (
            process.memory_info().rss
            / (1024 * 1024)
        )

    except Exception:

        return self_max_rss_mb()


# --------------------------------------------------------------------------
# RSS sampler
# --------------------------------------------------------------------------

class RssSampler:
    """
    Samples process RSS in a background thread.

    The peak RSS observed during the operation is stored in:

        sampler.peak_mb
    """

    def __init__(
        self,
        interval_s: float = 0.005,
    ):

        self.interval = interval_s

        self._stop = threading.Event()

        self._thread: Optional[
            threading.Thread
        ] = None

        self.peak_mb = 0.0

    def __enter__(self):

        # Capture RSS before the operation starts.
        self.peak_mb = current_rss_mb()

        self._stop.clear()

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

        self._thread.start()

        return self

    def _run(self):

        while not self._stop.is_set():

            try:

                current = current_rss_mb()

                self.peak_mb = max(
                    self.peak_mb,
                    current,
                )

            except Exception:
                pass

            time.sleep(
                self.interval
            )

    def __exit__(
        self,
        *exc,
    ):

        self._stop.set()

        if self._thread:

            self._thread.join()

        # Final measurement.
        self.peak_mb = max(
            self.peak_mb,
            current_rss_mb(),
        )

        return False