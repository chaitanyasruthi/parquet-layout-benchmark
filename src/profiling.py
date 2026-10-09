"""Parsing of DuckDB query profiles (``enable_profiling = 'json'``).

The JSON emitted by DuckDB is an operator tree whose exact key names differ
between versions (``operator_type`` vs ``name`` vs ``operator_name``,
``extra_info`` as a dict vs a multi-line string, ...).  Therefore this module:

* walks the tree *recursively* and never assumes a fixed depth;
* identifies Parquet scans by looking at several name-like keys and at
  ``extra_info['Function']``;
* looks up metrics by a list of candidate key names and returns ``None``
  (never a made-up number) when a metric is absent.

What DuckDB does **not** report (as of the version this project was developed
against, 1.5.x) and therefore is not parsed here:

* the number of row groups in a scan, or how many were skipped by min/max
  statistics.  See ``zonemap.py`` for the footer-derived estimate and README
  "Limitations".
* per-operator bytes read (``total_bytes_read`` exists only at query level).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator, Optional

# --------------------------------------------------------------------------
# Tree helpers
# --------------------------------------------------------------------------
_NAME_KEYS = ("operator_type", "operator_name", "name", "type")


def walk(node: Any) -> Iterator[dict]:
    """Depth-first, recursive traversal of a profile tree (dict with
    ``children``).  Tolerates lists, missing ``children`` and non-dict junk."""
    if isinstance(node, list):
        for item in node:
            yield from walk(item)
        return
    if not isinstance(node, dict):
        return
    yield node
    for child in node.get("children") or []:
        yield from walk(child)


def _names(node: dict) -> str:
    parts = [str(node.get(k, "")) for k in _NAME_KEYS]
    extra = node.get("extra_info")
    if isinstance(extra, dict):
        parts.append(str(extra.get("Function", "")))
    elif isinstance(extra, str):
        parts.append(extra)
    return " ".join(parts).upper()


def is_parquet_scan(node: dict) -> bool:
    """True for READ_PARQUET / PARQUET_SCAN table scans (any DuckDB version)."""
    names = _names(node)
    if "PARQUET" in names and ("SCAN" in names or "READ" in names):
        return True
    return False


def _extra_dict(node: dict) -> dict:
    """Normalise extra_info to a dict.  Old versions emit one string with
    ``Key:\\nvalue`` lines; new versions emit a dict."""
    extra = node.get("extra_info")
    if isinstance(extra, dict):
        return extra
    out: dict = {}
    if isinstance(extra, str):
        key = None
        for line in extra.splitlines():
            m = re.match(r"^\s*([A-Za-z][\w \-\(\)/]*?):\s*(.*)$", line)
            if m:
                key = m.group(1).strip()
                out[key] = m.group(2).strip()
            elif key and line.strip():
                out[key] = (str(out.get(key, "")) + " " + line.strip()).strip()
    return out


def _first(d: dict, keys: tuple[str, ...]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _to_num(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d+(?:\.\d+)?", str(v).replace(",", ""))
    return float(m.group(0)) if m else None


def _as_list(v: Any) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [s for s in str(v).splitlines() if s.strip()]


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def extract_profile_metrics(profile: dict) -> dict:
    """Summarise a DuckDB JSON profile.  Missing metrics are ``None``.

    Returned keys
    -------------
    latency_s            query latency according to the profiler
    bytes_read           query-level total bytes read from storage
    rows_scanned         sum of ``operator_rows_scanned`` of Parquet scans
    rows_after_pushdown  sum of ``operator_cardinality`` of Parquet scans
    scan_operators       number of Parquet scan operators found
    files_read           sum of 'Total Files Read' (files actually opened)
    files_total          files known to the scan before file-level pruning
    files_scanned_hint   raw 'Scanning Files' text e.g. '1/90' (hive pruning)
    filters / projections  pushdown info as reported by the scan operator
    filter_pushdown      True if any scan carries a Filters / File Filters entry
    projection_columns   max number of columns projected by any scan
    peak_buffer_mb       system_peak_buffer_memory (DuckDB buffer manager)
    """
    nodes = list(walk(profile))
    root = nodes[0] if nodes else {}

    latency = _first(root, ("latency", "query_time", "total_time", "timing"))
    bytes_read = _first(root, ("total_bytes_read", "bytes_read"))
    if bytes_read is None:  # older versions: sum operator-level counters if any
        vals = [n.get("total_bytes_read") for n in nodes if "total_bytes_read" in n]
        bytes_read = sum(v for v in vals if isinstance(v, (int, float))) if vals else None
    peak_buf = _first(root, ("system_peak_buffer_memory", "peak_buffer_memory"))

    scans = [n for n in nodes if is_parquet_scan(n)]
    rows_scanned = rows_card = None
    files_read = files_total = None
    filters: list[str] = []
    projections: list[str] = []
    hints: list[str] = []
    proj_cols = None
    for s in scans:
        ex = _extra_dict(s)
        rs = _to_num(_first(s, ("operator_rows_scanned", "rows_scanned")))
        rc = _to_num(_first(s, ("operator_cardinality", "cardinality")))
        rows_scanned = (rows_scanned or 0) + rs if rs is not None else rows_scanned
        rows_card = (rows_card or 0) + rc if rc is not None else rows_card

        fr = _to_num(ex.get("Total Files Read"))
        if fr is not None:
            files_read = (files_read or 0) + fr
        sf = ex.get("Scanning Files")
        if sf:
            hints.append(str(sf))
            m = re.match(r"\s*(\d+)\s*/\s*(\d+)", str(sf))
            if m:
                files_total = (files_total or 0) + int(m.group(2))
        for key in ("Filters", "File Filters"):
            filters += [f"{key}: {x}" for x in _as_list(ex.get(key))]
        p = _as_list(ex.get("Projections"))
        projections += p
        proj_cols = len(p) if proj_cols is None else max(proj_cols, len(p))

    if files_total is None and files_read is not None:
        files_total = files_read  # no file-level pruning reported
    return {
        "latency_s": float(latency) if latency is not None else None,
        "bytes_read": int(bytes_read) if bytes_read is not None else None,
        "rows_scanned": int(rows_scanned) if rows_scanned is not None else None,
        "rows_after_pushdown": int(rows_card) if rows_card is not None else None,
        "scan_operators": len(scans),
        "files_read": int(files_read) if files_read is not None else None,
        "files_total": int(files_total) if files_total is not None else None,
        "files_scanned_hint": ";".join(hints) or None,
        "filters": filters,
        "projections": projections,
        "filter_pushdown": bool(filters),
        "projection_columns": proj_cols,
        "peak_buffer_mb": (float(peak_buf) / 1048576.0) if peak_buf is not None else None,
    }


def load_profile(path: Path) -> dict:
    text = Path(path).read_text()
    if not text.strip():
        raise ValueError(f"empty profile file: {path}")
    return json.loads(text)
