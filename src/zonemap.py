"""Footer-statistics (zone-map) evaluation of Parquet row groups.

DuckDB's profile does not report how many row groups a scan skipped.  This
module computes that number *from the real Parquet footers* of the files in a
layout:

* every row group carries per-column ``min``/``max`` statistics;
* a row group can be skipped for an equality / range predicate iff the
  predicate interval does not intersect ``[min, max]``;
* Hive partition predicates (``date=...``, ``user_id=...``, ``country=...``)
  skip *whole files* before any footer is read.

This is a deterministic function of the on-disk data and the predicate - not a
measurement of DuckDB.  DuckDB may skip additional row groups through
dictionary / bloom-filter checks, so the figure is a *lower bound* on what
DuckDB can skip using min/max alone.  DuckDB's own ``total_bytes_read``
(see ``profiling.py``) remains the ground truth for I/O.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq

log = logging.getLogger("plb.zonemap")


@dataclass(frozen=True)
class Predicate:
    """column == lo   (op='eq')   or   lo <= column <= hi   (op='between')."""
    column: str
    op: str
    lo: Any
    hi: Any = None


@dataclass
class FileEntry:
    path: str
    partition: dict            # hive key -> raw string value
    # one dict per row group: column -> (min, max) or None if no stats
    row_groups: list = field(default_factory=list)


def parse_hive_partitions(root: Path, file: Path) -> dict:
    out = {}
    for part in file.relative_to(root).parts[:-1]:
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


class FooterIndex:
    """Min/max statistics for the columns we filter on, for every file."""

    STAT_COLUMNS = ("event_ts", "user_id", "country")

    def __init__(self, root: Path, files: list[Path]):
        self.root = Path(root)
        self.entries: list[FileEntry] = []
        for f in files:
            md = pq.read_metadata(f)
            names = [md.schema.column(i).name for i in range(md.num_columns)]
            idx = {n: i for i, n in enumerate(names) if n in self.STAT_COLUMNS}
            fe = FileEntry(str(f), parse_hive_partitions(self.root, f))
            for rg_i in range(md.num_row_groups):
                rg = md.row_group(rg_i)
                stats = {}
                for name, ci in idx.items():
                    st = rg.column(ci).statistics
                    if st is not None and st.has_min_max:
                        stats[name] = (st.min, st.max)
                    else:
                        stats[name] = None
                fe.row_groups.append(stats)
            self.entries.append(fe)

    @property
    def total_row_groups(self) -> int:
        return sum(len(e.row_groups) for e in self.entries)

    # ------------------------------------------------------------------
    def evaluate(self, predicates: list[Predicate]) -> dict:
        """Return counts of row groups / files skipped by the predicates."""
        total = pruned = files_pruned_by_path = 0
        files_with_survivors = 0
        for e in self.entries:
            n_rg = len(e.row_groups)
            total += n_rg
            if self._path_excludes(e, predicates):
                pruned += n_rg
                files_pruned_by_path += 1
                continue
            survivors = 0
            for stats in e.row_groups:
                if self._stats_exclude(stats, predicates, e):
                    pruned += 1
                else:
                    survivors += 1
            if survivors:
                files_with_survivors += 1
        return {
            "row_groups_total": total,
            "row_groups_pruned": pruned,
            "files_total": len(self.entries),
            "files_pruned_by_path": files_pruned_by_path,
            "files_with_surviving_row_groups": files_with_survivors,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_partition(col: str, raw: str, like: Any):
        if isinstance(like, int) and not isinstance(like, bool):
            return int(raw)
        if isinstance(like, dt.datetime):
            return dt.datetime.fromisoformat(raw)
        if isinstance(like, dt.date):
            return dt.date.fromisoformat(raw)
        return raw

    def _path_excludes(self, e: FileEntry, preds: list[Predicate]) -> bool:
        for p in preds:
            if p.column not in e.partition:
                continue
            try:
                v = self._coerce_partition(p.column, e.partition[p.column], p.lo)
            except ValueError:
                continue
            if p.op == "eq" and v != p.lo:
                return True
            if p.op == "between" and not (p.lo <= v <= p.hi):
                return True
        return False

    @staticmethod
    def _stats_exclude(stats: dict, preds: list[Predicate], e: FileEntry) -> bool:
        for p in preds:
            if p.column in e.partition:
                continue  # column is not stored in the file
            mm = stats.get(p.column)
            if mm is None:
                continue  # no statistics -> cannot prune
            mn, mx = mm
            try:
                if p.op == "eq" and (p.lo < mn or p.lo > mx):
                    return True
                if p.op == "between" and (p.hi < mn or p.lo > mx):
                    return True
            except TypeError:
                continue
        return False
