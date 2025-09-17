#!/usr/bin/env python3
from __future__ import annotations

"""
Alignment checker for merged group dscalars.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .logger import get_logger

log = get_logger(__name__)


def _read_columns_map(path: Path) -> List[Tuple[int, str, str]]:
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((int(row["col_index"]), row["sid"], row["state"]))
    rows.sort(key=lambda x: x[0])
    return rows


def _sniff_delim(path: Path) -> str:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")[:2048]
    return "\t" if txt.count("\t") >= txt.count(",") else ","


def _colname(keys, candidates) -> str | None:
    lkeys = [k.strip().lower() for k in keys]
    for cand in candidates:
        if cand in lkeys:
            return keys[lkeys.index(cand)]
    return None


# def _read_subjects_used(path: Path) -> List[str]:
#     delim = _sniff_delim(path)
#     with open(path, newline="") as f:
#         r = csv.DictReader(f, delimiter=delim)
#         keys = list(r.fieldnames or [])
#         sid_key = _colname(keys, ["sid", "subject", "participant", "participant_id", "subject_id", "id", "subid"])
#         if sid_key is None:
#             # Fallback: take the first column
#             sid_key = keys[0] if keys else None
#         if sid_key is None:
#             return []
        # return [ (row.get(sid_key, "").strip()) for row in r if row.get(sid_key) ]
def _read_subjects_used(path: Path) -> List[str]:
    delim = _sniff_delim(path)
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter=delim)
        keys = list(r.fieldnames or [])
        sid_key = _colname(keys, ["sid","subject","participant","participant_id","subject_id","id","subid"])
        if sid_key is None:
            sid_key = keys[0] if keys else None
        if sid_key is None:
            return []
        # gather then dedupe while preserving order
        seen = set()
        uniq = []
        for row in r:
            sid = (row.get(sid_key, "") or "").strip()
            if sid and sid not in seen:
                seen.add(sid)
                uniq.append(sid)
        return uniq

@dataclass
class AlignmentConfig:
    columns_map_csv: Path
    subjects_used_csv: Path
    K: int


class AlignmentChecker:
    def __init__(self, cfg: AlignmentConfig):
        self.cfg = cfg

    def check(self) -> bool:
        cols = _read_columns_map(self.cfg.columns_map_csv)
        sids = _read_subjects_used(self.cfg.subjects_used_csv)
        expected = []
        col_idx = 1
        for sid in sids:
            for k in range(self.cfg.K):
                expected.append((col_idx, sid, f"state{k}"))
                col_idx += 1
        ok = True
        mismatches = []
        for got, exp in zip(cols, expected):
            if got != exp:
                ok = False
                mismatches.append((got, exp))
                if len(mismatches) >= 10:
                    break
        if ok and len(cols) == len(expected):
            log.info("alignment_ok")
            return True
        log.error("alignment_mismatch", extra={
            "merged_columns": len(cols),
            "expected": len(expected),
            "examples": mismatches,
        })
        return False
