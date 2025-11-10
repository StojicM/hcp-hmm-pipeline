#!/usr/bin/env python3
from __future__ import annotations

"""
Alignment checker for merged group dscalars.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .logger import get_logger

log = get_logger(__name__)


def _read_columns_map(path: Path) -> List[Tuple[str, int, str, str]]:
    rows: List[Tuple[str, int, str, str]] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            return rows

        fields = {name.lower(): name for name in r.fieldnames}
        merged_key = fields.get("merged_file") or fields.get("output")
        col_key = fields.get("col_index")
        sid_key = fields.get("sid") or fields.get("subject")
        state_key = fields.get("state")

        if not col_key or not sid_key or not state_key:
            raise SystemExit(f"columns_map {path} is missing required headers")

        for row in r:
            merged = row.get(merged_key, "") if merged_key else "combined"
            sid = (row.get(sid_key) or "").strip()
            state = (row.get(state_key) or "").strip()
            col_raw = (row.get(col_key) or "").strip()
            if not sid or not state or not col_raw:
                continue
            try:
                col_idx = int(col_raw)
            except ValueError:
                continue
            rows.append((merged, col_idx, sid, state))
    rows.sort(key=lambda x: (x[0], x[1]))
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
        if not cols:
            log.error("alignment_no_columns", extra={"csv": str(self.cfg.columns_map_csv)})
            return False

        state_to_rows: Dict[str, List[Tuple[str, int, str, str]]] = {}
        for merged, col_idx, sid, state in cols:
            state_to_rows.setdefault(state, []).append((merged, col_idx, sid, state))

        ok = True
        mismatches = []
        for k in range(self.cfg.K):
            state_name = f"state{k}"
            rows = state_to_rows.get(state_name, [])
            if not rows:
                ok = False
                mismatches.append({"state": state_name, "issue": "missing_state"})
                continue
            rows.sort(key=lambda x: x[1])
            for expect_idx, sid in enumerate(sids, start=1):
                if expect_idx > len(rows):
                    ok = False
                    mismatches.append({
                        "state": state_name,
                        "issue": "missing_subject",
                        "expected_sid": sid,
                        "expected_col": expect_idx,
                    })
                    break
                _, col_idx, got_sid, _ = rows[expect_idx - 1]
                if col_idx != expect_idx or got_sid != sid:
                    ok = False
                    mismatches.append({
                        "state": state_name,
                        "issue": "order_mismatch",
                        "got_sid": got_sid,
                        "got_col": col_idx,
                        "expected_sid": sid,
                        "expected_col": expect_idx,
                    })
                    break

        unexpected_states = sorted(k for k in state_to_rows.keys() if not k.startswith("state"))
        if unexpected_states:
            log.warning("alignment_unknown_states", extra={"states": unexpected_states})

        if ok:
            log.info("alignment_ok")
            return True

        log.error("alignment_mismatch", extra={
            "csv": str(self.cfg.columns_map_csv),
            "examples": mismatches[:10],
        })
        return False
