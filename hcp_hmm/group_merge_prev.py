#!/usr/bin/env python3
from __future__ import annotations

"""
Merge subject dscalars into a group dscalar and emit a columns map CSV.
Replicates src/08_group_merge.sh but in Python with logging.
"""

import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .logger import get_logger

log = get_logger(__name__)


def _check_wb():
    if shutil.which("wb_command") is None:
        raise SystemExit("wb_command not found in PATH. Install Workbench.")


@dataclass
class GroupMergeConfig:
    betas_dir: Path
    K: int
    out_dir: Path
    subjects_used_csv: Path


class GroupMerger:
    def __init__(self, cfg: GroupMergeConfig):
        self.cfg = cfg

    @staticmethod
    def _sniff_delim(path: Path) -> str:
        txt = Path(path).read_text(encoding="utf-8", errors="ignore")[:2048]
        return "\t" if txt.count("\t") >= txt.count(",") else ","

    @staticmethod
    def _read_sids(path: Path) -> List[str]:
        delim = GroupMerger._sniff_delim(path)
        with open(path, newline="") as f:
            r = csv.DictReader(f, delimiter=delim)
            keys = list(r.fieldnames or [])
            # Prefer 'sid' else 'Subject' else first column
            sid_key = None
            for cand in ("sid", "Subject", "subject", "participant", "participant_id", "id"):
                if cand in keys:
                    sid_key = cand; break
            if sid_key is None:
                sid_key = keys[0] if keys else None
            if sid_key is None:
                return []
            sids = [str((row.get(sid_key) or "").strip()) for row in r]
        return [x for x in sids if x]

    def run(self) -> Path:
        _check_wb()
        out_dir = self.cfg.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        sids = self._read_sids(self.cfg.subjects_used_csv)
        Ktag = f"{self.cfg.K}S"
        merged = out_dir / f"allsubs_{Ktag}_zscored.dscalar.nii"
        mapcsv = out_dir / "columns_map.csv"

        cmd = ["wb_command", "-cifti-merge", str(merged)]
        with open(mapcsv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["col_index", "sid", "state"])
            col = 1
            for sid in sids:
                fpath = self.cfg.betas_dir / f"{sid}_state_betas_{Ktag}_zscored.dscalar.nii"
                if not fpath.exists():
                    log.warning("missing_subject_dscalar", extra={"sid": sid, "path": str(fpath)})
                    continue
                cmd += ["-cifti", str(fpath)]
                for s in range(self.cfg.K):
                    w.writerow([col, sid, f"state{s}"])
                    col += 1

        log.info("run_shell", extra={"cmd": cmd})
        subprocess.run(cmd, check=True)
        log.info("group_merged", extra={"merged": str(merged), "map": str(mapcsv)})
        return merged
