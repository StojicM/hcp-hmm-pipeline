#!/usr/bin/env python3
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .logger import get_logger
from .workbench import ensure_workbench, run_wb
log = get_logger(__name__)

def _sniff_delim(path: Path) -> str:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")[:2048]
    return "\t" if txt.count("\t") >= txt.count(",") else ","

# def _read_sids(path: Path) -> List[str]:
#     delim = _sniff_delim(path)
#     with open(path, newline="") as f:
#         r = csv.DictReader(f, delimiter=delim)
#         keys = list(r.fieldnames or [])
#         sid_key = next((k for k in ("sid","Subject","subject","participant","participant_id","id") if k in keys), None)
#         if sid_key is None:
#             sid_key = keys[0] if keys else None
#         if sid_key is None:
#             return []
#         sids = [str((row.get(sid_key) or "").strip()) for row in r]
#     return [x for x in sids if x]

def _read_sids(path: Path) -> List[str]:
    delim = _sniff_delim(path)
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter=delim)
        keys = list(r.fieldnames or [])
        sid_key = next((k for k in ("sid","Subject","subject","participant","participant_id","id") if k in keys), None)
        if sid_key is None:
            sid_key = keys[0] if keys else None
        if sid_key is None:
            return []
        sids = [str((row.get(sid_key) or "").strip()) for row in r]
    # order-preserving unique
    seen = set()
    uniq = []
    for x in sids:
        if x and x not in seen:
            seen.add(x); uniq.append(x)
    return uniq

def _count_maps(dscalar: Path) -> int:
    # Count columns (maps) in a CIFTI-2 dscalar
    proc = run_wb(
        ["-file-information", str(dscalar), "-only-map-names"],
        capture_output=True,
    )
    stdout = proc.stdout if proc and proc.stdout is not None else ""
    return len([ln for ln in stdout.splitlines() if ln.strip()])

@dataclass
class GroupMergeConfig:
    betas_dir: Path            # folder with per-subject dscalars
    K: int                     # number of states you want to keep per subject (e.g., 6)
    out_dir: Path              # group folder
    subjects_used_csv: Path    # file with subject order (same order as design/EB)

class GroupMerger:
    def __init__(self, cfg: GroupMergeConfig):
        self.cfg = cfg

    def run(self) -> Path:
        """
        Build a (Nsub × K)-map group dscalar  by taking exactly the first K columns
        from each subject dscalar, in the subject order given by subjects_used_csv.
        """
        ensure_workbench()
        out_dir = self.cfg.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        sids = _read_sids(self.cfg.subjects_used_csv)
        if not sids:
            raise SystemExit(f"No subject IDs found in {self.cfg.subjects_used_csv}")

        Ktag = f"{self.cfg.K}S"
        merged = out_dir / f"allsubs_{Ktag}_zscored.dscalar.nii"
        mapcsv = out_dir / "columns_map.csv"

        cmd = ["-cifti-merge", str(merged)]
        with open(mapcsv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["col_index", "sid", "state"])  # 1-based column index in the MERGED file
            col = 1

            for sid in sids:
                fpath = self.cfg.betas_dir / f"{sid}_state_betas_{Ktag}_zscored.dscalar.nii"
                if not fpath.exists():
                    log.warning("missing_subject_dscalar", extra={"sid": sid, "path": str(fpath)})
                    continue

                n_maps = _count_maps(fpath)
                
                if n_maps < self.cfg.K:
                    raise SystemExit(f"{fpath} has only {n_maps} maps; need at least {self.cfg.K}.")

                # IMPORTANT: select only the first K columns (states) explicitly
                for s in range(1, self.cfg.K + 1):     # wb columns are 1-based
                    cmd += ["-cifti", str(fpath), "-column", str(s)]
                    # keep your 0..K-1 state naming convention in the CSV
                    w.writerow([col, sid, f"state{s-1}"])
                    col += 1

        log.info("run_shell", extra={"cmd": cmd})
        run_wb(cmd)

        # sanity check: expect len(sids)*K maps (e.g., 6*6=36)
        total = _count_maps(merged)
        expect = len(sids) * self.cfg.K
        if total != expect:
            log.warning("merged_map_count_mismatch", extra={"got": total, "expected": expect, "file": str(merged)})
        else:
            log.info("group_merged", extra={"merged": str(merged), "maps": total, "map_csv": str(mapcsv)})

        return merged
