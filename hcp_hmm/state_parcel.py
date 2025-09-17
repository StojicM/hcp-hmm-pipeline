#!/usr/bin/env python3
from __future__ import annotations

"""
Parcellate subject-level state maps (dscalar with K columns) into parcel tables.

Replaces src/10_parcellate_states.py with an OOP runner that uses
Connectome Workbench `wb_command` to:
  - export a parcel label table
  - parcellate each subject's z-scored state maps to parcels (MEAN)
  - write per-subject TSVs (rows: parcels, cols: states)

Outputs go under `<betas_dir>/parcel`.
"""

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .logger import get_logger
from .workbench import ensure_workbench, export_label_table, run_wb

log = get_logger(__name__)


@dataclass
class ParcellateStatesConfig:
    in_dir: Path            # betas_dir containing *_zscored.dscalar.nii
    out_dir: Path           # betas_dir/parcel
    labels_dlabel: Path     # atlas .dlabel.nii
    K: int
    method: str = "MEAN"


class StateParcellator:
    def __init__(self, cfg: ParcellateStatesConfig):
        self.cfg = cfg

    def export_labels(self) -> Path:
        out = self.cfg.out_dir / "parcel_labels.tsv"
        if not out.exists():
            log.info("export_parcel_labels", extra={"dlabel": str(self.cfg.labels_dlabel), "out": str(out)})
            out.parent.mkdir(parents=True, exist_ok=True)
            export_label_table(self.cfg.labels_dlabel, out)
        return out

    def _iter_inputs(self) -> List[Path]:
        pat = str(self.cfg.in_dir / f"*_state_betas_{self.cfg.K}S_zscored.dscalar.nii")
        files = sorted(glob.glob(pat))
        return [Path(p) for p in files]

    def run(self) -> List[Path]:
        ensure_workbench()
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.export_labels()

        inputs = self._iter_inputs()
        if not inputs:
            log.warning("no_state_dscalars_found", extra={"dir": str(self.cfg.in_dir), "K": int(self.cfg.K)})
            return []

        outs: List[Path] = []
        for dscalar in inputs:
            sid = dscalar.name.split("_")[0]
            out_pscalar = self.cfg.out_dir / dscalar.name.replace(".dscalar.nii", ".pscalar.nii")
            out_tsv = self.cfg.out_dir / dscalar.name.replace("_zscored.dscalar.nii", "_z.parc.tsv")

            if out_tsv.exists():
                log.info("skip_parcellate_states_subject", extra={"sid": sid, "tsv": str(out_tsv)})
                outs.append(out_tsv)
                continue

            log.info("parcellate_states_subject", extra={"sid": sid, "in": str(dscalar)})
            run_wb([
                "-cifti-parcellate",
                str(dscalar),
                str(self.cfg.labels_dlabel),
                "COLUMN",
                str(out_pscalar),
                "-method",
                self.cfg.method,
            ])
            run_wb(["-cifti-convert", "-to-text", str(out_pscalar), str(out_tsv)])
            outs.append(out_tsv)
        log.info("parcellate_states_done", extra={"n": len(outs), "out_dir": str(self.cfg.out_dir)})
        return outs

