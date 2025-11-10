#!/usr/bin/env python3
from __future__ import annotations

"""Export per-subject parcel TSVs directly from z-scored parcel betas."""

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .logger import get_logger
from .workbench import export_label_table

log = get_logger(__name__)


@dataclass
class ParcellateStatesConfig:
    in_dir: Path            # betas_dir containing *_zscored.txt
    out_dir: Path           # betas_dir/parcel
    labels_dlabel: Path     # atlas .dlabel.nii
    K: int


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
        pat = str(self.cfg.in_dir / f"*_state_betas_{self.cfg.K}S_zscored.txt")
        files = sorted(glob.glob(pat))
        return [Path(p) for p in files]

    def run(self) -> List[Path]:
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.export_labels()

        inputs = self._iter_inputs()
        if not inputs:
            log.warning("no_state_txt_found", extra={"dir": str(self.cfg.in_dir), "K": int(self.cfg.K)})
            return []

        outs: List[Path] = []
        for txt in inputs:
            sid = txt.name.split("_")[0]
            arr = np.loadtxt(txt, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != self.cfg.K:
                log.warning("bad_shape", extra={"sid": sid, "shape": str(arr.shape)})
                continue

            out_tsv = self.cfg.out_dir / txt.name.replace("_zscored.txt", "_z.parc.tsv")
            np.savetxt(out_tsv, arr.astype(np.float32), fmt="%.6f", delimiter="\t")
            outs.append(out_tsv)
            log.info("parcel_tsv_written", extra={"sid": sid, "out": str(out_tsv)})

        log.info("parcellate_states_done", extra={"n": len(outs), "out_dir": str(self.cfg.out_dir)})
        return outs
