#!/usr/bin/env python3
from __future__ import annotations

"""
Z-score V×K beta maps and export as K-map dscalar per subject.
Replicates src/07_zscore_export.py with structured logging.
"""

import glob
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .logger import get_logger
from .dtseries import get_dt_path, index_dtseries
from .workbench import ensure_workbench, merge_cifti, run_wb

log = get_logger(__name__)


def _make_template(dtseries_path: Path, out_tmpl: Path):
    if not out_tmpl.exists():
        run_wb(["-cifti-reduce", str(dtseries_path), "MEAN", str(out_tmpl)])


def _zscore_cols(VK: np.ndarray) -> np.ndarray:
    mu = VK.mean(axis=0)
    sd = VK.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return (VK - mu) / sd


@dataclass
class ZScoreConfig:
    dtseries_dir: Path
    betas_dir: Path
    K: int
    redo: bool = False


class ZScoreExporter:
    def __init__(self, cfg: ZScoreConfig):
        self.cfg = cfg

    def run(self) -> None:
        ensure_workbench()
        dt_dir = self.cfg.dtseries_dir
        betas_dir = self.cfg.betas_dir

        txts = sorted(glob.glob(str(betas_dir / f"*_state_betas_{self.cfg.K}S.txt")))
        if not txts:
            raise SystemExit(f"No files matching *_state_betas_{self.cfg.K}S.txt in {betas_dir}")

        dt_lookup = index_dtseries(dt_dir)

        for txt in txts:
            txt = Path(txt)
            sid = txt.name.split("_")[0]
            dt = dt_lookup.get(sid) or get_dt_path(dt_dir, sid)
            if dt is None:
                log.warning("missing_dtseries", extra={"sid": sid, "dir": str(dt_dir)})
                continue

            ztxt = txt.with_name(txt.stem + "_zscored.txt")
            zdscalar = txt.with_name(txt.stem + "_zscored.dscalar.nii")
            tmpl = txt.with_name(txt.stem + "_zscored.tmpl.dscalar.nii")

            if zdscalar.exists() and not self.cfg.redo:
                log.info(f"zscore exists {sid}", extra={"sid": sid, "dscalar": str(zdscalar)})
                continue

            VK = np.loadtxt(txt, dtype=np.float32)
            if VK.ndim != 2 or VK.shape[1] != self.cfg.K:
                raise SystemExit(f"{txt.name}: expected V×K with K={self.cfg.K}, got {VK.shape}")
            VKz = _zscore_cols(VK)
            np.savetxt(ztxt, VKz.astype(np.float32), fmt="%.6f")

            _make_template(dt, tmpl)

            single_maps: List[Path] = []
            with tempfile.TemporaryDirectory() as tdir:
                for k in range(self.cfg.K):
                    col_txt = Path(tdir) / f"col_{k}.txt"
                    np.savetxt(col_txt, VKz[:, [k]], fmt="%.6f")
                    col_dc = Path(tdir) / f"map_{k}.dscalar.nii"
                    run_wb(["-cifti-convert", "-from-text", str(col_txt), str(tmpl), str(col_dc)])
                    single_maps.append(col_dc)
                merge_cifti(zdscalar, single_maps)

            log.info(f"zscore {sid}", extra={"sid": sid, "dscalar": str(zdscalar)})
