#!/usr/bin/env python3
from __future__ import annotations

"""Z-score parcel beta maps and export as CIFTI pscalar files."""

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np
from nibabel.cifti2.cifti2_axes import ScalarAxis

from .logger import get_logger
from .ptseries import get_pt_path, index_ptseries

log = get_logger(__name__)


def _zscore_cols(PK: np.ndarray) -> np.ndarray:
    mu = PK.mean(axis=0)
    sd = PK.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return (PK - mu) / sd


def _write_pscalar(data: np.ndarray, template_img: nib.cifti2.Cifti2Image, labels: List[str], out_path: Path) -> None:
    if data.ndim != 2:
        raise SystemExit(f"Expected 2D array for pscalar export; got {data.shape}")
    K, P = data.shape
    if len(labels) != K:
        raise SystemExit(f"State labels length {len(labels)} does not match K={K}")
    parcels_axis = template_img.header.get_axis(1)
    state_axis = ScalarAxis(labels)
    header = nib.cifti2.Cifti2Header.from_axes([state_axis, parcels_axis])
    img = nib.Cifti2Image(data.astype(np.float32, copy=False), header=header)
    nib.save(img, str(out_path))


@dataclass
class ZScoreConfig:
    betas_dir: Path
    K: int
    ptseries_dir: Optional[Path] = None
    redo: bool = True
    state_labels: Optional[List[str]] = None
    dtseries_dir: Optional[Path] = None  # backward compatibility fallback

    def resolve_pt_dir(self) -> Path:
        if self.ptseries_dir is not None:
            return Path(self.ptseries_dir)
        if self.dtseries_dir is not None:
            log.warning("zscore_using_dtseries_fallback", extra={"dt_dir": str(self.dtseries_dir)})
            return Path(self.dtseries_dir)
        raise SystemExit("ZScoreConfig requires ptseries_dir (or dtseries_dir for backward compatibility)")

    def state_names(self) -> List[str]:
        if self.state_labels:
            return list(self.state_labels)
        return [f"State{k+1}" for k in range(self.K)]


class ZScoreExporter:
    def __init__(self, cfg: ZScoreConfig):
        self.cfg = cfg

    def run(self) -> None:
        betas_dir = self.cfg.betas_dir
        pt_dir = self.cfg.resolve_pt_dir()

        txts = sorted(glob.glob(str(betas_dir / f"*_state_betas_{self.cfg.K}S.txt")))
        if not txts:
            raise SystemExit(f"No files matching *_state_betas_{self.cfg.K}S.txt in {betas_dir}")

        pt_lookup = index_ptseries(pt_dir)
        labels = self.cfg.state_names()

        for txt in txts:
            txt = Path(txt)
            sid = txt.name.split("_")[0]
            pt = pt_lookup.get(sid) or get_pt_path(pt_dir, sid)
            if pt is None:
                log.warning("missing_ptseries", extra={"sid": sid, "dir": str(pt_dir)})
                continue

            ztxt = txt.with_name(txt.stem + "_zscored.txt")
            zpscalar = txt.with_name(txt.stem + "_zscored.pscalar.nii")

            if zpscalar.exists() and not self.cfg.redo:
                log.info("zscore_exists", extra={"sid": sid, "pscalar": str(zpscalar)})
                continue

            PK = np.loadtxt(txt, dtype=np.float32)
            if PK.ndim != 2 or PK.shape[1] != self.cfg.K:
                raise SystemExit(f"{txt.name}: expected P×K with K={self.cfg.K}, got {PK.shape}")

            PKz = _zscore_cols(PK)
            np.savetxt(ztxt, PKz.astype(np.float32), fmt="%.6f")

            img = nib.load(str(pt))
            data = PKz.T  # K × P
            _write_pscalar(data, img, labels, zpscalar)

            log.info("zscore_parcel", extra={"sid": sid, "pscalar": str(zpscalar)})
