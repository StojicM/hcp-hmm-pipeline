#!/usr/bin/env python3
from __future__ import annotations

"""
Concatenate CIFTI .ptseries.nii files into a single memmapped array and index.

Matches outputs of src/01_concat_ptseries.py:
    #TODO da li je tran_X.py 300 parcela times series nastakovano preko svih subjects.
  - <outdir>/train_X.npy 
  - <outdir>/subjects_index.csv (Subject,start,end,nTR)
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib # library for dealing with neuroimaging file formats NIfTI, CIFTI, GIFTI, MINC, Analyze...
from .logger import get_logger

log = get_logger(__name__)


@dataclass
class PtConcatConfig:
    indir: Path
    outdir: Path


class PtSeriesConcatenator:
    def __init__(self, cfg: PtConcatConfig):
        self.cfg = cfg

    @staticmethod
    def load_ptseries(path: Path) -> np.ndarray:
        img = nib.load(str(path))
        X = img.get_fdata(dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"{path.name}: expected 2D array; got {X.shape}")
        if X.shape[0] < X.shape[1]:
            X = X.T
        return X.astype(np.float32, copy=False)

    @staticmethod
    def zscore_cols(X: np.ndarray) -> np.ndarray:
        mu = X.mean(axis=0, dtype=np.float32)
        sd = X.std(axis=0, ddof=1, dtype=np.float32)
        sd[sd == 0] = np.float32(1.0)
        return (X - mu) / sd

    def run(self) -> tuple[Path, Path]:
        indir = self.cfg.indir
        outdir = self.cfg.outdir
        outdir.mkdir(parents=True, exist_ok=True)

        files = sorted(indir.glob("*.ptseries.nii"))
        if not files:
            raise FileNotFoundError(f"No .ptseries.nii found in {indir}")

        # pass 1: preflight
        kept, total_TR, n_parcels = [], 0, None
        for f in files:
            img = nib.load(str(f))
            sh = img.shape
            if len(sh) != 2:
                log.warning("skip_non2d_ptseries", extra={"file": f.name, "shape": str(sh)})
                continue
            T, P = (sh if sh[0] >= sh[1] else (sh[1], sh[0]))
            if n_parcels is None:
                n_parcels = P
            elif P != n_parcels:
                log.warning("skip_mismatched_parcels", extra={"file": f.name, "parcels": P, "expected": n_parcels})
                continue
            kept.append((f, T))
            total_TR += T

        if not kept:
            raise RuntimeError("No usable ptseries after preflight.")

        Xpath = outdir / "train_X.npy"
        Xall = np.lib.format.open_memmap(
            filename=str(Xpath), mode="w+", dtype=np.float32, shape=(total_TR, n_parcels)
        )

        rows = []
        row_ptr = 0
        for f, T in kept:
            X = self.load_ptseries(f)
            Xz = self.zscore_cols(X).astype(np.float32, copy=False)
            nTR = Xz.shape[0]
            Xall[row_ptr:row_ptr+nTR, :] = Xz
            sid = f.stem.split("_")[0]
            rows.append(dict(Subject=sid, start=row_ptr, end=row_ptr+nTR, nTR=nTR))
            row_ptr += nTR
            log.info(f"concat {sid}", extra={"file": f.name, "sid": sid, "nTR": int(nTR), "parcels": int(n_parcels)})

        Xall.flush()
        idx_path = outdir / "subjects_index.csv"
        pd.DataFrame(rows).to_csv(idx_path, index=False)
        log.info("concat_done", extra={"total_TR": int(total_TR), "parcels": int(n_parcels)})
        return Xpath, idx_path
