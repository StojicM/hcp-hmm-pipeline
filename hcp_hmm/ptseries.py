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
from functools import lru_cache
from pathlib import Path
from typing import Dict, Sequence

import nibabel as nib # library for dealing with neuroimaging file formats NIfTI, CIFTI, GIFTI, MINC, Analyze...
import numpy as np
import pandas as pd

from .logger import get_logger

log = get_logger(__name__)


@dataclass
class PtConcatConfig:
    indir: Path
    outdir: Path


class PtSeriesConcatenator:
    def __init__(self, config: PtConcatConfig):
        self.config = config

    #Defines a static helper that takes a file path and returns a NumPy 2D array:
    @staticmethod
    def load_ptseries(path: Path) -> np.ndarray:
        # Uses nibabel to load the CIFTI/NIfTI file. 
        # - str(path) is used because nibabel expects a string path.
        _ifti_img = nib.load(str(path))
        # Extracts the data as a NumPy array, converting/scaling to float32 to save memory.
        X = _ifti_img.get_fdata(dtype=np.float32)
        # Ensures the file contains a 2D matrix (time × parcels or parcels × time).
        if X.ndim != 2:
            #If not 2D, fail early with a clear error message that includes the shape.
            raise ValueError(f"{path.name}: expected 2D array; got {X.shape}")

        # Checks orientation the dumb way 
        # - If rows < columns, rows are likely parcels and columns time.
        #ToDo: make it smart
        if X.shape[0] < X.shape[1]:
            # Transposes so rows represent time points and 
            # columns represent parcels (standardized orientation).
            X = X.T
        # Returns the data as float32 (no extra copy if already float32).
        return X.astype(np.float32, copy=False)

    @staticmethod
    def zscore_cols(X: np.ndarray) -> np.ndarray:
        mu = X.mean(axis=0, dtype=np.float32)
        sd = X.std(axis=0, ddof=1, dtype=np.float32)
        sd[sd == 0] = np.float32(1.0)
        return (X - mu) / sd

    def run(self) -> tuple[Path, Path]:
        indir = self.config.indir
        outdir = self.config.outdir
        outdir.mkdir(parents=True, exist_ok=True)

        files = sorted(indir.glob("*.ptseries.nii"))
        if not files:
            raise FileNotFoundError(f"No .ptseries.nii found in {indir}")

        # Preflight Loop
        # Goal (scan/filter/size): Validate files, ensure same parcel count, 
        # -and sum total TRs before allocating output.
        # pass 1: preflight
        kept, total_TR, n_parcels = [], 0, None # kept = files to use
        for f in files: #Loop over every candidate file found earlier.
            _ifti_img = nib.load(str(f)) #Load the file header/data via nibabel (no data copied yet).
            _ifti_shape = _ifti_img.shape
            #If not 2D, log a warning and skip (not a time-by-parcel matrix).
            if len(_ifti_shape) != 2:
                log.warning("skip_non2d_ptseries", extra={"file": f.name, "shape": str(_ifti_shape)})
                continue
            # Determine (T, P) so that T is the longer dimension (time) and 
            # - P the shorter (parcels). 
            # If needed, swap.
            T, P = (_ifti_shape if _ifti_shape[0] >= _ifti_shape[1] else (_ifti_shape[1], _ifti_shape[0]))
            #For the first valid file, set n_parcels = P
            if n_parcels is None:
                n_parcels = P
            elif P != n_parcels:
                #For later files, if P doesn’t match n_parcels, warn and skip 
                # (can’t concatenate differing parcel counts).
                log.warning("skip_mismatched_parcels", extra={"file": f.name, "parcels": P, "expected": n_parcels})
                continue
            #Keep the file with its time length (f, T) for the second pass.
            kept.append((f, T))
            #Accumulate total_TR += T across kept files.
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


_PREFERRED_KEYWORDS: Sequence[str] = ("rest", "REST")


def _choose_preferred(paths: Sequence[Path]) -> Path:
    if len(paths) == 1:
        return paths[0]
    for key in _PREFERRED_KEYWORDS:
        key_lower = key.lower()
        for path in paths:
            if key_lower in path.name.lower():
                return path
    return sorted(paths)[0]


@lru_cache(maxsize=32)
def _index_cached(ptseries_dir: str) -> Dict[str, Path]:
    pt_dir = Path(ptseries_dir)
    if not pt_dir.exists():
        log.warning("ptseries_dir_missing", extra={"path": ptseries_dir})
        return {}

    grouped: Dict[str, list[Path]] = {}
    for path in sorted(pt_dir.glob("*.ptseries.nii")):
        sid = path.name.split("_")[0]
        grouped.setdefault(sid, []).append(path)

    best = {sid: _choose_preferred(paths) for sid, paths in grouped.items()}
    return best


def index_ptseries(pt_dir: Path) -> Dict[str, Path]:
    """Return a mapping Subject → ptseries path (cached per directory)."""

    return dict(_index_cached(str(Path(pt_dir))))


def get_pt_path(pt_dir: Path, subject: str) -> Path | None:
    """Fetch the ptseries path for a subject, if present."""

    if not subject:
        return None
    lookup = _index_cached(str(Path(pt_dir)))
    return lookup.get(subject)
