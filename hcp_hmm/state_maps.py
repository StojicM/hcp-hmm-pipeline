#!/usr/bin/env python3
from __future__ import annotations

"""
Compute state β maps from posteriors (Γ) and dtseries via β = pinv(Γ) @ Y.
Replicates src/06_state_maps.py behavior with structured logging.
"""

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import nibabel as nib

from .logger import get_logger
from .dtseries import get_dt_path, index_dtseries
from .workbench import ensure_workbench, merge_cifti, run_wb

log = get_logger(__name__)


def _load_gamma(states_dir: Path, sid: str, K: int) -> np.ndarray:
    for ext in ("txt", "tsv"):
        f = states_dir / f"{sid}_state_probs_{K}S.{ext}"
        if f.exists():
            G = np.loadtxt(f)
            if G.ndim == 1:
                G = G.reshape(-1, K)
            if G.shape[1] != K:
                raise SystemExit(f"{f.name}: expected K={K} columns, got {G.shape[1]}")
            return G.astype(np.float32)
    raise SystemExit(f"Posteriors not found for {sid} in {states_dir}")


def _compute_betas_chunked(G: np.ndarray, img: "nib.cifti2.Cifti2Image", chunk: int, rcond: float) -> np.ndarray:
    T, K = G.shape
    G_pinv = np.linalg.pinv(G, rcond=rcond).astype(np.float32)
    shape = img.shape
    if len(shape) != 2:
        raise SystemExit(f"dtseries expected 2D (T×V); got shape {shape}")
    T_dt, V = shape
    if T_dt != T:
        raise SystemExit(f"TR mismatch: dtseries T={T_dt}, gamma T={T}.")
    dataobj = img.dataobj
    B = np.empty((K, V), dtype=np.float32)
    for start in range(0, V, chunk):
        stop = min(start + chunk, V)
        Y_block = np.asarray(dataobj[:, start:stop], dtype=np.float32)
        B[:, start:stop] = G_pinv @ Y_block
    return B


def _write_dscalar_from_vk(Bvk_txt: Path, dtseries_path: Path, out_path: Path, K: int) -> None:
    tmpl = out_path.with_suffix(".tmpl.dscalar.nii")
    if not tmpl.exists():
        run_wb(["-cifti-reduce", str(dtseries_path), "MEAN", str(tmpl)])

    B = np.loadtxt(Bvk_txt, dtype=np.float32)        # V×K
    if B.ndim != 2 or B.shape[1] != K:
        raise SystemExit(f"{Bvk_txt.name}: expected V×K, got {B.shape}")

    # Write each column (V×1) to its own dscalar, then merge
    import tempfile
    with tempfile.TemporaryDirectory() as tdir:
        one_maps: List[Path] = []
        for k in range(K):
            col_txt = Path(tdir) / f"col_{k}.txt"
            np.savetxt(col_txt, B[:, [k]], fmt="%.6f")
            col_dc = Path(tdir) / f"map_{k}.dscalar.nii"
            run_wb(["-cifti-convert", "-from-text", str(col_txt), str(tmpl), str(col_dc)])
            one_maps.append(col_dc)
        merge_cifti(out_path, one_maps)


@dataclass
class StateMapConfig:
    dtseries_dir: Path
    states_dir: Path
    out_dir: Path
    K: int
    chunk: int = 60000
    rcond: float = 1e-6


class StateMapEstimator:
    def __init__(self, cfg: StateMapConfig):
        self.cfg = cfg

    def run(self) -> None:
        ensure_workbench()
        dt_dir = self.cfg.dtseries_dir
        st_dir = self.cfg.states_dir
        out_dir = self.cfg.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        pat = str(st_dir / f"*_state_probs_{self.cfg.K}S.*")
        files = sorted(glob.glob(pat))
        if not files:
            raise SystemExit(f"No posterior files matching {pat}")

        sids = sorted({Path(f).name.split("_")[0] for f in files})

        dt_lookup = index_dtseries(dt_dir)
        log.debug("state_maps_dt_index", extra={"subjects": len(dt_lookup)})

        log.info("state_maps_subjects", extra={"N": len(sids)})
        for sid in sids:
            dt = dt_lookup.get(sid) or get_dt_path(dt_dir, sid)
            if dt is None:
                log.warning("missing_dtseries", extra={"sid": sid, "dir": str(dt_dir)})
                continue
            G = _load_gamma(st_dir, sid, self.cfg.K)
            img = nib.load(str(dt))
            B = _compute_betas_chunked(G, img, self.cfg.chunk, self.cfg.rcond)
            betas_txt = out_dir / f"{sid}_state_betas_{self.cfg.K}S.txt"
            np.savetxt(betas_txt, B.T, fmt="%.6f")  # V×K
            out_dc = out_dir / f"{sid}_state_betas_{self.cfg.K}S.dscalar.nii"
            _write_dscalar_from_vk(betas_txt, dt, out_dc, self.cfg.K)
            log.info(f"state-maps {sid}", extra={"sid": sid, "txt": str(betas_txt), "dscalar": str(out_dc)})
