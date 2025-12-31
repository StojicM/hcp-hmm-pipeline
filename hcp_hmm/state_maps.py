#!/usr/bin/env python3
from __future__ import annotations

"""Compute state β maps in parcel space using ptseries inputs."""

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np
from nibabel.cifti2.cifti2_axes import ScalarAxis

from .logger import get_logger
from .ptseries import PtSeriesConcatenator, get_pt_path, index_ptseries

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


def _load_ptseries(path: Path) -> tuple[np.ndarray, nib.cifti2.Cifti2Image]:
    img = nib.load(str(path))
    data = PtSeriesConcatenator.load_ptseries(path)
    return data, img


def _compute_betas(G: np.ndarray, Y: np.ndarray, rcond: float) -> np.ndarray:
    T, K = G.shape
    if Y.ndim != 2:
        raise SystemExit(f"ptseries expected 2D (T×P); got shape {Y.shape}")
    T_pt, P = Y.shape
    if T_pt != T:
        raise SystemExit(f"TR mismatch: ptseries T={T_pt}, gamma T={T}.")
    G_pinv = np.linalg.pinv(G, rcond=rcond)
    B = (G_pinv @ Y).astype(np.float32, copy=False)
    return B


def _write_pscalar(B: np.ndarray, template_img: nib.cifti2.Cifti2Image, out_path: Path, labels: List[str]) -> None:
    if B.ndim != 2:
        raise SystemExit(f"Expected 2D beta array; got shape {B.shape}")
    K, P = B.shape
    if len(labels) != K:
        raise SystemExit(f"State labels length {len(labels)} does not match K={K}")
    parcels_axis = template_img.header.get_axis(1)
    state_axis = ScalarAxis(labels)
    header = nib.cifti2.Cifti2Header.from_axes([state_axis, parcels_axis])
    img = nib.Cifti2Image(B.astype(np.float32, copy=False), header=header)
    nib.save(img, str(out_path))


@dataclass
class StateMapConfig:
    states_dir: Path
    out_dir: Path
    K: int
    ptseries_dir: Optional[Path] = None
    rcond: float = 1e-6
    chunk: int = 60000  # retained for CLI compatibility; unused
    state_labels: Optional[List[str]] = None
    dtseries_dir: Optional[Path] = None  # backward compatibility fallback
    center_betas: bool = True            # remove intercept-like bias across states

    def resolve_pt_dir(self) -> Path:
        if self.ptseries_dir is not None:
            return Path(self.ptseries_dir)
        if self.dtseries_dir is not None:
            log.warning("state_maps_using_dtseries_fallback", extra={"dt_dir": str(self.dtseries_dir)})
            return Path(self.dtseries_dir)
        raise SystemExit("StateMapConfig requires ptseries_dir (or dtseries_dir for backward compatibility)")

    def state_names(self) -> List[str]:
        if self.state_labels:
            return list(self.state_labels)
        return [f"State{k+1}" for k in range(self.K)]


class StateMapEstimator:
    def __init__(self, cfg: StateMapConfig):
        self.cfg = cfg

    def run(self) -> None:
        pt_dir = self.cfg.resolve_pt_dir()
        st_dir = self.cfg.states_dir
        out_dir = self.cfg.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        pat = str(st_dir / f"*_state_probs_{self.cfg.K}S.*")
        files = sorted(glob.glob(pat))
        if not files:
            raise SystemExit(f"No posterior files matching {pat}")

        sids = sorted({Path(f).name.split("_")[0] for f in files})

        pt_lookup = index_ptseries(pt_dir)
        log.debug("state_maps_pt_index", extra={"subjects": len(pt_lookup)})

        labels = self.cfg.state_names()

        log.info("state_maps_subjects", extra={"N": len(sids)})
        for sid in sids:
            pt_path = pt_lookup.get(sid) or get_pt_path(pt_dir, sid)
            if pt_path is None:
                log.warning("missing_ptseries", extra={"sid": sid, "dir": str(pt_dir)})
                continue

            G = _load_gamma(st_dir, sid, self.cfg.K)
            Y, img = _load_ptseries(pt_path)
            B = _compute_betas(G, Y, self.cfg.rcond)
            # Optional centering across states to remove intercept-like component
            if self.cfg.center_betas:
                w = G.mean(axis=0).astype(np.float32, copy=False)  # K
                m = w @ B  # V
                B = (B - m[None, :]).astype(np.float32, copy=False)

            betas_txt = out_dir / f"{sid}_state_betas_{self.cfg.K}S.txt"
            np.savetxt(betas_txt, B.T, fmt="%.6f")  # P×K

            out_pscalar = out_dir / f"{sid}_state_betas_{self.cfg.K}S.pscalar.nii"
            _write_pscalar(B, img, out_pscalar, labels)

            log.info(
                "state_maps_parcel",
                extra={"sid": sid, "txt": str(betas_txt), "pscalar": str(out_pscalar)},
            )
