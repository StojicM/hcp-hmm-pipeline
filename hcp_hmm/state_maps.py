#!/usr/bin/env python3
from __future__ import annotations

"""Compute state β maps in parcel space using ptseries inputs."""

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    # Optional for BrainSpace rendering of subject betas
    render_brainspace: bool = False
    atlas_dlabel: Optional[Path] = None
    surface_dir: Optional[Path] = None
    surface_left: Optional[Path] = None
    surface_right: Optional[Path] = None
    surface_left_inflated: Optional[Path] = None
    surface_right_inflated: Optional[Path] = None

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

    # ==== BrainSpace helpers (optional) ====
    @staticmethod
    def _apply_vtk_compat_shim() -> None:
        try:
            import importlib
            modules = [
                "vtkmodules.vtkCommonCore",
                "vtkmodules.vtkCommonDataModel",
                "vtkmodules.vtkCommonExecutionModel",
                "vtkmodules.vtkFiltersCore",
                "vtkmodules.vtkRenderingCore",
                "vtkmodules.vtkIOGeometry",
                "vtkmodules.vtkIOXML",
            ]
            for modname in modules:
                try:
                    m = importlib.import_module(modname)
                except Exception:
                    continue
                for name, obj in vars(m).items():
                    if isinstance(obj, type) and name.startswith("vtk") and not hasattr(obj, "__vtkname__"):
                        try:
                            setattr(obj, "__vtkname__", name)
                        except Exception:
                            pass
        except Exception:
            pass

    def _resolve_surface_paths(self) -> Optional[Dict[str, Path]]:
        def _valid(p: Optional[Path]) -> Optional[Path]:
            if p is None:
                return None
            pp = Path(p)
            return pp if pp.exists() else None
        left = _valid(self.cfg.surface_left)
        right = _valid(self.cfg.surface_right)
        left_inf = _valid(self.cfg.surface_left_inflated)
        right_inf = _valid(self.cfg.surface_right_inflated)
        base = _valid(self.cfg.surface_dir)
        def _find(root: Path, hemi: str, keys: List[str]) -> Optional[Path]:
            pats = []
            for kw in keys:
                pats += [f"*{hemi}*.{kw}*.surf.gii", f"*{hemi}*{kw}*.surf.gii", f"*{kw}*.{hemi}*.surf.gii"]
            pats.append(f"*{hemi}*.surf.gii")
            for pat in pats:
                hits = sorted(root.glob(pat))
                if hits:
                    return hits[0]
            return None
        if base is not None:
            if left is None:
                left = _find(base, "L", ["midthickness", "32k", "fs_LR"])
            if right is None:
                right = _find(base, "R", ["midthickness", "32k", "fs_LR"])
            if left_inf is None:
                left_inf = _find(base, "L", ["inflated", "veryinflated"])
            if right_inf is None:
                right_inf = _find(base, "R", ["inflated", "veryinflated"])
        if not left or not right:
            return None
        d: Dict[str, Path] = {"lh": left, "rh": right}
        if left_inf: d["lh_inflated"] = left_inf
        if right_inf: d["rh_inflated"] = right_inf
        return d

    def _load_brainspace_context(self) -> Optional[Dict[str, Any]]:
        if not self.cfg.atlas_dlabel:
            return None
        surfs = self._resolve_surface_paths()
        if not surfs:
            return None
        try:
            os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
            self._apply_vtk_compat_shim()
            from brainspace.mesh.mesh_io import read_surface  # type: ignore
            from brainspace.plotting import plot_hemispheres  # type: ignore
        except Exception as e:
            log.info("state_maps_brainspace_unavailable", extra={"err": str(e)})
            return None
        try:
            surf_lh = read_surface(str(surfs["lh"]))
            surf_rh = read_surface(str(surfs["rh"]))
        except Exception as e:
            log.warning("state_maps_surface_load_failed", extra={"err": str(e)})
            return None
        # Load dlabel to get vertex parcel codes
        try:
            img = nib.load(str(self.cfg.atlas_dlabel))
            data = np.asarray(img.get_fdata()).reshape(-1).astype(int)
            bm = img.header.get_axis(1)
            lh_slice = None; rh_slice=None
            for name, sl, _ in bm.iter_structures():
                s = str(name)
                if "CORTEX_LEFT" in s:
                    lh_slice = sl
                elif "CORTEX_RIGHT" in s:
                    rh_slice = sl
            if lh_slice is None or rh_slice is None:
                return None
            codes_lh = data[lh_slice]
            codes_rh = data[rh_slice]
        except Exception as e:
            log.warning("state_maps_dlabel_load_failed", extra={"err": str(e)})
            return None
        return {
            "surf_lh": surf_lh,
            "surf_rh": surf_rh,
            "plot": plot_hemispheres,
            "codes_lh": codes_lh,
            "codes_rh": codes_rh,
        }

    @staticmethod
    def _zscore_vec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        mu = float(np.nanmean(x)) if x.size else 0.0
        sd = float(np.nanstd(x, ddof=1)) if x.size else 1.0
        if not np.isfinite(sd) or sd == 0.0:
            sd = 1.0
        return (x - mu) / sd

    @staticmethod
    def _values_to_vertices(vals: np.ndarray, codes: np.ndarray) -> np.ndarray:
        P = vals.shape[0]
        out = np.full(codes.shape[0], np.nan, dtype=np.float32)
        mask = (codes > 0) & (codes <= P)
        idx = codes[mask] - 1
        out[mask] = vals[idx]
        return out
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

            # Optional: render per-subject betas on BrainSpace surfaces if configured
            ctx = None
            if self.cfg.render_brainspace:
                try:
                    ctx = self._load_brainspace_context()
                except Exception:
                    ctx = None
            if ctx is not None:
                fig_dir = out_dir / "fig_brainspace"
                fig_dir.mkdir(exist_ok=True)
                for s in range(self.cfg.K):
                    v = B[s, :]
                    for kind, arr in (("betas", v), ("betas_z", self._zscore_vec(v))):
                        lh = self._values_to_vertices(arr, ctx["codes_lh"])
                        rh = self._values_to_vertices(arr, ctx["codes_rh"])
                        dense = np.concatenate([lh, rh])
                        fname = fig_dir / f"{sid}_state{s}_{kind}.png"
                        try:
                            ctx["plot"](
                                ctx["surf_lh"], ctx["surf_rh"],
                                array_name=dense,
                                color_bar=True,
                                color_range=("sym" if kind == "betas_z" else None),
                                cmap=("RdBu_r" if kind == "betas_z" else "viridis"),
                                nan_color=(0.85, 0.85, 0.85, 1.0),
                                zoom=1.1,
                                background=(1, 1, 1),
                                size=(900, 450),
                                layout_style="grid",
                                label_text={"top": [f"{sid} State {s} {('(z)' if kind=='betas_z' else '')}", ""]},
                                interactive=False,
                                embed_nb=False,
                                screenshot=True,
                                filename=str(fname),
                                transparent_bg=False,
                            )
                        except Exception as e:
                            log.warning("state_maps_brainspace_failed", extra={"sid": sid, "state": int(s), "kind": kind, "err": str(e)})
