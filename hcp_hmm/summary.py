#!/usr/bin/env python3
from __future__ import annotations

"""
Build a compact per-state summary that combines:
  - WHEN (temporal): subject-averaged state metrics (FO, DT, SR, ...)
  - WHERE (spatial): PALM p-map highlights (counts and top parcels)

Outputs into <hmm_dir>/summary:
  - state_temporal_summary.csv  (wide summary by state and metric)
  - state_spatial_summary.csv   (per state counts and top parcels)
  - state_summary.csv           (merged one-row-per-state overview)
  - summary.html                (lightweight HTML overview)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib

from .logger import get_logger

log = get_logger(__name__)


def _read_labels(tsv: Path) -> List[str]:
    if not tsv.exists():
        return []
    names: List[str] = []
    with open(tsv, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # Workbench label table alternates: NAME line, then "index R G B A".
    # Extract names in-order of appearance.
    for i in range(0, len(lines), 2):
        names.append(lines[i])
    return names


def _labels_from_pscalar(betas_dir: Path, K: int) -> List[str]:
    """Try to read parcel names directly from a pscalar header (more reliable order)."""
    # Prefer a subject-level pscalar (KxP) if present
    pats = [
        f"*_state_betas_{K}S_zscored.pscalar.nii",
        f"*_state_betas_{K}S.pscalar.nii",
    ]
    for pat in pats:
        files = sorted((betas_dir).glob(pat))
        if files:
            try:
                img = nib.load(str(files[0]))
                ax = img.header.get_axis(1)
                names = []
                for i in range(len(ax)):
                    el = ax.get_element(i)
                    name = el[0] if isinstance(el, tuple) else str(el)
                    names.append(str(name))
                if names:
                    return names
            except Exception:
                pass
    # Fallback: try a group per-state file (NxP), still read axis 1
    grp_files = sorted((betas_dir / "group").glob(f"allsubs_state*_{K}S_zscored.pscalar.nii"))
    if grp_files:
        try:
            img = nib.load(str(grp_files[0]))
            ax = img.header.get_axis(1)
            names = []
            for i in range(len(ax)):
                el = ax.get_element(i)
                name = el[0] if isinstance(el, tuple) else str(el)
                names.append(str(name))
            if names:
                return names
        except Exception:
            pass
    return []


def _read_inputs_list(group_dir: Path, K: int) -> List[Path]:
    Ktag = f"{K}S"
    lst = group_dir / f"inputs_{Ktag}_pscalar.txt"
    if lst.exists():
        return [Path(ln.strip()) for ln in lst.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return []


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    if p.size == 0:
        return p
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(p) + 1)
    q = p * len(p) / ranks
    # monotone
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    qvals = np.empty_like(q_sorted)
    qvals[order] = q_sorted
    return np.clip(qvals, 0, 1)


def _perm_two_sample(x: np.ndarray, g: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    # two-sided on mean difference |m1-m2|
    mask = (g == 0) | (g == 1)
    x = x[mask]; g = g[mask]
    if x.size == 0 or np.unique(g).size < 2:
        return np.nan
    x1 = x[g == 1]; x0 = x[g == 0]
    obs = abs(np.nanmean(x1) - np.nanmean(x0))
    idx = np.arange(x.size)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(idx)
        g_perm = g[idx]
        d = abs(np.nanmean(x[g_perm == 1]) - np.nanmean(x[g_perm == 0]))
        if d >= obs:
            cnt += 1
    return (cnt + 1.0) / (n_perm + 1.0)


def _perm_anova_oneway(x: np.ndarray, groups: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    # return p-value for one-way ANOVA F statistic across >=2 groups
    vals = x.copy()
    g = groups.copy()
    uniq = np.array(sorted([u for u in np.unique(g) if str(u) != 'nan' and u != '']))
    if uniq.size < 2:
        return np.nan
    # compute F
    def _f_stat(v, gg):
        mask = ~(np.isnan(v))
        v = v[mask]; gg = gg[mask]
        cats = [v[gg == u] for u in uniq]
        n = sum(len(c) for c in cats)
        if n <= len(cats):
            return np.nan
        grand = np.mean(v)
        ssb = sum(len(c)*(np.mean(c)-grand)**2 for c in cats)
        ssw = sum(sum((c - np.mean(c))**2) for c in cats)
        dfb = len(cats)-1; dfw = n - len(cats)
        if dfw <= 0 or ssw <= 0:
            return np.nan
        return (ssb/dfb) / (ssw/dfw)
    obs = _f_stat(vals, g)
    if not np.isfinite(obs):
        return np.nan
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(vals)
        f = _f_stat(vals, g)
        if np.isfinite(f) and f >= obs:
            cnt += 1
    return (cnt + 1.0) / (n_perm + 1.0)


def _infer_network(label: str) -> str:
    """Rudimentary mapping from parcel label to Yeo-7 network name.
    Tries to find a network token in the label string.
    """
    s = label.replace('-', '_')
    up = s.upper()
    if 'VIS' in up or 'VISUAL' in up:
        return 'Visual'
    if 'SOM' in up or 'SOMATOMOTOR' in up or 'MOT' in up:
        return 'SomMot'
    if 'DOR' in up or 'DORS' in up:
        return 'DorsAttn'
    if 'VENT' in up or 'SAL' in up:
        return 'VentAttn'
    if 'LIMB' in up:
        return 'Limbic'
    if 'CONT' in up or 'CONTROL' in up or 'FRONTOPARIETAL' in up or 'FP' in up:
        return 'Control'
    if 'DEFAULT' in up or 'DMN' in up:
        return 'Default'
    return 'Other'


def _load_state_patterns(hmm_dir: Path, K: int) -> Tuple[np.ndarray, pd.DataFrame]:
    Ktag = f"{K}S"
    path = hmm_dir / f"state_mean_patterns_{Ktag}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if 'state' in df.columns:
        df = df.set_index('state')
    arr = df.to_numpy(dtype=np.float32)
    return arr, df


def _spatial_for_state(group_dir: Path, state: int, K: int, labels: List[str], p_thr: float = 0.05) -> Dict[str, object]:
    Ktag = f"{K}S"
    fwe = group_dir / "palm" / f"palm_{Ktag}_state{state}_dat_tstat_fwep.pscalar.nii"
    unc = group_dir / "palm" / f"palm_{Ktag}_state{state}_dat_tstat_uncp.pscalar.nii"
    row: Dict[str, object] = {"state": state}
    if fwe.exists():
        arr = np.asarray(nib.load(str(fwe)).get_fdata(dtype=np.float32)).ravel()
        P = arr.shape[0]
        row["n_sig_fwe"] = int(np.count_nonzero(arr < p_thr))
        # top 5 parcels by smallest p
        idx = np.argsort(arr)[: min(5, P)]
        tops = []
        for j in idx:
            nm = labels[j] if j < len(labels) else f"parcel_{j+1}"
            tops.append(f"{nm}:{arr[j]:.3g}")
        row["top5_fwe"] = "; ".join(tops)
    else:
        row["n_sig_fwe"] = 0
        row["top5_fwe"] = ""
    if unc.exists():
        arru = np.asarray(nib.load(str(unc)).get_fdata(dtype=np.float32)).ravel()
        row["n_sig_unc05"] = int(np.count_nonzero(arru < p_thr))
    else:
        row["n_sig_unc05"] = 0
    return row


def _temporal_summary(metrics_state_csv: Path, K: int) -> pd.DataFrame:
    if not metrics_state_csv.exists():
        raise FileNotFoundError(metrics_state_csv)
    df = pd.read_csv(metrics_state_csv)
    if "state" not in df.columns:
        raise SystemExit(f"state column missing in {metrics_state_csv}")
    metrics = [
        c for c in (
            "FO", "DT_mean", "DT_median", "DT_var", "n_visits", "IV_mean", "SR_state",
            "row_entropy_bits", "self_transition"
        ) if c in df.columns
    ]
    pieces: List[pd.DataFrame] = []
    for m in metrics:
        g = df.groupby("state")[m].agg(["mean", "std", "count"]).reset_index()
        g.insert(1, "metric", m)
        pieces.append(g)
    if pieces:
        out = pd.concat(pieces, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["state", "metric", "mean", "std", "count"])
    return out


@dataclass
class SummaryConfig:
    hmm_dir: Path
    betas_dir: Path
    K: int
    atlas_dlabel: Optional[Path] = None
    surface_left: Optional[Path] = None
    surface_right: Optional[Path] = None
    surface_left_inflated: Optional[Path] = None
    surface_right_inflated: Optional[Path] = None
    surface_dir: Optional[Path] = None
    n_perm: int = 2000
    seed: int = 42


@dataclass
class BrainSpaceContext:
    surf_lh: Any
    surf_rh: Any
    plot_hemispheres: Any
    lh_n: int
    rh_n: int
    codes_lh: Optional[np.ndarray] = None
    codes_rh: Optional[np.ndarray] = None


class SummaryBuilder:
    def __init__(self, cfg: SummaryConfig):
        self.cfg = cfg

    @staticmethod
    def _apply_vtk_compat_shim() -> None:
        """Best-effort shim to make BrainSpace work with newer VTK (>=9.4/9.5).

        Older BrainSpace versions expect VTK classes to expose ``__vtkname__``.
        VTK 9.4+ dropped this attribute; we recreate it on common classes so
        BrainSpace's wrappers don't crash. No-op if already present.
        """
        try:
            import importlib  # noqa: WPS433
            modules = [
                "vtkmodules.vtkCommonCore",
                "vtkmodules.vtkCommonDataModel",
                "vtkmodules.vtkCommonExecutionModel",
                "vtkmodules.vtkFiltersCore",
                "vtkmodules.vtkFiltersSources",
                "vtkmodules.vtkRenderingCore",
                "vtkmodules.vtkInteractionStyle",
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
        except Exception as e:
            # Silent fallback – only relevant for BrainSpace plotting; never fail summary
            try:
                log.debug("vtk_shim_failed", extra={"err": str(e)})
            except Exception:
                pass

    def _resolve_surface_paths(self) -> Optional[Dict[str, Path]]:
        def _valid(path: Optional[Path]) -> Optional[Path]:
            if path is None:
                return None
            p = Path(path)
            return p if p.exists() else None

        left = _valid(self.cfg.surface_left)
        right = _valid(self.cfg.surface_right)
        left_inf = _valid(self.cfg.surface_left_inflated)
        right_inf = _valid(self.cfg.surface_right_inflated)
        base_dir = _valid(self.cfg.surface_dir)

        def _find_surface(root: Path, hemi: str, keywords: List[str]) -> Optional[Path]:
            patterns: List[str] = []
            for kw in keywords:
                patterns.extend([
                    f"*{hemi}*.{kw}*.surf.gii",
                    f"*{hemi}*{kw}*.surf.gii",
                    f"*{kw}*.{hemi}*.surf.gii",
                ])
            patterns.append(f"*{hemi}*.surf.gii")
            for pat in patterns:
                hits = sorted(root.glob(pat))
                if hits:
                    return hits[0]
            # Fallback to recursive search only for specific keywords to avoid large scans
            for kw in keywords:
                pat = f"**/*{hemi}*{kw}*.surf.gii"
                hits = sorted(root.glob(pat))
                if hits:
                    return hits[0]
            return None

        if base_dir is not None:
            if left is None:
                left = _find_surface(base_dir, "L", ["midthickness", "32k", "fs_LR"])
            if right is None:
                right = _find_surface(base_dir, "R", ["midthickness", "32k", "fs_LR"])
            if left_inf is None:
                left_inf = _find_surface(base_dir, "L", ["inflated", "veryinflated"])
            if right_inf is None:
                right_inf = _find_surface(base_dir, "R", ["inflated", "veryinflated"])

        if left is None or right is None:
            return None

        paths = {"lh": left, "rh": right}
        if left_inf is not None:
            paths["lh_inflated"] = left_inf
        if right_inf is not None:
            paths["rh_inflated"] = right_inf
        return paths

    def _load_brainspace_context(self) -> Optional[BrainSpaceContext]:
        surface_paths = self._resolve_surface_paths()
        if not surface_paths:
            return None
        try:
            os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
            # Apply compatibility shim before importing BrainSpace/VTK wrappers
            self._apply_vtk_compat_shim()
            from brainspace.mesh.mesh_io import read_surface  # type: ignore
            from brainspace.plotting import plot_hemispheres  # type: ignore
            # Patch BrainSpace VTK wrapper to tolerate missing __vtkname__ (VTK>=9.4)
            try:
                from brainspace.vtk_interface import wrappers as _bs_wrap  # type: ignore
                base_mod = getattr(_bs_wrap, 'base', None)
                old_wrap = getattr(base_mod, 'BSWrapVTKObject', None) if base_mod else None
                if callable(old_wrap):
                    def _patched_wrap(obj, _old=old_wrap):  # type: ignore
                        try:
                            return _old(obj)
                        except AttributeError:
                            # Best-effort: attach __vtkname__ to MRO classes and retry
                            try:
                                for sc in obj.__class__.mro()[:-3]:
                                    if not hasattr(sc, '__vtkname__'):
                                        nm = sc.__name__
                                        setattr(sc, '__vtkname__', nm if nm.startswith('vtk') else f'vtk{nm}')
                            except Exception:
                                pass
                            return _old(obj)
                    setattr(base_mod, 'BSWrapVTKObject', _patched_wrap)
            except Exception:
                pass
        except Exception as e:
            log.info("summary_brainspace_unavailable", extra={"err": str(e)})
            return None

        try:
            surf_lh = read_surface(str(surface_paths["lh"]))
            surf_rh = read_surface(str(surface_paths["rh"]))
        except Exception as e:
            log.warning("summary_brainspace_surface_load_failed", extra={"err": str(e)})
            return None

        def _n_points(surf: Any) -> int:
            if hasattr(surf, "n_points"):
                n = getattr(surf, "n_points")
                return int(n() if callable(n) else n)
            if hasattr(surf, "points"):
                pts = getattr(surf, "points")
                arr = pts() if callable(pts) else pts
                return int(len(arr))
            if hasattr(surf, "Points"):
                pts = getattr(surf, "Points")
                arr = pts() if callable(pts) else pts
                return int(len(arr))
            raise AttributeError("surface missing point data")

        try:
            lh_n = _n_points(surf_lh)
            rh_n = _n_points(surf_rh)
        except Exception as e:
            log.warning("summary_brainspace_surface_points_failed", extra={"err": str(e)})
            return None

        # Optional: load atlas dlabel to get per-vertex parcel codes for mapping parcel values
        codes_lh = None
        codes_rh = None
        if getattr(self.cfg, 'atlas_dlabel', None):
            try:
                img = nib.load(str(self.cfg.atlas_dlabel))
                data = np.asarray(img.get_fdata()).reshape(-1).astype(int)
                bm = img.header.get_axis(1)
                lh_slice = None; rh_slice = None
                for name, sl, _ in bm.iter_structures():
                    s = str(name)
                    if 'CORTEX_LEFT' in s:
                        lh_slice = sl
                    elif 'CORTEX_RIGHT' in s:
                        rh_slice = sl
                if lh_slice is not None and rh_slice is not None:
                    codes_lh = data[lh_slice]
                    codes_rh = data[rh_slice]
            except Exception as e:
                log.warning("summary_brainspace_codes_failed", extra={"err": str(e)})

        return BrainSpaceContext(
            surf_lh=surf_lh,
            surf_rh=surf_rh,
            plot_hemispheres=plot_hemispheres,
            lh_n=lh_n,
            rh_n=rh_n,
            codes_lh=codes_lh,
            codes_rh=codes_rh,
        )

    @staticmethod
    def _parcel_values_to_vertices(values: np.ndarray, parcels_axis: Any, lh_n: int, rh_n: int) -> Tuple[np.ndarray, np.ndarray]:
        lh_vals = np.full(lh_n, np.nan, dtype=np.float32)
        rh_vals = np.full(rh_n, np.nan, dtype=np.float32)
        if parcels_axis is None or not hasattr(parcels_axis, "vertices"):
            return lh_vals, rh_vals

        vertices = parcels_axis.vertices
        for idx, val in enumerate(values):
            if not np.isfinite(val):
                continue
            if idx >= len(vertices):
                break
            mapping = vertices[idx]
            if not isinstance(mapping, dict):
                continue
            for struct, verts in mapping.items():
                if verts.size == 0:
                    continue
                if "CORTEX_LEFT" in str(struct):
                    tgt = lh_vals
                elif "CORTEX_RIGHT" in str(struct):
                    tgt = rh_vals
                else:
                    continue
                valid = verts[(verts >= 0) & (verts < tgt.shape[0])]
                if valid.size:
                    tgt[valid] = val
        return lh_vals, rh_vals

    def _load_palm_state_arrays(self, group_dir: Path, state: int) -> Optional[Dict[str, Any]]:
        Ktag = f"{self.cfg.K}S"
        palm_dir = group_dir / "palm"
        candidates = [
            ("fwe", palm_dir / f"palm_{Ktag}_state{state}_dat_tstat_fwep.pscalar.nii"),
            ("unc", palm_dir / f"palm_{Ktag}_state{state}_dat_tstat_uncp.pscalar.nii"),
        ]
        parcels_axis = None
        p_vals = None
        correction = None
        for label, path in candidates:
            if not path.exists():
                continue
            try:
                img = nib.load(str(path))
                p_vals = np.asarray(img.get_fdata(dtype=np.float32)).squeeze()
                parcels_axis = img.header.get_axis(1)
                correction = label
                break
            except Exception as e:
                log.warning("summary_brainspace_load_p_failed", extra={"state": int(state), "path": str(path), "err": str(e)})
                return None
        if p_vals is None or parcels_axis is None:
            return None

        t_path = palm_dir / f"palm_{Ktag}_state{state}_dat_tstat.pscalar.nii"
        t_vals = None
        if t_path.exists():
            try:
                t_vals = np.asarray(nib.load(str(t_path)).get_fdata(dtype=np.float32)).squeeze()
            except Exception as e:
                log.warning("summary_brainspace_load_t_failed", extra={"state": int(state), "path": str(t_path), "err": str(e)})
                t_vals = None

        return {"p": p_vals, "t": t_vals, "axis": parcels_axis, "correction": correction}

    def _brainspace_render_state(
        self,
        context: BrainSpaceContext,
        state: int,
        data: Dict[str, Any],
        out_path: Path,
        p_threshold: float = 0.05,
    ) -> bool:
        p_vals = np.asarray(data.get("p"), dtype=np.float32)
        if p_vals.ndim != 1:
            p_vals = p_vals.reshape(-1)
        if p_vals.size == 0:
            return False
        p_clip = np.clip(p_vals, 1e-12, 1.0, out=p_vals.copy())
        sig_mask = np.isfinite(p_clip) & (p_clip < p_threshold)
        no_sig = not bool(sig_mask.any())

        neg_log = -np.log10(p_clip, out=p_clip.copy())
        neg_log[~sig_mask] = np.nan

        t_vals = data.get("t")
        if isinstance(t_vals, np.ndarray) and t_vals.shape == neg_log.shape:
            signed = neg_log * np.sign(t_vals.astype(np.float32))
            cmap = "RdBu_r"
            color_range = "sym"
        else:
            signed = neg_log
            cmap = "magma"
            color_range = None

        lh_vals, rh_vals = self._parcel_values_to_vertices(signed, data.get("axis"), context.lh_n, context.rh_n)
        dense = np.concatenate([lh_vals, rh_vals])
        try:
            # Compose concise label text with top significant parcels
            label_text = None
            try:
                axis = data.get("axis")
                names = []
                if hasattr(axis, "name"):
                    names = [str(axis.name[i]) for i in range(len(axis))]
                else:
                    names = []
                # fallback to generic parcel_i if names missing
                if not names:
                    names = [f"parcel_{i+1}" for i in range(neg_log.size)]
                order = np.argsort(p_vals)
                topk = []
                for j in order[: min(8, order.size)]:
                    if not sig_mask[j]:
                        continue
                    nm = names[j] if j < len(names) else f"parcel_{j+1}"
                    topk.append(f"{nm} (p={p_vals[j]:.2g})")
                if topk:
                    txt = "Top sig: " + ", ".join(topk)
                    # BrainSpace grid layout expects 2 labels for 'bottom'
                    label_text = {"bottom": [txt, ""]}
            except Exception:
                label_text = None
            if no_sig:
                # Render clean surfaces with a note; avoid all-NaN data
                context.plot_hemispheres(
                    context.surf_lh,
                    context.surf_rh,
                    array_name=None,
                    color_bar=False,
                    zoom=1.1,
                    background=(1, 1, 1),
                    size=(900, 450),
                    layout_style="grid",
                    label_text={"bottom": ["No significant parcels (p<%.2f)" % p_threshold, ""]},
                    interactive=False,
                    embed_nb=False,
                    screenshot=True,
                    filename=str(out_path),
                    transparent_bg=False,
                )
            else:
                context.plot_hemispheres(
                    context.surf_lh,
                    context.surf_rh,
                    array_name=dense,
                    color_bar=True,
                    color_range=color_range,
                    cmap=cmap,
                    nan_color=(0.85, 0.85, 0.85, 1.0),
                    zoom=1.1,
                    background=(1, 1, 1),
                    size=(900, 450),
                    layout_style="grid",
                    label_text=label_text,
                    interactive=False,
                    embed_nb=False,
                    screenshot=True,
                    filename=str(out_path),
                    transparent_bg=False,
                )
        except Exception as e:
            log.warning("summary_brainspace_plot_failed", extra={"state": int(state), "err": str(e)})
            return False
        return out_path.exists()

    # NOTE: Removed parcel heatmap generation per request — only BrainSpace surfaces remain.

    @staticmethod
    def _zscore_vec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        mu = float(np.nanmean(x)) if x.size else 0.0
        sd = float(np.nanstd(x, ddof=1)) if x.size else 1.0
        if not np.isfinite(sd) or sd == 0.0:
            sd = 1.0
        return (x - mu) / sd

    def _brainspace_render_values(
        self,
        context: BrainSpaceContext,
        state: int,
        values: np.ndarray,
        out_path: Path,
        title: str = "",
        zscore: bool = True,
    ) -> bool:
        vals = np.asarray(values, dtype=np.float32).reshape(-1)
        if zscore:
            vals = self._zscore_vec(vals)
        # Prefer dlabel codes mapping if available
        lh_vals = np.full(context.lh_n, np.nan, dtype=np.float32)
        rh_vals = np.full(context.rh_n, np.nan, dtype=np.float32)
        if context.codes_lh is not None and context.codes_rh is not None:
            P = vals.shape[0]
            mask_l = (context.codes_lh > 0) & (context.codes_lh <= P)
            idx_l = context.codes_lh[mask_l] - 1
            lh_vals[mask_l] = vals[idx_l]
            mask_r = (context.codes_rh > 0) & (context.codes_rh <= P)
            idx_r = context.codes_rh[mask_r] - 1
            rh_vals[mask_r] = vals[idx_r]
        else:
            # Fallback: no mapping available → skip
            return False
        if not (np.isfinite(lh_vals).any() or np.isfinite(rh_vals).any()):
            return False
        dense = np.concatenate([lh_vals, rh_vals])
        try:
            context.plot_hemispheres(
                context.surf_lh,
                context.surf_rh,
                array_name=dense,
                color_bar=True,
                color_range="sym",
                cmap="RdBu_r",
                nan_color=(0.85, 0.85, 0.85, 1.0),
                zoom=1.1,
                background=(1, 1, 1),
                size=(900, 450),
                layout_style="grid",
                label_text=({"top": [title, ""]} if title else None),
                interactive=False,
                embed_nb=False,
                screenshot=True,
                filename=str(out_path),
                transparent_bg=False,
            )
        except Exception as e:
            log.warning("summary_brainspace_values_failed", extra={"state": int(state), "err": str(e)})
            return False
        return out_path.exists()

    def _brainspace_render_all(
        self,
        group_dir: Path,
        out_dir: Path,
        state_figs: Dict[int, Dict[str, Path | None]],
        labels: List[str] | None = None,
    ) -> None:
        context = self._load_brainspace_context()
        if context is None:
            context = None  # fall through to parcel heatmaps only

        rendered_any = False
        for state in range(self.cfg.K):
            data = self._load_palm_state_arrays(group_dir, state)
            if not data:
                continue
            # BrainSpace surface (if context ok)
            if context is not None:
                out_path = out_dir / f"fig_state{state}_brainspace.png"
                try:
                    created = self._brainspace_render_state(context, state, data, out_path)
                except Exception as e:
                    log.warning("summary_brainspace_state_failed", extra={"state": int(state), "err": str(e)})
                    created = False
                if created:
                    state_figs.setdefault(state, {"net": None, "brain": None, "parcel": None})
                    state_figs[state]["brain"] = out_path
                    rendered_any = True
            # Parcel heatmap removed per user request

        if not rendered_any:
            log.info("summary_brainspace_no_figures")

    def run(self) -> Tuple[Path, Path, Path, Path]:
        # Write summaries under the analysis root (hmm_dir)
        out_dir = self.cfg.hmm_dir / "summary"
        out_dir.mkdir(parents=True, exist_ok=True)

        # WHEN: temporal metrics per state
        metrics_csv = self.cfg.hmm_dir / "metrics" / f"metrics_state_{self.cfg.K}S.csv"
        # Subject-level metrics (for sex-split visuals) and their aggregated temporal summary
        df_metrics = pd.read_csv(metrics_csv)
        df_temporal = _temporal_summary(metrics_csv, self.cfg.K)
        temporal_out = out_dir / "state_temporal_summary.csv"
        df_temporal.to_csv(temporal_out, index=False)

        # WHERE: spatial PALM highlights per state
        # Prefer labels from a reference pscalar header to ensure correct order
        labels = _labels_from_pscalar(self.cfg.betas_dir, self.cfg.K)
        if not labels:
            labels_tsv = self.cfg.betas_dir / "parcel" / "parcel_labels.tsv"
            labels = _read_labels(labels_tsv)
        group_dir = self.cfg.betas_dir / "group"
        rows = [
            _spatial_for_state(group_dir, s, self.cfg.K, labels)
            for s in range(self.cfg.K)
        ]
        df_spatial = pd.DataFrame(rows)
        spatial_out = out_dir / "state_spatial_summary.csv"
        df_spatial.to_csv(spatial_out, index=False)

        state_figs: Dict[int, Dict[str, Path | None]] = {}
        self._brainspace_render_all(group_dir, out_dir, state_figs, labels)

        # State mean patterns → network profiles + top parcels
        tops_out = None
        netfig_out = None  # deprecated: removed network-mean heatmap based on parcel z-averages
        try:
            patterns, df_patterns = _load_state_patterns(self.cfg.hmm_dir, self.cfg.K)
            P = patterns.shape[1]
            parcel_nets = [
                _infer_network(labels[j]) if j < len(labels) else 'Other'
                for j in range(P)
            ]
            net_order = ["Visual", "SomMot", "DorsAttn", "VentAttn", "Limbic", "Control", "Default", "Other"]
            net_to_idx = {n: i for i, n in enumerate(net_order)}
            # Build profiles: K × N
            N = len(net_order)
            prof = np.zeros((self.cfg.K, N), dtype=np.float32)
            counts = np.zeros(N, dtype=np.int32)
            # Precompute parcel indices per network
            net_indices: Dict[int, List[int]] = {i: [] for i in range(N)}
            for j, net in enumerate(parcel_nets):
                idx = net_to_idx.get(net, net_to_idx['Other'])
                net_indices[idx].append(j)

            for n_idx, idxs in net_indices.items():
                counts[n_idx] = len(idxs)
                if idxs:
                    prof[:, n_idx] = patterns[:, idxs].mean(axis=1)

            # Also render BrainSpace value maps per state using model mean patterns
            try:
                ctx = self._load_brainspace_context()
            except Exception:
                ctx = None
            if ctx is not None:
                for s in range(self.cfg.K):
                    v = patterns[s, :]
                    outp = out_dir / f"fig_state{s}_brainspace_betas.png"
                    self._brainspace_render_values(ctx, s, v, outp, title="Model mean pattern (z-scored)")

            # Drop previous "top parcels" CSV and figures to simplify interpretation.
            for s in range(self.cfg.K):
                pass
            # Subject-level network means (signed) and magnitudes (L2)
            subj_rows: List[Dict[str, object]] = []
            subj_inputs = _read_inputs_list(self.cfg.betas_dir / "group", self.cfg.K)
            for fpath in subj_inputs:
                try:
                    img = nib.load(str(fpath))
                    data = np.asarray(img.get_fdata(dtype=np.float32))  # K × P
                    if data.ndim != 2 or data.shape[0] != self.cfg.K:
                        continue
                    for s in range(self.cfg.K):
                        vals = data[s, :]
                        for n_idx, idxs in net_indices.items():
                            if not idxs:
                                continue
                            v = vals[idxs]
                            mean_signed = float(np.nanmean(v))
                            l2 = float(np.sqrt(np.nanmean(v**2)))
                            subj_rows.append({
                                "subject_file": fpath.name,
                                "state": s,
                                "network": net_order[n_idx],
                                "mean": mean_signed,
                                "l2": l2,
                                "n_parc": int(len(idxs)),
                            })
                except Exception:
                    continue
            if subj_rows:
                df_subj = pd.DataFrame(subj_rows)
                # Aggregate by state × network
                agg_rows: List[Dict[str, object]] = []
                for (s, net), subdf in df_subj.groupby(["state", "network"]):
                    agg_rows.append({
                        "state": int(s),
                        "network": str(net),
                        "mean_mean": float(subdf["mean"].mean()),
                        "mean_sd": float(subdf["mean"].std(ddof=1) if len(subdf) > 1 else 0.0),
                        "l2_mean": float(subdf["l2"].mean()),
                        "l2_sd": float(subdf["l2"].std(ddof=1) if len(subdf) > 1 else 0.0),
                        "subjects": int(len(subdf)),
                    })
                df_net_subj = pd.DataFrame(agg_rows)
                net_subj_out = out_dir / "state_network_subject_summary.csv"
                df_net_subj.to_csv(net_subj_out, index=False)

                # Per-state subject mean bars
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    for s in range(self.cfg.K):
                        sub = df_net_subj[df_net_subj["state"] == s]
                        if sub.empty:
                            continue
                        # Signed means ± sd (save SD figure)
                        order_idx = [net_order.index(n) for n in sub["network"]]
                        sort_idx = np.argsort(order_idx)
                        sub = sub.iloc[sort_idx]
                        x = np.arange(len(sub))
                        fig, ax = plt.subplots(figsize=(7.2, 3.2))
                        ax.bar(x, sub["mean_mean"].to_numpy(), yerr=sub["mean_sd"].to_numpy(), capsize=3, color="#72B7B2")
                        ax.set_xticks(x)
                        ax.set_xticklabels(sub["network"].tolist(), rotation=30, ha='right')
                        ax.set_title(f"State {s}: network means across subjects (±SD)")
                        ax.set_ylabel("mean z")
                        ax.grid(True, axis='y', alpha=0.2)
                        fig.tight_layout()
                        outp = out_dir / f"fig_state{s}_net_subject_means.png"
                        fig.savefig(outp, dpi=150)
                        plt.close(fig)
                        # Merge with existing entry for this state if present
                        state_figs.setdefault(s, {"net": None, "brain": None})
                        state_figs[s]["net"] = outp

                except Exception as e:
                    log.warning("summary_net_subject_fig_failed", extra={"err": str(e)})

        except FileNotFoundError:
            pass

        # Compact overview per state
        # Pick a few temporal metrics to pivot wide
        keep_metrics = [m for m in ("FO", "DT_mean", "SR_state") if m in set(df_temporal["metric"]) ]
        wide = df_temporal[df_temporal["metric"].isin(keep_metrics)].pivot_table(
            index="state", columns="metric", values="mean"
        )
        wide.columns = [f"{c}_mean" for c in wide.columns]
        overview = wide.reset_index().merge(df_spatial, on="state", how="left")
        overview_out = out_dir / "state_summary.csv"
        overview.to_csv(overview_out, index=False)

        # Optional visuals (bar charts) if matplotlib is available
        figures: List[Path] = []
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Helper to save a bar chart
            def _barplot(state_vals: Dict[int, Tuple[float, float]], title: str, ylabel: str, fname: str) -> Path:
                xs = sorted(state_vals.keys())
                means = [state_vals[s][0] for s in xs]
                sds = [state_vals[s][1] for s in xs]
                labels = [f"S{s}" for s in xs]
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(range(len(xs)), means, yerr=sds, capsize=3, color="#4C78A8")
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels(labels)
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.grid(True, axis='y', alpha=0.2)
                fig.tight_layout()
                outp = out_dir / fname
                fig.savefig(outp, dpi=150)
                plt.close(fig)
                return outp

            # Build per-state mean/std lookups from df_temporal
            def _mean_std(metric_name: str) -> Dict[int, Tuple[float, float]]:
                sub = df_temporal[df_temporal["metric"] == metric_name]
                d: Dict[int, Tuple[float, float]] = {}
                for _, r in sub.iterrows():
                    d[int(r["state"])] = (float(r.get("mean", np.nan)), float(r.get("std", 0.0)))
                # Ensure all states present
                for s in range(self.cfg.K):
                    d.setdefault(s, (np.nan, 0.0))
                return d

            if not df_temporal.empty:
                # FO plot
                if (df_temporal["metric"] == "FO").any():
                    figures.append(_barplot(_mean_std("FO"), "Fractional Occupancy (mean±sd)", "FO", "fig_fo.png"))
                # DT_mean plot
                if (df_temporal["metric"] == "DT_mean").any():
                    figures.append(_barplot(_mean_std("DT_mean"), "Dwell Time (mean±sd)", "TRs", "fig_dt_mean.png"))

            # Sex-split grouped bars (M vs F) per state if Sex present
            if not df_metrics.empty and "Sex" in df_metrics.columns:
                def _sex_group_plot(metric: str, title: str, ylabel: str, fname: str) -> Path | None:
                    sub = df_metrics[["state", "Sex", metric]].dropna()
                    if sub.empty:
                        return None
                    # Normalize Sex values to {M,F}
                    sx = sub["Sex"].astype(str).str.upper().map({"MALE": "M", "FEMALE": "F", "M": "M", "F": "F", "1": "M", "0": "F"}).fillna("")
                    sub = sub.assign(Sex=sx)
                    if (sub["Sex"] == "").all():
                        return None
                    # Aggregate mean and std per state and sex
                    agg = sub.groupby(["state", "Sex"]).agg(mean=(metric, "mean"), sd=(metric, "std"), n=(metric, "count")).reset_index()
                    # Ensure both sexes present per state (fill NaNs)
                    xs = list(range(self.cfg.K))
                    means_M = []; sds_M = []; means_F = []; sds_F = []
                    for s in xs:
                        rowM = agg[(agg["state"] == s) & (agg["Sex"] == "M")]
                        rowF = agg[(agg["state"] == s) & (agg["Sex"] == "F")]
                        means_M.append(float(rowM["mean"].values[0]) if not rowM.empty else np.nan)
                        sds_M.append(float(rowM["sd"].values[0]) if not rowM.empty else 0.0)
                        means_F.append(float(rowF["mean"].values[0]) if not rowF.empty else np.nan)
                        sds_F.append(float(rowF["sd"].values[0]) if not rowF.empty else 0.0)
                    # Plot grouped bars (M vs F)
                    labels = [f"S{s}" for s in xs]
                    x = np.arange(len(xs))
                    width = 0.38
                    fig, ax = plt.subplots(figsize=(7, 3.2))
                    ax.bar(x - width/2, means_M, width, yerr=sds_M, capsize=3, label="M", color="#4C78A8")
                    ax.bar(x + width/2, means_F, width, yerr=sds_F, capsize=3, label="F", color="#E45756")
                    ax.set_xticks(x)
                    ax.set_xticklabels(labels)
                    ax.set_title(title)
                    ax.set_ylabel(ylabel)
                    ax.legend(frameon=False)
                    ax.grid(True, axis='y', alpha=0.2)
                    fig.tight_layout()
                    outp = out_dir / fname
                    fig.savefig(outp, dpi=150)
                    plt.close(fig)
                    return outp

                if "FO" in df_metrics.columns:
                    p = _sex_group_plot("FO", "Fractional Occupancy by Sex", "FO", "fig_fo_by_sex.png")
                    if p: figures.append(p)
                if "DT_mean" in df_metrics.columns:
                    p = _sex_group_plot("DT_mean", "Dwell Time by Sex", "TRs", "fig_dt_mean_by_sex.png")
                    if p: figures.append(p)

            # Significant parcel counts
            if not df_spatial.empty and "n_sig_fwe" in df_spatial.columns:
                xs = list(range(self.cfg.K))
                counts = [int(df_spatial.set_index("state").get("n_sig_fwe", pd.Series()).get(s, 0)) for s in xs]
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(range(len(xs)), counts, color="#F58518")
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels([f"S{s}" for s in xs])
                ax.set_title("Significant Parcels (FWE<0.05)")
                ax.set_ylabel("Count")
                ax.grid(True, axis='y', alpha=0.2)
                fig.tight_layout()
                outp = out_dir / "fig_sig_counts.png"
                fig.savefig(outp, dpi=150)
                plt.close(fig)
                figures.append(outp)

        except Exception as e:
            log.warning("summary_plots_failed", extra={"err": str(e)})

        # HTML overview (very lightweight)
        html_out = out_dir / "summary.html"
        try:
            parts: List[str] = []
            parts.append("<html><head><meta charset='utf-8'><title>HMM Summary</title>"
                         "<style>body{font-family:system-ui,Arial,sans-serif;margin:1.5rem;}"
                         "h1,h2{margin-top:1.2rem;} table{border-collapse:collapse;}"
                         "th,td{border:1px solid #ccc;padding:4px 8px;}"
                         "th{background:#f5f5f5;}</style></head><body>")
            parts.append(f"<h1>HMM Summary (K={self.cfg.K})</h1>")
            parts.append("<h2>Temporal (WHEN)</h2>")
            parts.append(f"<p>Source: {temporal_out}</p>")
            parts.append(df_temporal.to_html(index=False))
            parts.append("<h2>Spatial (WHERE) — PALM highlights</h2>")
            parts.append(f"<p>Source: {spatial_out}</p>")
            parts.append(df_spatial.to_html(index=False))
            parts.append("<h2>Overview</h2>")
            parts.append(f"<p>Source: {overview_out}</p>")
            parts.append(overview.to_html(index=False))
            if figures:
                parts.append("<h2>Visuals</h2>")
                for fig in figures:
                    parts.append(f"<div><img src='{fig.name}' style='max-width:900px;'></div>")
            # Per-state figure gallery
            parts.append("<h2>State mean patterns</h2>")
            if state_figs:
                for s in sorted(state_figs):
                    entry = state_figs[s]
                    parts.append(f"<h3>State {s}</h3>")
                    net_p = entry.get("net")
                    brain_p = entry.get("brain")
                    if net_p:
                        parts.append(f"<div><img src='{Path(net_p).name}' style='max-width:900px;'></div>")
                    if brain_p:
                        parts.append(f"<div><img src='{Path(brain_p).name}' style='max-width:900px;'></div>")
                # Enrichment
                enrich_csv = out_dir / "state_network_enrichment.csv"
                if enrich_csv.exists():
                    parts.append("<h2>Network enrichment (permutation)</h2>")
                    parts.append(f"<p>See: {enrich_csv} — p_mean_2s (two-sided for signed mean), p_l2_1s (one-sided for magnitude). Z-scores are relative to the permutation null.</p>")
            parts.append("</body></html>")
            html_out.write_text("\n".join(parts), encoding="utf-8")
        except Exception as e:
            # Don't fail the pipeline just for HTML rendering
            log.warning("summary_html_failed", extra={"err": str(e)})

        log.info(
            "summary_written",
            extra={
                "temporal": str(temporal_out),
                "spatial": str(spatial_out),
                "overview": str(overview_out),
                "html": str(html_out),
            },
        )
        return temporal_out, spatial_out, overview_out, html_out
