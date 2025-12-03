#!/usr/bin/env python3
from __future__ import annotations

"""
HMM fitting and export with metrics.

Loads the concatenated parcel time series, fits a `GaussianHMM` (hmmlearn),
exports the fitted model, per-subject state sequences and probabilities, and
computes a panel of global and statewise metrics per subject.
"""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy #as np
import pandas #as pd
from hmmlearn.hmm import GaussianHMM
import nibabel #as nib

from .logger import get_logger
from .subjects import load_subject_covariates

log = get_logger(__name__)


def runs_for_state(states: numpy.ndarray, s: int):
    """Return run lengths (dwell times) for consecutive visits to state `s`."""
    runs, c = [], 0
    for v in states:
        if v == s:
            c += 1
        elif c > 0:
            runs.append(c); c = 0
    if c > 0:
        runs.append(c)
    return runs


def intervisit_intervals(states: numpy.ndarray, s: int):
    """Return intervals (in TRs) between successive visits to state `s`."""
    idx = numpy.where(states == s)[0]
    if idx.size == 0:
        return []
    starts = idx[(numpy.r_[True, numpy.diff(idx) > 1])]
    if len(starts) < 2:
        return []
    return list(numpy.diff(starts))


def stationary_distribution(P: numpy.ndarray, tol=1e-12):
    """Compute stationary distribution of a Markov chain with transition `P`.
    Solves `(P^T - I) pi = 0` with a sum-to-one constraint via least squares,
    then normalizes and clips for numerical stability.
    """
    # SPEED: linear solve instead of eig; numerically equivalent for our use
    K = P.shape[0]
    A = (P.T - numpy.eye(K))
    A[-1, :] = 1.0
    b = numpy.zeros(K); b[-1] = 1.0
    pi = numpy.linalg.lstsq(A, b, rcond=None)[0]
    pi = numpy.clip(pi, 0.0, None)
    s = pi.sum()
    return (pi / s) if s >= tol else numpy.ones(K) / K


def entropy_bits(p, eps=1e-12):
    """Shannon entropy (bits) of a probability vector `p` (safe/clipped)."""
    p = numpy.asarray(p, dtype=float)
    p = numpy.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-(p * numpy.log2(p)).sum())


def lzc_binary(states: numpy.ndarray) -> float:
    """Lempel–Ziv complexity of the binary switch signal (state changes)."""
    if len(states) < 2:
        return 0.0
    sw = (states[1:] != states[:-1]).astype(int)
    s = ''.join(str(x) for x in sw.tolist())
    i = k = 1
    l = 1
    c = 1
    n = len(s)
    while True:
        if i + k > n:
            c += 1
            break
        if s[i:i+k] in s[0:l-1]:
            k += 1
        else:
            c += 1
            l += k
            i = l - 1
            k = 1
        if l > n:
            break
    return c / (n / math.log2(n)) if n > 1 else float(c)


@dataclass
class HMMConfig:
    """Configuration for HMM fitting and outputs.

    - `in_dir` / `out_dir`: where to read `train_X.npy` + index and write outputs
    - `K`, `cov`, `max_iter`, `tol`, `seed`: hmmlearn hyperparameters
    - `tr_sec`: TR in seconds (for converting TR counts to seconds in reports)
    - Optional atlas/surfaces used only to render state betas (if available)
    """
    in_dir: Path
    out_dir: Path
    K: int
    cov: str = "diag"
    max_iter: int = 500
    tol: float = 1e-3
    seed: int = 42
    tr_sec: float = 0.72
    subjects_csv: Path | None = None
    # Optional: surfaces + atlas dlabel for BrainSpace rendering of state betas
    atlas_dlabel: Optional[Path] = None
    surface_dir: Optional[Path] = None
    surface_left: Optional[Path] = None
    surface_right: Optional[Path] = None
    surface_left_inflated: Optional[Path] = None
    surface_right_inflated: Optional[Path] = None


class HMMRunner:
    """Runs HMM fit and exports model, states, and metrics."""
    def __init__(self, cfg: HMMConfig):
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
        d = {"lh": left, "rh": right}
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
            log.info("fit_brainspace_unavailable", extra={"err": str(e)})
            return None
        try:
            surf_lh = read_surface(str(surfs["lh"]))
            surf_rh = read_surface(str(surfs["rh"]))
        except Exception as e:
            log.warning("fit_surface_load_failed", extra={"err": str(e)})
            return None
        # Load dlabel vertex→parcel codes
        try:
            img = nibabel.load(str(self.cfg.atlas_dlabel))
            data = numpy.asarray(img.get_fdata()).reshape(-1).astype(int)
            bm = img.header.get_axis(1)
            # Split into LH/RH by structures
            lh_slice = None; rh_slice = None
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
            log.warning("fit_dlabel_load_failed", extra={"err": str(e)})
            return None
        return {
            "surf_lh": surf_lh,
            "surf_rh": surf_rh,
            "plot": plot_hemispheres,
            "codes_lh": codes_lh,
            "codes_rh": codes_rh,
        }

    @staticmethod
    def _zscore_vec(x: numpy.ndarray) -> numpy.ndarray:
        x = numpy.asarray(x, dtype=numpy.float32)
        mu = float(numpy.nanmean(x)) if x.size else 0.0
        sd = float(numpy.nanstd(x, ddof=1)) if x.size else 1.0
        if not numpy.isfinite(sd) or sd == 0.0:
            sd = 1.0
        return (x - mu) / sd

    def _values_to_vertices(self, vals: numpy.ndarray, codes: numpy.ndarray) -> numpy.ndarray:
        # codes are 0..P, with 0=background. Parcel index j corresponds to code j+1
        P = vals.shape[0]
        out = numpy.full(codes.shape[0], numpy.nan, dtype=numpy.float32)
        mask = (codes > 0) & (codes <= P)
        idx = codes[mask] - 1
        out[mask] = vals[idx]
        return out

    def _render_state_betas(self, means: numpy.ndarray) -> None:
        """Optionally render BrainSpace PNGs for state betas if configured."""
        ctx = self._load_brainspace_context()
        if ctx is None:
            return
        surf_lh = ctx["surf_lh"]
        surf_rh = ctx["surf_rh"]
        plot = ctx["plot"]
        codes_lh = ctx["codes_lh"]
        codes_rh = ctx["codes_rh"]
        out_dir = self.cfg.out_dir / "summary"
        out_dir.mkdir(parents=True, exist_ok=True)
        for s in range(self.cfg.K):
            v = numpy.asarray(means[s, :], dtype=numpy.float32)
            for kind, arr in (("betas", v), ("betas_z", self._zscore_vec(v))):
                lh = self._values_to_vertices(arr, codes_lh)
                rh = self._values_to_vertices(arr, codes_rh)
                dense = numpy.concatenate([lh, rh])
                fname = out_dir / f"fig_state{s}_brainspace_{kind}.png"
                try:
                    plot(
                        surf_lh, surf_rh,
                        array_name=dense,
                        color_bar=True,
                        color_range=("sym" if kind == "betas_z" else None),
                        cmap=("RdBu_r" if kind == "betas_z" else "viridis"),
                        nan_color=(0.85, 0.85, 0.85, 1.0),
                        zoom=1.1,
                        background=(1, 1, 1),
                        size=(900, 450),
                        layout_style="grid",
                        label_text={"top": ("State %d %s" % (s, ("(z-scored)" if kind=="betas_z" else "")) )},
                        interactive=False,
                        embed_nb=False,
                        screenshot=True,
                        filename=str(fname),
                        transparent_bg=False,
                    )
                except Exception as e:
                    # Log both the tag and the error string for visibility in plain format
                    log.warning(f"fit_brainspace_betas_failed: {e}", extra={"state": int(s), "kind": kind, "err": str(e)})

    def _prepare_covariates(self) -> Tuple[List[str], Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
        """Load optional subject covariates and expose lookups by Subject/UID."""
        df = load_subject_covariates(getattr(self.cfg, "subjects_csv", None))
        if df.empty:
            return [], {}, {}

        covar_cols = [c for c in df.columns if c != "Subject"]
        subj_lookup = df.set_index("Subject")[covar_cols].to_dict("index")
        uid_lookup = {}
        if "UID" in df.columns:
            uid_lookup = df.groupby("UID")[covar_cols].first().to_dict("index")
        return covar_cols, subj_lookup, uid_lookup

    def fit_and_export(self):
        """Fit the HMM and write model, per-subject states, and metrics CSVs."""
        # SPEED: memory-map large array to reduce RAM pressure; dtype left unchanged
        X = numpy.load(self.cfg.in_dir / "train_X.npy", mmap_mode="r")
        idx = pandas.read_csv(self.cfg.in_dir / "subjects_index.csv")
        # Normalize index columns to a standard schema
        cols = {c.lower(): c for c in idx.columns}
        sid_key = None
        # for k in ("subject", "sid", "subject_id", "uid", "id"):
        for k in ("subject_id"):
            if k in cols:
                sid_key = cols[k]; break
        if sid_key is None:
            raise SystemExit(f"subjects_index.csv missing subject column. Found: {list(idx.columns)}")
        start_key = cols.get("start") or cols.get("start_tr") or cols.get("begin")
        end_key = cols.get("end") or cols.get("end_tr") or cols.get("stop")
        ntr_key = cols.get("ntr") or cols.get("n_tr") or cols.get("ntrs") or cols.get("n")
        if ntr_key is not None:
            lengths = idx[ntr_key].astype(int).tolist()
        elif start_key is not None and end_key is not None:
            lengths = (idx[end_key].astype(int) - idx[start_key].astype(int)).tolist()
        else:
            raise SystemExit("subjects_index.csv must have either nTR or start/end columns.")
        # Rename to canonical names for downstream code
        ren = {sid_key: "Subject"}
        if start_key: ren[start_key] = "start"
        if end_key: ren[end_key] = "end"
        if ntr_key: ren[ntr_key] = "nTR"
        idx = idx.rename(columns=ren)

        log.info("hmm_train_begin", extra={"K": self.cfg.K, "cov": self.cfg.cov, "seed": self.cfg.seed})
        model = GaussianHMM(
            n_components=self.cfg.K,
            covariance_type=self.cfg.cov,
            n_iter=self.cfg.max_iter,
            tol=self.cfg.tol,
            random_state=self.cfg.seed,
            verbose=True,
        )
        model.fit(X, lengths)
        logL = model.score(X, lengths)

        ## Make the output directory
        out = self.cfg.out_dir
        out.mkdir(parents=True, exist_ok=True)
        states_dir = out / "per_subject_states"; states_dir.mkdir(exist_ok=True)
        metrics_dir = out / "metrics"; metrics_dir.mkdir(exist_ok=True)

        # Save model and summaries
        Ktag = f"{self.cfg.K}S"
        joblib.dump(model, out / "model.joblib")
        joblib.dump(model, out / f"hmm_model_{Ktag}.pkl")

        summary = {
            "K": self.cfg.K,
            "covariance_type": self.cfg.cov,
            "max_iter": self.cfg.max_iter,
            "tol": self.cfg.tol,
            "seed": self.cfg.seed,
            "TR_sec": self.cfg.tr_sec,
            "n_parcels": int(X.shape[1]),
            "n_timepoints_total": int(X.shape[0]),
            "n_subjects": int(len(idx)),
            "loglik": float(logL),
            "transmat": model.transmat_.tolist(),
            "startprob": model.startprob_.tolist(),
            "means_shape": list(model.means_.shape),
            "covars_shape": list(numpy.asarray(model.covars_).shape),
        }
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        numpy.savetxt(out / f"state_transition_matrix_{Ktag}.txt", model.transmat_, fmt="%.6f", delimiter="\t")
        pandas.DataFrame(model.means_).to_csv(out / f"state_mean_patterns_{Ktag}.csv", index_label="state")
        # Optional: BrainSpace surfaces of state betas (raw and z-scored)
        try:
            self._render_state_betas(model.means_)
        except Exception as e:
            log.info("fit_betas_render_skipped", extra={"err": str(e)})
        
        log.info("hmm_train_done", extra={"loglik": float(logL), "subjects": int(len(idx)), "parcels": int(X.shape[1])})

        rows_state, rows_global, rows_trans = [], [], []

        # SPEED: predict once over all sequences, then slice by subject
        states_all = model.predict(X, lengths=lengths).astype(int)
        probs_all = model.predict_proba(X, lengths=lengths)
        offsets = numpy.cumsum([0] + lengths)

        covar_cols, covars_lookup, covars_uid_lookup = self._prepare_covariates()
        default_covars = {col: pandas.NA for col in covar_cols}
        missing_covars = set()

        def covars_for_subject(sid: str) -> Dict[str, object]:
            if not covar_cols:
                return {}
            info = covars_lookup.get(sid)
            if info is not None:
                return info
            token = sid.split("_")[0]
            info = covars_uid_lookup.get(token)
            if info is not None:
                return info
            missing_covars.add(sid)
            return default_covars

        for i, row in idx.reset_index(drop=True).iterrows():
            sid = str(row["Subject"]) if "Subject" in row else str(row.get("sid", ""))
            s = offsets[i]
            e = offsets[i + 1]
            Xs = X[s:e, :]  # kept (unused, but preserves exact structure)
            states = states_all[s:e]
            probs = probs_all[s:e, :]
            T = len(states)

            stem = f"{sid}"
            numpy.savetxt(states_dir / f"{stem}_state_vector_{Ktag}.txt", states, fmt="%d")
            # keep original (TXT) format for probs to avoid any downstream changes
            numpy.savetxt(states_dir / f"{stem}_state_probs_{Ktag}.txt", probs, fmt="%.6f")

            K = self.cfg.K
            # SPEED: vectorized transition counts
            if T >= 2:
                a = states[:-1].astype(numpy.int64)
                b = states[1:].astype(numpy.int64)
                flat = a * K + b
                counts = numpy.bincount(flat, minlength=K*K).reshape(K, K).astype(numpy.float64)
            else:
                counts = numpy.zeros((K, K), dtype=numpy.float64)

            alpha = 1.0
            denom = counts.sum(axis=1, keepdims=True) + alpha * K
            P = (counts + alpha) / numpy.maximum(denom, 1.0)

            # Precompute metadata for CSV rows (subject + covariates)
            subj_cov = covars_for_subject(sid)
            meta = {"Subject": sid}
            if covar_cols:
                meta.update({col: subj_cov.get(col, pandas.NA) for col in covar_cols})
            meta_with_K = {**meta, "K": K}

            # transitions long-form (unchanged semantics apart from metadata)
            for ii in range(K):
                for jj in range(K):
                    rows_trans.append({
                        **meta_with_K,
                        "from_state": ii,
                        "to_state": jj,
                        "P": float(P[ii, jj])
                    })

            # SPEED: numpy.count_nonzero
            switches_total = int(numpy.count_nonzero(states[1:] != states[:-1])) if T > 1 else 0
            SR_global = switches_total / max(T - 1, 1)

            pi = stationary_distribution(P)
            # SPEED: vectorized row entropies once
            P_clip = numpy.clip(P, 1e-12, 1.0)
            row_H = -(P_clip * numpy.log2(P_clip)).sum(axis=1)
            occ_entropy_bits = entropy_bits(pi)
            entropy_rate_bits = float((pi * row_H).sum())

            mean_self = float(numpy.trace(P) / K)
            evals = numpy.linalg.eigvals(P)
            evals = numpy.sort(numpy.abs(numpy.real(evals)))[::-1]
            spectral_gap = float(1.0 - (evals[1] if len(evals) > 1 else 0.0))
            LZC = lzc_binary(states)

            rows_global.append({
                **meta_with_K,
                "SR_global": SR_global,
                "occ_entropy_bits": occ_entropy_bits,
                "entropy_rate_bits": entropy_rate_bits,
                "mean_self_transition": mean_self,
                "spectral_gap": spectral_gap,
                "LZC_switches": LZC
            })

            # SPEED: bincount for FO
            bc = numpy.bincount(states, minlength=K).astype(numpy.float64)
            FO_all = bc / max(T, 1)
            for s_ in range(K):
                FO = float(FO_all[s_])
                runs = runs_for_state(states, s_)
                n_visits = int(len(runs))
                if n_visits:
                    DT_mean = float(numpy.mean(runs))
                    DT_median = float(numpy.median(runs))
                    DT_var = float(numpy.var(runs, ddof=1)) if n_visits > 1 else 0.0
                else:
                    DT_mean = DT_median = DT_var = 0.0
                iv = intervisit_intervals(states, s_)
                IV_mean = float(numpy.mean(iv)) if iv else 0.0

                if T > 1:
                    entries = int(numpy.count_nonzero((states[:-1] != s_) & (states[1:] == s_)))
                    exits   = int(numpy.count_nonzero((states[:-1] == s_) & (states[1:] != s_)))
                    SR_state = (entries + exits) / (T - 1)
                else:
                    SR_state = 0.0

                rows_state.append({
                    **meta_with_K,
                    "state": s_,
                    "FO": FO,
                    "DT_mean": DT_mean,
                    "DT_median": DT_median,
                    "DT_var": DT_var,
                    "n_visits": n_visits,
                    "IV_mean": IV_mean,
                    "SR_state": float(SR_state),
                    "row_entropy_bits": float(row_H[s_]),
                    "self_transition": float(P[s_, s_])
                })

        Ktag = f"{self.cfg.K}S"

        # Tidy potential duplicate identifier columns before writing
        def _tidy_id_cols(df: pandas.DataFrame) -> pandas.DataFrame:
            d = df.copy()
            # Prefer canonical 'Subject'; drop/rename variants
            if "subject" in d.columns:
                if "Subject" not in d.columns:
                    d = d.rename(columns={"subject": "Subject"})
                else:
                    # If both exist and are identical after str-cast, drop the lowercase one
                    try:
                        if d["subject"].astype(str).equals(d["Subject"].astype(str)):
                            d = d.drop(columns=["subject"])
                        else:
                            # Fall back to the capitalized version as canonical
                            d = d.drop(columns=["subject"])
                    except Exception:
                        d = d.drop(columns=["subject"])
            # Drop UID if it is redundant with Subject (exact or first-token match)
            if "UID" in d.columns and "Subject" in d.columns:
                try:
                    uid_s = d["UID"].astype(str)
                    subj_s = d["Subject"].astype(str)
                    if uid_s.equals(subj_s) or uid_s.equals(subj_s.str.split("_").str[0]):
                        d = d.drop(columns=["UID"])
                except Exception:
                    pass
            if "Subject" in d.columns:
                d["Subject"] = d["Subject"].astype(str)
            return d

        df_state = _tidy_id_cols(pandas.DataFrame(rows_state))
        df_glob = _tidy_id_cols(pandas.DataFrame(rows_global))
        df_trans = _tidy_id_cols(pandas.DataFrame(rows_trans))

        df_state.to_csv(self.cfg.out_dir / "metrics" / f"metrics_state_{Ktag}.csv", index=False)
        df_glob.to_csv(self.cfg.out_dir / "metrics" / f"metrics_global_{Ktag}.csv", index=False)
        df_trans.to_csv(self.cfg.out_dir / "metrics" / f"transitions_long_{Ktag}.csv", index=False)
        if missing_covars:
            sample = sorted(missing_covars)
            log.warning("covars_missing_subjects", extra={
                "missing_n": len(missing_covars),
                "sample": sample[:5]
            })
        log.info("hmm_outputs_written", extra={"out_dir": str(self.cfg.out_dir)})
