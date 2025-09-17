#!/usr/bin/env python3
from __future__ import annotations

"""
HMM fitting and export, matching src/02_fit_hmm.py outputs, with metrics.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from .logger import get_logger
from .subjects import load_subject_covariates

log = get_logger(__name__)


def runs_for_state(states: np.ndarray, s: int):
    runs, c = [], 0
    for v in states:
        if v == s:
            c += 1
        elif c > 0:
            runs.append(c); c = 0
    if c > 0:
        runs.append(c)
    return runs


def intervisit_intervals(states: np.ndarray, s: int):
    idx = np.where(states == s)[0]
    if idx.size == 0:
        return []
    starts = idx[(np.r_[True, np.diff(idx) > 1])]
    if len(starts) < 2:
        return []
    return list(np.diff(starts))


def stationary_distribution(P: np.ndarray, tol=1e-12):
    # SPEED: linear solve instead of eig; numerically equivalent for our use
    K = P.shape[0]
    A = (P.T - np.eye(K))
    A[-1, :] = 1.0
    b = np.zeros(K); b[-1] = 1.0
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    pi = np.clip(pi, 0.0, None)
    s = pi.sum()
    return (pi / s) if s >= tol else np.ones(K) / K


def entropy_bits(p, eps=1e-12):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def lzc_binary(states: np.ndarray) -> float:
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
    in_dir: Path
    out_dir: Path
    K: int
    cov: str = "diag"
    max_iter: int = 500
    tol: float = 1e-3
    seed: int = 42
    tr_sec: float = 0.72
    subjects_csv: Path | None = None


class HMMRunner:
    def __init__(self, cfg: HMMConfig):
        self.cfg = cfg

    def _prepare_covariates(self) -> Tuple[List[str], Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
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
        # SPEED: memory-map large array to reduce RAM pressure; dtype left unchanged
        X = np.load(self.cfg.in_dir / "train_X.npy", mmap_mode="r")
        idx = pd.read_csv(self.cfg.in_dir / "subjects_index.csv")
        # Normalize index columns to a standard schema
        cols = {c.lower(): c for c in idx.columns}
        sid_key = None
        for k in ("subject", "sid", "subject_id", "uid", "id"):
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
            "covars_shape": list(np.asarray(model.covars_).shape),
        }
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        np.savetxt(out / f"state_transition_matrix_{Ktag}.txt", model.transmat_, fmt="%.6f", delimiter="\t")
        pd.DataFrame(model.means_).to_csv(out / f"state_mean_patterns_{Ktag}.csv", index_label="state")
        
        log.info("hmm_train_done", extra={"loglik": float(logL), "subjects": int(len(idx)), "parcels": int(X.shape[1])})

        rows_state, rows_global, rows_trans = [], [], []

        # SPEED: predict once over all sequences, then slice by subject
        states_all = model.predict(X, lengths=lengths).astype(int)
        probs_all = model.predict_proba(X, lengths=lengths)
        offsets = np.cumsum([0] + lengths)

        covar_cols, covars_lookup, covars_uid_lookup = self._prepare_covariates()
        default_covars = {col: pd.NA for col in covar_cols}
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
            np.savetxt(states_dir / f"{stem}_state_vector_{Ktag}.txt", states, fmt="%d")
            # keep original (TXT) format for probs to avoid any downstream changes
            np.savetxt(states_dir / f"{stem}_state_probs_{Ktag}.txt", probs, fmt="%.6f")

            K = self.cfg.K
            # SPEED: vectorized transition counts
            if T >= 2:
                a = states[:-1].astype(np.int64)
                b = states[1:].astype(np.int64)
                flat = a * K + b
                counts = np.bincount(flat, minlength=K*K).reshape(K, K).astype(np.float64)
            else:
                counts = np.zeros((K, K), dtype=np.float64)

            alpha = 1.0
            denom = counts.sum(axis=1, keepdims=True) + alpha * K
            P = (counts + alpha) / np.maximum(denom, 1.0)

            # Precompute metadata for CSV rows (subject + covariates)
            subj_cov = covars_for_subject(sid)
            meta = {"subject": sid, "Subject": sid}
            if covar_cols:
                meta.update({col: subj_cov.get(col, pd.NA) for col in covar_cols})
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

            # SPEED: np.count_nonzero
            switches_total = int(np.count_nonzero(states[1:] != states[:-1])) if T > 1 else 0
            SR_global = switches_total / max(T - 1, 1)

            pi = stationary_distribution(P)
            # SPEED: vectorized row entropies once
            P_clip = np.clip(P, 1e-12, 1.0)
            row_H = -(P_clip * np.log2(P_clip)).sum(axis=1)
            occ_entropy_bits = entropy_bits(pi)
            entropy_rate_bits = float((pi * row_H).sum())

            mean_self = float(np.trace(P) / K)
            evals = np.linalg.eigvals(P)
            evals = np.sort(np.abs(np.real(evals)))[::-1]
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
            bc = np.bincount(states, minlength=K).astype(np.float64)
            FO_all = bc / max(T, 1)
            for s_ in range(K):
                FO = float(FO_all[s_])
                runs = runs_for_state(states, s_)
                n_visits = int(len(runs))
                if n_visits:
                    DT_mean = float(np.mean(runs))
                    DT_median = float(np.median(runs))
                    DT_var = float(np.var(runs, ddof=1)) if n_visits > 1 else 0.0
                else:
                    DT_mean = DT_median = DT_var = 0.0
                iv = intervisit_intervals(states, s_)
                IV_mean = float(np.mean(iv)) if iv else 0.0

                if T > 1:
                    entries = int(np.count_nonzero((states[:-1] != s_) & (states[1:] == s_)))
                    exits   = int(np.count_nonzero((states[:-1] == s_) & (states[1:] != s_)))
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
        pd.DataFrame(rows_state).to_csv(self.cfg.out_dir / "metrics" / f"metrics_state_{Ktag}.csv", index=False)
        pd.DataFrame(rows_global).to_csv(self.cfg.out_dir / "metrics" / f"metrics_global_{Ktag}.csv", index=False)
        pd.DataFrame(rows_trans).to_csv(self.cfg.out_dir / "metrics" / f"transitions_long_{Ktag}.csv", index=False)
        if missing_covars:
            sample = sorted(missing_covars)
            log.warning("covars_missing_subjects", extra={
                "missing_n": len(missing_covars),
                "sample": sample[:5]
            })
        log.info("hmm_outputs_written", extra={"out_dir": str(self.cfg.out_dir)})
