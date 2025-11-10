#!/usr/bin/env python3
from __future__ import annotations

"""Summarise states that satisfy both temporal (\"when\") and spatial (\"where\") criteria."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Tuple

import json
import math

import numpy as np
import pandas as pd
import nibabel as nib

from .logger import get_logger

log = get_logger(__name__)


def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR control (q-values)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    ranks = np.arange(1, m + 1, dtype=float)
    q = p[order] * m / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def _load_numerical_data(path: Path) -> Optional[np.ndarray]:
    """Load numeric array from PALM outputs (CIFTI, GIFTI, or text)."""
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv"}:
            return pd.read_csv(path, header=None).to_numpy(dtype=float)
        if suffix in {".txt"}:
            return np.loadtxt(path, dtype=float, ndmin=1)
        # Handle compressed nifti/cifti names (e.g., .nii, .nii.gz, .dscalar.nii)
        if suffix in {".nii", ".gii"} or path.name.endswith(".nii.gz") or path.name.endswith(".dscalar.nii"):
            img = nib.load(str(path))
            data = np.asanyarray(img.dataobj, dtype=float)
            return data
    except Exception as exc:  # pragma: no cover - best effort
        log.warning("useful_states_load_failed", extra={"file": str(path), "err": str(exc)})
    return None


@dataclass
class UsefulStatesConfig:
    metrics_state_csv: Path
    stats_rm_csv: Path
    palm_dir: Path
    K: int
    out_csv: Path
    q_threshold: float = 0.05
    n_perm_temporal: int = 2000
    out_json: Optional[Path] = None


class UsefulStatesAnalyzer:
    def __init__(self, cfg: UsefulStatesConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(42)

    # --------------------------- Temporal ("when") ---------------------------
    def _metric_columns(self, df: pd.DataFrame) -> list[str]:
        ignore = {"Subject", "UID", "state", "Age", "AgeGroup", "Sex", "mean_FD", "FD"}
        ignore |= {c.lower() for c in ignore}
        cols: list[str] = []
        for col in df.columns:
            if col in ignore or col.lower() in ignore:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                cols.append(col)
        return cols

    def _temporal_stats(self, df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
        results = []
        all_subjects = df["Subject"].astype(str).unique()
        for metric in metrics:
            sub = df[["Subject", "state", "mean_FD", metric]].dropna()
            if sub.empty:
                continue
            sub = sub.copy()
            sub["Subject"] = sub["Subject"].astype(str)

            # Regress out mean FD (global nuisance)
            y = sub[metric].to_numpy(dtype=float)
            fd = sub["mean_FD"].fillna(sub["mean_FD"].mean()).to_numpy(dtype=float)
            X = np.c_[np.ones(len(sub), dtype=float), fd]
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            sub["resid"] = y - X @ beta

            pivot = sub.pivot_table(index="Subject", columns="state", values="resid", aggfunc="mean")
            pivot = pivot.dropna(axis=0, how="any")
            if pivot.empty:
                continue

            states = [int(s) for s in pivot.columns]
            values = pivot.to_numpy(dtype=float)
            n_subjects = values.shape[0]
            row_mean = values.mean(axis=1, keepdims=True)
            obs = (values - row_mean).mean(axis=0)
            obs_abs = np.abs(obs)

            perm_abs = np.zeros((self.cfg.n_perm_temporal, len(states)), dtype=float)
            for b in range(self.cfg.n_perm_temporal):
                permuted = values.copy()
                for i in range(permuted.shape[0]):
                    self.rng.shuffle(permuted[i])
                perm_diff = permuted - permuted.mean(axis=1, keepdims=True)
                perm_abs[b] = np.abs(perm_diff.mean(axis=0))

            counts = (perm_abs >= obs_abs).sum(axis=0)
            pvals = (counts + 1.0) / (self.cfg.n_perm_temporal + 1.0)

            for idx, state in enumerate(states):
                results.append(
                    dict(
                        metric=metric,
                        state=int(state),
                        p_perm=float(pvals[idx]),
                        effect=float(obs[idx]),
                        n_subjects=int(n_subjects),
                        n_subjects_dropped=int(len(all_subjects) - n_subjects),
                    )
                )

        df_res = pd.DataFrame(results)
        if df_res.empty:
            return df_res
        df_res["q_bh"] = _bh_qvalues(df_res["p_perm"].to_numpy())
        return df_res

    # ----------------------------- Spatial ("where") -----------------------------
    def _palm_min_p(self, state: int) -> Tuple[Optional[float], Optional[str], Optional[int]]:
        Ktag = f"{self.cfg.K}S"
        base = f"palm_{Ktag}_state{state}"
        palm_dir = self.cfg.palm_dir
        if not palm_dir.exists():
            return None, None, None

        patterns = [
            f"{base}*_logp*.nii*",
            f"{base}*_logp*.gii",
            f"{base}*_logp*.txt",
            f"{base}*_fdrp*.nii*",
            f"{base}*_fdrp*.txt",
            f"{base}*_fwep*.nii*",
            f"{base}*_fwep*.txt",
            f"{base}*_pvalue*.nii*",
            f"{base}*_pvalue*.txt",
        ]

        min_p = math.inf
        best_file: Optional[str] = None
        best_nonzero_count: Optional[int] = None

        for pattern in patterns:
            for path in palm_dir.glob(pattern):
                data = _load_numerical_data(path)
                if data is None:
                    continue
                arr = np.asarray(data, dtype=float).flatten()
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                name_low = path.name.lower()
                if "logp" in name_low:
                    arr = np.power(10.0, -arr)
                arr = arr[arr >= 0]
                if arr.size == 0:
                    continue
                candidate = float(arr.min())
                if candidate < min_p:
                    min_p = candidate
                    best_file = str(path)
                    best_nonzero_count = int(np.sum(arr <= self.cfg.q_threshold))

        if not np.isfinite(min_p):
            return None, None, None
        return min_p, best_file, best_nonzero_count

    # ----------------------------- Public API -----------------------------
    def run(self) -> Path:
        if not self.cfg.metrics_state_csv.exists():
            raise FileNotFoundError(self.cfg.metrics_state_csv)
        if not self.cfg.stats_rm_csv.exists():
            raise FileNotFoundError(self.cfg.stats_rm_csv)
        df = pd.read_csv(self.cfg.metrics_state_csv)
        if "Subject" not in df.columns or "state" not in df.columns:
            raise SystemExit("metrics_state CSV must contain 'Subject' and 'state' columns")
        df["Subject"] = df["Subject"].astype(str)
        df["state"] = df["state"].astype(int)

        metrics = self._metric_columns(df)
        if not metrics:
            raise SystemExit("No metric columns detected in metrics_state CSV")
        temporal = self._temporal_stats(df, metrics)

        states = sorted(df["state"].unique().tolist())

        summary_rows = []
        for state in states:
            row = {"state": int(state)}

            when_info = temporal[temporal["state"] == state]
            if not when_info.empty:
                best = when_info.sort_values("q_bh", kind="mergesort").iloc[0]
                row.update(
                    when_pass=bool(best["q_bh"] <= self.cfg.q_threshold),
                    when_metric=str(best["metric"]),
                    when_q=float(best["q_bh"]),
                    when_effect=float(best["effect"]),
                    when_n_subjects=int(best["n_subjects"]),
                    when_n_subjects_dropped=int(best["n_subjects_dropped"]),
                )
            else:
                row.update(
                    when_pass=False,
                    when_metric=None,
                    when_q=None,
                    when_effect=None,
                    when_n_subjects=0,
                    when_n_subjects_dropped=int(df["Subject"].nunique()),
                )

            min_p, source, count_sig = self._palm_min_p(state)
            if min_p is not None:
                row.update(
                    where_pass=bool(min_p <= self.cfg.q_threshold),
                    where_min_p=float(min_p),
                    where_source=str(source),
                    where_n_sig=int(count_sig) if count_sig is not None else None,
                )
            else:
                row.update(
                    where_pass=False,
                    where_min_p=None,
                    where_source=None,
                    where_n_sig=None,
                )

            row["useful"] = bool(row["when_pass"]) and bool(row["where_pass"])
            summary_rows.append(row)

        summary = pd.DataFrame(summary_rows).sort_values("state").reset_index(drop=True)
        summary.to_csv(self.cfg.out_csv, index=False)
        log.info("useful_states_written", extra={"out": str(self.cfg.out_csv), "rows": int(len(summary))})

        if self.cfg.out_json:
            with open(self.cfg.out_json, "w", encoding="utf-8") as f:
                json.dump(summary.to_dict(orient="records"), f, indent=2)
            log.info("useful_states_json_written", extra={"out": str(self.cfg.out_json)})

        return self.cfg.out_csv


__all__ = ["UsefulStatesAnalyzer", "UsefulStatesConfig"]
