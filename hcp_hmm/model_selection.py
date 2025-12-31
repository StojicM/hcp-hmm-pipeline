#!/usr/bin/env python3
from __future__ import annotations

"""
Model selection / evaluation sweep for HMM K (and seeds).

This module fits HMMs for multiple K/seed combinations and computes a
scorecard of detectors:
  1) Junk-State Detectors
  2) Randomness Detectors
  3) Indecision Detectors
  4) Clone Detectors
  5) Reality Check: Test–Retest reliability (run-splitting)
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .config import EvaluationParams, HMMParams
from .hmm_fit import HMMConfig as _HMMConfig
from .hmm_fit import HMMRunner
from .logger import get_logger

log = get_logger(__name__)


def _json_default(x: object) -> object:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (Path,)):
        return str(x)
    return str(x)


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size or x.size < 2:
        return float("nan")
    xs = float(np.std(x, ddof=1))
    ys = float(np.std(y, ddof=1))
    if not np.isfinite(xs) or not np.isfinite(ys) or xs == 0.0 or ys == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _entropy_bits_rows(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    P = np.clip(P, eps, 1.0)
    P = P / np.maximum(P.sum(axis=1, keepdims=True), eps)
    return -(P * np.log2(P)).sum(axis=1)


def _entropy_bits_vec(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float).ravel()
    if p.size == 0:
        return float("nan")
    p = np.clip(p, eps, 1.0)
    p = p / np.maximum(p.sum(), eps)
    return float(-(p * np.log2(p)).sum())


def _load_gamma_txt(path: Path, K: int) -> np.ndarray:
    G = np.loadtxt(path)
    if G.ndim == 1:
        G = G.reshape(-1, K)
    if G.shape[1] != K:
        raise SystemExit(f"{path.name}: expected K={K} columns, got {G.shape[1]}")
    G = G.astype(np.float32, copy=False)
    # txt export is rounded; re-normalize rows for safety
    row_sums = np.sum(G, axis=1, keepdims=True, dtype=np.float32)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return (G / row_sums).astype(np.float32, copy=False)


def _subject_ids_from_states_dir(states_dir: Path, Ktag: str) -> list[str]:
    suf = f"_state_probs_{Ktag}.txt"
    out: list[str] = []
    for p in sorted(states_dir.glob(f"*{suf}")):
        if p.name.endswith(suf):
            out.append(p.name[: -len(suf)])
    return out


def _soft_transition_counts_from_gamma(G: np.ndarray) -> np.ndarray:
    """Soft transition counts C[i,j] ~= sum_t gamma_t(i) * gamma_{t+1}(j)."""
    G = np.asarray(G, dtype=float)
    if G.ndim != 2 or G.shape[0] < 2:
        K = int(G.shape[1]) if G.ndim == 2 else 0
        return np.zeros((K, K), dtype=np.float64)
    return (G[:-1, :].T @ G[1:, :]).astype(np.float64, copy=False)


@dataclass
class DetectorResult:
    name: str
    summary: Dict[str, object] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: Dict[str, Path] = field(default_factory=dict)


@dataclass
class EvalContext:
    run_dir: Path
    eval_dir: Path
    K: int
    seed: int
    cfg: EvaluationParams
    hmm_summary: Dict[str, object]
    metrics_state: pd.DataFrame
    metrics_global: pd.DataFrame
    transitions_long: pd.DataFrame
    means: np.ndarray  # K×P
    _gamma_cache: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    @property
    def Ktag(self) -> str:
        return f"{self.K}S"

    @property
    def states_dir(self) -> Path:
        return self.run_dir / "per_subject_states"

    def subject_ids(self) -> list[str]:
        if not self.metrics_global.empty and "Subject" in self.metrics_global.columns:
            try:
                vals = self.metrics_global["Subject"].astype(str).tolist()
                # Preserve order while deduplicating
                seen = set()
                out = []
                for v in vals:
                    if v in seen:
                        continue
                    seen.add(v)
                    out.append(v)
                if out:
                    return out
            except Exception:
                pass
        return _subject_ids_from_states_dir(self.states_dir, self.Ktag)

    def gamma(self, sid: str) -> Optional[np.ndarray]:
        sid = str(sid)
        if sid in self._gamma_cache:
            return self._gamma_cache[sid]
        f = self.states_dir / f"{sid}_state_probs_{self.Ktag}.txt"
        if not f.exists():
            return None
        try:
            G = _load_gamma_txt(f, int(self.K))
        except Exception as e:
            log.warning("eval_load_gamma_failed", extra={"sid": sid, "path": str(f), "err": str(e)})
            return None
        self._gamma_cache[sid] = G
        return G


class BaseDetector:
    name: str = "detector"

    def run(self, ctx: EvalContext) -> DetectorResult:  # pragma: no cover - interface
        raise NotImplementedError


class JunkStateDetector(BaseDetector):
    name = "junk"

    def run(self, ctx: EvalContext) -> DetectorResult:
        K = int(ctx.K)
        thr = ctx.cfg.junk

        rows: list[dict[str, object]] = []
        for sid in ctx.subject_ids():
            G = ctx.gamma(str(sid))
            if G is None:
                continue
            T = int(G.shape[0])
            if T < 2:
                continue

            fo = np.mean(G.astype(float, copy=False), axis=0)
            exp_counts = fo * float(T)
            present = exp_counts >= 1.0  # >= 1 expected TR in this state

            C = _soft_transition_counts_from_gamma(G)
            diag = np.diag(C)
            row_sum = np.sum(C, axis=1)
            entries = np.sum(C, axis=0) - diag
            exits = np.sum(C, axis=1) - diag

            for s in range(K):
                is_present = bool(present[s])
                fo_s = float(fo[s]) if is_present else 0.0

                dt_mean = float("nan")
                n_visits = 0.0
                sr_state = 0.0
                if is_present and float(row_sum[s]) > 1e-6:
                    p_self = float(diag[s] / row_sum[s])
                    p_self = min(max(p_self, 0.0), 1.0 - 1e-9)
                    dt_mean = float(1.0 / (1.0 - p_self))
                    n_visits = float(float(G[0, s]) + float(entries[s]))
                    sr_state = float((float(entries[s]) + float(exits[s])) / max(T - 1, 1))

                rows.append(
                    {
                        "Subject": str(sid),
                        "state": int(s),
                        "FO": fo_s,
                        "present": is_present,
                        "DT_mean": dt_mean,
                        "n_visits": n_visits,
                        "SR_state": sr_state,
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            return DetectorResult(self.name, summary={"n_junk_states": 0, "junk_states": []})

        g = df.groupby("state", sort=True)
        states = list(range(K))
        out = pd.DataFrame({"state": states})
        out["FO_median"] = g["FO"].median().reindex(states).fillna(0.0).to_numpy(dtype=float)
        out["FO_mean"] = g["FO"].mean().reindex(states).fillna(0.0).to_numpy(dtype=float)
        out["FO_q05"] = g["FO"].quantile(0.05).reindex(states).fillna(0.0).to_numpy(dtype=float)
        out["presence_frac"] = g["present"].mean().reindex(states).fillna(0.0).to_numpy(dtype=float)
        out["DT_mean_median_tr"] = g["DT_mean"].median().reindex(states).to_numpy(dtype=float)
        out["DT_mean_mean_tr"] = g["DT_mean"].mean().reindex(states).to_numpy(dtype=float)
        out["n_visits_median"] = g["n_visits"].median().reindex(states).fillna(0.0).to_numpy(dtype=float)
        out["n_visits_mean"] = g["n_visits"].mean().reindex(states).fillna(0.0).to_numpy(dtype=float)
        out["SR_state_mean"] = g["SR_state"].mean().reindex(states).fillna(0.0).to_numpy(dtype=float)

        out["flag_low_fo"] = out["FO_median"] < float(thr.fo_median_min)
        out["flag_short_dt"] = out["DT_mean_median_tr"] < float(thr.dt_mean_min_tr)
        out["flag_low_presence"] = out["presence_frac"] < float(thr.presence_min)
        out["junk_state"] = out[["flag_low_fo", "flag_short_dt", "flag_low_presence"]].any(axis=1)

        junk_states = out.loc[out["junk_state"], "state"].astype(int).tolist()

        # Plots
        figs: Dict[str, Path] = {}
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            if "FO_median" in out.columns:
                axes[0].bar(out["state"].astype(int), out["FO_median"].astype(float), color="#4C78A8")
                axes[0].axhline(float(thr.fo_median_min), color="red", lw=1, ls="--", alpha=0.7)
                axes[0].set_title("State FO (median across subjects)")
                axes[0].set_xlabel("State")
                axes[0].set_ylabel("FO")
            if "DT_mean_median_tr" in out.columns:
                axes[1].bar(out["state"].astype(int), out["DT_mean_median_tr"].astype(float), color="#F58518")
                axes[1].axhline(float(thr.dt_mean_min_tr), color="red", lw=1, ls="--", alpha=0.7)
                axes[1].set_title("State Dwell (median DT_mean, TR)")
                axes[1].set_xlabel("State")
                axes[1].set_ylabel("TR")
            for ax in axes:
                ax.grid(True, axis="y", alpha=0.2)
            fig.tight_layout()
            p = ctx.eval_dir / "fig_junk_state_bars.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            figs["fig_junk_state_bars"] = p
        except Exception as e:
            log.warning("eval_junk_plots_failed", extra={"err": str(e)})

        return DetectorResult(
            name=self.name,
            summary={
                "n_junk_states": int(len(junk_states)),
                "junk_states": junk_states,
            },
            tables={"junk_states": out.sort_values("state").reset_index(drop=True)},
            figures=figs,
        )


class RandomnessDetector(BaseDetector):
    name = "randomness"

    @staticmethod
    def _asymmetry(P: np.ndarray) -> tuple[float, float]:
        P = np.asarray(P, dtype=float)
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            return float("nan"), float("nan")
        K = P.shape[0]
        tri = np.triu_indices(K, 1)
        num = float(np.abs(P - P.T)[tri].mean())
        den = float(np.abs(P + P.T)[tri].mean())
        return num, (num / (den + 1e-12))

    def run(self, ctx: EvalContext) -> DetectorResult:
        K = int(ctx.K)
        rows = []
        for sid in ctx.subject_ids():
            G = ctx.gamma(str(sid))
            if G is None:
                continue
            T = int(G.shape[0])
            if T < 2:
                continue

            pi = np.mean(G.astype(float, copy=False), axis=0)
            pi = pi / np.maximum(np.sum(pi), 1e-12)
            occ_entropy_bits = _entropy_bits_vec(pi)

            C = _soft_transition_counts_from_gamma(G)
            row_sum = np.sum(C, axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                P = C / np.maximum(row_sum, 1e-12)
            # rows with very low expected occupancy -> NaN (avoid meaningless metrics)
            P[np.squeeze(row_sum, axis=1) < 1.0] = np.nan

            row_H = _entropy_bits_rows(P)
            entropy_rate_bits = float(np.sum(pi * np.nan_to_num(row_H, nan=0.0)))

            diag = np.diag(P).astype(float, copy=False)
            mean_self = float(np.nanmean(diag)) if np.isfinite(diag).any() else float("nan")

            spectral_gap = float("nan")
            try:
                if not np.isnan(P).any():
                    evals = np.linalg.eigvals(P)
                    evals = np.sort(np.abs(np.real(evals)))[::-1]
                    spectral_gap = float(1.0 - (evals[1] if len(evals) > 1 else 0.0))
            except Exception:
                spectral_gap = float("nan")

            a_raw, a_norm = (float("nan"), float("nan"))
            if not np.isnan(P).any():
                try:
                    a_raw, a_norm = self._asymmetry(P)
                except Exception:
                    pass

            # Expected switching probability per TR (soft analogue)
            same_p = np.sum(G[:-1, :] * G[1:, :], axis=1, dtype=np.float64)
            switch_rate_soft = float(np.mean(1.0 - same_p)) if same_p.size else float("nan")

            rows.append(
                {
                    "Subject": str(sid),
                    "occ_entropy_bits": occ_entropy_bits,
                    "entropy_rate_bits": entropy_rate_bits,
                    "mean_self_transition": mean_self,
                    "spectral_gap": spectral_gap,
                    "asym_raw": a_raw,
                    "asym_norm": a_norm,
                    "switch_rate_soft": switch_rate_soft,
                }
            )

        df = pd.DataFrame(rows)

        summary: Dict[str, object] = {}
        if not df.empty:
            for col in (
                "occ_entropy_bits",
                "entropy_rate_bits",
                "mean_self_transition",
                "spectral_gap",
                "asym_norm",
                "switch_rate_soft",
            ):
                summary[f"{col}_mean"] = float(pd.to_numeric(df[col], errors="coerce").mean())
                summary[f"{col}_std"] = float(pd.to_numeric(df[col], errors="coerce").std(ddof=1))
            denom = math.log2(max(int(ctx.K), 2))
            if denom > 0 and "entropy_rate_bits_mean" in summary:
                summary["entropy_rate_norm_mean"] = float(summary["entropy_rate_bits_mean"]) / denom
                if "entropy_rate_bits_std" in summary:
                    summary["entropy_rate_norm_std"] = float(summary["entropy_rate_bits_std"]) / denom

        figs: Dict[str, Path] = {}
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if not df.empty and "asym_norm" in df.columns:
                fig, ax = plt.subplots(figsize=(6, 3))
                vals = pd.to_numeric(df["asym_norm"], errors="coerce").dropna().to_numpy(dtype=float)
                ax.hist(vals, bins=30, color="#54A24B", alpha=0.9)
                ax.set_title("Transition Asymmetry (normalized) across subjects")
                ax.set_xlabel("asym_norm")
                ax.set_ylabel("Count")
                ax.grid(True, axis="y", alpha=0.2)
                fig.tight_layout()
                p = ctx.eval_dir / "fig_transition_asymmetry_hist.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
                figs["fig_transition_asymmetry_hist"] = p
        except Exception as e:
            log.warning("eval_randomness_plots_failed", extra={"err": str(e)})

        return DetectorResult(
            name=self.name,
            summary=summary,
            tables={"randomness_subject": df} if not df.empty else {},
            figures=figs,
        )


class IndecisionDetector(BaseDetector):
    name = "indecision"

    def run(self, ctx: EvalContext) -> DetectorResult:
        K = int(ctx.K)
        thr = ctx.cfg.indecision

        rows = []
        for sid in ctx.subject_ids():
            G = ctx.gamma(str(sid))
            if G is None:
                continue
            P = np.clip(G.astype(float, copy=False), 1e-12, 1.0)
            P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
            H = -(P * np.log2(P)).sum(axis=1)
            mx = np.max(P, axis=1)
            rows.append(
                dict(
                    Subject=str(sid),
                    posterior_entropy_bits_mean=float(np.mean(H)),
                    posterior_entropy_bits_median=float(np.median(H)),
                    posterior_max_mean=float(np.mean(mx)),
                    posterior_max_median=float(np.median(mx)),
                    frac_max_ge_dom=float(np.mean(mx >= float(thr.dominance_thr))),
                    frac_max_lt_amb=float(np.mean(mx < float(thr.ambiguous_thr))),
                    T=int(G.shape[0]),
                )
            )

        df = pd.DataFrame(rows)
        summary: Dict[str, object] = {}
        if not df.empty:
            for col in (
                "posterior_entropy_bits_mean",
                "posterior_max_mean",
                "frac_max_ge_dom",
                "frac_max_lt_amb",
            ):
                summary[f"{col}_mean"] = float(pd.to_numeric(df[col], errors="coerce").mean())
                summary[f"{col}_std"] = float(pd.to_numeric(df[col], errors="coerce").std(ddof=1))

        figs: Dict[str, Path] = {}
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if not df.empty:
                fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                axes[0].hist(
                    pd.to_numeric(df["posterior_max_mean"], errors="coerce").dropna().to_numpy(dtype=float),
                    bins=30,
                    color="#E45756",
                    alpha=0.9,
                )
                axes[0].axvline(float(thr.dominance_thr), color="black", lw=1, ls="--", alpha=0.7)
                axes[0].set_title("Posterior dominance (mean max prob)")
                axes[0].set_xlabel("mean max posterior")
                axes[0].set_ylabel("Count")

                axes[1].hist(
                    pd.to_numeric(df["posterior_entropy_bits_mean"], errors="coerce").dropna().to_numpy(dtype=float),
                    bins=30,
                    color="#72B7B2",
                    alpha=0.9,
                )
                axes[1].set_title("Posterior entropy (mean, bits)")
                axes[1].set_xlabel("mean posterior entropy")
                axes[1].set_ylabel("Count")
                for ax in axes:
                    ax.grid(True, axis="y", alpha=0.2)
                fig.tight_layout()
                p = ctx.eval_dir / "fig_posterior_indecision_hists.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
                figs["fig_posterior_indecision_hists"] = p
        except Exception as e:
            log.warning("eval_indecision_plots_failed", extra={"err": str(e)})

        return DetectorResult(name=self.name, summary=summary, tables={"posterior_subject": df} if not df.empty else {}, figures=figs)


class CloneDetector(BaseDetector):
    name = "clone"

    def run(self, ctx: EvalContext) -> DetectorResult:
        K = int(ctx.K)
        X = np.asarray(ctx.means, dtype=float)
        if X.ndim != 2 or X.shape[0] != K:
            raise SystemExit(f"Expected means shape (K,P) with K={K}; got {X.shape}")

        C = np.corrcoef(X)
        C = np.clip(C, -1.0, 1.0)
        thr = float(ctx.cfg.clone.corr_thr)
        pairs = []
        for i in range(K):
            for j in range(i + 1, K):
                r = float(C[i, j])
                if np.isfinite(r) and r >= thr:
                    pairs.append({"state_i": i, "state_j": j, "corr": r})

        df_pairs = pd.DataFrame(pairs).sort_values("corr", ascending=False) if pairs else pd.DataFrame(columns=["state_i", "state_j", "corr"])
        max_offdiag = float(np.max(C[np.triu_indices(K, 1)])) if K > 1 else float("nan")

        figs: Dict[str, Path] = {}
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_title("State mean-pattern similarity (corr)")
            ax.set_xlabel("State")
            ax.set_ylabel("State")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            p = ctx.eval_dir / "fig_state_similarity_heatmap.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            figs["fig_state_similarity_heatmap"] = p
        except Exception as e:
            log.warning("eval_clone_plots_failed", extra={"err": str(e)})

        return DetectorResult(
            name=self.name,
            summary={
                "n_clone_pairs": int(len(df_pairs)),
                "max_state_mean_corr": max_offdiag,
                "clone_corr_thr": thr,
            },
            tables={"clone_pairs": df_pairs},
            figures=figs,
        )


class ReliabilityDetector(BaseDetector):
    name = "reliability"

    @staticmethod
    def _pairwise_corr_mean(mat: np.ndarray) -> float:
        # mat: R×K
        R = int(mat.shape[0])
        vals = []
        for i in range(R):
            for j in range(i + 1, R):
                vals.append(_safe_pearsonr(mat[i], mat[j]))
        if not vals:
            return float("nan")
        return float(np.nanmean(np.asarray(vals, dtype=float)))

    def run(self, ctx: EvalContext) -> DetectorResult:
        K = int(ctx.K)
        run_len = int(ctx.cfg.reliability.run_len_tr)
        n_runs = int(ctx.cfg.reliability.n_runs)
        expected_T = int(run_len * n_runs)

        rows = []
        for sid in ctx.subject_ids():
            G = ctx.gamma(str(sid))
            if G is None:
                continue
            if int(G.shape[0]) != expected_T:
                # keep the sweep usable even if a few subjects differ
                log.warning(
                    "eval_reliability_length_mismatch",
                    extra={"sid": str(sid), "T": int(G.shape[0]), "expected_T": expected_T},
                )
                continue

            fo_gamma = np.zeros((n_runs, K), dtype=float)
            for r in range(n_runs):
                a = r * run_len
                b = (r + 1) * run_len
                segG = G[a:b, :]
                fo_gamma[r] = np.mean(segG, axis=0)

            rows.append(
                dict(
                    Subject=str(sid),
                    fo_gamma_r=float(self._pairwise_corr_mean(fo_gamma)),
                )
            )

        df = pd.DataFrame(rows)
        summary: Dict[str, object] = {}
        if not df.empty:
            for col in ("fo_gamma_r",):
                summary[f"{col}_mean"] = float(pd.to_numeric(df[col], errors="coerce").mean())
                summary[f"{col}_std"] = float(pd.to_numeric(df[col], errors="coerce").std(ddof=1))
        summary["run_len_tr"] = run_len
        summary["n_runs"] = n_runs

        figs: Dict[str, Path] = {}
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if not df.empty:
                fig, ax = plt.subplots(figsize=(6, 3))
                vals = pd.to_numeric(df["fo_gamma_r"], errors="coerce").dropna().to_numpy(dtype=float)
                ax.hist(vals, bins=30, color="#B279A2", alpha=0.9)
                ax.set_title("Stage-5 reliability (FO gamma), within-subject")
                ax.set_xlabel("mean run–run corr")
                ax.set_ylabel("Count")
                ax.grid(True, axis="y", alpha=0.2)
                fig.tight_layout()
                p = ctx.eval_dir / "fig_reliability_fo_gamma_hist.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
                figs["fig_reliability_fo_gamma_hist"] = p
        except Exception as e:
            log.warning("eval_reliability_plots_failed", extra={"err": str(e)})

        return DetectorResult(
            name=self.name,
            summary=summary,
            tables={"reliability_subject": df} if not df.empty else {},
            figures=figs,
        )


def _read_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_means(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "state" in df.columns:
        df = df.sort_values("state").set_index("state")
    return df.to_numpy(dtype=float)


@dataclass
class ModelSelectionConfig:
    """Configuration for running a multi-K / multi-seed sweep."""

    in_dir: Path
    out_dir: Path
    hmm: HMMParams
    evaluation: EvaluationParams
    subjects_csv: Optional[Path] = None
    atlas_dlabel: Optional[Path] = None
    surface_dir: Optional[Path] = None
    surface_left: Optional[Path] = None
    surface_right: Optional[Path] = None
    surface_left_inflated: Optional[Path] = None
    surface_right_inflated: Optional[Path] = None
    force: bool = False


class ModelSelectionRunner:
    def __init__(self, cfg: ModelSelectionConfig):
        self.cfg = cfg

    def _Ks(self) -> list[int]:
        Ks = list(self.cfg.evaluation.K_values) if self.cfg.evaluation.K_values else [int(self.cfg.hmm.K)]
        Ks = sorted({int(k) for k in Ks})
        return Ks

    def _seeds(self) -> list[int]:
        seeds = list(self.cfg.evaluation.seeds) if self.cfg.evaluation.seeds else [int(self.cfg.hmm.seed)]
        seeds = sorted({int(s) for s in seeds})
        return seeds

    def _run_dir(self, K: int, seed: int) -> Path:
        return self.cfg.out_dir / "runs" / f"K{int(K):02d}" / f"seed{int(seed)}"

    def _fit_one(self, K: int, seed: int, out_dir: Path) -> None:
        Ktag = f"{int(K)}S"
        required = [
            out_dir / "metrics" / f"metrics_state_{Ktag}.csv",
            out_dir / "metrics" / f"metrics_global_{Ktag}.csv",
            out_dir / "metrics" / f"transitions_long_{Ktag}.csv",
            out_dir / f"state_mean_patterns_{Ktag}.csv",
            out_dir / "summary.json",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if not missing and not self.cfg.force:
            log.info(
                "model_select_skip_fit K=%d seed=%d dir=%s",
                int(K),
                int(seed),
                str(out_dir),
                extra={"K": int(K), "seed": int(seed), "dir": str(out_dir)},
            )
            return

        # If a previous run completed training but got interrupted during export,
        # resume from the saved model instead of refitting.
        model_path = out_dir / "model.joblib"
        if model_path.exists() and not self.cfg.force:
            try:
                log.info(
                    "model_select_resume_export K=%d seed=%d model=%s missing_n=%d",
                    int(K),
                    int(seed),
                    str(model_path),
                    int(len(missing)),
                    extra={"K": int(K), "seed": int(seed), "path": str(model_path), "missing": missing},
                )
                cfg = _HMMConfig(
                    in_dir=self.cfg.in_dir,
                    out_dir=out_dir,
                    K=int(K),
                    cov=self.cfg.hmm.cov,
                    max_iter=int(self.cfg.hmm.max_iter),
                    tol=float(self.cfg.hmm.tol),
                    seed=int(seed),
                    backend=str(getattr(self.cfg.hmm, "backend", "dynamax_arhmm")),
                    tr_sec=float(self.cfg.hmm.tr_sec),
                    ar_order=int(getattr(self.cfg.hmm, "ar_order", 1)),
                    slds_latent_dim=int(getattr(self.cfg.hmm, "slds_latent_dim", 4)),
                    subjects_csv=self.cfg.subjects_csv,
                    atlas_dlabel=self.cfg.atlas_dlabel,
                    surface_dir=self.cfg.surface_dir,
                    surface_left=self.cfg.surface_left,
                    surface_right=self.cfg.surface_right,
                    surface_left_inflated=self.cfg.surface_left_inflated,
                    surface_right_inflated=self.cfg.surface_right_inflated,
                )
                HMMRunner(cfg).export_only(model_path=model_path)
                missing_after = [str(p) for p in required if not p.exists()]
                if not missing_after:
                    return
                log.warning(
                    "model_select_resume_export_incomplete K=%d seed=%d missing_n=%d",
                    int(K),
                    int(seed),
                    int(len(missing_after)),
                    extra={"K": int(K), "seed": int(seed), "missing": missing_after},
                )
            except Exception as e:
                log.warning(
                    "model_select_resume_failed K=%d seed=%d err=%s",
                    int(K),
                    int(seed),
                    str(e),
                    extra={"K": int(K), "seed": int(seed), "err": str(e)},
                )

        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = _HMMConfig(
            in_dir=self.cfg.in_dir,
            out_dir=out_dir,
            K=int(K),
            cov=self.cfg.hmm.cov,
            max_iter=int(self.cfg.hmm.max_iter),
            tol=float(self.cfg.hmm.tol),
            seed=int(seed),
            backend=str(getattr(self.cfg.hmm, "backend", "dynamax_arhmm")),
            tr_sec=float(self.cfg.hmm.tr_sec),
            ar_order=int(getattr(self.cfg.hmm, "ar_order", 1)),
            slds_latent_dim=int(getattr(self.cfg.hmm, "slds_latent_dim", 4)),
            subjects_csv=self.cfg.subjects_csv,
            atlas_dlabel=self.cfg.atlas_dlabel,
            surface_dir=self.cfg.surface_dir,
            surface_left=self.cfg.surface_left,
            surface_right=self.cfg.surface_right,
            surface_left_inflated=self.cfg.surface_left_inflated,
            surface_right_inflated=self.cfg.surface_right_inflated,
        )
        HMMRunner(cfg).fit_and_export()

    def _evaluate_one(self, K: int, seed: int, run_dir: Path) -> Path:
        Ktag = f"{int(K)}S"
        eval_dir = run_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        scorecard_csv = eval_dir / "scorecard.csv"
        if scorecard_csv.exists() and not self.cfg.force:
            log.info(
                "model_select_skip_eval K=%d seed=%d path=%s",
                int(K),
                int(seed),
                str(scorecard_csv),
                extra={"K": int(K), "seed": int(seed), "path": str(scorecard_csv)},
            )
            return scorecard_csv

        metrics_state = run_dir / "metrics" / f"metrics_state_{Ktag}.csv"
        metrics_global = run_dir / "metrics" / f"metrics_global_{Ktag}.csv"
        transitions_long = run_dir / "metrics" / f"transitions_long_{Ktag}.csv"
        means_csv = run_dir / f"state_mean_patterns_{Ktag}.csv"
        summary_json = run_dir / "summary.json"
        # Evaluation detectors should not depend on hard Viterbi outputs; keep
        # the required inputs minimal (means + summary + posteriors).
        required = [means_csv, summary_json]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing evaluation inputs: " + ", ".join(missing))

        if not list((run_dir / "per_subject_states").glob(f"*_state_probs_{Ktag}.txt")):
            raise FileNotFoundError(f"Missing per-subject posteriors: {run_dir / 'per_subject_states'}/*_state_probs_{Ktag}.txt")

        def _read_csv_optional(path: Path) -> pd.DataFrame:
            if not path.exists():
                return pd.DataFrame()
            try:
                return pd.read_csv(path)
            except Exception as e:
                log.warning("eval_read_csv_failed", extra={"path": str(path), "err": str(e)})
                return pd.DataFrame()

        ctx = EvalContext(
            run_dir=run_dir,
            eval_dir=eval_dir,
            K=int(K),
            seed=int(seed),
            cfg=self.cfg.evaluation,
            hmm_summary=_read_json(summary_json),
            metrics_state=_read_csv_optional(metrics_state),
            metrics_global=_read_csv_optional(metrics_global),
            transitions_long=_read_csv_optional(transitions_long),
            means=_load_means(means_csv),
        )

        detectors: list[BaseDetector] = [
            JunkStateDetector(),
            RandomnessDetector(),
            IndecisionDetector(),
            CloneDetector(),
            ReliabilityDetector(),
        ]

        results: list[DetectorResult] = []
        for det in detectors:
            res = det.run(ctx)
            results.append(res)
            # write tables as we go
            for name, df in res.tables.items():
                df.to_csv(eval_dir / f"{name}.csv", index=False)

        # Run-level scorecard (1 row)
        card: Dict[str, object] = {
            "K": int(K),
            "seed": int(seed),
            "cov": str(self.cfg.hmm.cov),
            "backend": str(getattr(self.cfg.hmm, "backend", "dynamax_arhmm")),
        }
        for k0 in ("loglik", "AIC", "BIC", "n_params", "n_subjects", "n_timepoints_total", "n_parcels"):
            if k0 in ctx.hmm_summary:
                card[k0] = ctx.hmm_summary[k0]

        by_name = {r.name: r for r in results}
        for k, v in by_name.get("junk", DetectorResult("junk")).summary.items():
            card[f"junk__{k}"] = v
        for k, v in by_name.get("randomness", DetectorResult("randomness")).summary.items():
            card[f"randomness__{k}"] = v
        for k, v in by_name.get("indecision", DetectorResult("indecision")).summary.items():
            card[f"indecision__{k}"] = v
        for k, v in by_name.get("clone", DetectorResult("clone")).summary.items():
            card[f"clone__{k}"] = v
        for k, v in by_name.get("reliability", DetectorResult("reliability")).summary.items():
            card[f"reliability__{k}"] = v

        pd.DataFrame([card]).to_csv(scorecard_csv, index=False)
        with open(eval_dir / "scorecard.json", "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2, default=_json_default)

        # Lightweight per-run HTML
        try:
            parts = []
            parts.append("<html><head><meta charset='utf-8'><title>Model Selection</title>"
                         "<style>body{font-family:system-ui,Arial,sans-serif;margin:1.5rem;}"
                         "h1,h2{margin-top:1.2rem;} table{border-collapse:collapse;}"
                         "th,td{border:1px solid #ccc;padding:4px 8px;}"
                         "th{background:#f5f5f5;}</style></head><body>")
            parts.append(f"<h1>Model Selection Run (K={int(K)}, seed={int(seed)})</h1>")
            parts.append("<h2>Scorecard</h2>")
            parts.append(pd.DataFrame([card]).to_html(index=False))
            for res in results:
                if res.tables:
                    parts.append(f"<h2>{res.name}</h2>")
                    for name, df in res.tables.items():
                        parts.append(f"<h3>{name}</h3>")
                        parts.append(df.to_html(index=False))
                if res.figures:
                    parts.append("<h3>Figures</h3>")
                    for _, fig in res.figures.items():
                        parts.append(f"<div><img src='{fig.name}' style='max-width:1100px;'></div>")
            parts.append("</body></html>")
            (eval_dir / "report.html").write_text("".join(parts), encoding="utf-8")
        except Exception as e:
            log.warning("model_select_run_html_failed", extra={"err": str(e)})

        log.info("model_select_eval_done", extra={"K": int(K), "seed": int(seed), "dir": str(eval_dir)})
        return scorecard_csv

    def run(self) -> Path:
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)

        X = self.cfg.in_dir / "train_X.npy"
        idx = self.cfg.in_dir / "subjects_index.csv"
        for required in (X, idx):
            if not required.exists():
                raise FileNotFoundError(required)

        scorecards: list[Path] = []
        for K in self._Ks():
            for seed in self._seeds():
                run_dir = self._run_dir(K, seed)
                try:
                    self._fit_one(K, seed, run_dir)
                except Exception as e:
                    log.warning(
                        "model_select_fit_failed K=%d seed=%d err=%s",
                        int(K),
                        int(seed),
                        str(e),
                        extra={"K": int(K), "seed": int(seed), "err": str(e)},
                    )
                    continue
                try:
                    scorecards.append(self._evaluate_one(K, seed, run_dir))
                except Exception as e:
                    log.warning(
                        "model_select_eval_failed K=%d seed=%d err=%s",
                        int(K),
                        int(seed),
                        str(e),
                        extra={"K": int(K), "seed": int(seed), "err": str(e)},
                    )
                    continue

        # Collate across all runs
        rows = []
        for p in scorecards:
            try:
                rows.append(pd.read_csv(p).iloc[0].to_dict())
            except Exception:
                continue
        df_run = pd.DataFrame(rows)
        df_run = df_run.sort_values(["K", "seed"], kind="mergesort").reset_index(drop=True) if not df_run.empty else df_run
        out_run = self.cfg.out_dir / "summary_by_run.csv"
        df_run.to_csv(out_run, index=False)

        # Aggregate across seeds per K
        out_K = self.cfg.out_dir / "summary_by_K.csv"
        if not df_run.empty:
            numeric_cols = [c for c in df_run.columns if c not in {"K", "seed", "cov", "backend"} and pd.api.types.is_numeric_dtype(df_run[c])]
            grouped = df_run.groupby("K", sort=True)
            df_K = grouped[numeric_cols].agg(["mean", "std"]).reset_index()
            # Flatten multiindex columns
            df_K.columns = ["K" if c[0] == "K" else f"{c[0]}__{c[1]}" for c in df_K.columns.to_list()]
            df_K.to_csv(out_K, index=False)
        else:
            pd.DataFrame(columns=["K"]).to_csv(out_K, index=False)

        # Plots + HTML report
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            figs_dir = self.cfg.out_dir / "figures"
            figs_dir.mkdir(exist_ok=True)

            if not df_run.empty:
                dfK = pd.read_csv(out_K)
                x = dfK["K"].to_numpy(dtype=int)

                def _plot_metric(ax, col_mean: str, title: str):
                    if col_mean not in dfK.columns:
                        ax.set_axis_off()
                        return
                    y = pd.to_numeric(dfK[col_mean], errors="coerce").to_numpy(dtype=float)
                    col_std = col_mean.replace("__mean", "__std")
                    yerr = pd.to_numeric(dfK[col_std], errors="coerce").to_numpy(dtype=float) if col_std in dfK.columns else None
                    ax.errorbar(x, y, yerr=yerr, marker="o", lw=1.5, capsize=3)
                    ax.set_title(title)
                    ax.set_xlabel("K")
                    ax.grid(True, axis="y", alpha=0.2)

                fig, axes = plt.subplots(2, 2, figsize=(10, 6))
                _plot_metric(axes[0, 0], "BIC__mean", "BIC (mean±std across seeds)")
                _plot_metric(axes[0, 1], "junk__n_junk_states__mean", "Junk states (#, mean±std)")
                _plot_metric(axes[1, 0], "clone__n_clone_pairs__mean", "Clone pairs (#, mean±std)")
                _plot_metric(axes[1, 1], "reliability__fo_gamma_r_mean__mean", "Reliability: FO gamma (mean±std)")
                fig.tight_layout()
                p = figs_dir / "fig_metrics_vs_K.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
        except Exception as e:
            log.warning("model_select_plots_failed", extra={"err": str(e)})

        try:
            parts = []
            parts.append("<html><head><meta charset='utf-8'><title>Model Selection</title>"
                         "<style>body{font-family:system-ui,Arial,sans-serif;margin:1.5rem;}"
                         "h1,h2{margin-top:1.2rem;} table{border-collapse:collapse;}"
                         "th,td{border:1px solid #ccc;padding:4px 8px;}"
                         "th{background:#f5f5f5;}</style></head><body>")
            parts.append("<h1>HMM Model Selection Summary</h1>")
            parts.append(f"<p>Out: {self.cfg.out_dir}</p>")
            parts.append("<h2>By K (aggregated across seeds)</h2>")
            try:
                parts.append(pd.read_csv(out_K).to_html(index=False))
            except Exception:
                pass
            parts.append("<h2>By run (K×seed)</h2>")
            try:
                parts.append(pd.read_csv(out_run).to_html(index=False))
            except Exception:
                pass
            fig = self.cfg.out_dir / "figures" / "fig_metrics_vs_K.png"
            if fig.exists():
                parts.append("<h2>Visuals</h2>")
                parts.append(f"<div><img src='figures/{fig.name}' style='max-width:1100px;'></div>")
            parts.append("</body></html>")
            (self.cfg.out_dir / "report.html").write_text("".join(parts), encoding="utf-8")
        except Exception as e:
            log.warning("model_select_html_failed", extra={"err": str(e)})

        log.info("model_select_done", extra={"out_dir": str(self.cfg.out_dir)})
        return out_run


__all__ = ["ModelSelectionConfig", "ModelSelectionRunner"]
