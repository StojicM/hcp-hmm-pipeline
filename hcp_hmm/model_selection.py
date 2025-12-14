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
from .hmm_fit import HMMRunner, runs_for_state
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

    @property
    def Ktag(self) -> str:
        return f"{self.K}S"

    @property
    def states_dir(self) -> Path:
        return self.run_dir / "per_subject_states"


class BaseDetector:
    name: str = "detector"

    def run(self, ctx: EvalContext) -> DetectorResult:  # pragma: no cover - interface
        raise NotImplementedError


class JunkStateDetector(BaseDetector):
    name = "junk"

    def run(self, ctx: EvalContext) -> DetectorResult:
        df = ctx.metrics_state.copy()
        if df.empty:
            return DetectorResult(self.name, summary={"n_junk_states": 0})
        if "state" not in df.columns:
            raise SystemExit("metrics_state is missing 'state' column")

        needed = [c for c in ("FO", "DT_mean", "n_visits", "IV_mean", "SR_state") if c in df.columns]
        if not needed:
            raise SystemExit("metrics_state is missing temporal metric columns (FO/DT_mean/n_visits/...)")

        df["state"] = df["state"].astype(int)

        # State-level aggregates across subjects
        g = df.groupby("state", sort=True)
        out = pd.DataFrame({"state": sorted(df["state"].unique().tolist())})
        if "FO" in df.columns:
            out["FO_median"] = g["FO"].median().to_numpy()
            out["FO_mean"] = g["FO"].mean().to_numpy()
            out["FO_q05"] = g["FO"].quantile(0.05).to_numpy()
            out["presence_frac"] = g["FO"].apply(lambda x: float(np.mean(np.asarray(x) > 0))).to_numpy()
        if "DT_mean" in df.columns:
            out["DT_mean_median_tr"] = g["DT_mean"].median().to_numpy()
            out["DT_mean_mean_tr"] = g["DT_mean"].mean().to_numpy()
        if "n_visits" in df.columns:
            out["n_visits_median"] = g["n_visits"].median().to_numpy()
            out["n_visits_mean"] = g["n_visits"].mean().to_numpy()
        if "IV_mean" in df.columns:
            out["IV_mean_median_tr"] = g["IV_mean"].median().to_numpy()
        if "SR_state" in df.columns:
            out["SR_state_mean"] = g["SR_state"].mean().to_numpy()

        thr = ctx.cfg.junk
        out["flag_low_fo"] = out.get("FO_median", pd.Series(False, index=out.index)) < float(thr.fo_median_min)
        out["flag_short_dt"] = out.get("DT_mean_median_tr", pd.Series(False, index=out.index)) < float(thr.dt_mean_min_tr)
        out["flag_low_presence"] = out.get("presence_frac", pd.Series(False, index=out.index)) < float(thr.presence_min)
        out["junk_state"] = out[["flag_low_fo", "flag_short_dt", "flag_low_presence"]].any(axis=1)

        junk_states = out[out["junk_state"]]["state"].astype(int).tolist()

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
        df_g = ctx.metrics_global.copy()
        df_t = ctx.transitions_long.copy()

        per_subject = []
        for sid, sub in df_t.groupby("Subject", sort=False):
            try:
                K = int(ctx.K)
                Pmat = np.full((K, K), np.nan, dtype=float)
                for _, r in sub.iterrows():
                    Pmat[int(r["from_state"]), int(r["to_state"])] = float(r["P"])
                if np.isnan(Pmat).any():
                    continue
                a_raw, a_norm = self._asymmetry(Pmat)
                per_subject.append({"Subject": str(sid), "asym_raw": a_raw, "asym_norm": a_norm})
            except Exception:
                continue

        df_asym = pd.DataFrame(per_subject)

        summary: Dict[str, object] = {}
        for col in ("entropy_rate_bits", "occ_entropy_bits", "mean_self_transition", "spectral_gap", "LZC_switches"):
            if col in df_g.columns:
                summary[f"{col}_mean"] = float(pd.to_numeric(df_g[col], errors="coerce").mean())
                summary[f"{col}_std"] = float(pd.to_numeric(df_g[col], errors="coerce").std(ddof=1))
        if "entropy_rate_bits_mean" in summary:
            denom = math.log2(max(int(ctx.K), 2))
            summary["entropy_rate_norm_mean"] = float(summary["entropy_rate_bits_mean"]) / denom if denom > 0 else float("nan")
        if not df_asym.empty:
            summary["asym_norm_mean"] = float(pd.to_numeric(df_asym["asym_norm"], errors="coerce").mean())
            summary["asym_norm_std"] = float(pd.to_numeric(df_asym["asym_norm"], errors="coerce").std(ddof=1))

        figs: Dict[str, Path] = {}
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if not df_asym.empty:
                fig, ax = plt.subplots(figsize=(6, 3))
                vals = pd.to_numeric(df_asym["asym_norm"], errors="coerce").dropna().to_numpy(dtype=float)
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
            tables={"transition_asymmetry_subject": df_asym} if not df_asym.empty else {},
            figures=figs,
        )


class IndecisionDetector(BaseDetector):
    name = "indecision"

    def _load_gamma(self, path: Path, K: int) -> np.ndarray:
        G = np.loadtxt(path)
        if G.ndim == 1:
            G = G.reshape(-1, K)
        if G.shape[1] != K:
            raise SystemExit(f"{path.name}: expected K={K} columns, got {G.shape[1]}")
        return G.astype(np.float32, copy=False)

    def run(self, ctx: EvalContext) -> DetectorResult:
        K = int(ctx.K)
        thr = ctx.cfg.indecision

        rows = []
        for sid in ctx.metrics_global["Subject"].astype(str).unique():
            f = ctx.states_dir / f"{sid}_state_probs_{ctx.Ktag}.txt"
            if not f.exists():
                continue
            G = self._load_gamma(f, K)
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

    def _load_gamma(self, path: Path, K: int) -> np.ndarray:
        G = np.loadtxt(path)
        if G.ndim == 1:
            G = G.reshape(-1, K)
        if G.shape[1] != K:
            raise SystemExit(f"{path.name}: expected K={K} columns, got {G.shape[1]}")
        return G.astype(np.float32, copy=False)

    def _load_states(self, path: Path) -> np.ndarray:
        v = np.loadtxt(path, dtype=int)
        if v.ndim != 1:
            v = np.asarray(v).reshape(-1)
        return v.astype(int, copy=False)

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
        for sid in ctx.metrics_global["Subject"].astype(str).unique():
            fG = ctx.states_dir / f"{sid}_state_probs_{ctx.Ktag}.txt"
            fS = ctx.states_dir / f"{sid}_state_vector_{ctx.Ktag}.txt"
            if not fG.exists() or not fS.exists():
                continue
            G = self._load_gamma(fG, K)
            states = self._load_states(fS)
            if int(G.shape[0]) != int(states.shape[0]):
                continue
            if int(G.shape[0]) != expected_T:
                # keep the sweep usable even if a few subjects differ
                log.warning(
                    "eval_reliability_length_mismatch",
                    extra={"sid": str(sid), "T": int(G.shape[0]), "expected_T": expected_T},
                )
                continue

            fo_gamma = np.zeros((n_runs, K), dtype=float)
            fo_vit = np.zeros((n_runs, K), dtype=float)
            dt_mean = np.zeros((n_runs, K), dtype=float)
            for r in range(n_runs):
                a = r * run_len
                b = (r + 1) * run_len
                segG = G[a:b, :]
                segS = states[a:b]
                fo_gamma[r] = np.mean(segG, axis=0)
                # viterbi FO and DT
                for s in range(K):
                    fo_vit[r, s] = float(np.mean(segS == s))
                    runs = runs_for_state(segS, s)
                    dt_mean[r, s] = float(np.mean(runs)) if runs else 0.0

            rows.append(
                dict(
                    Subject=str(sid),
                    fo_gamma_r=float(self._pairwise_corr_mean(fo_gamma)),
                    fo_viterbi_r=float(self._pairwise_corr_mean(fo_vit)),
                    dt_mean_r=float(self._pairwise_corr_mean(dt_mean)),
                )
            )

        df = pd.DataFrame(rows)
        summary: Dict[str, object] = {}
        if not df.empty:
            for col in ("fo_gamma_r", "fo_viterbi_r", "dt_mean_r"):
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
        probe = out_dir / "metrics" / f"metrics_state_{Ktag}.csv"
        if probe.exists() and not self.cfg.force:
            log.info("model_select_skip_fit", extra={"K": int(K), "seed": int(seed), "path": str(probe)})
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = _HMMConfig(
            in_dir=self.cfg.in_dir,
            out_dir=out_dir,
            K=int(K),
            cov=self.cfg.hmm.cov,
            max_iter=int(self.cfg.hmm.max_iter),
            tol=float(self.cfg.hmm.tol),
            seed=int(seed),
            backend=str(getattr(self.cfg.hmm, "backend", "hmmlearn")),
            tr_sec=float(self.cfg.hmm.tr_sec),
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

        metrics_state = run_dir / "metrics" / f"metrics_state_{Ktag}.csv"
        metrics_global = run_dir / "metrics" / f"metrics_global_{Ktag}.csv"
        transitions_long = run_dir / "metrics" / f"transitions_long_{Ktag}.csv"
        means_csv = run_dir / f"state_mean_patterns_{Ktag}.csv"
        summary_json = run_dir / "summary.json"
        required = [metrics_state, metrics_global, transitions_long, means_csv, summary_json]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing evaluation inputs: " + ", ".join(missing))

        ctx = EvalContext(
            run_dir=run_dir,
            eval_dir=eval_dir,
            K=int(K),
            seed=int(seed),
            cfg=self.cfg.evaluation,
            hmm_summary=_read_json(summary_json),
            metrics_state=pd.read_csv(metrics_state),
            metrics_global=pd.read_csv(metrics_global),
            transitions_long=pd.read_csv(transitions_long),
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
            "backend": str(getattr(self.cfg.hmm, "backend", "hmmlearn")),
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

        scorecard_csv = eval_dir / "scorecard.csv"
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
                self._fit_one(K, seed, run_dir)
                scorecards.append(self._evaluate_one(K, seed, run_dir))

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
