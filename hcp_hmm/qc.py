#!/usr/bin/env python3
from __future__ import annotations

"""
QC reporting for HMM fits: fingerprints, posterior sharpness, FO summaries, FD correlations, HTML report.
Replaces src/qc_hmm.py usage with an in-package implementation.
"""

import glob
import html
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .logger import get_logger

log = get_logger(__name__)


def _load_label_names(labels_path: Path) -> Optional[List[str]]:
    # Accept a simple one-column TSV/CSV (first column = label names)
    if not labels_path.exists():
        return None
    try:
        txt = labels_path.read_text("utf-8", errors="ignore")
        delim = "\t" if txt.count("\t") >= txt.count(",") else ","
        df = pd.read_csv(labels_path, sep=delim)
        col = df.columns[0]
        names = df[col].astype(str).tolist()
        # Drop empty/NA
        names = [n for n in names if isinstance(n, str) and n.strip()]
        return names if names else None
    except Exception as e:
        log.warning("qc_label_read_failed", extra={"path": str(labels_path), "err": str(e)})
        return None


def _yeo7_fingerprints(means_df: pd.DataFrame, label_names: List[str]) -> pd.DataFrame:
    nets = [
        ("Default",      ["Default"]),
        ("Visual",       ["Vis", "Visual"]),
        ("SomMot",       ["SomMot", "Somatomotor"]),
        ("DorsAttn",     ["DorsAttn", "DorsalAttn", "DAN"]),
        ("SalVentAttn",  ["SalVentAttn", "VentralAttn", "Salience", "VAN"]),
        ("Limbic",       ["Limbic"]),
        ("Cont",         ["Cont", "Control", "Frontoparietal"]),
    ]
    lname = [str(x) for x in label_names]
    net2cols = {}
    for net, tokens in nets:
        cols = [i for i, nm in enumerate(lname) if any(tok in nm for tok in tokens)]
        net2cols[net] = cols
    rows = []
    for s in range(means_df.shape[0]):
        row = {"state": s}
        for net, cols in net2cols.items():
            row[net] = float(np.nan) if len(cols) == 0 else float(means_df.iloc[s, cols].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("state")


def _posterior_sharpness(gamma_dir: Path, K: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    Ktag = f"{K}S"
    max_posts = []
    for f in glob.glob(str(gamma_dir / f"*state_probs_{Ktag}.txt")):
        P = np.loadtxt(f)
        if P.ndim == 1:
            P = P.reshape(-1, K)
        max_posts.append(P.max(axis=1))
    if not max_posts:
        return pd.DataFrame(), pd.DataFrame()
    mx = np.concatenate(max_posts)
    summary = pd.DataFrame({
        "N_TR": [len(mx)],
        "median": [np.median(mx)],
        "p25": [np.percentile(mx, 25)],
        "p75": [np.percentile(mx, 75)],
        "min": [mx.min()],
        "max": [mx.max()],
    })
    bins = np.linspace(0.0, 1.0, 21)
    counts, edges = np.histogram(mx, bins=bins)
    hist = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts})
    return summary, hist


def _fo_balance(gamma_dir: Path, K: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    Ktag = f"{K}S"
    rows = []
    for f in glob.glob(str(gamma_dir / f"*state_vector_{Ktag}.txt")):
        sid = Path(f).stem.split("_")[0]
        z = np.loadtxt(f, dtype=int)
        if z.ndim != 1:
            z = z.ravel()
        for k in range(K):
            rows.append({"Subject": sid, "state": k, "FO": (z == k).mean()})
    fo_df = pd.DataFrame(rows)
    if fo_df.empty:
        return fo_df, fo_df
    summ = fo_df.groupby("state")["FO"].describe().reset_index()
    return fo_df, summ


def _fd_correlations(hmm_dir: Path, K: int, fd_csv: Optional[Path]) -> pd.DataFrame:
    if fd_csv is None or not Path(fd_csv).exists():
        return pd.DataFrame()
    Ktag = f"{K}S"
    metrics_path = hmm_dir / f"metrics/metrics_global_{Ktag}.csv"
    if not metrics_path.exists():
        return pd.DataFrame()
    metrics = pd.read_csv(metrics_path)
    if "Subject" not in metrics.columns:
        return pd.DataFrame()
    fd = pd.read_csv(fd_csv)
    if "Subject" not in fd.columns or "mean_FD" not in fd.columns:
        return pd.DataFrame()
    df = metrics.merge(fd, on="Subject", how="left")
    cols = [c for c in ["SR_global","occ_entropy_bits","entropy_rate_bits",
                        "mean_self_transition","spectral_gap","LZC_switches"] if c in df.columns]
    rows = []
    for c in cols:
        if df["mean_FD"].notna().sum() >= 2 and df[c].notna().sum() >= 2:
            r = df[[c, "mean_FD"]].corr().iloc[0, 1]
            rows.append({"metric": c, "r_with_mean_FD": float(r)})
    return pd.DataFrame(rows)


def _write_html_report(outdir: Path, K: int,
                       fp_df: pd.DataFrame,
                       post_sum: pd.DataFrame,
                       post_hist: pd.DataFrame,
                       fo_df: pd.DataFrame,
                       fo_sum: pd.DataFrame,
                       fd_corr: pd.DataFrame):
    Ktag = f"{K}S"
    html_parts = []
    css = """
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
      h1 { margin-bottom: 0; }
      h2 { margin-top: 28px; }
      table { border-collapse: collapse; margin: 8px 0 16px 0; }
      th, td { border: 1px solid #ddd; padding: 6px 10px; font-size: 14px; }
      th { background: #f3f3f3; }
      .note { color: #666; font-size: 13px; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    </style>
    """
    html_parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>"+css+"</head><body>")
    html_parts.append(f"<h1>HMM QC report ({html.escape(Ktag)})</h1>")
    html_parts.append(f"<div class='note'>Generated: {html.escape(datetime.datetime.now().isoformat(timespec='seconds'))}</div>")

    # Fingerprints
    if not fp_df.empty:
        html_parts.append("<h2>Yeo7 fingerprints</h2>")
        html_parts.append(fp_df.to_html(float_format=lambda x: f"{x:.3f}", border=0))
    # Posterior sharpness
    if not post_sum.empty:
        html_parts.append("<h2>Posterior sharpness</h2>")
        html_parts.append(post_sum.to_html(index=False, float_format=lambda x: f"{x:.3f}", border=0))
        if not post_hist.empty:
            html_parts.append("<h3>Histogram</h3>")
            html_parts.append(post_hist.to_html(index=False, float_format=lambda x: f"{x:.3f}", border=0))
    # FO summary
    if not fo_sum.empty:
        html_parts.append("<h2>FO balance</h2>")
        html_parts.append(fo_sum.to_html(index=False, float_format=lambda x: f"{x:.3f}", border=0))
    # FD correlations
    if not fd_corr.empty:
        html_parts.append("<h2>Correlation with mean FD</h2>")
        html_parts.append(fd_corr.to_html(index=False, float_format=lambda x: f"{x:.3f}", border=0))

    html_parts.append("</body></html>")
    outpath = outdir / f"report_{Ktag}.html"
    outpath.write_text("\n".join(html_parts), encoding="utf-8")
    return outpath


@dataclass
class QCConfig:
    hmm_dir: Path
    K: int
    atlas_labels: Optional[Path] = None  # path to TSV with label names (first column)
    fd_csv: Optional[Path] = None


class QCReporter:
    def __init__(self, cfg: QCConfig):
        self.cfg = cfg

    def run(self) -> None:
        outdir = self.cfg.hmm_dir / "qc"
        outdir.mkdir(parents=True, exist_ok=True)
        Ktag = f"{self.cfg.K}S"

        # Fingerprints (optional)
        fp_df = pd.DataFrame()
        means_path = self.cfg.hmm_dir / f"state_mean_patterns_{Ktag}.csv"
        if self.cfg.atlas_labels and means_path.exists():
            label_names = _load_label_names(self.cfg.atlas_labels)
            if label_names:
                means_df = pd.read_csv(means_path, index_col=0)
                fp_df = _yeo7_fingerprints(means_df, label_names)
                fp_df.to_csv(outdir / f"fingerprints_Yeo7_{Ktag}.csv")

        # Posterior sharpness
        post_sum, post_hist = _posterior_sharpness(self.cfg.hmm_dir / "per_subject_states", self.cfg.K)
        if not post_sum.empty:
            post_sum.to_csv(outdir / f"posterior_sharpness_summary_{Ktag}.csv", index=False)
        if not post_hist.empty:
            post_hist.to_csv(outdir / f"posterior_sharpness_hist_{Ktag}.csv", index=False)

        # FO balance
        fo_df, fo_sum = _fo_balance(self.cfg.hmm_dir / "per_subject_states", self.cfg.K)
        if not fo_df.empty:
            fo_df.to_csv(outdir / f"fo_per_subject_{Ktag}.csv", index=False)
            fo_sum.to_csv(outdir / f"fo_summary_{Ktag}.csv", index=False)

        # FD correlations (optional)
        fd_corr = _fd_correlations(self.cfg.hmm_dir, self.cfg.K, self.cfg.fd_csv)
        if not fd_corr.empty:
            fd_corr.to_csv(outdir / f"fd_correlations_{Ktag}.csv", index=False)

        # HTML report
        _write_html_report(outdir, self.cfg.K, fp_df, post_sum, post_hist, fo_df, fo_sum, fd_corr)
        log.info("qc_report_written", extra={"dir": str(outdir)})

