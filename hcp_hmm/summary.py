#!/usr/bin/env python3
from __future__ import annotations

"""
Lightweight HMM summary:
  - Temporal metrics per state (CSV)
  - Spatial PALM highlights per state (CSV)
  - Combined overview (CSV)
  - Simple HTML with tables and optional bar plots (matplotlib)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    for i in range(0, len(lines), 2):
        names.append(lines[i])
    return names


def _labels_from_pscalar(betas_dir: Path, K: int) -> List[str]:
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


def _spatial_for_state(group_dir: Path, state: int, K: int, labels: List[str], p_thr: float = 0.05) -> Dict[str, object]:
    Ktag = f"{K}S"
    fwe = group_dir / "palm" / f"palm_{Ktag}_state{state}_dat_tstat_fwep.pscalar.nii"
    unc = group_dir / "palm" / f"palm_{Ktag}_state{state}_dat_tstat_uncp.pscalar.nii"
    row: Dict[str, object] = {"state": state}
    if fwe.exists():
        arr = np.asarray(nib.load(str(fwe)).get_fdata(dtype=np.float32)).ravel()
        P = arr.shape[0]
        row["n_sig_fwe"] = int(np.count_nonzero(arr < p_thr))
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
    n_perm: int = 500
    seed: int = 42


class SummaryBuilder:
    def __init__(self, cfg: SummaryConfig):
        self.cfg = cfg

    def run(self) -> Tuple[Path, Path, Path, Path]:
        out_dir = self.cfg.hmm_dir / "summary"
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics_csv = self.cfg.hmm_dir / "metrics" / f"metrics_state_{self.cfg.K}S.csv"
        df_temporal = _temporal_summary(metrics_csv, self.cfg.K)
        temporal_out = out_dir / "state_temporal_summary.csv"
        df_temporal.to_csv(temporal_out, index=False)

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

        keep_metrics = [m for m in ("FO", "DT_mean", "SR_state") if m in set(df_temporal["metric"])]
        wide = df_temporal[df_temporal["metric"].isin(keep_metrics)].pivot_table(
            index="state", columns="metric", values="mean"
        )
        wide.columns = [f"{c}_mean" for c in wide.columns]
        overview = wide.reset_index().merge(df_spatial, on="state", how="left")
        overview_out = out_dir / "state_summary.csv"
        overview.to_csv(overview_out, index=False)

        figures: List[Path] = []
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
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
            parts.append("</body></html>")
            html_out.write_text("".join(parts), encoding="utf-8")
        except Exception as e:
            log.warning("summary_html_failed", extra={"err": str(e)})

        log.info("summary_done", extra={"out_dir": str(out_dir)})
        return temporal_out, spatial_out, overview_out, html_out
