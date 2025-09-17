#!/usr/bin/env python3
from __future__ import annotations

"""
Compute parcel-level summary stats across subjects for each state.

Replaces src/11_parcel_stats.py with a Python implementation that:
  - reads per-subject parcel TSVs produced by state parcellation
  - restricts to subjects listed in group/subjects_used.csv
  - parses parcel labels to include names
  - writes a compact summary CSV per state plus one wide summary

Output directory: `<betas_dir>/parcel/stats`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .logger import get_logger

log = get_logger(__name__)


def _sniff_delim(path: Path) -> str:
    txt = path.read_text(encoding="utf-8", errors="ignore")[:2048]
    return "\t" if txt.count("\t") >= txt.count(",") else ","


def _read_subjects_used(path: Path) -> Tuple[List[str], Dict[str, float]]:
    if not path.exists():
        return [], {}
    delim = _sniff_delim(path)
    df = pd.read_csv(path, sep=delim)
    sid_col = None
    for c in df.columns:
        if c.lower() in ("sid", "subject", "subject_id", "participant", "participant_id", "id"):
            sid_col = c; break
    if sid_col is None:
        sid_col = df.columns[0]
    sids = df[sid_col].astype(str).tolist()
    sex_map: Dict[str, float] = {}
    for c in df.columns:
        if c.lower() == "sex":
            m: Dict[str, float] = {}
            for s, v in zip(df[sid_col].astype(str), df[c]):
                try:
                    # Accept numeric or textual sex values
                    if isinstance(v, str):
                        vv = v.strip().upper()
                        if vv in ("M", "MALE"): m[s] = 1.0; continue
                        if vv in ("F", "FEMALE"): m[s] = 0.0; continue
                    m[s] = float(v)
                except Exception:
                    # Unparseable -> leave unspecified
                    continue
            sex_map = m
            break
    return sids, sex_map


def _parse_label_table(path: Path) -> List[str]:
    """Parse Workbench label table text into a list of parcel names ordered by index.
    The format alternates lines: NAME, then "index R G B A".
    We return a list of names where position (index-1) holds the name.
    """
    if not path.exists():
        return []
    names: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    i = 0
    # Build a dict index->name then expand to dense list
    idx_to_name: Dict[int, str] = {}
    while i + 1 < len(lines):
        name = lines[i]
        meta = lines[i + 1].split()
        try:
            idx = int(meta[0])
        except Exception:
            # if unexpected format, bail to simple sequential names
            break
        idx_to_name[idx] = name
        i += 2
    if idx_to_name:
        max_idx = max(idx_to_name)
        names = [idx_to_name.get(j + 1, f"parcel_{j+1}") for j in range(max_idx)]
    else:
        # Fallback: return lines that look like names
        names = [ln for j, ln in enumerate(lines) if j % 2 == 0]
    return names


@dataclass
class ParcelStatsConfig:
    parcel_dir: Path            # betas_dir/parcel
    subjects_used_csv: Path     # betas_dir/group/subjects_used.csv
    K: int
    labels_tsv: Path            # parcel_labels.tsv
    out_dir: Path | None = None


class ParcelStatsRunner:
    def __init__(self, cfg: ParcelStatsConfig):
        self.cfg = cfg

    def _find_subject_tsv(self, sid: str) -> Path | None:
        cand1 = self.cfg.parcel_dir / f"{sid}_state_betas_{self.cfg.K}S_z.parc.tsv"
        cand2 = self.cfg.parcel_dir / f"{sid}_state_betas_{self.cfg.K}S_zscored.parc.tsv"
        if cand1.exists():
            return cand1
        if cand2.exists():
            return cand2
        # Fallback: any file starting with sid and ending with parc.tsv
        for p in sorted(self.cfg.parcel_dir.glob(f"{sid}*parc.tsv")):
            return p
        return None

    def run(self) -> Path:
        out_dir = self.cfg.out_dir or (self.cfg.parcel_dir / "stats")
        out_dir.mkdir(parents=True, exist_ok=True)

        sids, sex_map = _read_subjects_used(self.cfg.subjects_used_csv)
        if not sids:
            # If no subjects_used.csv, use all TSVs in the folder
            sids = [p.name.split("_")[0] for p in self.cfg.parcel_dir.glob("*_z.parc.tsv")]
        labels = _parse_label_table(self.cfg.labels_tsv)

        # Load per-subject arrays
        X_list: List[np.ndarray] = []
        kept_sids: List[str] = []
        for sid in sids:
            tsv = self._find_subject_tsv(sid)
            if tsv is None:
                log.warning("missing_parcel_tsv", extra={"sid": sid})
                continue
            arr = np.loadtxt(tsv, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != self.cfg.K:
                log.warning("bad_shape", extra={"sid": sid, "shape": str(arr.shape)})
                continue
            X_list.append(arr)
            kept_sids.append(sid)

        if not X_list:
            raise SystemExit("No parcel TSVs found for selected subjects.")

        X = np.stack(X_list, axis=0)  # N × P × K
        N, P, K = X.shape
        assert K == self.cfg.K

        # Basic summary per state
        rows: List[dict] = []
        for k in range(K):
            Xk = X[:, :, k]  # N × P
            mean_all = Xk.mean(axis=0)
            sd_all = Xk.std(axis=0, ddof=1)
            # Optional sex split if available (expects 0/1 coding)
            if sex_map:
                sex = np.array([sex_map.get(s, np.nan) for s in kept_sids])
                m_mask = np.isclose(sex, 1.0); f_mask = np.isclose(sex, 0.0)
                Xm = Xk[m_mask]; Xf = Xk[f_mask]
                mean_m = Xm.mean(axis=0) if Xm.size else np.full(P, np.nan, np.float32)
                mean_f = Xf.mean(axis=0) if Xf.size else np.full(P, np.nan, np.float32)
            else:
                mean_m = np.full(P, np.nan, np.float32)
                mean_f = np.full(P, np.nan, np.float32)

            for p in range(P):
                rows.append({
                    "state": k,
                    "parcel_index": p + 1,
                    "parcel_label": (labels[p] if p < len(labels) else f"parcel_{p+1}"),
                    "N": N,
                    "mean": float(mean_all[p]),
                    "sd": float(sd_all[p]),
                    "mean_m": float(mean_m[p]) if not np.isnan(mean_m[p]) else np.nan,
                    "mean_f": float(mean_f[p]) if not np.isnan(mean_f[p]) else np.nan,
                })

        df = pd.DataFrame(rows)
        wide = df.pivot_table(index=["parcel_index", "parcel_label"], columns="state", values="mean")
        wide.columns = [f"state{s}_mean" for s in wide.columns]
        wide = wide.reset_index()

        out_all = out_dir / "stats_state_summary.csv"
        out_per_state: List[Path] = []
        df.to_csv(out_all, index=False)
        for s in range(K):
            d = df[df["state"] == s].drop(columns=["state"]).copy()
            out_s = out_dir / f"stats_state{s}.csv"
            d.to_csv(out_s, index=False)
            out_per_state.append(out_s)

        log.info("parcel_stats_written", extra={"rows": int(len(df)), "out": str(out_all)})
        return out_all
