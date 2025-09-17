#!/usr/bin/env python3
"""
Merge HMM metrics with subject covariates (AgeGroup, Sex, mean_FD).

Writes two files into <hmm_dir>/metrics:
  - metrics_state_<K>S_with_covars.csv
  - metrics_global_<K>S_with_covars.csv

Inputs:
  - metrics_state_<K>S.csv and metrics_global_<K>S.csv from the HMM step
  - subjects.csv with columns: uid, age, sex
  - optional fd_csv with columns: Subject,mean_FD
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

from .logger import get_logger
from .subjects import load_subject_covariates

log = get_logger(__name__)


def merge_covars(hmm_dir: Path, subjects_csv: Path, K: int, fd_csv: Path | None = None) -> tuple[Path, Path]:
    mdir = hmm_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)

    Ktag = f"{K}S"
    state_path = mdir / f"metrics_state_{Ktag}.csv"
    global_path = mdir / f"metrics_global_{Ktag}.csv"
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    if not global_path.exists():
        raise FileNotFoundError(global_path)

    df_state = pd.read_csv(state_path)
    df_glob = pd.read_csv(global_path)

    subs = load_subject_covariates(subjects_csv, strict=True)
    if subs.empty:
        raise SystemExit("subjects CSV did not yield any records")

    if "AgeGroup" not in subs.columns or subs["AgeGroup"].isna().all():
        if "Age" in subs.columns:
            subs["AgeGroup"] = subs["Age"].astype(str)

    fd = None
    if fd_csv and Path(fd_csv).exists():
        fd = pd.read_csv(fd_csv)
        if not {"Subject", "mean_FD"}.issubset(fd.columns):
            fd = None
        else:
            fd["Subject"] = fd["Subject"].astype(str)

    def attach(df):
        df = df.copy()
        subj_col = "subject" if "subject" in df.columns else ("Subject" if "Subject" in df.columns else None)
        if subj_col is None:
            raise SystemExit("metrics table lacks Subject/subject column")
        df["Subject"] = df[subj_col].astype(str)
        # Heuristic: if Subject contains underscores, extract UID as the first token for merging
        df["UID"] = df["Subject"].str.split("_").str[0]
        df = df.drop(columns=[c for c in ["subject"] if c in df.columns])
        # Try merge on full Subject first; fallback to UID if many NaNs
        out = df.merge(subs, on="Subject", how="left")
        if "AgeGroup" in out.columns and out["AgeGroup"].isna().mean() > 0.5:
            out = df.drop(columns=["Subject"]).rename(columns={"UID": "Subject"}).merge(subs, on="Subject", how="left")
        if fd is not None and "mean_FD" in fd.columns:
            out = out.merge(fd, on="Subject", how="left")
        # If FD not provided separately but present in subjects CSV, it is already included.
        if "mean_FD" not in out.columns:
            out["mean_FD"] = pd.NA
        return out

    st_cov = attach(df_state)
    gl_cov = attach(df_glob)

    out_state = mdir / f"metrics_state_{Ktag}_with_covars.csv"
    out_glob = mdir / f"metrics_global_{Ktag}_with_covars.csv"
    st_cov.to_csv(out_state, index=False)
    gl_cov.to_csv(out_glob, index=False)
    log.info("merge_covars_done", extra={"state_csv": str(out_state), "global_csv": str(out_glob)})
    return out_state, out_glob


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hmm-dir", required=True)
    ap.add_argument("--subjects-csv", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--fd-csv", default=None)
    args = ap.parse_args()
    merge_covars(Path(args.hmm_dir), Path(args.subjects_csv), args.K, Path(args.fd_csv) if args.fd_csv else None)
