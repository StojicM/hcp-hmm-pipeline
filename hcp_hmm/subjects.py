#!/usr/bin/env python3
"""Utilities for normalizing and loading subject-level covariates."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import pandas as pd

from .logger import get_logger

log = get_logger(__name__)

_SUBJECT_KEYS: Sequence[str] = (
    "subject",
    "subject_id",
    "sid",
    "uid",
    "id",
    "participant",
    "participant_id",
    "sub",
    "subid",
)


def coerce_sex_str(value) -> str:
    """Normalize sex/gender values to {"M", "F"}."""
    s = str(value).strip()
    if not s:
        return ""
    up = s.upper()
    if up in {"M", "MALE", "1", "1.0"}:
        return "M"
    if up in {"F", "FEMALE", "0", "0.0"}:
        return "F"
    return up


def _infer_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    cols = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def load_subject_covariates(csv_path: Path | None, *, strict: bool = False) -> pd.DataFrame:
    """Load and normalize subject covariates from CSV.

    Returns a DataFrame with at least the columns:
      - Subject (str)
      - UID (str; first token before '_')
      - Sex (str in {"M", "F", ""})
      - AgeGroup (str or NA)
      - mean_FD (float or NA)

    Any additional columns from the source CSV are preserved.
    """

    if not csv_path:
        return pd.DataFrame(columns=["Subject", "UID", "Sex", "AgeGroup", "mean_FD"])
    csv_path = Path(csv_path)
    if not csv_path.exists():
        msg = f"subjects CSV not found: {csv_path}"
        if strict:
            raise FileNotFoundError(msg)
        log.warning("subjects_csv_missing", extra={"path": str(csv_path)})
        return pd.DataFrame(columns=["Subject", "UID", "Sex", "AgeGroup", "mean_FD"])

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - logged, but bail gracefully
        if strict:
            raise
        log.warning("subjects_csv_read_failed", extra={"path": str(csv_path), "err": str(exc)})
        return pd.DataFrame(columns=["Subject", "UID", "Sex", "AgeGroup", "mean_FD"])

    if df.empty:
        log.warning("subjects_csv_empty", extra={"path": str(csv_path)})
        return pd.DataFrame(columns=["Subject", "UID", "Sex", "AgeGroup", "mean_FD"])

    subj_col = _infer_column(df.columns, _SUBJECT_KEYS)
    if subj_col is None:
        msg = f"subjects CSV missing a subject identifier column. Columns: {list(df.columns)}"
        if strict:
            raise SystemExit(msg)
        log.warning("subjects_csv_missing_subject", extra={"path": str(csv_path), "columns": list(df.columns)})
        return pd.DataFrame(columns=["Subject", "UID", "Sex", "AgeGroup", "mean_FD"])

    age_col = _infer_column(df.columns, ("agegroup", "age_group", "age", "agecategory", "age_cat", "agebin", "age_bin"))
    sex_col = _infer_column(df.columns, ("sex", "gender"))
    fd_col = _infer_column(df.columns, ("mean_FD", "mean_fd", "fd"))

    rename: Dict[str, str] = {subj_col: "Subject"}
    if age_col:
        rename[age_col] = "AgeGroup"
    if sex_col:
        rename[sex_col] = "Sex"
    if fd_col:
        rename[fd_col] = "mean_FD"
    df = df.rename(columns=rename)

    df["Subject"] = df["Subject"].astype(str)
    df["UID"] = df["Subject"].str.split("_").str[0]

    if "Sex" not in df.columns:
        df["Sex"] = ""
    df["Sex"] = df["Sex"].map(coerce_sex_str)

    if "AgeGroup" not in df.columns:
        df["AgeGroup"] = pd.NA
    else:
        df["AgeGroup"] = df["AgeGroup"].astype(str)

    if "mean_FD" not in df.columns:
        df["mean_FD"] = pd.NA
    else:
        df["mean_FD"] = pd.to_numeric(df["mean_FD"], errors="coerce")

    # Ensure canonical ordering first, extras afterwards
    canonical = ["Subject", "UID", "Sex", "AgeGroup", "mean_FD"]
    extras = [c for c in df.columns if c not in canonical]
    out = df[canonical + extras]
    return out

