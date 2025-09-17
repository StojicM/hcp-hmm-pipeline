#!/usr/bin/env python3
from __future__ import annotations

"""
Build group design files (FSL-compatible) from a subjects CSV.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import csv
import numpy as np

from .logger import get_logger

log = get_logger(__name__)


def _write_fsl_mat(path: Path, X: np.ndarray) -> None:
    with open(path, "w") as f:
        #X.shape[0] => number of maps in the dscalar
        f.write(f"/NumWaves\t{X.shape[1]}\n/NumPoints\t{X.shape[0]}\n/PPHeights\t")
        f.write("\t".join(["1"] * X.shape[1]) + "\n/Matrix\n")
        for row in X:
            f.write("\t".join(f"{v:.6f}" for v in row) + "\n")


def _write_fsl_con(path: Path, C: np.ndarray, names: List[str]) -> None:
    with open(path, "w") as f:
        f.write("/ContrastName1\t" + names[0] + "\n")
        f.write(f"/NumWaves\t{C.shape[1]}\n/NumContrasts\t{C.shape[0]}\n/PPHeights\t1.0\n/Matrix\n")
        f.write("\t".join(f"{v:.6f}" for v in C[0]) + "\n")


def _write_fsl_grp(path: Path, g: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write(f"/NumWaves\t1\n/NumPoints\t{len(g)}\n/Matrix\n")
        for v in g:
            f.write(f"{int(v)}\n")


def _sniff_delim(path: Path) -> str:
    txt = path.read_text(encoding="utf-8", errors="ignore")[:2048]
    return "\t" if txt.count("\t") >= txt.count(",") else ","


def _read_subjects(csv_path: Path) -> Tuple[List[str], List[str], List[float], Optional[List[float]]]:
    sids: List[str] = []
    ages: List[str] = []
    sexes: List[float] = []
    fds: List[float] = []
    have_fd = False
    delim = _sniff_delim(csv_path)
    with open(csv_path, newline="", encoding="utf-8") as f:
        print(csv_path)
        reader = csv.DictReader(f, delimiter=delim)
        # Case-insensitive column access with flexible candidates
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        def pick(candidates: List[str]) -> Optional[str]:
            for k in candidates:
                if k.lower() in cols:
                    return cols[k.lower()]
            return None
        id_key = pick(["uid","sid","subject","participant","participant_id","subject_id","subid","id"]) or next(iter(cols.values()), None)
        age_key = pick(["age","age_group","agegroup","age_bin","agebin"]) or "age"
        sex_key = pick(["sex","gender"]) or "sex"
        fd_key = pick(["mean_fd","mean_FD","fd"])  # optional
        have_fd = fd_key is not None
        
        if id_key is None or id_key not in (reader.fieldnames or []):
            raise SystemExit(f"subjects CSV missing an ID column. Found columns: {list(reader.fieldnames or [])}")
        if age_key not in (reader.fieldnames or []) or sex_key not in (reader.fieldnames or []):
            raise SystemExit(f"subjects CSV must have 'age' and 'sex' columns (case-insensitive). Found: {list(reader.fieldnames or [])}")

        for row in reader:
            sid = str(row.get(id_key, "")).strip()
            age = str(row.get(age_key, "")).strip()
            sex_raw = str(row.get(sex_key, "")).strip()
            if sex_raw.upper() in ("M", "MALE"):
                sex = 1.0
            elif sex_raw.upper() in ("F", "FEMALE"):
                sex = 0.0
            else:
                try:
                    sex = float(sex_raw)
                except Exception:
                    sex = 0.0
            if not sid:
                continue
            sids.append(sid); ages.append(age); sexes.append(sex)
            if have_fd:
                try:
                    fds.append(float((row.get(fd_key) or "").strip()))
                except Exception:
                    fds.append(0.0)
    return sids, ages, sexes, (fds if have_fd else None)


def _one_hot_drop_first(labels: np.ndarray, prefix: str):
    labels = np.asarray(labels, dtype=str)
    order, seen = [], set()
    for v in labels:
        if v not in seen:
            seen.add(v); order.append(v)
    K = len(order)
    N = labels.shape[0]
    X = np.zeros((N, max(K - 1, 0)), dtype=float)
    idx = {c: j for j, c in enumerate(order[1:])}
    for i, v in enumerate(labels):
        j = idx.get(v)
        if j is not None:
            X[i, j] = 1.0
    colnames = [f"{prefix}[{c}]" for c in order[1:]]
    ref = order[0] if order else None
    return X, colnames, ref, order


@dataclass
class GroupDesignConfig:
    subjects_csv: Path
    out_dir: Path
    demean: List[str] | None = None
    contrast: str | None = None


class GroupDesignBuilder:
    def __init__(self, cfg: GroupDesignConfig):
        self.cfg = cfg

    def run(self) -> None:
        sids, ages, sexes, fds = _read_subjects(self.cfg.subjects_csv)
        if not sids:
            raise SystemExit("No rows found in subjects CSV.")
        ages = np.array(ages, dtype=str)
        sexes = np.array(sexes, dtype=float)

        X_cols = []
        colnames = []

        X_cols.append(np.ones(len(ages), dtype=float)[:, None])
        colnames.append("intercept")

        A_age, age_names, age_ref, age_order = _one_hot_drop_first(ages, prefix="age")
        if A_age.size:
            X_cols.append(A_age)
            colnames.extend(age_names)

        x_sex = sexes.copy()
        if self.cfg.demean and ("sex" in self.cfg.demean):
            x_sex = x_sex - np.mean(x_sex)
        X_cols.append(x_sex[:, None])
        colnames.append("sex")

        X = np.column_stack(X_cols)

        if self.cfg.contrast:
            if self.cfg.contrast not in colnames:
                raise SystemExit(f"Requested contrast '{self.cfg.contrast}' not in columns: {colnames}")
            j = colnames.index(self.cfg.contrast)
            C = np.zeros((1, X.shape[1]), dtype=float); C[0, j] = 1.0
            con_names = [self.cfg.contrast]
        else:
            if age_names:
                j = colnames.index(age_names[0])
                C = np.zeros((1, X.shape[1]), dtype=float); C[0, j] = 1.0
                con_names = [age_names[0]]
            else:
                j = colnames.index("sex")
                C = np.zeros((1, X.shape[1]), dtype=float); C[0, j] = 1.0
                con_names = ["sex"]

        out = self.cfg.out_dir; out.mkdir(parents=True, exist_ok=True)

        _write_fsl_mat(out / "design.mat", X)
        _write_fsl_con(out / "design.con", C, con_names)
        _write_fsl_grp(out / "design.grp", np.ones(len(ages), dtype=int))

        with open(out / "design_cols.txt", "w") as f:
            f.write("\n".join(colnames) + "\n")
        with open(out / "subjects_used.csv", "w", encoding="utf-8") as f:
            if fds is not None:
                f.write("sid\tage\tsex\tmean_FD\n")
                for sid, a, s, fd in zip(sids, ages, sexes, fds):
                    f.write(f"{sid}\t{a}\t{s:.6f}\t{fd:.6f}\n")
            else:
                f.write("sid\tage\tsex\n")
                for sid, a, s in zip(sids, ages, sexes):
                    f.write(f"{sid}\t{a}\t{s:.6f}\n")
        with open(out / "age_levels.txt", "w", encoding="utf-8") as f:
            f.write("reference\t" + (age_ref if age_ref is not None else "NA") + "\n")
            for lvl in age_order:
                f.write(lvl + "\n")

        log.info("group_design_written", extra={"dir": str(out)})
