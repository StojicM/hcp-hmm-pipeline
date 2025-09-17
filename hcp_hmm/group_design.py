#!/usr/bin/env python3
from __future__ import annotations

"""
Build group-level design files (FSL-compatible) for PALM with repeated measures.

Key idea: one design row per OBSERVATION in the input .dscalar
          (subject × state), plus exchangeability blocks (eb.txt)
          to tell PALM that rows are grouped by subject.

Outputs (in cfg.out_dir):
  - design.mat   (FSL matrix; rows == #maps in .dscalar)
  - design.con   (one or more contrasts)
  - design.grp   (FSL-style group file; not used by PALM but harmless)
  - eb.txt       (PALM exchangeability blocks: subject IDs per row)
  - mapping.tsv  (row ↔ subject ↔ state bookkeeping; source of truth)
  - design_cols.txt (column names, in order)
  - subjects_used.csv, age_levels.txt (same diagnostics you had)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
import csv
import numpy as np

# Your project logger (kept as-is)
from .logger import get_logger
log = get_logger(__name__)


# ------------------------------- Writers -------------------------------

def _write_fsl_mat(path: Path, X: np.ndarray) -> None:
    """Write a FSL-style .mat file (text)."""
    with open(path, "w") as f:
        # NumPoints MUST equal the number of maps in your .dscalar (obs count).
        f.write(f"/NumWaves\t{X.shape[1]}\n/NumPoints\t{X.shape[0]}\n/PPHeights\t")
        f.write("\t".join(["1"] * X.shape[1]) + "\n/Matrix\n")
        # print(f"/NumWaves\t{X.shape[1]}\n/NumPoints\t{X.shape[0]}\n/PPHeights\t")
        # print("\t".join(["1"] * X.shape[1]) + "\n/Matrix\n")
        for row in X:
            f.write("\t".join(f"{v:.6f}" for v in row) + "\n")
            # print("\t".join(f"{v:.6f}" for v in row) + "\n")


def _write_fsl_con(path: Path, C: np.ndarray, names: List[str]) -> None:
    """Write a FSL-style .con file; supports multiple contrasts."""
    assert C.ndim == 2 and C.shape[0] == len(names), "C rows must match names."
    with open(path, "w") as f:
        for i, nm in enumerate(names, 1):
            f.write(f"/ContrastName{i}\t{nm}\n")
        f.write(f"/NumWaves\t{C.shape[1]}\n/NumContrasts\t{C.shape[0]}\n")
        # One PPHeight per contrast is fine; PALM ignores it anyway.
        f.write("/PPHeights\t" + "\t".join(["1.0"] * C.shape[0]) + "\n/Matrix\n")
        for row in C:
            f.write("\t".join(f"{v:.6f}" for v in row) + "\n")


def _write_fsl_grp(path: Path, g: np.ndarray) -> None:
    """Write a FSL-style .grp file. PALM doesn't need it, but harmless."""
    with open(path, "w") as f:
        f.write(f"/NumWaves\t1\n/NumPoints\t{len(g)}\n/Matrix\n")
        for v in g:
            f.write(f"{int(v)}\n")


def _write_eb_txt(path: Path, eb: np.ndarray) -> None:
    """Write PALM exchangeability blocks (one integer per row)."""
    np.savetxt(path, eb.astype(int), fmt="%d")


# ------------------------------ Utilities ------------------------------

def _sniff_delim(path: Path) -> str:
    txt = path.read_text(encoding="utf-8", errors="ignore")[:2048]
    return "\t" if txt.count("\t") >= txt.count(",") else ","


def _read_subjects(csv_path: Path) -> Tuple[List[str], List[str], List[float], Optional[List[float]]]:
    """
    Load subjects and covariates from CSV.

    Returns:
      sids:  list[str]     subject identifiers (strings)
      ages:  list[str]     categorical (e.g., Y/M/O)
      sexes: list[float]   numeric (M→1, F→0; else float if provided)
      fds:   list[float] or None (optional mean FD)
    """
    sids: List[str] = []
    ages: List[str] = []
    sexes: List[float] = []
    fds: List[float] = []
    have_fd = False
    delim = _sniff_delim(csv_path)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}

        def pick(cands: List[str]) -> Optional[str]:
            for k in cands:
                if k.lower() in cols:
                    return cols[k.lower()]
            return None

        id_key  = pick(["uid","sid","subject","participant","participant_id","subject_id","subid","id"]) or next(iter(cols.values()), None)
        age_key = pick(["age","age_group","agegroup","age_bin","agebin"]) or "age"
        sex_key = pick(["sex","gender"]) or "sex"
        fd_key  = pick(["mean_fd","mean_FD","fd"])  # optional
        have_fd = fd_key is not None

        if id_key is None or id_key not in (reader.fieldnames or []):
            raise SystemExit(f"subjects CSV missing an ID column. Found: {list(reader.fieldnames or [])}")
        if age_key not in (reader.fieldnames or []) or sex_key not in (reader.fieldnames or []):
            raise SystemExit(f"subjects CSV must have 'age' and 'sex' columns (case-insensitive). Found: {list(reader.fieldnames or [])}")

        for row in reader:
            sid = str(row.get(id_key, "")).strip()
            age = str(row.get(age_key, "")).strip()
            sex_raw = str(row.get(sex_key, "")).strip()
            if not sid:
                continue
            # Encode sex: M→1, F→0, else try float fallback
            if sex_raw.upper() in ("M","MALE"):
                sex = 1.0
            elif sex_raw.upper() in ("F","FEMALE"):
                sex = 0.0
            else:
                try:    sex = float(sex_raw)
                except: sex = 0.0

            sids.append(sid); ages.append(age); sexes.append(sex)
            if have_fd:
                try:    fds.append(float((row.get(fd_key) or "").strip()))
                except: fds.append(0.0)

    return sids, ages, sexes, (fds if have_fd else None)


def _one_hot_drop_first(labels: np.ndarray, prefix: str):
    """
    One-hot encode string labels, dropping the first level as reference.
    Returns (X, colnames, ref, order).
    """
    labels = np.asarray(labels, dtype=str)
    order, seen = [], set()
    for v in labels:
        if v not in seen:
            seen.add(v); order.append(v)
    K = len(order)
    N = labels.shape[0]
    X = np.zeros((N, max(K - 1, 0)), dtype=float)
    idx = {c: j for j, c in enumerate(order[1:])}  # ref = order[0]
    for i, v in enumerate(labels):
        j = idx.get(v)
        if j is not None:
            X[i, j] = 1.0
    colnames = [f"{prefix}[{c}]" for c in order[1:]]
    ref = order[0] if order else None
    return X, colnames, ref, order


def _read_subject_order(path: Optional[Path]) -> Optional[List[str]]:
    """Optional file with subject IDs (one per line) to fix merge order."""
    if path is None:
        return None
    ids = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return ids or None


# ------------------------------- Config -------------------------------

@dataclass
class GroupDesignConfig:
    subjects_csv: Path
    out_dir: Path
    demean: List[str] | None = None          # e.g., ["sex", "fd"]
    contrast: str | None = None              # name from design_cols (e.g., "sex", "state[6]")
    K: int = 6                               # number of states per subject
    stacking: str = "subject-major"          # or "state-major"
    subject_order_file: Optional[Path] = None  # optional explicit subject order (merge order)
    include_fd: bool = False                 # replicate mean FD per state if available


# ------------------------------ Builder -------------------------------

class GroupDesignBuilder:
    def __init__(self, cfg: GroupDesignConfig):
        self.cfg = cfg

    def _build_mapping(self, ordered_sids: List[str]) -> List[Tuple[str, int]]:
        """
        Build the (subject, state) row order that matches how the .dscalar was stacked.
        - subject-major: sub1 s1..K, sub2 s1..K, ...
        - state-major:   s1 all subs, s2 all subs, ...
        Returns list of tuples (sid, state) of length Nobs = Nsub × K.
        """
        rows: List[Tuple[str, int]] = []
        K = int(self.cfg.K)
        if self.cfg.stacking.lower().startswith("subject"):
            for sid in ordered_sids:
                for s in range(1, K + 1):
                    rows.append((sid, s))
        elif self.cfg.stacking.lower().startswith("state"):
            for s in range(1, K + 1):
                for sid in ordered_sids:
                    rows.append((sid, s))
        else:
            raise SystemExit("stacking must be 'subject-major' or 'state-major'")
        return rows

    def run(self) -> None:
        # ---- 1) Read subject-level covariates --------------------------------
        sids, ages, sexes, fds = _read_subjects(self.cfg.subjects_csv)
        if not sids:
            raise SystemExit("No rows found in subjects CSV.")

        # Preserve a dict lookup by subject ID for fast per-row access later.
        # Keep ages as strings (e.g., Y/M/O), sex numeric (1/0 or float), fd optional.
        age_by_sid = {sid: age for sid, age in zip(sids, ages)}
        sex_by_sid = {sid: float(sex) for sid, sex in zip(sids, sexes)}
        fd_by_sid  = ({sid: float(fd) for sid, fd in zip(sids, fds)} if fds is not None else None)

        # ---- 2) Determine *merge order* of subjects (MUST match the .dscalar stack) ----
        order = _read_subject_order(self.cfg.subject_order_file)
        if order is None:
            # Fallback: use order as they appear in subjects_csv
            ordered_sids = sids
        else:
            # Trust the provided order, but keep only those present in CSV
            ordered_sids = [sid for sid in order if sid in age_by_sid]
            missing = [sid for sid in order if sid not in age_by_sid]
            if missing:
                log.warning("subject_order_file IDs missing in CSV; ignoring: %s", missing)
        Nsub = len(ordered_sids)
        if Nsub == 0:
            raise SystemExit("No subjects left after applying subject_order_file.")

        # ---- 3) Build row mapping (subject, state) in the exact stacking order -------
        mapping = self._build_mapping(ordered_sids)   # length = Nobs = Nsub × K
        Nobs = len(mapping)
        K = int(self.cfg.K)

        # Save mapping for transparency/debugging (row indices are 1-based for human-friendliness)
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cfg.out_dir / "mapping.tsv", "w", encoding="utf-8") as f:
            f.write("row\tsubid\tstate\n")
            for i, (sid, st) in enumerate(mapping, 1):
                f.write(f"{i}\t{sid}\t{st}\n")

        # ---- 4) Expand covariates to one row per (subject, state) --------------------
        # Categorical AGE: one-hot (drop first) after expansion (so levels reflect actual rows)
        age_rows = np.array([age_by_sid[sid] for sid, _ in mapping], dtype=str)
        A_age, age_names, age_ref, age_order = _one_hot_drop_first(age_rows, prefix="age")

        # SEX (numeric); replicate per state, then optional demeaning
        sex_rows = np.array([sex_by_sid[sid] for sid, _ in mapping], dtype=float)
        if self.cfg.demean and ("sex" in self.cfg.demean):
            sex_rows = sex_rows - float(np.mean(sex_rows))

        # Optional mean FD (demeaned if requested)
        if self.cfg.include_fd and (fd_by_sid is not None):
            fd_rows = np.array([fd_by_sid[sid] for sid, _ in mapping], dtype=float)
            if self.cfg.demean and ("fd" in self.cfg.demean):
                fd_rows = fd_rows - float(np.mean(fd_rows))
        else:
            fd_rows = None

        # STATE dummies (drop first) so you can test within-subject state effects)
        state_rows = np.array([str(st) for _, st in mapping], dtype=str)
        A_state, state_names, state_ref, state_order = _one_hot_drop_first(state_rows, prefix="state")

        # ---- 5) Assemble design matrix columns --------------------------------------
        X_cols = []
        colnames: List[str] = []

        # Intercept
        X_cols.append(np.ones((Nobs, 1), dtype=float)); colnames.append("intercept")

        # Age one-hot (if >1 level present)
        if A_age.size:
            X_cols.append(A_age); colnames.extend(age_names)

        # Sex
        X_cols.append(sex_rows[:, None]); colnames.append("sex")

        # Optional FD
        if fd_rows is not None:
            X_cols.append(fd_rows[:, None]); colnames.append("fd")

        # State dummies (K-1)
        if A_state.size:
            X_cols.append(A_state); colnames.extend(state_names)

        X = np.column_stack(X_cols)

        # ---- 6) Build contrasts ------------------------------------------------------
        # If user requested a contrast by name (e.g., "sex" or "state[6]"), use it.
        # Else default to the last state dummy (e.g., state[K]) if present,
        # otherwise fall back to "sex", otherwise first age dummy if present.
        con_names: List[str]
        if self.cfg.contrast:
            if self.cfg.contrast not in colnames:
                raise SystemExit(f"Requested contrast '{self.cfg.contrast}' not in columns: {colnames}")
            j = colnames.index(self.cfg.contrast)
            C = np.zeros((1, X.shape[1]), dtype=float); C[0, j] = 1.0
            con_names = [self.cfg.contrast]
        else:
            if state_names:
                j = colnames.index(state_names[-1])  # e.g., "state[6]" with ref "state[1]"
                C = np.zeros((1, X.shape[1]), dtype=float); C[0, j] = 1.0
                con_names = [state_names[-1]]
            elif "sex" in colnames:
                j = colnames.index("sex")
                C = np.zeros((1, X.shape[1]), dtype=float); C[0, j] = 1.0
                con_names = ["sex"]
            elif age_names:
                j = colnames.index(age_names[0])
                C = np.zeros((1, X.shape[1]), dtype=float); C[0, j] = 1.0
                con_names = [age_names[0]]
            else:
                # Shouldn’t happen (at least intercept+sex present), but guard anyway
                C = np.zeros((1, X.shape[1]), dtype=float); C[0, 0] = 1.0
                con_names = ["intercept"]

        # ---- 7) Exchangeability blocks (PALM -eb) -----------------------------------
        # One integer per row: rows that share the same subject ID get same block number.
        # The actual block value is arbitrary; we use 1..Nsub in the chosen subject order.
        block_id = {sid: i+1 for i, sid in enumerate(ordered_sids)}   # 1-based
        eb = np.array([block_id[sid] for sid, _ in mapping], dtype=int)

        # ---- 8) Write all outputs ----------------------------------------------------
        out = self.cfg.out_dir; out.mkdir(parents=True, exist_ok=True)

        _write_fsl_mat(out / "design.mat", X)
        _write_fsl_con(out / "design.con", C, con_names)
        _write_fsl_grp(out / "design.grp", np.ones(Nobs, dtype=int))  # harmless
        _write_eb_txt(out / "eb.txt", eb)

        with open(out / "design_cols.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(colnames) + "\n")

        # Diagnostics (subject-level + state replicated view)
        with open(out / "subjects_used.csv", "w", encoding="utf-8") as f:
            if fd_by_sid is not None and self.cfg.include_fd:
                f.write("row\tsid\tstate\tage\tsex\tfd\n")
                for i, (sid, st) in enumerate(mapping, 1):
                    f.write(f"{i}\t{sid}\t{st}\t{age_by_sid[sid]}\t{sex_by_sid[sid]:.6f}\t{fd_by_sid[sid]:.6f}\n")
            else:
                f.write("row\tsid\tstate\tage\tsex\n")
                for i, (sid, st) in enumerate(mapping, 1):
                    f.write(f"{i}\t{sid}\t{st}\t{age_by_sid[sid]}\t{sex_by_sid[sid]:.6f}\n")

        with open(out / "age_levels.txt", "w", encoding="utf-8") as f:
            f.write("reference\t" + ("NA" if age_order is None else (age_order[0] if len(age_order) else "NA")) + "\n")
            for lvl in age_order:
                f.write(lvl + "\n")

        log.info("group_design_written", extra={"dir": str(out)})


# ------------------------------- CLI (optional) -------------------------------

def _parse_args() -> GroupDesignConfig:
    p = argparse.ArgumentParser(description="Build PALM/FSL group design with repeated measures.")
    p.add_argument("--subjects-csv", type=Path, required=True, help="CSV with subject_id, age, sex (and optional mean_fd).")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for design files.")
    p.add_argument("--states", type=int, default=6, help="Number of states per subject (K).")
    p.add_argument("--stacking", choices=["subject-major","state-major"], default="subject-major",
                   help="Order used to stack maps into the group .dscalar.")
    p.add_argument("--subject-order", type=Path, default=None,
                   help="Optional text file with subject IDs in EXACT merge order (one per line).")
    p.add_argument("--demean", nargs="*", default=[], help="Columns to demean (choose from: sex, fd).")
    p.add_argument("--contrast", type=str, default=None,
                   help="Column name to use as a 1-df contrast (e.g., 'sex', 'state[6]', 'age[M]').")
    p.add_argument("--include-fd", action="store_true", help="Include mean FD as a covariate (replicated per state).")
    args = p.parse_args()

    return GroupDesignConfig(
        subjects_csv=args.subjects_csv,
        out_dir=args.out_dir,
        demean=args.demean or None,
        contrast=args.contrast,
        K=args.states,
        stacking=args.stacking,
        subject_order_file=args.subject_order,
        include_fd=bool(args.include_fd),
    )


if __name__ == "__main__":
    cfg = _parse_args()
    GroupDesignBuilder(cfg).run()