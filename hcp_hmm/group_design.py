#!/usr/bin/env python3
from __future__ import annotations

"""
Build group-level design files (FSL-compatible) for PALM with repeated measures.

Key idea: one design row per OBSERVATION in the input .pscalar
          (subject x state), plus exchangeability blocks (eb.txt)
          to tell PALM that rows are grouped by subject.

Outputs (in config.out_dir):
  - design.mat   (FSL matrix; rows == #maps in .pscalar)
  - design.con   (one or more contrasts)
  - design.grp   (FSL-style group file; not used by PALM but harmless)
  - eb.txt       (PALM exchangeability blocks: subject IDs per row)
  - mapping.tsv  (row ↔ subject ↔ state bookkeeping; source of truth)
  - design_cols.txt (column names, in order)
  - subjects_used.csv, age_levels.txt (same diagnostics you had)

Notes
- State effects are encoded with K-1 dummy columns (reference is state[1]).
- Age is treated as categorical here: one-hot with drop-first; adjust upstream if continuous.
- Sex is numeric (M->1, F->0) and can be demeaned; FD is optional and can be demeaned.
- Repeated measures are handled by PALM via '-eb' using per-row subject IDs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
import csv
import numpy #as np

# Project's logger
from .logger import get_logger
log = get_logger(__name__)


# ------------------------------- Writers -------------------------------

def _write_fsl_mat(path: Path, X: numpy.ndarray) -> None:
    """Write an FSL-style .mat file (text)."""
    with open(path, "w") as f:
        # NumPoints MUST equal the number of maps in your .pscalar (obs count).
        f.write(f"/NumWaves\t{X.shape[1]}\n/NumPoints\t{X.shape[0]}\n/PPHeights\t")
        f.write("\t".join(["1"] * X.shape[1]) + "\n/Matrix\n")
        for row in X:
            f.write("\t".join(f"{v:.6f}" for v in row) + "\n")


def _write_fsl_con(path: Path, C: numpy.ndarray, names: List[str]) -> None:
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


def _write_fsl_grp(path: Path, g: numpy.ndarray) -> None:
    """Write a FSL-style .grp file. PALM doesn't need it, but harmless."""
    with open(path, "w") as f:
        f.write(f"/NumWaves\t1\n/NumPoints\t{len(g)}\n/Matrix\n")
        for v in g:
            f.write(f"{int(v)}\n")


def _write_eb_txt(path: Path, eb: numpy.ndarray) -> None:
    """Write PALM exchangeability blocks (one integer per row)."""
    numpy.savetxt(path, eb.astype(int), fmt="%d")


# ------------------------------ Utilities ------------------------------

def _sniff_delim(path: Path) -> str:
    txt = path.read_text(encoding="utf-8", errors="ignore")[:2048]
    return "\t" if txt.count("\t") >= txt.count(",") else ","


def _read_subjects(csv_path: Path) -> Tuple[List[str], List[str], List[float], Optional[List[float]]]:
    """Read subject-level covariates: subject_id, age, sex, optional mean_fd.
    - Sex is coerced to numeric: M/MALE→1.0, F/FEMALE→0.0, else float or 0.0.
    - Age stays as string labels (e.g., Y/M/O), later one-hot encoded.
    - mean_fd is optional and interpreted as a float when present.
    """
    subject_ids: List[str] = []
    ages: List[str] = []
    sexes: List[float] = []

    delim = _sniff_delim(csv_path)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        fields = reader.fieldnames or []
        id_key, age_key, sex_key, fd_key = "subject_id", "age", "sex", "mean_fd"

        # required columns
        missing = [k for k in (id_key, age_key, sex_key) if k not in fields]
        if missing:
            raise SystemExit(f"subjects CSV missing required columns: {missing}. Found: {fields}")

        # optional FD covariate
        have_fd = fd_key in fields
        fds: Optional[List[float]] = [] if have_fd else None

        for row in reader:
            sub_id = (row.get(id_key) or "").strip()
            if not sub_id: #skip if missing subject id
                continue
            age = (row.get(age_key) or "").strip()
            sex_raw = (row.get(sex_key) or "").strip().upper()

            # M→1.0, F→0.0; else try float, fallback 0.0
            if sex_raw in ("M", "MALE"):
                sex_val = 1.0
            elif sex_raw in ("F", "FEMALE"):
                sex_val = 0.0
            else:
                print(f"Missing Sex Column...skipping subject ${sub_id}")
                continue
            #     try:
            #         sex_val = float(sex_raw)
            #     except ValueError:
            #         sex_val = 0.0

            subject_ids.append(sub_id)
            ages.append(age)
            sexes.append(sex_val)

            if have_fd:
                val = (row.get(fd_key) or "").strip()
                try:
                    fds.append(float(val))
                except ValueError:
                    fds.append(0.0)

    return subject_ids, ages, sexes, fds


def _one_hot_drop_first(labels: numpy.ndarray, prefix: str):
    """One-hot encode labels, dropping first level as reference.
    Returns tuple (X, colnames, ref, order) where X has shape N×(K-1),
    `colnames` are like `prefix[level]`, `ref` is the dropped level,
    and `order` lists all levels in the discovered order.
    """
    labels = numpy.asarray(labels, dtype=str)
    order, seen = [], set()
    for v in labels:
        if v not in seen:
            seen.add(v); order.append(v)
    K = len(order)
    N = labels.shape[0]
    X = numpy.zeros((N, max(K - 1, 0)), dtype=float)
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
    return ids #or None #?


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
    include_fd: Optional[bool] = None        # True: force include; False: skip; None: auto if present


# ------------------------------ Builder -------------------------------

class GroupDesignBuilder:
    def __init__(self, config: GroupDesignConfig):
        self.config = config

    def _build_mapping(self, ordered_sids: List[str]) -> List[Tuple[str, int]]:
        """
        Build the (subject, state) row order that matches how the .dscalar was stacked.
        - subject-major: sub1 s1..K, sub2 s1..K, ...
        - state-major: s1 all subs, s2 all subs, ...
        Returns list of tuples (sid, state) of length Nobs = Nsub × K.
        """
        rows: List[Tuple[str, int]] = []
        K = int(self.config.K)
        if self.config.stacking.lower().startswith("subject"):
            for subj_id in ordered_sids:
                for s in range(1, K + 1):
                    rows.append((subj_id, s))
        elif self.config.stacking.lower().startswith("state"):
            for s in range(1, K + 1):
                for subj_id in ordered_sids:
                    rows.append((subj_id, s))
        else:
            raise SystemExit("stacking must be 'subject-major' or 'state-major'")
        return rows

    def _build_contrasts(self, colnames: List[str], state_names: List[str], ncols: int) -> Tuple[numpy.ndarray, List[str]]:
        """Choose a simple 1-df contrast.

        If `config.contrast` is set, build a unit vector for that column.
        Otherwise, prefer the last state dummy, then `sex`, `fd`, any `age[...]`,
        falling back to `intercept` or the first column.
        """
        if ncols == 0:
            raise SystemExit("Design matrix has zero columns; cannot build contrasts.")

        if self.config.contrast:
            if self.config.contrast not in colnames:
                raise SystemExit(f"Requested contrast '{self.config.contrast}' not in columns: {colnames}")
            C = numpy.zeros((1, ncols), dtype=float)
            C[0, colnames.index(self.config.contrast)] = 1.0
            return C, [self.config.contrast]

        priority: List[str] = []
        if state_names:
            priority.extend(state_names[::-1])
        for cand in ("sex", "fd"):
            if cand in colnames:
                priority.append(cand)
        priority.extend(name for name in colnames if name.startswith("age"))
        if "intercept" in colnames:
            priority.append("intercept")

        for name in priority:
            if name in colnames:
                C = numpy.zeros((1, ncols), dtype=float)
                C[0, colnames.index(name)] = 1.0
                return C, [name]

        C = numpy.zeros((1, ncols), dtype=float)
        C[0, 0] = 1.0
        return C, [colnames[0]]

    def run(self) -> None:
        """Build design/contrast/group/eb files and diagnostics in `out_dir`."""
        # ---- 1) Read subject-level covariates --------------------------------
        subject_ids, ages, sexes, fds = _read_subjects(self.config.subjects_csv)
        if not subject_ids:
            raise SystemExit("No rows found in subjects CSV.")

        # Preserve a dict lookup by subject ID for fast per-row access later.
        # Keep ages as strings (e.g., Y/M/O), sex numeric (1/0 or float), fd optional.
        age_by_sid = {sid: age for sid, age in zip(subject_ids, ages)}
        sex_by_sid = {sid: float(sex) for sid, sex in zip(subject_ids, sexes)}
        fd_by_sid  = ({sid: float(fd) for sid, fd in zip(subject_ids, fds)} if fds is not None else None)

        # ---- 2) Determine *merge order* of subjects (MUST match the .dscalar stack) ----
        order = _read_subject_order(self.config.subject_order_file)
        if order is None:
            # Fallback: use order as they appear in subjects_csv
            ordered_sids = subject_ids
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
        K = int(self.config.K)

        # Save mapping for transparency/debugging (row indices are 1-based for human-friendliness)
        self.config.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config.out_dir / "mapping.tsv", "w", encoding="utf-8") as f:
            f.write("row\tsubid\tstate\n")
            for i, (sid, st) in enumerate(mapping, 1):
                f.write(f"{i}\t{sid}\t{st}\n")

        # ---- 4) Expand covariates to one row per (subject, state) --------------------
        # Categorical AGE: one-hot (drop first) after expansion (so levels reflect actual rows)
        age_rows = numpy.array([age_by_sid[sid] for sid, _ in mapping], dtype=str)
        A_age, age_names, age_ref, age_order = _one_hot_drop_first(age_rows, prefix="age")

        # SEX (numeric); replicate per state, then optional demeaning
        sex_rows = numpy.array([sex_by_sid[sid] for sid, _ in mapping], dtype=float)
        if self.config.demean and ("sex" in self.config.demean):
            sex_rows = sex_rows - float(numpy.mean(sex_rows))

        # Optional mean FD (demeaned if requested)
        want_fd = self.config.include_fd
        has_fd = fd_by_sid is not None
        use_fd = has_fd and (want_fd is None or want_fd is True)
        if want_fd and not has_fd:
            log.warning("group_design_fd_missing", extra={"reason": "column absent"})

        if use_fd:
            fd_rows = numpy.array([fd_by_sid[sid] for sid, _ in mapping], dtype=float)
            if self.config.demean and ("fd" in self.config.demean):
                fd_rows = fd_rows - float(numpy.mean(fd_rows))
                log.info("group_design_fd", extra={"included": True, "demeaned": True})
            else:
                log.info("group_design_fd", extra={"included": True, "demeaned": False})
        else:
            fd_rows = None
            reason = "missing" if not has_fd else "disabled"
            log.info("group_design_fd", extra={"included": False, "reason": reason})

        # STATE dummies (drop first) so you can test within-subject state effects)
        state_rows = numpy.array([str(st) for _, st in mapping], dtype=str)
        A_state, state_names, state_ref, state_order = _one_hot_drop_first(state_rows, prefix="state")

        # ---- 5) Assemble design matrix columns --------------------------------------
        X_cols = []
        colnames: List[str] = []

        # Intercept
        X_cols.append(numpy.ones((Nobs, 1), dtype=float)); colnames.append("intercept")

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

        X = numpy.column_stack(X_cols)

        # ---- 6) Build contrasts ------------------------------------------------------
        C, con_names = self._build_contrasts(colnames, state_names, X.shape[1])

        # ---- 7) Exchangeability blocks (PALM -eb) -----------------------------------
        # One integer per row: rows that share the same subject ID get same block number.
        # The actual block value is arbitrary; we use 1..Nsub in the chosen subject order.
        block_id = {sid: i+1 for i, sid in enumerate(ordered_sids)}   # 1-based
        eb = numpy.array([block_id[sid] for sid, _ in mapping], dtype=int)

        # ---- 8) Write all outputs ----------------------------------------------------
        out = self.config.out_dir; out.mkdir(parents=True, exist_ok=True)

        _write_fsl_mat(out / "design.mat", X)
        _write_fsl_con(out / "design.con", C, con_names)
        _write_fsl_grp(out / "design.grp", numpy.ones(Nobs, dtype=int))  # harmless
        _write_eb_txt(out / "eb.txt", eb)

        with open(out / "design_cols.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(colnames) + "\n")

        # Diagnostics (subject-level + state replicated view)
        with open(out / "subjects_used.csv", "w", encoding="utf-8") as f:
            if fd_rows is not None:
                f.write("row\tsid\tstate\tage\tsex\tfd_raw\tfd_design\n")
                for idx, (sid, st) in enumerate(mapping, 1):
                    fd_raw = fd_by_sid[sid]
                    fd_design = float(fd_rows[idx - 1])
                    f.write(f"{idx}\t{sid}\t{st}\t{age_by_sid[sid]}\t{sex_by_sid[sid]:.6f}\t{fd_raw:.6f}\t{fd_design:.6f}\n")
            else:
                f.write("row\tsid\tstate\tage\tsex\n")
                for i, (sid, st) in enumerate(mapping, 1):
                    f.write(f"{i}\t{sid}\t{st}\t{age_by_sid[sid]}\t{sex_by_sid[sid]:.6f}\n")

        with open(out / "age_levels.txt", "w", encoding="utf-8") as f:
            f.write("reference\t" + ("NA" if age_order is None else (age_order[0] if len(age_order) else "NA")) + "\n")
            for lvl in age_order:
                f.write(lvl + "\n")

        # Subject-level design files (one row per subject, drop state dummies)
        subject_row_idx = [idx for idx, (_sid, st) in enumerate(mapping) if int(st) == 1]
        if len(subject_row_idx) != Nsub:
            log.warning(
                "group_design_subject_rows_mismatch",
                extra={"expected": int(Nsub), "found": int(len(subject_row_idx))},
            )
        keep_cols_subject = [j for j, name in enumerate(colnames) if not name.startswith("state[")]
        if not keep_cols_subject:
            keep_cols_subject = [0]

        X_subject = X[subject_row_idx][:, keep_cols_subject]
        colnames_subject = [colnames[j] for j in keep_cols_subject]
        if not colnames_subject:
            colnames_subject = ["intercept"]
            X_subject = numpy.ones((len(subject_row_idx), 1), dtype=float)

        C_subject, con_names_subject = self._build_contrasts(colnames_subject, [], X_subject.shape[1])

        _write_fsl_mat(out / "design_subjects.mat", X_subject)
        _write_fsl_con(out / "design_subjects.con", C_subject, con_names_subject)
        _write_fsl_grp(out / "design_subjects.grp", numpy.ones(X_subject.shape[0], dtype=int))
        with open(out / "design_subjects_cols.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(colnames_subject) + "\n")

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
    config = _parse_args()
    GroupDesignBuilder(config).run()
