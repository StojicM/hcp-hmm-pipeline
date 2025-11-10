#!/usr/bin/env python3
from __future__ import annotations

"""
Between-subject permutation tests for global metrics.

Approach
- Regress out motion (mean FD) as a nuisance covariate.
- Test Sex: absolute mean difference (M vs F) on residuals; report Cohen's d.
- Test AgeGroup: variance explained (η²) across age levels on residuals.

Permutation scheme shuffles group labels to form the null. Replicates
src/05b_stats_global_between.py.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .logger import get_logger

log = get_logger(__name__)


def _cohen_d(x, y):
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y); sx, sy = x.std(ddof=1), y.std(ddof=1)
    sp = np.sqrt(((nx - 1) * sx ** 2 + (ny - 1) * sy ** 2) / max(nx + ny - 2, 1))
    return 0.0 if sp == 0 else (x.mean() - y.mean()) / sp


def _eta_sq(y, g):
    y = np.asarray(y, float); g = np.asarray(g)
    grand = y.mean(); sst = ((y - grand) ** 2).sum()
    if sst == 0: return 0.0
    ssb = 0.0
    for lvl in pd.unique(g):
        yi = y[g == lvl]
        if yi.size: ssb += yi.size * (yi.mean() - grand) ** 2
    return float(ssb / sst)


@dataclass
class StatsBetweenConfig:
    in_csv: Path
    out_csv: Path
    n_perm: int = 5000


class StatsBetween:
    def __init__(self, cfg: StatsBetweenConfig):
        self.cfg = cfg

    def run(self) -> Path:
        df = pd.read_csv(self.cfg.in_csv)
        need = {"Subject", "AgeGroup", "Sex", "mean_FD"}
        miss = need - set(df.columns)
        if miss: raise SystemExit(f"Missing columns: {miss}")

        # Exclude identifiers and non-metrics to avoid treating them as outcomes
        ignore = {"mean_FD", "Subject", "AgeGroup", "Sex", "UID", "K"}
        metrics = [
            c for c in df.columns
            if c not in ignore
            and c.lower() not in {"subject", "uid", "age", "agegroup", "sex", "mean_fd", "k"}
            and np.issubdtype(df[c].dtype, np.number)
        ]

        rng = np.random.default_rng(42)
        rows = []
        for m in metrics:
            d = df[~df[m].isna()].copy()
            X = np.c_[np.ones(len(d)), d["mean_FD"].fillna(d["mean_FD"].mean())]
            beta = np.linalg.lstsq(X, d[m].to_numpy(float), rcond=None)[0]
            resid = d[m].to_numpy(float) - X @ beta

            lab = d["Sex"].astype(str).str.upper().to_numpy()
            xm = resid[lab == "M"]; xf = resid[lab == "F"]
            if len(xm) > 0 and len(xf) > 0:
                obs = abs(xm.mean() - xf.mean())
                null = []
                lab_perm = lab.copy()
                for _ in range(self.cfg.n_perm):
                    rng.shuffle(lab_perm)
                    null.append(abs(resid[lab_perm == "M"].mean() - resid[lab_perm == "F"].mean()))
                p = (np.sum(np.array(null) >= obs) + 1) / (self.cfg.n_perm + 1)
                rows.append(dict(metric=m, factor="Sex", stat=obs, p_perm=p,
                                 effect=_cohen_d(xm, xf), effect_name="Cohen_d"))

            grp = d["AgeGroup"].to_numpy()
            obs = _eta_sq(resid, grp)
            null = []
            grp_perm = grp.copy()
            for _ in range(self.cfg.n_perm):
                rng.shuffle(grp_perm)
                null.append(_eta_sq(resid, grp_perm))
            p = (np.sum(np.array(null) >= obs) + 1) / (self.cfg.n_perm + 1)
            rows.append(dict(metric=m, factor="AgeGroup", stat=obs, p_perm=p,
                             effect=obs, effect_name="eta_squared"))

        out = pd.DataFrame(rows)
        if not out.empty:
            p = out["p_perm"].values
            m = len(p); order = np.argsort(p)
            ranks = np.empty_like(order); ranks[order] = np.arange(1, m + 1)
            q = p * m / ranks
            q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
            qvals = np.empty_like(q_sorted); qvals[order] = q_sorted
            out["q_bh"] = np.clip(qvals, 0, 1)
        out.to_csv(self.cfg.out_csv, index=False)
        log.info("stats_between_written", extra={"out": str(self.cfg.out_csv), "rows": int(len(out))})
        return self.cfg.out_csv
