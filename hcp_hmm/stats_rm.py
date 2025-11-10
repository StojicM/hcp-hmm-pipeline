#!/usr/bin/env python3
from __future__ import annotations

"""
Repeated-measures permutation tests (subject-blocked) for statewise metrics.

Design
- Within-subject repeated measures across states; blocking by subject.
- Omnibus State effects and demographic covariates tested via F-tests with
  permutation nulls that preserve subject structure (shuffle state labels per
  subject, or shuffle subject-level labels across subjects for between-subject
  covariates like Sex and AgeGroup).

Outputs include BH-adjusted q-values. Replicates src/05a_stats_state_rm.py.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from .logger import get_logger

log = get_logger(__name__)

from .logger import get_logger

log = get_logger(__name__)

def _cohen_d(x, y):
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    sp = np.sqrt(((nx - 1) * sx ** 2 + (ny - 1) * sy ** 2) / max(nx + ny - 2, 1))
    return 0.0 if sp == 0 else float((x.mean() - y.mean()) / sp)

def _eta_sq(y, g):
    y = np.asarray(y, float)
    g = np.asarray(g)
    if y.size == 0:
        return 0.0
    grand = y.mean()
    sst = float(((y - grand) ** 2).sum())
    if sst == 0:
        return 0.0
    ssb = 0.0
    for lvl in pd.unique(g):
        yi = y[g == lvl]
        if yi.size:
            ssb += yi.size * (yi.mean() - grand) ** 2
    return float(ssb / sst)


def _X_mat(df: pd.DataFrame, K: int, terms: set[str]) -> np.ndarray:
    cols = []
    if 'Intercept' in terms:
        cols.append(np.ones((len(df), 1), np.float64))
    if 'Subject' in terms:
        cat = pd.Categorical(df['Subject'])
        S = np.eye(len(cat.categories), dtype=np.float64)[cat.codes][:, 1:]
        cols.append(S)
    if 'State' in terms:
        cat = pd.Categorical(df['state'])
        U = np.eye(len(cat.categories), dtype=np.float64)[cat.codes][:, 1:]
        cols.append(U)
    if 'FD' in terms:
        fd_series = df['mean_FD']
        if fd_series.notna().any():
            fd_vals = fd_series.fillna(fd_series.mean()).to_numpy(np.float64)
        else:
            # All missing: treat as 0 (centered nuisance)
            fd_vals = np.zeros(len(df), dtype=np.float64)
        cols.append(fd_vals[:, None])
    if 'Sex' in terms:
        sx = df['Sex'].astype(str).str.upper().map({'F': 0, 'M': 1}).fillna(0).to_numpy(np.float64)[:, None]
        cols.append(sx)
    if 'AgeGroup' in terms:
        ag = pd.get_dummies(df['AgeGroup'], drop_first=True, dtype=np.float64).to_numpy()
        if ag.size:
            cols.append(ag)
    if 'Sex:State' in terms:
        cat = pd.Categorical(df['state']); U = np.eye(len(cat.categories), dtype=np.float64)[cat.codes][:, 1:]
        sx = df['Sex'].astype(str).str.upper().map({'F': 0, 'M': 1}).fillna(0).to_numpy(np.float64)[:, None]
        cols.append(U * sx)
    if 'AgeGroup:State' in terms:
        cat = pd.Categorical(df['state']); U = np.eye(len(cat.categories), dtype=np.float64)[cat.codes][:, 1:]
        AG = pd.get_dummies(df['AgeGroup'], drop_first=True, dtype=np.float64).to_numpy()
        if AG.size:
            cols.append(np.concatenate([U * AG[:, i:i + 1] for i in range(AG.shape[1])], axis=1))
    return np.concatenate(cols, axis=1) if cols else np.zeros((len(df), 1), np.float64)


def _safe_solve(X: np.ndarray, y: np.ndarray, rcond: float = 1e-6) -> np.ndarray:
    """Solve X beta ≈ y with stability safeguards.
    - Try float32 lstsq (fast) with rcond.
    - Fallback to float32 pinv.
    - If still problematic (NaN/Inf), escalate to float64 pinv, then cast back to float32.
    """
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=rcond)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X, rcond=rcond) @ y
    if not np.all(np.isfinite(beta)):
        X64 = X.astype(np.float64, copy=False)
        y64 = y.astype(np.float64, copy=False)
        beta64 = np.linalg.pinv(X64, rcond=rcond) @ y64
        beta = beta64.astype(np.float32)
        log.debug("stats_rm_escalated_to_float64")
    return beta


def _f_test(y: np.ndarray, Xr: np.ndarray, Xf: np.ndarray):
    br = _safe_solve(Xr, y)
    bf = _safe_solve(Xf, y)
    rr = y - Xr @ br; rf = y - Xf @ bf
    RSSr = float(rr.T @ rr); RSSf = float(rf.T @ rf)
    df_num = Xf.shape[1] - Xr.shape[1]; df_den = len(y) - Xf.shape[1]
    if df_num <= 0 or df_den <= 0 or RSSf <= 0: return np.nan, np.nan, df_num, df_den
    F = ((RSSr - RSSf) / df_num) / (RSSf / df_den)
    SS_eff = RSSr - RSSf
    return F, SS_eff, df_num, df_den


def _part_eta2(SS_eff, RSS_full):
    if SS_eff is None or np.isnan(SS_eff): return np.nan
    denom = SS_eff + RSS_full
    return float(SS_eff / denom) if denom > 0 else np.nan


@dataclass
class StatsRMConfig:
    in_csv: Path
    K: int
    out_csv: Path
    n_perm: int = 5000
    # Optional: also write per-state simple effects (Sex / AgeGroup)
    out_simple_csv: Path | None = None


class StatsRM:
    def __init__(self, cfg: StatsRMConfig):
        self.cfg = cfg

    def run(self) -> Path:
        df = pd.read_csv(self.cfg.in_csv)
        for c in ["Subject", "state", "Sex", "AgeGroup", "mean_FD"]:
            if c not in df.columns: raise SystemExit(f"Missing column: {c}")
        df["Subject"] = df["Subject"].astype(str); df["state"] = df["state"].astype(int)

        metrics = [m for m in [
            # FO variants (older pipelines) and current FO
            "FO", "FO_hard", "FO_soft", "FO_gap",
            # Dwell time and visits metrics
            "DT_mean", "DT_median", "DT_var", "DT_std", "DT_CV",
            "n_visits", "IV_mean", "SR_state", "sharpness",
            # Row entropy and self-transition from current exporter
            "row_entropy_bits", "self_transition"
        ] if m in df.columns]

        base_terms_no_state = {"Intercept", "Subject", "FD"}
        base_terms_with_state = base_terms_no_state | {"State"}
        rng = np.random.default_rng(42)
        rows = []

        for m in metrics:
            d = df[~df[m].isna()].copy()
            y = d[m].to_numpy(np.float64)[:, None]
            X_base_no_state = _X_mat(d, self.cfg.K, base_terms_no_state)
            Xb = _X_mat(d, self.cfg.K, base_terms_with_state)

            subs = d["Subject"].unique()
            ag_map0 = d.groupby("Subject")["AgeGroup"].first().to_dict()
            sx_map0 = d.groupby("Subject")["Sex"].first().to_dict()
            subj_rows = d.groupby("Subject").groups

            # State main effect (omnibus)
            Xr = X_base_no_state
            Xf = Xb
            F_obs, SS_eff, df_num, df_den = _f_test(y, Xr, Xf)
            permF = []
            for _ in range(self.cfg.n_perm):
                dp = d.copy()
                for sid, idx in subj_rows.items():
                    vals = dp.loc[idx, "state"].to_numpy()
                    if len(vals) > 1:
                        dp.loc[idx, "state"] = rng.permutation(vals)
                Xrp = _X_mat(dp, self.cfg.K, base_terms_no_state)
                Xfp = _X_mat(dp, self.cfg.K, base_terms_with_state)
                Fp, _, _, _ = _f_test(y, Xrp, Xfp)
                permF.append(Fp if np.isfinite(Fp) else -np.inf)
            p = (np.sum(np.asarray(permF) >= F_obs) + 1) / (self.cfg.n_perm + 1)
            beta_full = _safe_solve(Xf, y)
            RSSf = float(((y - Xf @ beta_full).T @ (y - Xf @ beta_full)))
            rows.append(dict(metric=m, term="State", F_obs=float(F_obs), p_perm=float(p),
                             df_num=int(df_num), df_den=int(df_den), partial_eta2=_part_eta2(SS_eff, RSSf)))

            # AgeGroup main
            Xr = Xb
            Xf = _X_mat(d, self.cfg.K, base_terms_with_state | {"AgeGroup"})
            F_obs, SS_eff, df_num, df_den = _f_test(y, Xr, Xf)
            permF = []
            for _ in range(self.cfg.n_perm):
                psubs = subs.copy(); rng.shuffle(psubs)
                ag_new = {s: ag_map0[p] for s, p in zip(subs, psubs)}
                dp = d.copy(); dp["AgeGroup"] = dp["Subject"].map(ag_new)
                Xfp = _X_mat(dp, self.cfg.K, base_terms_with_state | {"AgeGroup"})
                Fp, _, _, _ = _f_test(y, Xr, Xfp); permF.append(Fp if np.isfinite(Fp) else -np.inf)
            p = (np.sum(np.asarray(permF) >= F_obs) + 1) / (self.cfg.n_perm + 1)
            beta_full = _safe_solve(Xf, y)
            RSSf = float(((y - Xf @ beta_full).T @ (y - Xf @ beta_full)))
            rows.append(dict(metric=m, term="AgeGroup", F_obs=float(F_obs), p_perm=float(p),
                             df_num=int(df_num), df_den=int(df_den), partial_eta2=_part_eta2(SS_eff, RSSf)))

            # AgeGroup × State
            Xr = _X_mat(d, self.cfg.K, base_terms_with_state | {"AgeGroup"})
            Xf = _X_mat(d, self.cfg.K, base_terms_with_state | {"AgeGroup", "AgeGroup:State"})
            F_obs, SS_eff, df_num, df_den = _f_test(y, Xr, Xf)
            permF = []
            for _ in range(self.cfg.n_perm):
                psubs = subs.copy(); rng.shuffle(psubs)
                ag_new = {s: ag_map0[p] for s, p in zip(subs, psubs)}
                dp = d.copy(); dp["AgeGroup"] = dp["Subject"].map(ag_new)
                Xrp = _X_mat(dp, self.cfg.K, base_terms_with_state | {"AgeGroup"})
                Xfp = _X_mat(dp, self.cfg.K, base_terms_with_state | {"AgeGroup", "AgeGroup:State"})
                Fp, _, _, _ = _f_test(y, Xrp, Xfp); permF.append(Fp if np.isfinite(Fp) else -np.inf)
            p = (np.sum(np.asarray(permF) >= F_obs) + 1) / (self.cfg.n_perm + 1)
            beta_full = _safe_solve(Xf, y)
            RSSf = float(((y - Xf @ beta_full).T @ (y - Xf @ beta_full)))
            rows.append(dict(metric=m, term="AgeGroup:State", F_obs=float(F_obs), p_perm=float(p),
                             df_num=int(df_num), df_den=int(df_den), partial_eta2=_part_eta2(SS_eff, RSSf)))

            # Sex main
            Xr = Xb
            Xf = _X_mat(d, self.cfg.K, base_terms_with_state | {"Sex"})
            F_obs, SS_eff, df_num, df_den = _f_test(y, Xr, Xf)
            permF = []
            for _ in range(self.cfg.n_perm):
                psubs = subs.copy(); rng.shuffle(psubs)
                sx_new = {s: sx_map0[p] for s, p in zip(subs, psubs)}
                dp = d.copy(); dp["Sex"] = dp["Subject"].map(sx_new)
                Xfp = _X_mat(dp, self.cfg.K, base_terms_with_state | {"Sex"})
                Fp, _, _, _ = _f_test(y, Xr, Xfp); permF.append(Fp if np.isfinite(Fp) else -np.inf)
            p = (np.sum(np.asarray(permF) >= F_obs) + 1) / (self.cfg.n_perm + 1)
            beta_full = _safe_solve(Xf, y)
            RSSf = float(((y - Xf @ beta_full).T @ (y - Xf @ beta_full)))
            rows.append(dict(metric=m, term="Sex", F_obs=float(F_obs), p_perm=float(p),
                             df_num=int(df_num), df_den=int(df_den), partial_eta2=_part_eta2(SS_eff, RSSf)))

            # Sex × State
            Xr = _X_mat(d, self.cfg.K, base_terms_with_state | {"Sex"})
            Xf = _X_mat(d, self.cfg.K, base_terms_with_state | {"Sex", "Sex:State"})
            F_obs, SS_eff, df_num, df_den = _f_test(y, Xr, Xf)
            permF = []
            for _ in range(self.cfg.n_perm):
                psubs = subs.copy(); rng.shuffle(psubs)
                sx_new = {s: sx_map0[p] for s, p in zip(subs, psubs)}
                dp = d.copy(); dp["Sex"] = dp["Subject"].map(sx_new)
                Xrp = _X_mat(dp, self.cfg.K, base_terms_with_state | {"Sex"})
                Xfp = _X_mat(dp, self.cfg.K, base_terms_with_state | {"Sex", "Sex:State"})
                Fp, _, _, _ = _f_test(y, Xrp, Xfp); permF.append(Fp if np.isfinite(Fp) else -np.inf)
            p = (np.sum(np.asarray(permF) >= F_obs) + 1) / (self.cfg.n_perm + 1)
            beta_full = _safe_solve(Xf, y)
            RSSf = float(((y - Xf @ beta_full).T @ (y - Xf @ beta_full)))
            rows.append(dict(metric=m, term="Sex:State", F_obs=float(F_obs), p_perm=float(p),
                             df_num=int(df_num), df_den=int(df_den), partial_eta2=_part_eta2(SS_eff, RSSf)))

        res = pd.DataFrame(rows)
        if not res.empty:
            pvals = res["p_perm"].to_numpy()
            order = np.argsort(pvals); ranks = np.empty_like(order); ranks[order] = np.arange(1, len(pvals) + 1)
            q = pvals * len(pvals) / ranks
            q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
            qvals = np.empty_like(q_sorted); qvals[order] = q_sorted
            res["q_bh"] = np.clip(qvals, 0, 1)
        res.to_csv(self.cfg.out_csv, index=False)
        log.info("stats_rm_written", extra={"out": str(self.cfg.out_csv), "rows": int(len(res))})

        # Additionally compute simple per-state effects (Sex, AgeGroup) for each metric
        simple_rows = []
        rng2 = np.random.default_rng(42)
        states = sorted(df["state"].dropna().unique().tolist())
        for m in metrics:
            for s in states:
                d = df[(df["state"] == s) & (~df[m].isna())].copy()
                if d.empty:
                    continue
                # Regress out FD (nuisance), as in between-subjects stats
                X = np.c_[np.ones(len(d)), d["mean_FD"].fillna(d["mean_FD"].mean())]
                beta = np.linalg.lstsq(X, d[m].to_numpy(float), rcond=None)[0]
                resid = d[m].to_numpy(float) - X @ beta

                # Sex simple effect within this state
                lab = d["Sex"].astype(str).str.upper().to_numpy()
                xm = resid[lab == "M"]; xf = resid[lab == "F"]
                if len(xm) > 0 and len(xf) > 0:
                    obs = abs(xm.mean() - xf.mean())
                    null = []
                    lab_perm = lab.copy()
                    for _ in range(self.cfg.n_perm):
                        rng2.shuffle(lab_perm)
                        null.append(abs(resid[lab_perm == "M"].mean() - resid[lab_perm == "F"].mean()))
                    p = (np.sum(np.array(null) >= obs) + 1) / (self.cfg.n_perm + 1)
                    simple_rows.append(dict(metric=m, state=int(s), factor="Sex", stat=float(obs), p_perm=float(p),
                                            effect=float(_cohen_d(xm, xf)), effect_name="Cohen_d"))

                # AgeGroup simple effect within this state
                grp = d["AgeGroup"].to_numpy()
                # Skip if all missing or only one level present
                if pd.isna(grp).all() or len(pd.unique(grp)) <= 1:
                    pass
                else:
                    obs = float(_eta_sq(resid, grp))
                    null = []
                    grp_perm = grp.copy()
                    for _ in range(self.cfg.n_perm):
                        rng2.shuffle(grp_perm)
                        null.append(_eta_sq(resid, grp_perm))
                    p = (np.sum(np.array(null) >= obs) + 1) / (self.cfg.n_perm + 1)
                    simple_rows.append(dict(metric=m, state=int(s), factor="AgeGroup", stat=float(obs), p_perm=float(p),
                                            effect=float(obs), effect_name="eta_squared"))

        if simple_rows:
            simple_df = pd.DataFrame(simple_rows)
            p = simple_df["p_perm"].to_numpy()
            order = np.argsort(p); ranks = np.empty_like(order); ranks[order] = np.arange(1, len(p) + 1)
            q = p * len(p) / ranks
            q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
            qvals = np.empty_like(q_sorted); qvals[order] = q_sorted
            simple_df["q_bh"] = np.clip(qvals, 0, 1)
            out_simple = self.cfg.out_simple_csv
            if out_simple is None:
                # Derive a reasonable default next to out_csv
                out_simple = self.cfg.out_csv.parent / f"stats_state_{self.cfg.K}S_simple.csv"
            simple_df.to_csv(out_simple, index=False)
            log.info("stats_state_simple_written", extra={"out": str(out_simple), "rows": int(len(simple_df))})

        return self.cfg.out_csv
