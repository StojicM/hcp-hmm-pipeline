#!/usr/bin/env python3
from __future__ import annotations

"""
HMM fitting and export with metrics.

Loads the concatenated parcel time series, fits a Dynamax HMM backend,
exports the fitted model, per-subject state sequences and probabilities, and
computes a panel of global and statewise metrics per subject.
"""

import inspect
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy #as np
import pandas #as pd

from .logger import get_logger
from .subjects import load_subject_covariates

log = get_logger(__name__)


def _split_sequences(X: numpy.ndarray, lengths: List[int]) -> List[numpy.ndarray]:
    """Split concatenated array into per-sequence arrays."""
    seqs: List[numpy.ndarray] = []
    start = 0
    for n in lengths:
        end = start + int(n)
        seqs.append(numpy.asarray(X[start:end, :]))
        start = end
    return seqs


def _pad_sequences(X: numpy.ndarray, lengths: List[int]) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Pad variable-length sequences to (B, T_max, P) with mask (B, T_max)."""
    seqs = _split_sequences(X, lengths)
    B = len(seqs)
    T_max = max(lengths) if lengths else 0
    P = X.shape[1] if X.ndim == 2 else 0
    padded = numpy.zeros((B, T_max, P), dtype=X.dtype)
    mask = numpy.zeros((B, T_max), dtype=bool)
    for i, seq in enumerate(seqs):
        L = seq.shape[0]
        padded[i, :L, :] = seq
        mask[i, :L] = True
    return padded, mask


def _require_dynamax():
    try:
        import dynamax  # type: ignore
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore
    except Exception as e:
        raise SystemExit(
            "dynamax backend requested but dynamax/jax is not available. "
            "Install dynamax (and jax/jaxlib) before using dynamax_* backends."
        ) from e
    return dynamax, jax, jnp


def _filter_kwargs(fn, kwargs: Dict[str, object]) -> Dict[str, object]:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _try_call(fn, variants: List[Tuple[Tuple[object, ...], Dict[str, object]]]) -> object:
    last_err: Exception | None = None
    for args, kwargs in variants:
        try:
            return fn(*args, **_filter_kwargs(fn, kwargs))
        except TypeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise TypeError("No callable variants for dynamax function.")


def _dynamax_find_class(mod, candidates: List[str], fallback_tokens: Optional[List[str]] = None) -> Optional[type]:
    for name in candidates:
        cls = getattr(mod, name, None)
        if isinstance(cls, type):
            return cls
    if not fallback_tokens:
        return None
    for name in dir(mod):
        if any(token in name.lower() for token in fallback_tokens):
            cls = getattr(mod, name, None)
            if isinstance(cls, type):
                return cls
    return None


def _build_dynamax_arhmm(K: int, D: int, ar_order: int):
    _require_dynamax()
    errors: List[str] = []
    modules = [
        "dynamax.hidden_markov_model.models",
        "dynamax.hidden_markov_model",
        "dynamax.hmm",
    ]
    candidates = ["LinearAutoregressiveHMM", "AutoregressiveHMM", "ARHMM"]
    for mod_name in modules:
        try:
            mod = __import__(mod_name, fromlist=["*"])
        except Exception as e:
            errors.append(str(e))
            continue
        cls = _dynamax_find_class(mod, candidates, fallback_tokens=["arhmm", "autoregressive", "autoregress"])
        if cls is None:
            continue
        variants = [
            ((int(K), int(D), int(ar_order)), {}),
            ((int(K), int(D)), {"num_lags": int(ar_order)}),
            ((int(K), int(D)), {"lags": int(ar_order)}),
            ((int(K), int(D)), {"order": int(ar_order)}),
        ]
        try:
            return _try_call(cls, variants)
        except TypeError as e:
            errors.append(str(e))
            continue
    msg = "; ".join(errors[-3:]) if errors else "unknown constructor failure"
    raise SystemExit(f"Could not construct dynamax ARHMM (K={K}, D={D}, ar_order={ar_order}). {msg}")


def _build_dynamax_slds(K: int, D: int, latent_dim: int):
    _require_dynamax()
    errors: List[str] = []
    modules = [
        "dynamax.slds",
        "dynamax.linear_gaussian_slds",
        "dynamax.ssm",
    ]
    candidates = ["LinearGaussianSLDS", "SLDS", "LinearSLDS"]
    for mod_name in modules:
        try:
            mod = __import__(mod_name, fromlist=["*"])
        except Exception as e:
            errors.append(str(e))
            continue
        cls = _dynamax_find_class(mod, candidates, fallback_tokens=["slds"])
        if cls is None:
            continue
        variants = [
            ((int(K), int(D), int(latent_dim)), {}),
            ((int(K), int(D)), {"latent_dim": int(latent_dim)}),
            ((int(K), int(D)), {"x_dim": int(latent_dim)}),
            ((int(K), int(D)), {"state_dim": int(latent_dim)}),
        ]
        try:
            return _try_call(cls, variants)
        except TypeError as e:
            errors.append(str(e))
            continue
    msg = "; ".join(errors[-3:]) if errors else "unknown constructor failure"
    raise SystemExit(f"Could not construct dynamax SLDS (K={K}, D={D}, latent_dim={latent_dim}). {msg}")


def _dynamax_initialize(model, key, emissions):
    fn = getattr(model, "initialize", None) or getattr(model, "init_params", None)
    if not callable(fn):
        raise SystemExit("dynamax model does not expose initialize/init_params.")
    variants = [
        ((key, emissions), {}),
        ((key,), {"emissions": emissions}),
        ((), {"key": key, "emissions": emissions}),
        ((key,), {}),
    ]
    return _try_call(fn, variants)


def _dynamax_fit_em(model, params, param_props, emissions, max_iter: int, tol: float, mask=None):
    fn = getattr(model, "fit_em", None) or getattr(model, "fit", None)
    if not callable(fn):
        raise SystemExit("dynamax model does not expose fit_em/fit.")
    kwargs = {
        "num_iters": int(max_iter),
        "max_iters": int(max_iter),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "tolerance": float(tol),
        "convergence_tol": float(tol),
    }
    if mask is not None:
        kwargs["mask"] = mask
        kwargs["masks"] = mask
        kwargs["emissions_mask"] = mask
    if param_props is not None:
        variants = [
            ((params, param_props, emissions), kwargs),
            ((params, param_props, emissions), {}),
        ]
        try:
            out = _try_call(fn, variants)
            if isinstance(out, tuple) and len(out) >= 1:
                return out[0]
            return out
        except Exception:
            pass
    variants = [
        ((params, emissions), kwargs),
        ((params, emissions), {}),
    ]
    out = _try_call(fn, variants)
    if isinstance(out, tuple) and len(out) >= 1:
        return out[0]
    return out


def _dynamax_logprob(model, params, emissions) -> float:
    for name in ("log_prob", "log_probability", "marginal_log_prob", "log_likelihood"):
        fn = getattr(model, name, None)
        if callable(fn):
            variants = [
                ((params, emissions), {}),
                ((params,), {"emissions": emissions}),
            ]
            val = _try_call(fn, variants)
            return float(numpy.sum(val)) if isinstance(val, (list, tuple, numpy.ndarray)) else float(val)
    raise SystemExit("dynamax backend does not expose log probability.")


def _dynamax_expected_states(model, params, emissions) -> numpy.ndarray:
    for name in ("smoother", "posterior", "smooth"):
        fn = getattr(model, name, None)
        if callable(fn):
            out = _try_call(fn, [((params, emissions), {}), ((params,), {"emissions": emissions})])
            if hasattr(out, "smoothed_probs"):
                return numpy.asarray(out.smoothed_probs)
            if isinstance(out, dict):
                for key in ("smoothed_probs", "posterior_probs", "gamma", "expected_states"):
                    if key in out:
                        return numpy.asarray(out[key])
            if isinstance(out, tuple):
                for item in out:
                    arr = numpy.asarray(item)
                    if arr.ndim == 2:
                        return arr
    fn = getattr(model, "expected_states", None)
    if callable(fn):
        out = _try_call(fn, [((params, emissions), {}), ((params,), {"emissions": emissions})])
        return numpy.asarray(out)
    raise SystemExit("dynamax backend did not return posterior state probabilities.")


def _dynamax_most_likely_states(model, params, emissions) -> numpy.ndarray:
    fn = getattr(model, "most_likely_states", None)
    if callable(fn):
        out = _try_call(fn, [((params, emissions), {}), ((params,), {"emissions": emissions})])
        return numpy.asarray(out)
    return numpy.argmax(_dynamax_expected_states(model, params, emissions), axis=1)


def _dynamax_get_attr(obj, path: str):
    cur = obj
    for token in path.split("."):
        if cur is None:
            return None
        if not hasattr(cur, token):
            return None
        cur = getattr(cur, token)
    if callable(cur):
        try:
            cur = cur()
        except Exception:
            pass
    return cur


def _dynamax_transmat(params) -> numpy.ndarray:
    for path in (
        "transitions.transition_matrix",
        "transitions.P",
        "transition_matrix",
        "P",
        "transition_probs",
    ):
        val = _dynamax_get_attr(params, path)
        if val is not None:
            return numpy.asarray(val)
    raise SystemExit("dynamax backend could not extract transition matrix.")


def _dynamax_startprob(params, transmat: numpy.ndarray) -> numpy.ndarray:
    for path in (
        "initial.probs",
        "initial.probabilities",
        "initial_state_probs",
        "init_state_probs",
        "pi0",
    ):
        val = _dynamax_get_attr(params, path)
        if val is not None:
            return numpy.asarray(val)
    return stationary_distribution(transmat)


class _DynamaxAdapter:
    def __init__(self, model, params, param_props, seed: int, max_iter: int, tol: float):
        self.model = model
        self.params = params
        self.param_props = param_props
        self.seed = int(seed)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.startprob_ = None
        self.transmat_ = None
        self.means_ = None
        self.covars_ = None
        self._refresh_params()

    def _refresh_params(self) -> None:
        if self.params is None:
            return
        trans = _dynamax_transmat(self.params)
        self.transmat_ = trans
        self.startprob_ = _dynamax_startprob(self.params, trans)

    def fit(self, X: numpy.ndarray, lengths: List[int]) -> None:
        _, jax, jnp = _require_dynamax()
        seqs = _split_sequences(X, lengths)
        emissions = [jnp.asarray(seq) for seq in seqs]
        key = jax.random.PRNGKey(self.seed)
        init = _dynamax_initialize(self.model, key, emissions)
        if isinstance(init, tuple) and len(init) >= 2:
            self.params, self.param_props = init[0], init[1]
        else:
            self.params, self.param_props = init, None
        try:
            self.params = _dynamax_fit_em(self.model, self.params, self.param_props, emissions, self.max_iter, self.tol)
        except Exception:
            pad, mask = _pad_sequences(X, lengths)
            emissions_pad = jnp.asarray(pad)
            try:
                self.params = _dynamax_fit_em(
                    self.model,
                    self.params,
                    self.param_props,
                    emissions_pad,
                    self.max_iter,
                    self.tol,
                    mask=mask,
                )
            except Exception as e:
                raise SystemExit(f"dynamax fit failed: {e}") from e
        self._refresh_params()

    def score(self, X: numpy.ndarray, lengths: List[int]) -> float:
        _, _, jnp = _require_dynamax()
        seqs = _split_sequences(X, lengths)
        total = 0.0
        for seq in seqs:
            total += float(_dynamax_logprob(self.model, self.params, jnp.asarray(seq)))
        return total

    def predict(self, X: numpy.ndarray, lengths: List[int]) -> numpy.ndarray:
        _, _, jnp = _require_dynamax()
        seqs = _split_sequences(X, lengths)
        states = [_dynamax_most_likely_states(self.model, self.params, jnp.asarray(seq)) for seq in seqs]
        return numpy.concatenate(states, axis=0)

    def predict_proba(self, X: numpy.ndarray, lengths: List[int]) -> numpy.ndarray:
        _, _, jnp = _require_dynamax()
        seqs = _split_sequences(X, lengths)
        probs = [_dynamax_expected_states(self.model, self.params, jnp.asarray(seq)) for seq in seqs]
        return numpy.concatenate(probs, axis=0)

def runs_for_state(states: numpy.ndarray, s: int):
    """Return run lengths (dwell times) for consecutive visits to state `s`."""
    runs, c = [], 0
    for v in states:
        if v == s:
            c += 1
        elif c > 0:
            runs.append(c); c = 0
    if c > 0:
        runs.append(c)
    return runs


def intervisit_intervals(states: numpy.ndarray, s: int):
    """Return intervals (in TRs) between successive visits to state `s`."""
    idx = numpy.where(states == s)[0]
    if idx.size == 0:
        return []
    starts = idx[(numpy.r_[True, numpy.diff(idx) > 1])]
    if len(starts) < 2:
        return []
    return list(numpy.diff(starts))


def stationary_distribution(P: numpy.ndarray, tol=1e-12):
    """Compute stationary distribution of a Markov chain with transition `P`.
    Solves `(P^T - I) pi = 0` with a sum-to-one constraint via least squares,
    then normalizes and clips for numerical stability.
    """
    # SPEED: linear solve instead of eig; numerically equivalent for our use
    K = P.shape[0]
    A = (P.T - numpy.eye(K))
    A[-1, :] = 1.0
    b = numpy.zeros(K); b[-1] = 1.0
    pi = numpy.linalg.lstsq(A, b, rcond=None)[0]
    pi = numpy.clip(pi, 0.0, None)
    s = pi.sum()
    return (pi / s) if s >= tol else numpy.ones(K) / K


def entropy_bits(p, eps=1e-12):
    """Shannon entropy (bits) of a probability vector `p` (safe/clipped)."""
    p = numpy.asarray(p, dtype=float)
    p = numpy.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-(p * numpy.log2(p)).sum())


def lzc_binary(states: numpy.ndarray) -> float:
    """Lempel–Ziv complexity of the binary switch signal (state changes)."""
    if len(states) < 2:
        return 0.0
    sw = (states[1:] != states[:-1]).astype(int)
    s = ''.join(str(x) for x in sw.tolist())
    i = k = 1
    l = 1
    c = 1
    n = len(s)
    while True:
        if i + k > n:
            c += 1
            break
        if s[i:i+k] in s[0:l-1]:
            k += 1
        else:
            c += 1
            l += k
            i = l - 1
            k = 1
        if l > n:
            break
    return c / (n / math.log2(n)) if n > 1 else float(c)


@dataclass
class HMMConfig:
    """Configuration for HMM fitting and outputs.

    - `in_dir` / `out_dir`: where to read `train_X.npy` + index and write outputs
    - `K`, `cov`, `max_iter`, `tol`, `seed`: HMM hyperparameters
    - `backend`: hmm implementation to use ("dynamax_arhmm" | "dynamax_slds")
    - `tr_sec`: TR in seconds (for converting TR counts to seconds in reports)
    - `ar_order`: AR lag order for dynamax_arhmm
    - `slds_latent_dim`: latent dimension for dynamax_slds
    - Atlas/surface paths kept for compatibility; rendering is disabled
    """
    in_dir: Path
    out_dir: Path
    K: int
    cov: str = "diag"
    max_iter: int = 500
    tol: float = 1e-3
    seed: int = 42
    backend: str = "dynamax_arhmm"
    tr_sec: float = 0.72
    ar_order: int = 1
    slds_latent_dim: int = 4
    subjects_csv: Path | None = None
    # Legacy BrainSpace inputs (unused; kept to preserve config compatibility)
    atlas_dlabel: Optional[Path] = None
    surface_dir: Optional[Path] = None
    surface_left: Optional[Path] = None
    surface_right: Optional[Path] = None
    surface_left_inflated: Optional[Path] = None
    surface_right_inflated: Optional[Path] = None


class HMMRunner:
    """Runs HMM fit and exports model, states, and metrics."""
    def __init__(self, cfg: HMMConfig):
        self.cfg = cfg

    @staticmethod
    def _num_params(K: int, P: int) -> Optional[int]:
        return None

    @staticmethod
    def _estimate_means_covars(
        X: numpy.ndarray,
        probs: numpy.ndarray,
        cov_type: str,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Estimate state means/covars from posterior probabilities."""
        cov_type = cov_type.lower()
        K = int(probs.shape[1])
        P = int(X.shape[1])
        weights = probs
        sum_w = weights.sum(axis=0)
        sum_w_safe = numpy.where(sum_w > 0, sum_w, 1.0)
        means = (weights.T @ X) / sum_w_safe[:, None]
        if cov_type == "diag":
            sum_x2 = weights.T @ (X ** 2)
            vars_ = sum_x2 / sum_w_safe[:, None] - means ** 2
            vars_ = numpy.clip(vars_, 1e-6, None)
            return means, vars_

        covars = numpy.zeros((K, P, P), dtype=float)
        for k in range(K):
            denom = float(sum_w_safe[k])
            if denom <= 0:
                continue
            w = weights[:, k][:, None]
            cov = (X * w).T @ X / denom - numpy.outer(means[k], means[k])
            covars[k] = (cov + cov.T) / 2.0
        if cov_type == "tied":
            occ = sum_w / max(sum_w.sum(), 1.0)
            covars = (covars * occ[:, None, None]).sum(axis=0)
        return means, covars

    def _load_training_data(self) -> Tuple[numpy.ndarray, pandas.DataFrame, List[int]]:
        """Load concatenated data and normalize the subject index schema."""
        # SPEED: memory-map large array to reduce RAM pressure; dtype left unchanged
        X = numpy.load(self.cfg.in_dir / "train_X.npy", mmap_mode="r")
        idx = pandas.read_csv(self.cfg.in_dir / "subjects_index.csv")
        # Normalize index columns to a standard schema
        cols = {c.lower(): c for c in idx.columns}
        sid_key = None

        if "subject_id" in cols:
            sid_key = "subject_id"

        if sid_key is None:
            raise SystemExit(f"subjects_index.csv missing subject column. Found: {list(idx.columns)}")
        start_key = cols.get("start") or cols.get("start_tr") or cols.get("begin")
        end_key = cols.get("end") or cols.get("end_tr") or cols.get("stop")
        ntr_key = cols.get("ntr") or cols.get("n_tr") or cols.get("ntrs") or cols.get("n")
        if ntr_key is not None:
            lengths = idx[ntr_key].astype(int).tolist()
        elif start_key is not None and end_key is not None:
            lengths = (idx[end_key].astype(int) - idx[start_key].astype(int)).tolist()
        else:
            raise SystemExit("subjects_index.csv must have either nTR or start/end columns.")
        # Rename to canonical names for downstream code
        ren = {sid_key: "Subject"}
        if start_key:
            ren[start_key] = "start"
        if end_key:
            ren[end_key] = "end"
        if ntr_key:
            ren[ntr_key] = "nTR"
        idx = idx.rename(columns=ren)
        return X, idx, lengths

    def _export_from_fitted(
        self,
        model,
        X: numpy.ndarray,
        idx: pandas.DataFrame,
        lengths: List[int],
        backend: str,
        cov_type: str,
        logL: float,
    ) -> None:
        """Write model, per-subject states/probabilities, and metrics CSVs."""
        n_obs = int(X.shape[0])
        k_params = self._num_params(self.cfg.K, X.shape[1])
        if k_params is None:
            aic = None
            bic = None
        else:
            aic = 2 * k_params - 2 * logL
            bic = math.log(max(n_obs, 1)) * k_params - 2 * logL

        # Make output directories
        out = self.cfg.out_dir
        out.mkdir(parents=True, exist_ok=True)
        states_dir = out / "per_subject_states"
        states_dir.mkdir(exist_ok=True)
        metrics_dir = out / "metrics"
        metrics_dir.mkdir(exist_ok=True)

        # Save model and summaries
        Ktag = f"{self.cfg.K}S"
        joblib.dump(model, out / "model.joblib")
        joblib.dump(model, out / f"hmm_model_{Ktag}.pkl")

        rows_state, rows_global, rows_trans = [], [], []

        # SPEED: predict once over all sequences, then slice by subject
        log.info("hmm_decode_begin", extra={"K": int(self.cfg.K), "backend": backend})
        t0 = time.perf_counter()
        states_all = model.predict(X, lengths=lengths).astype(int)
        log.info("hmm_viterbi_done", extra={"elapsed_s": float(time.perf_counter() - t0), "T": int(states_all.shape[0])})
        t1 = time.perf_counter()
        probs_all = model.predict_proba(X, lengths=lengths)
        log.info("hmm_posteriors_done", extra={"elapsed_s": float(time.perf_counter() - t1), "shape": list(probs_all.shape)})
        offsets = numpy.cumsum([0] + lengths)

        updated_model = False
        if getattr(model, "means_", None) is None or getattr(model, "covars_", None) is None:
            means, covars = self._estimate_means_covars(X, probs_all, cov_type)
            model.means_ = means
            model.covars_ = covars
            updated_model = True
        if updated_model:
            joblib.dump(model, out / "model.joblib")
            joblib.dump(model, out / f"hmm_model_{Ktag}.pkl")

        summary = {
            "K": self.cfg.K,
            "covariance_type": self.cfg.cov,
            "max_iter": self.cfg.max_iter,
            "tol": self.cfg.tol,
            "seed": self.cfg.seed,
            "backend": backend,
            "TR_sec": self.cfg.tr_sec,
            "ar_order": int(getattr(self.cfg, "ar_order", 1)),
            "slds_latent_dim": int(getattr(self.cfg, "slds_latent_dim", 4)),
            "n_parcels": int(X.shape[1]),
            "n_timepoints_total": int(X.shape[0]),
            "n_subjects": int(len(idx)),
            "loglik": float(logL),
            "n_params": int(k_params) if k_params is not None else None,
            "AIC": float(aic) if aic is not None else None,
            "BIC": float(bic) if bic is not None else None,
            "transmat": model.transmat_.tolist(),
            "startprob": model.startprob_.tolist(),
            "means_shape": list(numpy.asarray(model.means_).shape),
            "covars_shape": list(numpy.asarray(model.covars_).shape),
        }
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        numpy.savetxt(out / f"state_transition_matrix_{Ktag}.txt", model.transmat_, fmt="%.6f", delimiter="\t")
        pandas.DataFrame(model.means_).to_csv(out / f"state_mean_patterns_{Ktag}.csv", index_label="state")

        log.info("hmm_train_done", extra={"loglik": float(logL), "subjects": int(len(idx)), "parcels": int(X.shape[1])})

        covar_cols, covars_lookup, covars_uid_lookup = self._prepare_covariates()
        default_covars = {col: pandas.NA for col in covar_cols}
        missing_covars = set()

        def covars_for_subject(sid: str) -> Dict[str, object]:
            if not covar_cols:
                return {}
            info = covars_lookup.get(sid)
            if info is not None:
                return info
            token = sid.split("_")[0]
            info = covars_uid_lookup.get(token)
            if info is not None:
                return info
            missing_covars.add(sid)
            return default_covars

        for i, row in idx.reset_index(drop=True).iterrows():
            sid = str(row["Subject"]) if "Subject" in row else str(row.get("sid", ""))
            s = offsets[i]
            e = offsets[i + 1]
            Xs = X[s:e, :]  # kept (unused, but preserves exact structure)
            states = states_all[s:e]
            probs = probs_all[s:e, :]
            T = len(states)

            stem = f"{sid}"
            numpy.savetxt(states_dir / f"{stem}_state_vector_{Ktag}.txt", states, fmt="%d")
            # keep original (TXT) format for probs to avoid any downstream changes
            numpy.savetxt(states_dir / f"{stem}_state_probs_{Ktag}.txt", probs, fmt="%.6f")

            K = self.cfg.K
            # SPEED: vectorized transition counts
            if T >= 2:
                a = states[:-1].astype(numpy.int64)
                b = states[1:].astype(numpy.int64)
                flat = a * K + b
                counts = numpy.bincount(flat, minlength=K*K).reshape(K, K).astype(numpy.float64)
            else:
                counts = numpy.zeros((K, K), dtype=numpy.float64)

            alpha = 1.0
            denom = counts.sum(axis=1, keepdims=True) + alpha * K
            P = (counts + alpha) / numpy.maximum(denom, 1.0)

            # Precompute metadata for CSV rows (subject + covariates)
            subj_cov = covars_for_subject(sid)
            meta = {"Subject": sid}
            if covar_cols:
                meta.update({col: subj_cov.get(col, pandas.NA) for col in covar_cols})
            meta_with_K = {**meta, "K": K}

            # transitions long-form (unchanged semantics apart from metadata)
            for ii in range(K):
                for jj in range(K):
                    rows_trans.append({
                        **meta_with_K,
                        "from_state": ii,
                        "to_state": jj,
                        "P": float(P[ii, jj])
                    })

            # SPEED: numpy.count_nonzero
            switches_total = int(numpy.count_nonzero(states[1:] != states[:-1])) if T > 1 else 0
            SR_global = switches_total / max(T - 1, 1)

            pi = stationary_distribution(P)
            # SPEED: vectorized row entropies once
            P_clip = numpy.clip(P, 1e-12, 1.0)
            row_H = -(P_clip * numpy.log2(P_clip)).sum(axis=1)
            occ_entropy_bits = entropy_bits(pi)
            entropy_rate_bits = float((pi * row_H).sum())

            mean_self = float(numpy.trace(P) / K)
            evals = numpy.linalg.eigvals(P)
            evals = numpy.sort(numpy.abs(numpy.real(evals)))[::-1]
            spectral_gap = float(1.0 - (evals[1] if len(evals) > 1 else 0.0))
            LZC = lzc_binary(states)

            rows_global.append({
                **meta_with_K,
                "SR_global": SR_global,
                "occ_entropy_bits": occ_entropy_bits,
                "entropy_rate_bits": entropy_rate_bits,
                "mean_self_transition": mean_self,
                "spectral_gap": spectral_gap,
                "LZC_switches": LZC
            })

            # SPEED: bincount for FO
            bc = numpy.bincount(states, minlength=K).astype(numpy.float64)
            FO_all = bc / max(T, 1)
            for s_ in range(K):
                FO = float(FO_all[s_])
                runs = runs_for_state(states, s_)
                n_visits = int(len(runs))
                if n_visits:
                    DT_mean = float(numpy.mean(runs))
                    DT_median = float(numpy.median(runs))
                    DT_var = float(numpy.var(runs, ddof=1)) if n_visits > 1 else 0.0
                else:
                    DT_mean = DT_median = DT_var = 0.0
                iv = intervisit_intervals(states, s_)
                IV_mean = float(numpy.mean(iv)) if iv else 0.0

                if T > 1:
                    entries = int(numpy.count_nonzero((states[:-1] != s_) & (states[1:] == s_)))
                    exits   = int(numpy.count_nonzero((states[:-1] == s_) & (states[1:] != s_)))
                    SR_state = (entries + exits) / (T - 1)
                else:
                    SR_state = 0.0

                rows_state.append({
                    **meta_with_K,
                    "state": s_,
                    "FO": FO,
                    "DT_mean": DT_mean,
                    "DT_median": DT_median,
                    "DT_var": DT_var,
                    "n_visits": n_visits,
                    "IV_mean": IV_mean,
                    "SR_state": float(SR_state),
                    "row_entropy_bits": float(row_H[s_]),
                    "self_transition": float(P[s_, s_])
                })

        Ktag = f"{self.cfg.K}S"

        # Tidy potential duplicate identifier columns before writing
        def _tidy_id_cols(df: pandas.DataFrame) -> pandas.DataFrame:
            d = df.copy()
            # Prefer canonical 'Subject'; drop/rename variants
            if "subject" in d.columns:
                if "Subject" not in d.columns:
                    d = d.rename(columns={"subject": "Subject"})
                else:
                    # If both exist and are identical after str-cast, drop the lowercase one
                    try:
                        if d["subject"].astype(str).equals(d["Subject"].astype(str)):
                            d = d.drop(columns=["subject"])
                        else:
                            # Fall back to the capitalized version as canonical
                            d = d.drop(columns=["subject"])
                    except Exception:
                        d = d.drop(columns=["subject"])
            # Drop UID if it is redundant with Subject (exact or first-token match)
            if "UID" in d.columns and "Subject" in d.columns:
                try:
                    uid_s = d["UID"].astype(str)
                    subj_s = d["Subject"].astype(str)
                    if uid_s.equals(subj_s) or uid_s.equals(subj_s.str.split("_").str[0]):
                        d = d.drop(columns=["UID"])
                except Exception:
                    pass
            if "Subject" in d.columns:
                d["Subject"] = d["Subject"].astype(str)
            return d

        df_state = _tidy_id_cols(pandas.DataFrame(rows_state))
        df_glob = _tidy_id_cols(pandas.DataFrame(rows_global))
        df_trans = _tidy_id_cols(pandas.DataFrame(rows_trans))

        df_state.to_csv(self.cfg.out_dir / "metrics" / f"metrics_state_{Ktag}.csv", index=False)
        df_glob.to_csv(self.cfg.out_dir / "metrics" / f"metrics_global_{Ktag}.csv", index=False)
        df_trans.to_csv(self.cfg.out_dir / "metrics" / f"transitions_long_{Ktag}.csv", index=False)
        if missing_covars:
            sample = sorted(missing_covars)
            log.warning("covars_missing_subjects", extra={
                "missing_n": len(missing_covars),
                "sample": sample[:5]
            })
        log.info("hmm_outputs_written", extra={"out_dir": str(self.cfg.out_dir)})

    def _prepare_covariates(self) -> Tuple[List[str], Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
        """Load optional subject covariates and expose lookups by Subject/UID."""
        df = load_subject_covariates(getattr(self.cfg, "subjects_csv", None))
        if df.empty:
            return [], {}, {}

        covar_cols = [c for c in df.columns if c != "Subject"]
        subj_lookup = df.set_index("Subject")[covar_cols].to_dict("index")
        uid_lookup = {}
        if "UID" in df.columns:
            uid_lookup = df.groupby("UID")[covar_cols].first().to_dict("index")
        return covar_cols, subj_lookup, uid_lookup

    def fit_and_export(self):
        """Fit the HMM and write model, per-subject states, and metrics CSVs."""
        X, idx, lengths = self._load_training_data()

        backend = str(getattr(self.cfg, "backend", "dynamax_arhmm")).lower().replace("-", "_")
        cov_type = self.cfg.cov
        log.info("hmm_train_begin", extra={"K": self.cfg.K, "cov": cov_type, "seed": self.cfg.seed, "backend": backend})
        if backend not in ("dynamax_arhmm", "dynamax_slds"):
            raise SystemExit(f"Unknown HMM backend '{backend}'. Use 'dynamax_arhmm' or 'dynamax_slds'.")
        D = int(X.shape[1])
        if backend == "dynamax_arhmm":
            base = _build_dynamax_arhmm(self.cfg.K, D, int(self.cfg.ar_order))
        else:
            base = _build_dynamax_slds(self.cfg.K, D, int(self.cfg.slds_latent_dim))
        model = _DynamaxAdapter(base, params=None, param_props=None, seed=self.cfg.seed, max_iter=self.cfg.max_iter, tol=self.cfg.tol)
        model.fit(X, lengths)
        logL = model.score(X, lengths)
        self._export_from_fitted(model, X, idx, lengths, backend, cov_type, float(logL))

    def export_only(self, model_path: Optional[Path] = None) -> None:
        """Export states and metrics from an already fitted model on disk."""
        X, idx, lengths = self._load_training_data()
        backend = str(getattr(self.cfg, "backend", "dynamax_arhmm")).lower().replace("-", "_")
        cov_type = self.cfg.cov
        path = Path(model_path) if model_path is not None else (self.cfg.out_dir / "model.joblib")
        if not path.exists():
            raise FileNotFoundError(path)
        log.info("hmm_export_only_begin", extra={"K": int(self.cfg.K), "path": str(path), "backend": backend})
        model = joblib.load(path)
        if backend.startswith("dynamax_") and not isinstance(model, _DynamaxAdapter):
            model = _DynamaxAdapter(model, params=None, param_props=None, seed=self.cfg.seed, max_iter=self.cfg.max_iter, tol=self.cfg.tol)
        logL = model.score(X, lengths)
        self._export_from_fitted(model, X, idx, lengths, backend, cov_type, float(logL))
