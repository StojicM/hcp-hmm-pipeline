#!/usr/bin/env python3
"""
Lightweight JAX-backed Gaussian HMM (EM) compatible with hmmlearn's interface.

Supported covariance types:
- diag  : per-state diagonal covariance
- tied  : single full covariance shared across states

This is intentionally minimal and only implements what the pipeline consumes:
`fit`, `score`, `predict`, `predict_proba`, and attributes
`startprob_`, `transmat_`, `means_`, `covars_`.
"""
from __future__ import annotations

import math
import os
import time
import logging
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

try:
    # Prefer CPU unless user explicitly requests another backend; avoids TPU/GPU warnings
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    import jax
    import jax.numpy as jnp
    import jax.scipy.special as jss
except Exception:  # pragma: no cover - lazy import guard
    jax = None
    jnp = None
    jss = None

log = logging.getLogger(__name__)


def _require_jax() -> None:
    if jnp is None or jax is None:
        raise ImportError(
            "JAX backend requested but jax/jaxlib is not available. "
            "Install jax and jaxlib to use backend='jax'."
        )


def _split_sequences(X: np.ndarray, lengths: Sequence[int]) -> List[np.ndarray]:
    seqs: List[np.ndarray] = []
    start = 0
    for n in lengths:
        end = start + int(n)
        seqs.append(X[start:end, :])
        start = end
    return seqs


def _pad_sequences(X: np.ndarray, lengths: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Pad variable-length sequences to (B, T_max, P) with mask (B, T_max)."""
    seqs = _split_sequences(X, lengths)
    B = len(seqs)
    T_max = max(lengths) if lengths else 0
    P = X.shape[1] if X.ndim == 2 else 0
    padded = np.zeros((B, T_max, P), dtype=np.float64)
    mask = np.zeros((B, T_max), dtype=bool)
    for i, seq in enumerate(seqs):
        L = seq.shape[0]
        padded[i, :L, :] = seq
        mask[i, :L] = True
    return padded, mask


def _log_prob_diag(X: jnp.ndarray, means: jnp.ndarray, covars: jnp.ndarray, min_covar: float) -> jnp.ndarray:
    # X: T×P, means: K×P, covars: K×P
    var = jnp.maximum(covars, min_covar)
    diff = X[:, None, :] - means[None, :, :]
    logdet = jnp.sum(jnp.log(var), axis=1)  # K
    inv_var = 1.0 / var
    mah = jnp.sum(diff * diff * inv_var[None, :, :], axis=2)  # T×K
    P = X.shape[1]
    return -0.5 * (mah + logdet[None, :] + P * math.log(2.0 * math.pi))


def _log_prob_tied(X: jnp.ndarray, means: jnp.ndarray, cov: jnp.ndarray, min_covar: float) -> jnp.ndarray:
    # X: T×P, means: K×P, cov: P×P
    P = X.shape[1]
    cov_reg = cov + min_covar * jnp.eye(P, dtype=cov.dtype)
    L = jnp.linalg.cholesky(cov_reg)
    # log |Sigma|
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    # Solve L^{-1} diff to compute Mahalanobis distances
    diff = (X[:, None, :] - means[None, :, :]).reshape(-1, P)  # (T*K)×P
    y = jnp.linalg.solve(L, diff.T).T  # (T*K)×P
    mah = jnp.sum(y * y, axis=1).reshape(X.shape[0], means.shape[0])  # T×K
    return -0.5 * (mah + logdet + P * math.log(2.0 * math.pi))


def _log_prob_padded(
    X_pad: jnp.ndarray,
    mask: jnp.ndarray,
    means: jnp.ndarray,
    covars: jnp.ndarray,
    cov_type: str,
    min_covar: float,
) -> jnp.ndarray:
    """Return padded log-likelihoods shape (B,T,K) with padding set to -inf."""
    B, T, P = X_pad.shape
    X_flat = X_pad.reshape(B * T, P)
    if cov_type == "diag":
        ll_flat = _log_prob_diag(X_flat, means, covars, min_covar)
    else:
        ll_flat = _log_prob_tied(X_flat, means, covars, min_covar)
    ll = ll_flat.reshape(B, T, means.shape[0])
    # set padding to -inf so it is ignored
    pad_mask = (~mask)[..., None]
    ll = jnp.where(pad_mask, -jnp.inf, ll)
    return ll


def _e_step_padded(
    log_startprob: jnp.ndarray,
    log_transmat: jnp.ndarray,
    log_likelihood: jnp.ndarray,
    mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Vectorized forward-backward over padded batch.

    Returns:
      gamma_pad: (B, T, K) posteriors (0 on padding)
      xi_sum: (K, K) expected transitions summed over batch/time
      log_c: (B, T) scaling terms per timestep (0 on padding)
    """
    _require_jax()
    B, T, K = log_likelihood.shape

    # Forward
    ll_tm = jnp.swapaxes(log_likelihood, 0, 1)  # T×B×K
    mask_tm = jnp.swapaxes(mask, 0, 1)          # T×B
    ll0 = ll_tm[0]
    m0 = mask_tm[0]
    alpha0 = log_startprob[None, :] + ll0
    c0 = jss.logsumexp(alpha0, axis=1)
    alpha0 = alpha0 - c0[:, None]
    alpha0 = jnp.where(m0[:, None], alpha0, -jnp.inf)
    log_c0 = jnp.where(m0, c0, 0.0)

    def fwd_step(alpha_prev, inputs):
        ll_t, m_t = inputs  # ll_t: B×K, m_t: B
        tmp = alpha_prev[:, :, None] + log_transmat[None, :, :]  # B×K×K
        la = jss.logsumexp(tmp, axis=1) + ll_t  # B×K
        ct = jss.logsumexp(la, axis=1)  # B
        la = la - ct[:, None]
        alpha_next = jnp.where(m_t[:, None], la, alpha_prev)
        log_c_t = jnp.where(m_t, ct, 0.0)
        return alpha_next, (alpha_next, log_c_t)

    alpha_end, (alpha_scan, log_c_scan) = jax.lax.scan(
        fwd_step,
        alpha0,
        (ll_tm[1:], mask_tm[1:]),
    )
    # Stack in time-major then swap back to batch-major
    alphas_tm = jnp.concatenate([alpha0[None, :, :], alpha_scan], axis=0)  # T×B×K
    log_c_tm = jnp.concatenate([log_c0[None, :], log_c_scan], axis=0)      # T×B
    alphas = jnp.swapaxes(alphas_tm, 0, 1)  # B×T×K
    log_c = jnp.swapaxes(log_c_tm, 0, 1)    # B×T

    # Backward
    beta_T = jnp.zeros((B, K))

    def bwd_step(beta_next, inputs):
        ll_tp1, m_tp1, ct_tp1, m_t = inputs
        tmp = log_transmat[None, :, :] + ll_tp1[:, None, :] + beta_next[:, None, :]
        lb = jss.logsumexp(tmp, axis=2) - ct_tp1[:, None]
        beta_t = jnp.where(m_t[:, None] & m_tp1[:, None], lb, beta_next)
        return beta_t, beta_t

    ll_rev = jnp.flip(ll_tm, axis=0)     # T×B×K
    mask_rev = jnp.flip(mask_tm, axis=0) # T×B
    log_c_rev_tm = jnp.flip(log_c_tm, axis=0)  # T×B
    _, beta_scan = jax.lax.scan(
        bwd_step,
        beta_T,
        (ll_rev[1:], mask_rev[1:], log_c_rev_tm[1:], mask_rev[:-1]),
    )
    beta_tm = jnp.concatenate([beta_scan, beta_T[None, ...]], axis=0)  # T×B×K reversed (includes beta_T)
    beta_tm = jnp.swapaxes(beta_tm, 0, 1)  # B×T×K reversed
    betas = jnp.flip(beta_tm, axis=1)

    # Posterior gamma
    log_gamma = alphas + betas
    log_gamma = log_gamma - jss.logsumexp(log_gamma, axis=2, keepdims=True)
    gamma = jnp.where(mask[..., None], jnp.exp(log_gamma), 0.0)

    # Expected transitions
    ll_shift = log_likelihood[:, 1:, :]
    mask_t = mask[:, :-1]
    mask_tp1 = mask[:, 1:]
    ct_tp1 = log_c[:, 1:]
    tmp = (
        alphas[:, :-1, :, None]
        + log_transmat[None, None, :, :]
        + ll_shift[:, :, None, :]
        + betas[:, 1:, None, :]
        - ct_tp1[:, :, None, None]
    )  # B×(T-1)×K×K
    xi = jnp.exp(tmp)
    xi = jnp.where((mask_t & mask_tp1)[:, :, None, None], xi, 0.0)
    xi_sum = jnp.sum(xi, axis=(0, 1))

    return gamma, xi_sum, log_c


def _viterbi(
    log_startprob: jnp.ndarray,
    log_transmat: jnp.ndarray,
    log_likelihood: jnp.ndarray,
    lengths: Sequence[int],
) -> np.ndarray:
    # NOTE: A pure-JAX per-timestep Viterbi loop is extremely slow without JIT.
    # We compute emission log-likelihoods in JAX, then run the dynamic program
    # in NumPy (fast for small K; avoids per-op dispatch overhead).
    K = int(log_startprob.shape[0])
    log_pi = np.asarray(log_startprob, dtype=np.float64).reshape(-1)
    log_A = np.asarray(log_transmat, dtype=np.float64)
    ll = np.asarray(log_likelihood, dtype=np.float64)
    lengths = [int(L) for L in lengths]
    if ll.ndim != 2 or ll.shape[1] != K:
        raise ValueError(f"Expected log_likelihood shape (T,K={K}); got {ll.shape}")

    out = np.empty((ll.shape[0],), dtype=np.int64)
    off = 0
    for L in lengths:
        if L <= 0:
            continue
        seg = ll[off:off + L, :]  # L×K
        delta = log_pi + seg[0]
        psi = np.empty((L, K), dtype=np.int32)
        psi[0, :] = 0
        for t in range(1, L):
            scores = delta[:, None] + log_A  # K×K
            psi[t, :] = np.argmax(scores, axis=0).astype(np.int32, copy=False)
            delta = np.max(scores, axis=0) + seg[t]
        st = np.empty((L,), dtype=np.int64)
        st[-1] = int(np.argmax(delta))
        for t in range(L - 1, 0, -1):
            st[t - 1] = int(psi[t, st[t]])
        out[off:off + L] = st
        off += L
    return out[:off]


@dataclass
class JAXGaussianHMM:
    n_components: int
    covariance_type: str = "tied"  # diag | tied
    n_iter: int = 1000
    tol: float = 1e-3
    random_state: int | None = None
    verbose: bool = False
    min_covar: float = 1e-3
    # Internal: float32 can cause a small "jitter floor" in total log-likelihood.
    # This factor inflates the effective tolerance based on logprob magnitude
    # to prevent extremely long tails when `tol` is tiny (e.g. 1e-3) but the
    # numerics cannot reliably resolve deltas that small.
    tol_floor_factor: float = 0.1

    def __post_init__(self):
        _require_jax()
        cov = self.covariance_type.lower()
        if cov not in ("diag", "tied"):
            raise ValueError(f"JAXGaussianHMM supports covariance_type in ['diag', 'tied'], got {self.covariance_type}")
        self.covariance_type = cov
        self.rng = np.random.RandomState(self.random_state)
        # Model params populated after fit
        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None

    def _init_params(self, X: np.ndarray, lengths: Sequence[int]) -> None:
        K = self.n_components
        P = X.shape[1]
        # Uniform start/trans
        self.startprob_ = np.full(K, 1.0 / K, dtype=np.float64)
        self.transmat_ = np.full((K, K), 1.0 / K, dtype=np.float64)
        # Means: pick random rows
        idx = self.rng.choice(X.shape[0], size=K, replace=False if X.shape[0] >= K else True)
        self.means_ = X[idx, :].astype(np.float64, copy=True)
        # Covariance init
        if self.covariance_type == "diag":
            var = np.var(X, axis=0, ddof=1)
            var[var < self.min_covar] = self.min_covar
            self.covars_ = np.tile(var, (K, 1))
        else:  # tied
            cov = np.cov(X, rowvar=False)
            cov = cov + self.min_covar * np.eye(P)
            self.covars_ = cov.astype(np.float64, copy=True)

    def _log_prob_flat(self, X: np.ndarray) -> jnp.ndarray:
        _require_jax()
        Xj = jnp.asarray(X)
        means = jnp.asarray(self.means_)
        if self.covariance_type == "diag":
            covars = jnp.asarray(self.covars_)
            return _log_prob_diag(Xj, means, covars, self.min_covar)
        else:
            cov = jnp.asarray(self.covars_)
            return _log_prob_tied(Xj, means, cov, self.min_covar)

    def _log_prob_padded(self, X_pad: np.ndarray, mask: np.ndarray) -> jnp.ndarray:
        return _log_prob_padded(
            jnp.asarray(X_pad),
            jnp.asarray(mask),
            jnp.asarray(self.means_),
            jnp.asarray(self.covars_),
            self.covariance_type,
            self.min_covar,
        )

    def fit(self, X: np.ndarray, lengths: Sequence[int]) -> "JAXGaussianHMM":
        _require_jax()
        X = np.asarray(X, dtype=np.float64)
        lengths = [int(l) for l in lengths]
        self._init_params(X, lengths)
        K = self.n_components
        prev_logprob = float("nan")
        fit_t0 = time.perf_counter()
        n_obs = int(X.shape[0])
        n_seq = int(len(lengths))
        n_feat = int(X.shape[1]) if X.ndim == 2 else 0
        tol = float(self.tol)
        if math.isfinite(tol) and tol > 0:
            tol_decimals = int(max(0.0, math.ceil(-math.log10(tol))))
            delta_decimals = min(12, int(tol_decimals + 1))
        else:
            delta_decimals = 4
        if self.verbose:
            seed_disp = int(self.random_state) if self.random_state is not None else "NA"
            log.info(
                "[JAX-HMM] begin_fit K=%d seed=%s cov=%s n_iter=%d tol=%g n_seq=%d",
                int(K),
                seed_disp,
                str(self.covariance_type),
                int(self.n_iter),
                float(self.tol),
                int(n_seq),
                extra={
                    "K": int(K),
                    "seed": seed_disp if seed_disp != "NA" else None,
                    "cov": str(self.covariance_type),
                    "iter": int(self.n_iter),
                    "tol": float(self.tol),
                    "n_seq": int(n_seq),
                    "n_obs": int(n_obs),
                    "n_feat": int(n_feat),
                },
            )
        for it in range(self.n_iter):
            iter_t0 = time.perf_counter()
            X_pad, mask = _pad_sequences(X, lengths)
            log_start = jnp.log(jnp.asarray(self.startprob_))
            log_trans = jnp.log(jnp.asarray(self.transmat_))
            log_lik_pad = self._log_prob_padded(X_pad, mask)
            gamma_pad, xi_sum, log_c = _e_step_padded(log_start, log_trans, log_lik_pad, jnp.asarray(mask))
            # IMPORTANT: accumulate in float64 on host for a stable delta signal.
            # NumPy sums float32 in float32 by default, which can quantize total logprob
            # for large datasets and make deltas appear as ...000 or jump to 0 early.
            log_c_np = np.asarray(log_c)
            logprob = float(np.sum(log_c_np, dtype=np.float64))
            delta = (logprob - prev_logprob) if math.isfinite(prev_logprob) else None
            gamma_np = np.asarray(gamma_pad)
            xi_np = np.asarray(xi_sum)

            # Unpad gamma to match flat X for M-step
            gamma_list = []
            for i, L in enumerate(lengths):
                gamma_list.append(gamma_np[i, :L, :])
            gamma_flat = np.concatenate(gamma_list, axis=0) if gamma_list else np.zeros((0, K))

            gamma_sum = gamma_flat.sum(axis=0)  # K
            # M-step: start/trans
            start_accum = []
            offset = 0
            for L in lengths:
                start_accum.append(gamma_flat[offset, :])
                offset += L
            self.startprob_ = np.clip(np.mean(start_accum, axis=0), 1e-12, None)
            self.startprob_ = self.startprob_ / self.startprob_.sum()
            trans = xi_np / np.maximum(xi_np.sum(axis=1, keepdims=True), 1e-12)
            self.transmat_ = np.clip(trans, 1e-12, None)
            self.transmat_ = self.transmat_ / self.transmat_.sum(axis=1, keepdims=True)

            # M-step: means
            weighted_sum = gamma_flat.T @ X  # K×P
            means = weighted_sum / np.maximum(gamma_sum[:, None], 1e-12)
            self.means_ = means

            if self.covariance_type == "diag":
                diff = X[None, :, :] - means[:, None, :]  # K×T×P
                var = np.sum(gamma_flat.T[:, :, None] * diff * diff, axis=1) / np.maximum(gamma_sum[:, None], 1e-12)
                var[var < self.min_covar] = self.min_covar
                self.covars_ = var
            else:
                # Tied full covariance shared across states
                P = X.shape[1]
                cov_accum = np.zeros((P, P), dtype=np.float64)
                offset = 0
                for k in range(K):
                    diff = X - means[k]
                    cov_accum += (gamma_flat[:, k][:, None] * diff).T @ diff
                denom = np.maximum(gamma_sum.sum(), 1e-12)
                cov = cov_accum / denom
                cov = cov + self.min_covar * np.eye(P)
                self.covars_ = cov

            if self.verbose:
                iter_s = float(time.perf_counter() - iter_t0)
                elapsed_s = float(time.perf_counter() - fit_t0)
                delta_str = f"{delta:+.{delta_decimals}f}" if delta is not None else "NA"
                log.info(
                    "[JAX-HMM] iter %d/%d logprob=%.0f delta=%s iter_s=%.1f elapsed_s=%.1f",
                    int(it + 1),
                    int(self.n_iter),
                    float(logprob),
                    delta_str,
                    iter_s,
                    elapsed_s,
                    extra={
                        "iter": int(it + 1),
                        "logprob": float(logprob),
                        "delta": float(delta) if delta is not None else None,
                        "iter_s": iter_s,
                        "elapsed_s": elapsed_s,
                    },
                )
            # hmmlearn uses `delta < tol` on the total log-likelihood.
            # With float32 JAX numerics, the total log-likelihood can "jitter" by
            # ~O(1) even after practical convergence. To keep `pipeline.yaml` tol
            # values usable across backends, apply a small tolerance floor that
            # scales with logprob magnitude when running in float32.
            tol_eff = float(self.tol)
            try:
                eps = float(np.finfo(log_c_np.dtype).eps)
            except Exception:
                eps = float(np.finfo(np.float32).eps)
            tol_floor = float(self.tol_floor_factor) * eps * float(abs(logprob))
            if tol_floor > tol_eff:
                tol_eff = tol_floor

            if it > 0 and delta is not None and abs(delta) < tol_eff:
                if self.verbose:
                    abs_delta_str = f"{abs(delta):.{delta_decimals}f}"
                    log.info(
                        "[JAX-HMM] converged at iter %d (|delta|=%s < tol_eff=%g, user_tol=%g)",
                        int(it + 1),
                        abs_delta_str,
                        float(tol_eff),
                        float(tol),
                        extra={
                            "iter": int(it + 1),
                            "logprob": float(logprob),
                            "delta": float(delta),
                            "tol_eff": float(tol_eff),
                            "tol": float(tol),
                            "elapsed_s": float(time.perf_counter() - fit_t0),
                        },
                    )
                break
            prev_logprob = logprob
        if self.verbose:
            log.info(
                "[JAX-HMM] end_fit iters=%d logprob=%.0f elapsed_s=%.1f",
                int(it + 1),
                float(logprob),
                float(time.perf_counter() - fit_t0),
                extra={
                    "iters": int(it + 1),
                    "logprob": float(logprob),
                    "elapsed_s": float(time.perf_counter() - fit_t0),
                },
            )
        return self

    def score(self, X: np.ndarray, lengths: Sequence[int]) -> float:
        X_pad, mask = _pad_sequences(np.asarray(X, dtype=np.float64), lengths)
        log_start = jnp.log(jnp.asarray(self.startprob_))
        log_trans = jnp.log(jnp.asarray(self.transmat_))
        log_lik_pad = self._log_prob_padded(X_pad, mask)
        _, _, log_c = _e_step_padded(log_start, log_trans, log_lik_pad, jnp.asarray(mask))
        return float(np.sum(np.asarray(log_c), dtype=np.float64))

    def predict_proba(self, X: np.ndarray, lengths: Sequence[int]) -> np.ndarray:
        X_pad, mask = _pad_sequences(np.asarray(X, dtype=np.float64), lengths)
        log_start = jnp.log(jnp.asarray(self.startprob_))
        log_trans = jnp.log(jnp.asarray(self.transmat_))
        log_lik_pad = self._log_prob_padded(X_pad, mask)
        gamma_pad, _, _ = _e_step_padded(log_start, log_trans, log_lik_pad, jnp.asarray(mask))
        # Unpad back to flat order
        gamma_np = np.asarray(gamma_pad)
        gamma_list = []
        for i, L in enumerate(lengths):
            gamma_list.append(gamma_np[i, :int(L), :])
        return np.concatenate(gamma_list, axis=0) if gamma_list else np.zeros((0, self.n_components))

    def predict(self, X: np.ndarray, lengths: Sequence[int]) -> np.ndarray:
        log_start = jnp.log(jnp.asarray(self.startprob_))
        log_trans = jnp.log(jnp.asarray(self.transmat_))
        log_lik = self._log_prob_flat(X)
        return _viterbi(log_start, log_trans, log_lik, lengths)
