#!/usr/bin/env python3
from __future__ import annotations

"""
Autoregressive HMM (ARHMM) with diagonal Gaussian noise and streaming EM.

This implementation fits per-subject sequences without holding the full
N×T×D array in memory, then decodes each subject on demand.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

EPS = 1e-12


def _logsumexp(values: np.ndarray, axis=None) -> np.ndarray:
    max_v = np.max(values, axis=axis, keepdims=True)
    out = max_v + np.log(np.sum(np.exp(values - max_v), axis=axis, keepdims=True))
    if axis is None:
        return out.reshape(())
    return np.squeeze(out, axis=axis)


def _split_sequences(X: np.ndarray, lengths: List[int]) -> List[np.ndarray]:
    sequences: List[np.ndarray] = []
    start = 0
    for n in lengths:
        end = start + int(n)
        sequences.append(np.asarray(X[start:end, :]))
        start = end
    return sequences


def _build_lagged(y: np.ndarray, order: int) -> np.ndarray:
    if order <= 0:
        return np.zeros((y.shape[0], 0), dtype=float)
    T, D = y.shape
    lagged = np.empty((T, D * order), dtype=float)
    for lag in range(1, order + 1):
        block = lagged[:, (lag - 1) * D: lag * D]
        # Pad with the first observation to keep lengths aligned.
        block[:lag, :] = y[0]
        block[lag:, :] = y[:-lag]
    return lagged


def _emission_loglik(
    y: np.ndarray,
    x_aug: np.ndarray,
    weights: np.ndarray,
    covars: np.ndarray,
) -> np.ndarray:
    T, D = y.shape
    K = weights.shape[0]
    loglik = np.empty((T, K), dtype=float)
    for k in range(K):
        mean = x_aug @ weights[k]
        diff = y - mean
        var = np.clip(covars[k], 1e-6, None)
        inv_var = 1.0 / var
        const = -0.5 * np.sum(np.log(2.0 * np.pi * var))
        loglik[:, k] = const - 0.5 * np.sum(diff * diff * inv_var, axis=1)
    return loglik


def _forward_backward(
    log_emission: np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    T, K = log_emission.shape
    log_start = np.log(np.clip(startprob, EPS, None))
    log_trans = np.log(np.clip(transmat, EPS, None))

    log_alpha = np.empty((T, K), dtype=float)
    log_c = np.empty(T, dtype=float)
    log_alpha[0] = log_start + log_emission[0]
    log_c[0] = _logsumexp(log_alpha[0])
    log_alpha[0] -= log_c[0]

    for t in range(1, T):
        log_alpha[t] = log_emission[t] + _logsumexp(
            log_alpha[t - 1][:, None] + log_trans, axis=0
        )
        log_c[t] = _logsumexp(log_alpha[t])
        log_alpha[t] -= log_c[t]

    log_beta = np.empty((T, K), dtype=float)
    log_beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        log_beta[t] = _logsumexp(
            log_trans + log_emission[t + 1] + log_beta[t + 1], axis=1
        ) - log_c[t + 1]

    log_gamma = log_alpha + log_beta
    gamma = np.exp(log_gamma - _logsumexp(log_gamma, axis=1)[:, None])

    xi_sum = np.zeros((K, K), dtype=float)
    for t in range(T - 1):
        log_xi = (
            log_alpha[t][:, None]
            + log_trans
            + log_emission[t + 1][None, :]
            + log_beta[t + 1][None, :]
        )
        log_xi -= _logsumexp(log_xi)
        xi_sum += np.exp(log_xi)

    loglik = float(np.sum(log_c))
    return gamma, xi_sum, loglik


@dataclass
class ARHMM:
    n_states: int
    ar_order: int = 1
    max_iter: int = 100
    seed: int = 42
    ridge: float = 1e-5
    min_var: float = 1e-6

    startprob_: np.ndarray | None = None
    transmat_: np.ndarray | None = None
    weights_: np.ndarray | None = None  # (K, P, D)
    covars_: np.ndarray | None = None   # (K, D), diagonal
    means_: np.ndarray | None = None    # (K, D)
    loglik_: float | None = None

    def _init_params(self, sequences: List[np.ndarray]) -> None:
        rng = np.random.default_rng(self.seed)
        K = int(self.n_states)
        D = int(sequences[0].shape[1])
        P = D * int(self.ar_order) + 1

        self.startprob_ = np.ones(K, dtype=float) / float(K)
        self.transmat_ = np.ones((K, K), dtype=float) / float(K)
        self.weights_ = rng.normal(scale=0.01, size=(K, P, D))

        sum_y = np.zeros(D, dtype=float)
        sum_y2 = np.zeros(D, dtype=float)
        total = 0
        for seq in sequences:
            seq_f = seq.astype(float, copy=False)
            sum_y += seq_f.sum(axis=0)
            sum_y2 += (seq_f ** 2).sum(axis=0)
            total += seq_f.shape[0]
        mean = sum_y / max(total, 1)
        var = sum_y2 / max(total, 1) - mean ** 2
        var = np.clip(var, self.min_var, None)

        self.covars_ = np.tile(var[None, :], (K, 1))
        self.means_ = np.tile(mean[None, :], (K, 1))

    def fit(self, X: np.ndarray, lengths: List[int]) -> "ARHMM":
        sequences = _split_sequences(X, lengths)
        if not sequences:
            raise SystemExit("ARHMM fit requires at least one sequence.")
        self._init_params(sequences)

        K = int(self.n_states)
        D = int(sequences[0].shape[1])
        P = D * int(self.ar_order) + 1

        for _ in range(int(self.max_iter)):
            gamma_sum = np.zeros(K, dtype=float)
            gamma_init = np.zeros(K, dtype=float)
            xi_sum = np.zeros((K, K), dtype=float)
            wxx_sum = np.zeros((K, P, P), dtype=float)
            wxy_sum = np.zeros((K, P, D), dtype=float)
            y_sum = np.zeros((K, D), dtype=float)
            y2_sum = np.zeros((K, D), dtype=float)
            loglik_total = 0.0

            for seq in sequences:
                y = seq.astype(float, copy=False)
                lagged = _build_lagged(y, int(self.ar_order))
                x_aug = np.concatenate([lagged, np.ones((y.shape[0], 1), dtype=float)], axis=1)
                log_emission = _emission_loglik(y, x_aug, self.weights_, self.covars_)
                gamma, xi_counts, loglik = _forward_backward(
                    log_emission, self.startprob_, self.transmat_
                )

                gamma_sum += gamma.sum(axis=0)
                gamma_init += gamma[0]
                xi_sum += xi_counts
                y_sum += gamma.T @ y
                y2_sum += gamma.T @ (y ** 2)

                for k in range(K):
                    weights = gamma[:, k][:, None]
                    xw = x_aug * weights
                    wxx_sum[k] += xw.T @ x_aug
                    wxy_sum[k] += xw.T @ y

                loglik_total += loglik

            self.loglik_ = float(loglik_total)
            self.startprob_ = gamma_init / max(gamma_init.sum(), EPS)

            row_sums = xi_sum.sum(axis=1, keepdims=True)
            self.transmat_ = np.where(row_sums > 0, xi_sum / row_sums, 1.0 / K)

            weights_new = np.zeros_like(self.weights_)
            covars_new = np.zeros_like(self.covars_)
            means_new = np.zeros_like(self.means_)
            eye = np.eye(P, dtype=float)

            for k in range(K):
                denom = max(gamma_sum[k], EPS)
                means_new[k] = y_sum[k] / denom

                wxx = wxx_sum[k] + self.ridge * eye
                wxy = wxy_sum[k]
                try:
                    weights_k = np.linalg.solve(wxx, wxy)
                except np.linalg.LinAlgError:
                    weights_k = np.linalg.lstsq(wxx, wxy, rcond=None)[0]
                weights_new[k] = weights_k

                mean2 = np.einsum("ij,ji->i", weights_k.T @ wxx_sum[k], weights_k)
                resid = y2_sum[k] - 2.0 * np.einsum("ij,ij->j", weights_k, wxy_sum[k]) + mean2
                covars_new[k] = np.clip(resid / denom, self.min_var, None)

            self.weights_ = weights_new
            self.covars_ = covars_new
            self.means_ = means_new

        return self

    def _predict_sequence(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lagged = _build_lagged(y, int(self.ar_order))
        x_aug = np.concatenate([lagged, np.ones((y.shape[0], 1), dtype=float)], axis=1)
        log_emission = _emission_loglik(y, x_aug, self.weights_, self.covars_)
        gamma, _, _ = _forward_backward(log_emission, self.startprob_, self.transmat_)
        return gamma, log_emission

    def score(self, X: np.ndarray, lengths: List[int]) -> float:
        sequences = _split_sequences(X, lengths)
        total = 0.0
        for seq in sequences:
            y = seq.astype(float, copy=False)
            lagged = _build_lagged(y, int(self.ar_order))
            x_aug = np.concatenate([lagged, np.ones((y.shape[0], 1), dtype=float)], axis=1)
            log_emission = _emission_loglik(y, x_aug, self.weights_, self.covars_)
            _, _, loglik = _forward_backward(log_emission, self.startprob_, self.transmat_)
            total += loglik
        return float(total)

    def predict_proba(self, X: np.ndarray, lengths: List[int]) -> np.ndarray:
        sequences = _split_sequences(X, lengths)
        probs = []
        for seq in sequences:
            y = seq.astype(float, copy=False)
            gamma, _ = self._predict_sequence(y)
            probs.append(gamma)
        return np.concatenate(probs, axis=0)

    def predict(self, X: np.ndarray, lengths: List[int]) -> np.ndarray:
        sequences = _split_sequences(X, lengths)
        states = []
        log_start = np.log(np.clip(self.startprob_, EPS, None))
        log_trans = np.log(np.clip(self.transmat_, EPS, None))
        for seq in sequences:
            y = seq.astype(float, copy=False)
            lagged = _build_lagged(y, int(self.ar_order))
            x_aug = np.concatenate([lagged, np.ones((y.shape[0], 1), dtype=float)], axis=1)
            log_emission = _emission_loglik(y, x_aug, self.weights_, self.covars_)
            T, K = log_emission.shape
            delta = np.empty((T, K), dtype=float)
            psi = np.zeros((T, K), dtype=int)
            delta[0] = log_start + log_emission[0]
            for t in range(1, T):
                scores = delta[t - 1][:, None] + log_trans
                psi[t] = np.argmax(scores, axis=0)
                delta[t] = log_emission[t] + np.max(scores, axis=0)
            path = np.zeros(T, dtype=int)
            path[T - 1] = int(np.argmax(delta[T - 1]))
            for t in range(T - 2, -1, -1):
                path[t] = int(psi[t + 1, path[t + 1]])
            states.append(path)
        return np.concatenate(states, axis=0)


__all__ = ["ARHMM"]
