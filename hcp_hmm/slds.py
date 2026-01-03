#!/usr/bin/env python3
from __future__ import annotations

"""
Placeholder SLDS backend.

SLDS training is not implemented yet in this pipeline.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SLDS:
    n_states: int
    latent_dim: int
    max_iter: int = 100
    seed: int = 42

    def fit(self, X: np.ndarray, lengths: List[int]) -> "SLDS":
        raise SystemExit("SLDS backend is not implemented yet.")

    def score(self, X: np.ndarray, lengths: List[int]) -> float:
        raise SystemExit("SLDS backend is not implemented yet.")

    def predict(self, X: np.ndarray, lengths: List[int]) -> np.ndarray:
        raise SystemExit("SLDS backend is not implemented yet.")

    def predict_proba(self, X: np.ndarray, lengths: List[int]) -> np.ndarray:
        raise SystemExit("SLDS backend is not implemented yet.")


__all__ = ["SLDS"]
