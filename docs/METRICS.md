## Metrics Reference

This document summarizes the subject-level metrics exported by the pipeline and how to interpret them. All metrics refer to a fitted K-state HMM on parcel time series.

Global metrics (per subject)
- `SR_global` — Global switch rate. Fraction of time-steps with a state change: count(states[t] != states[t-1]) / (T-1). Higher means more frequent switching.
- `occ_entropy_bits` — Entropy of stationary occupancy π implied by the subject’s transition matrix P. High when occupancy is spread across states; low when concentrated.
- `entropy_rate_bits` — Entropy rate under the Markov chain: ∑_s π_s H(P[s, :]). High when transitions are diffuse on average; low when persistent/deterministic.
- `mean_self_transition` — Mean of diagonal entries of P. Larger values imply more persistence in states.
- `spectral_gap` — 1 − |λ2|, where λ2 is the second-largest eigenvalue magnitude of P. Larger gap suggests faster mixing/less temporal autocorrelation.
- `LZC_switches` — Lempel–Ziv complexity of the binary switch signal (1 at a change, 0 otherwise). Higher indicates more complex, less predictable switching.

Statewise metrics (per subject, per state)
- `FO` — Fractional occupancy: time in state / total time. Sum over states equals 1.
- `DT_mean`, `DT_median`, `DT_var` — Dwell time statistics for consecutive runs within the state (in TRs).
- `n_visits` — Number of state entries (runs) during the scan(s).
- `IV_mean` — Mean inter-visit interval: average TRs between starts of consecutive runs in the state.
- `SR_state` — Switch rate attributable to the state: proportion of adjacent pairs that enter or exit that state.
- `row_entropy_bits` — Entropy of outgoing transitions from the state: H(P[s, :]). High when the next state is uncertain; low when the state is “sticky”.
- `self_transition` — P[s, s], probability of remaining in the same state at the next TR.

Notes
- All rates/entropies are computed from the empirical transition matrix estimated from the hard state sequence per subject (with a small Dirichlet prior for stability).
- Dwell and inter-visit metrics are in TR units. Multiply by TR_sec from config if seconds are desired.

Statistical testing
- Repeated-measures (statewise): `stats_state_{K}S_rm.csv` uses subject-blocked permutations (PALM-style `-eb` logic) to test omnibus state effects and demographic covariates across states.
- Between-subject (global): `stats_global_{K}S.csv` tests differences across Sex and AgeGroup after regressing out motion (mean FD).

Effect sizes
- `Cohen_d` — standardised difference between two groups (Sex) after residualising FD.
- `eta_squared` — proportion of variance explained by a categorical factor (AgeGroup) after residualising FD.

