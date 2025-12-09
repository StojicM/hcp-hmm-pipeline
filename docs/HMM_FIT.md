## hcp_hmm/hmm_fit.py

Fits a `GaussianHMM` (hmmlearn) on concatenated parcel time series and exports per-subject state sequences, probabilities, and metrics.

Inputs
- `in_dir/train_X.npy` — stacked time × parcel array
- `in_dir/subjects_index.csv` — subject index with `Subject` and either `nTR` or `start`/`end` columns

Subject index schema
- Required identifier: `Subject` (case-insensitive variants like `subject`, `sid`, `uid` are auto-detected and normalized).
- Segment length:
  - Provide `nTR` (or `n_tr`, `ntrs`) per row, OR
  - Provide `start`/`end` (or `start_tr`/`end_tr`) indices so that `nTR = end - start`.
- One row corresponds to one contiguous segment for a subject. If a subject has multiple runs, include multiple rows.

Outputs (under `out_dir`)
- `model.joblib`, `hmm_model_{K}S.pkl` — the fitted HMM
- `summary.json` — training summary and shapes
- `state_transition_matrix_{K}S.txt` — K×K transition probabilities
- `state_mean_patterns_{K}S.csv` — K×P state means
- `per_subject_states/` — per-subject state vectors and probabilities
- `metrics/` — three CSVs: statewise, global, and transitions long-form

Metrics (per subject)
- Global
  - `SR_global` — switch rate across time
  - `occ_entropy_bits` — entropy of stationary occupancy
  - `entropy_rate_bits` — entropy rate under the estimated Markov chain
  - `mean_self_transition` — average of diagonal `P`
  - `spectral_gap` — `1 - |λ2|` of `P`
  - `LZC_switches` — Lempel–Ziv complexity of the binary switch signal

- Statewise
  - `FO` — fractional occupancy (time in state / total time)
  - `DT_mean`, `DT_median`, `DT_var` — dwell time stats (run lengths in state)
  - `n_visits` — number of state entries
  - `IV_mean` — mean inter-visit interval
  - `SR_state` — state-specific switch rate
  - `row_entropy_bits` — entropy of transition probabilities from this state
  - `self_transition` — P[s, s]

Notes
- The training array is memory-mapped during fit to reduce RAM pressure.
- BrainSpace rendering of state betas has been disabled; only numeric outputs are produced.

Interpreting entropy metrics
- `occ_entropy_bits` (occupancy entropy): Shannon entropy of the stationary distribution `π` implied by `P`. Higher values indicate more uniform long-run occupancy across states; lower values indicate that the chain concentrates on a few states.
- `entropy_rate_bits`: average uncertainty per step under the chain, computed as `∑_s π_s H(P[s, :])`. It captures both how mixed the occupancy is and how diffuse the transitions are. High values suggest many possible next-state choices on average; low values suggest persistent or deterministic transitions.
