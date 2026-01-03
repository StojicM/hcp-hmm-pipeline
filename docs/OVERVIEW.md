# Pipeline Overview

This pipeline orchestrates the following stages:

1) Parcellation (dtseries → ptseries)
   - Input: dense timeseries `*.dtseries.nii` and a `.dlabel.nii` atlas.
   - Output: parcel timeseries `*.ptseries.nii` and an optional labels table.

2) Concatenation (subject runs → training matrix)
   - Input: `*.ptseries.nii` per subject/run.
   - Output: `hmm/train_X.npy` (T×P) and `hmm/subjects_index.csv` with subject ID, ranges and/or lengths.

3) HMM Fit + Exports
   - Input: `hmm/train_X.npy`, `hmm/subjects_index.csv`.
   - Outputs:
     - Model: `hmm/model.joblib`, `hmm/state_mean_patterns_{K}S.csv`.
     - Per-subject: `hmm/per_subject_states/{Subject}_state_vector_{K}S.txt` and `{Subject}_state_probs_{K}S.txt`.
     - Metrics: `hmm/metrics/metrics_state_{K}S.csv`, `hmm/metrics/metrics_global_{K}S.csv`, `hmm/metrics/transitions_long_{K}S.csv` (see docs/METRICS.md).

4) Model Selection (optional)
   - Runs a K/seed sweep when `evaluation.enabled` is true.
   - Outputs: `hmm/model_selection/report.html` and per-run artifacts.

5) State Maps (subject betas + group merge)
   - Input: ptseries and per-subject states.
   - Output: subject state betas `*_state_betas_{K}S.pscalar.nii` and a group-merged stack and tables.

6) Z-scored Betas
   - Output: `*_state_betas_{K}S_zscored.pscalar.nii`.

7) Group Design + Merge
   - Inputs: `subjects_info.csv` and betas.
   - Outputs: FSL-style design (`design*.{mat,con,grp}`), merged pscalar stacks and lists.

8) PALM (optional)
   - Runs PALM on the merged pscalars using provided designs.

Configuration
- Main configuration lives in `pipeline.yaml` under sections `paths`, `hmm`, `parcellate`, `palm`, `stats`, `logging`, `group_design`.
- Parsed into `hcp_hmm/config.py` dataclasses and consumed by `hcp_hmm/pipeline.py`.

Command Entry Points
- For a single-command run: `python -m hcp_hmm.cli run --config pipeline.yaml`.
- For stepwise control, use subcommands (see docs/CLI.md).
