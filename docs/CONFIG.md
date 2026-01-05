## Configuration (pipeline.yaml)

Top-level sections:

- `paths`
  - `raw_dtseries_dir` — directory with `*.dtseries.nii`
  - `atlas_dlabel` — path to atlas `.dlabel.nii`
  - `ptseries_dir` — output dir for `*.ptseries.nii`
  - `hmm_dir` — working/output dir for HMM (model, states, metrics)
  - `betas_dir` — output dir for per-subject/group betas
  - `parcel_labels_dlabel` — atlas used for painting labels (defaults to `atlas_dlabel`)
  - `subjects_csv` — subject-level covariates CSV
  - Optional volumetric labels: `parcel_labels_nii`

- `hmm`
  - `K` — number of HMM states
  - `backend` — `arhmm` (default) or `slds` (planned)
  - `max_iter`, `seed`, `tr_sec` — training hyperparameters
  - `ar_order` — AR lag order for `arhmm`
  - `slds_latent_dim` — latent dimension for `slds`

- `parcellate`
  - `method` — aggregation inside parcels (e.g., `MEAN`)
  - `suffix` — filename suffix for ptseries artifacts

- `palm`
  - `enabled`, `n_perm`, `two_tailed`, `tfce`, `tfce2D`, `cifti`, `palm_bin`, `ise`, `use_zscored`

- `stats`
  - `n_perm_rm` — permutations for repeated-measures state stats
  - `n_perm_between` — permutations for global between-subject stats

- `logging`
  - `format` (`plain` or `json`) and `level` (`DEBUG`…)

- `group_design`
  - `contrast` (e.g., `intercept`, `sex`), `demean` ([columns]), `include_fd` (bool), optional `stacking` and `subject_order_file`

- `evaluation` (optional K/seed sweep)
  - `enabled` — run the sweep when calling `hcp_hmm.cli run` or `hcp_hmm.cli model-select`
  - `K_values` — list of candidate K values (defaults to `hmm.K` if omitted)
  - `seeds` — list of RNG seeds for stability (defaults to `hmm.seed` if omitted)
  - `out_dir` — where to write sweep outputs (defaults to `<paths.hmm_dir>/model_selection`)
  - `junk` — thresholds for Stage 1 (FO/dwell/presence)
  - `indecision` — thresholds for Stage 3 (posterior dominance/ambiguity)
  - `clone` — thresholds for Stage 4 (mean-pattern correlation)
  - `reliability` — Stage 5 run-splitting (e.g., `run_len_tr: 1200`, `n_runs: 4`)

Loaded by: `hcp_hmm/pipeline.py:PipelineConfig.from_yaml`.
