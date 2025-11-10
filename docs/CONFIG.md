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
  - Optional surfaces: `surface_dir`, `surface_left`, `surface_right`, `surface_left_inflated`, `surface_right_inflated`
  - Optional volumetric labels: `parcel_labels_nii`

- `hmm`
  - `K` — number of HMM states
  - `cov` — covariance type (`diag` or `full`)
  - `max_iter`, `tol`, `seed`, `tr_sec` — training hyperparameters

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

Loaded by: `hcp_hmm/config.py:PipelineConfig.from_yaml`.

