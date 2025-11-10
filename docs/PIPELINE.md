## hcp_hmm/pipeline.py

High-level orchestrator that wires together the pipeline stages using typed configs.

Constructed from `PipelineConfig` (see `hcp_hmm/config.py`), invoked via the CLI or programmatically.

**Class: Pipeline**
- File: `hcp_hmm/pipeline.py`
- Fields:
  - `configs`: parsed `PipelineConfig` (paths, hmm params, etc.)
  - `force`: recompute even if outputs exist

**Methods (run order)**
- `parcellate_dtseries()`
  - Inputs: `paths.raw_dtseries_dir`, `paths.atlas_dlabel`
  - Outputs: `paths.ptseries_dir/*.ptseries.nii`, labels TSV
  - Skips if outputs already exist (unless `force`)

- `concat_ptseries()`
  - Inputs: `paths.ptseries_dir`
  - Outputs: `paths.hmm_dir/train_X.npy`, `paths.hmm_dir/subjects_index.csv`

- `fit_hmm()`
  - Inputs: `hmm_dir/train_X.npy`, `hmm_dir/subjects_index.csv`
  - Outputs: model files, per-subject states/probabilities, and metrics under `hmm_dir`
  - Implementation: constructs `HMMConfig` and calls `HMMRunner.fit_and_export()`

- `qc()`
  - Inputs: HMM outputs (metrics, states), optional FD CSV
  - Outputs: HTML report under `hmm_dir/qc`

- `state_maps()`
  - Inputs: ptseries, per-subject states
  - Outputs: per-subject state betas (`*.pscalar.nii`) in `paths.betas_dir`

- `zscore_export()`
  - Inputs: subject betas
  - Outputs: z-scored subject betas (`*_zscored.pscalar.nii`)

- `group_design()`
  - Inputs: `paths.subjects_csv`, HMM `K`, group design options
  - Outputs: design files in `betas_dir/group/`

- `group_merge()`
  - Inputs: per-subject z-scored betas, design
  - Outputs: merged pscalars, inputs list, and handy stacks in `betas_dir/group/`

Notes
- Each step guards with existence checks unless `force` is set.
- Downstream classes live under `hcp_hmm/*` and use dedicated configs to avoid tight coupling.

**Key Artifacts**
- `paths.hmm_dir/model.joblib` — fitted `GaussianHMM`
- `paths.hmm_dir/per_subject_states/` — states and probabilities
- `paths.hmm_dir/metrics/` — CSV summaries
- `paths.betas_dir/` — subject and group betas; group subdir holds designs/merges

**Related Code**
- `hcp_hmm/hmm_fit.py` — HMM fitting and metrics
- `hcp_hmm/state_maps.py` — betas estimation
- `hcp_hmm/group_design.py`, `group_merge.py` — design/merge utilities

