## hcp_hmm/cli.py

Subcommands and their main inputs/outputs.

General usage:

```
python -m hcp_hmm.cli <command> [args]
```

Key commands
- `parcellate` — Parcellate dtseries to ptseries
  - Inputs: `--indir`, `--dlabel`
  - Outputs: `--outdir/*.ptseries.nii` (+ labels table)

- `concat` — Concatenate ptseries into training matrix
  - Inputs: `--indir`
  - Outputs: `--outdir/train_X.npy`, `--outdir/subjects_index.csv`

- `fit` — Fit HMM and export per-subject states and metrics
  - Inputs: `--in-dir` (with `train_X.npy`), `--out-dir`, `--K`, optional surfaces/atlas
  - Outputs: model/joblib, per-subject states/probabilities, metrics CSVs

- `state-maps` — Estimate subject state betas and optional surfaces
  - Inputs: `--ptseries-dir`, `--states-dir`, `--out-dir`, `--K`
  - Outputs: `*_state_betas_{K}S.pscalar.nii`

- `zscore` — Export z-scored betas
  - Inputs: `--ptseries-dir`, `--betas-dir`, `--K`
  - Outputs: `*_state_betas_{K}S_zscored.pscalar.nii`

- `group-design` — Build FSL design matrices for group tests
  - Inputs: `--subjects-csv`, `--out`, optional `--contrast`, `--demean`, `--include-fd`
  - Outputs: `design*.{mat,con,grp}` under `--out`

- `group-merge` — Merge subject betas into group stacks and lists
  - Inputs: `--betas-dir`, `--K`, `--out`, `--subjects-used`
  - Outputs: merged pscalars + inputs lists

- `check-alignment` — Verify group order matches subjects used
  - Inputs: `--columns-map`, `--subjects-used`, `--K`

- `stats-rm` — Repeated-measures statewise stats
  - Inputs: `--in-csv`, `--K`, `--n-perm`
  - Outputs: CSV

- `stats-between` — Between-subject global stats
  - Inputs: `--in-csv`, `--n-perm`
  - Outputs: CSV

- `palm` — Run PALM
  - Inputs: `--group-dir` with design files and stacks, `--K`, `--n-perm`
  - Outputs: PALM result files per state

- `run` — Execute the full pipeline using `pipeline.yaml`
  - Inputs: `--config`, optional `--force`
  - Notes: runs model selection when `evaluation.enabled` is true

- `model-select` — Sweep K/seed combinations and write model-selection reports
  - Inputs: `--config` (with an `evaluation:` section), optional `--force`
  - Outputs: `<evaluation.out_dir or paths.hmm_dir/model_selection>/report.html` (+ per-run artifacts under `runs/`)
