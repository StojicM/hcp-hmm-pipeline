## Group Design (Repeated Measures, PALM/FSL)

This pipeline builds FSL-compatible design files to analyze subject-level state betas with repeated measures (each subject contributes K maps — one per HMM state).

Terminology
- Observation: a single row in the group pscalar — i.e., a specific subject’s map for a specific state.
- Subject-major stacking: rows ordered as sub1: state1..K, sub2: state1..K, …
- State-major stacking: rows ordered as state1: sub1..N, state2: sub1..N, …

Files Emitted
- 'design.mat' — Nobs × W matrix (rows = observations, columns = regressors)
- 'design.con' — 1×W (or multiple) contrasts
- 'design.grp' — not used by PALM but included for completeness
- 'eb.txt' — exchangeability blocks for PALM ('-eb') encoding subject IDs per row
- 'mapping.tsv' — observation bookkeeping: row ↔ subject ↔ state
- Diagnostics: 'design_cols.txt', 'subjects_used.csv', 'age_levels.txt'

Regressors Included
- Intercept — 1 per row
- Age — treated as categorical: one-hot with drop-first (reference is first observed level). If you need a continuous age regressor, prepare a numeric column upstream and replace the one-hot expansion.
- Sex — numeric: M→1, F→0; can be demeaned (optional)
- Mean FD — optional covariate; included when available or forced; can be demeaned
- State dummies — K-1 columns: 'state[2]..state[K]' (reference is 'state[1]')

Exchangeability Blocks (Repeated Measures)
PALM supports repeated-measures via exchangeability blocks. We write 'eb.txt' containing a per-row subject ID (1..Nsub). Rows with the same subject ID form a block within which PALM permutes appropriately, preserving within-subject dependence.

Contrasts: “What to test?”
- Specifying a column: set 'contrast' to a design column name (e.g., 'sex', 'fd', 'state[6]', 'age[M]'). The builder constructs a 1-df contrast selecting that column.
- Default choice: if no contrast is provided, the builder picks a simple, useful default: last 'state[...]' column, else 'sex', 'fd', any 'age[...]', else 'intercept'.

Interpretation Examples
- 'sex': tests mean difference between sexes, controlling for other included covariates and state coding.
- 'state[K]': within-subject effect comparing state K vs the reference state (state 1), averaged across subjects.
- 'age[M]': difference between age level “M” and the reference age level.
- 'fd': association between motion (mean FD) and the maps.

Demeaning
- Demeaning removes group mean from a numeric covariate, often improving interpretability of the intercept and reducing collinearity.
- Choices in YAML/CLI (e.g., 'demean: ["fd"]') affect only sex/fd; age levels are categorical dummies and not demeaned.

Stacking Matters
- Ensure the 'stacking' option matches how you stacked the group pscalar ('subject-major' vs 'state-major'); otherwise subject/state rows won’t align.
- You can supply an explicit 'subject_order_file' to force the merge order (one subject ID per line).

Edge Cases
- If only one age level appears after filtering, no age columns are added (drop-first yields zero columns).
- If FD is requested but not present in the CSV, it is skipped with a log note.

Related Code
- Builder and config: 'hcp_hmm/group_design.py'
- Invoked by pipeline: 'hcp_hmm/pipeline.py: group_design()'

