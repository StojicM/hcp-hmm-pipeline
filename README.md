# HCP HMM Pipeline

This repository implements a modular pipeline to fit Hidden Markov Models (HMM) on parcellated HCP time series, export per-subject state sequences and probabilities, compute subject/global metrics, build group-level state maps, and optionally run PALM-based statistics and model-selection sweeps.

Quick links:
- docs/OVERVIEW.md — end-to-end flow with key artifacts
- docs/PIPELINE.md — code-level view of `hcp_hmm/pipeline.py`
- docs/CLI.md — subcommands provided by `hcp_hmm/cli.py`

Typical entry point:

```
python -m hcp_hmm.cli run --config pipeline.yaml
```

See `pipeline.yaml` for an example configuration and `run-example.txt` for a minimal run sketch.
