#!/usr/bin/env python3
"""
Unified CLI for the HCP HMM pipeline.

Initial subcommand implemented: `parcellate` (replaces 00_parcellate.sh)

Examples:
  python -m hcp_hmm.cli parcellate \
      --indir data/raw \
      --dlabel data/atlas/300Parcels_Yeo2011_7Networks.dlabel.nii \
      --outdir data/derivatives/ptseries \
      --suffix REST_all_Yeo300 \
      --method MEAN
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .parcellation import ParcellationConfig, Parcellator
from .config import PipelineConfig
from .pipeline import Pipeline
from .ptseries import PtConcatConfig, PtSeriesConcatenator
from .hmm_fit import HMMConfig as _HMMConfig, HMMRunner
from .state_maps import StateMapConfig, StateMapEstimator
from .zscore import ZScoreConfig, ZScoreExporter
from .group_design import GroupDesignConfig, GroupDesignBuilder
from .group_merge import GroupMergeConfig, GroupMerger
from .alignment import AlignmentConfig, AlignmentChecker
from .stats_rm import StatsRMConfig, StatsRM
from .stats_between import StatsBetweenConfig, StatsBetween
from .palm import PalmConfig, PalmRunner


def cmd_parcellate(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="hcp-hmm parcellate",
                                 description="Parcellate all dtseries in a folder using a .dlabel atlas.")
    ap.add_argument("--indir", required=True, help="directory with *.dtseries.nii")
    ap.add_argument("--dlabel", required=True, help="path to *.dlabel.nii atlas")
    ap.add_argument("--outdir", required=True, help="output directory for *.ptseries.nii")
    ap.add_argument("--suffix", default="REST_all_Yeo300", help="suffix to append to outputs")
    ap.add_argument("--method", default="MEAN", help="aggregation method inside parcels (e.g., MEAN)")
    ap.add_argument("--no-export-labels", action="store_true",
                    help="do not export atlas label table alongside outputs")
    args = ap.parse_args(argv)

    cfg = ParcellationConfig(
        indir=Path(args.indir),
        dlabel=Path(args.dlabel),
        outdir=Path(args.outdir),
        method=args.method,
        suffix=args.suffix,
        export_labels=not args.no_export_labels,
    )
    px = Parcellator(cfg)
    outs = px.run()
    if not outs:
        print(f"[00] No .dtseries.nii found in: {cfg.indir}")
        return 0
    print(f"[00] Done. Wrote {len(outs)} ptseries to: {cfg.outdir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hcp-hmm",
                                     description="HCP HMM pipeline CLI (modular, OOP-backed)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # We only parse top-level command here; subcommands parse their own argv
    sub.add_parser("parcellate", help="Parcellate dtseries → ptseries using a dlabel atlas")
    p_concat = sub.add_parser("concat", help="Concatenate *.ptseries.nii into train_X.npy + index")
    p_concat.add_argument("--indir", required=True)
    p_concat.add_argument("--outdir", required=True)
    p_fit = sub.add_parser("fit", help="Fit HMM and export states/metrics")
    p_fit.add_argument("--in-dir", required=True)
    p_fit.add_argument("--out-dir", required=True)
    p_fit.add_argument("--K", type=int, required=True)
    p_fit.add_argument("--cov", choices=["full","diag"], default="diag")
    p_fit.add_argument("--max-iter", type=int, default=500)
    p_fit.add_argument("--tol", type=float, default=1e-3)
    p_fit.add_argument("--seed", type=int, default=42)
    p_fit.add_argument("--tr-sec", type=float, default=0.72)
    p_fit.add_argument("--subjects-csv", default=None,
                       help="Optional CSV with subject-level covariates (sex, age, etc.)")
    p_maps = sub.add_parser("state-maps", help="Compute β maps from posteriors and dtseries")
    p_maps.add_argument("--dtseries-dir", required=True)
    p_maps.add_argument("--states-dir", required=True)
    p_maps.add_argument("--out-dir", required=True)
    p_maps.add_argument("--K", type=int, required=True)
    p_maps.add_argument("--chunk", type=int, default=60000)
    p_maps.add_argument("--rcond", type=float, default=1e-6)
    p_z = sub.add_parser("zscore", help="Z-score β maps and export dscalars")
    p_z.add_argument("--dtseries-dir", required=True)
    p_z.add_argument("--betas-dir", required=True)
    p_z.add_argument("--K", type=int, required=True)
    p_z.add_argument("--redo", action="store_true")
    p_gd = sub.add_parser("group-design", help="Write FSL design files and subjects_used.csv")
    p_gd.add_argument("--subjects-csv", required=True)
    p_gd.add_argument("--out", required=True)
    p_gd.add_argument("--contrast", default=None, help="Contrast name to write (e.g., 'sex' or an age dummy)")
    p_gd.add_argument("--demean", nargs="*", default=None, help="Columns to demean (e.g., 'sex')")
    p_gm = sub.add_parser("group-merge", help="Merge z-scored subject dscalars into a group dscalar")
    p_gm.add_argument("--betas-dir", required=True)
    p_gm.add_argument("--K", type=int, required=True)
    p_gm.add_argument("--out", required=True)
    p_gm.add_argument("--subjects-used", required=True)
    p_align = sub.add_parser("check-alignment", help="Verify merged group order matches subjects order")
    p_align.add_argument("--columns-map", required=True)
    p_align.add_argument("--subjects-used", required=True)
    p_align.add_argument("--K", type=int, required=True)
    p_rm = sub.add_parser("stats-rm", help="Repeated-measures statewise stats")
    p_rm.add_argument("--in-csv", required=True)
    p_rm.add_argument("--K", type=int, required=True)
    p_rm.add_argument("--out", required=True)
    p_rm.add_argument("--n-perm", type=int, default=5000)
    p_b = sub.add_parser("stats-between", help="Between-subject global stats")
    p_b.add_argument("--in-csv", required=True)
    p_b.add_argument("--out", required=True)
    p_b.add_argument("--n-perm", type=int, default=5000)
    p_palm = sub.add_parser("palm", help="Run PALM on the merged group dscalar and design files")
    p_palm.add_argument("--group-dir", required=True, help="Directory with design.mat/.con/.grp and merged dscalar")
    p_palm.add_argument("--K", type=int, required=True)
    p_palm.add_argument("--n-perm", type=int, default=5000)
    p_palm.add_argument("--two-tailed", action="store_true")
    p_palm.add_argument("--tfce", action="store_true")
    p_palm.add_argument("--tfce2D", action="store_true")
    p_palm.add_argument("--no-cifti", action="store_true")
    p_palm.add_argument("--palm-bin", default="palm")
    p_run = sub.add_parser("run", help="Run the full pipeline from config YAML")
    p_run.add_argument("--config", required=True, help="YAML with paths and parameters")
    p_run.add_argument("--force", action="store_true", help="recompute even if outputs exist")

    ns, rest = parser.parse_known_args(argv)
    if ns.cmd == "parcellate":
        # This subcommand uses its own dedicated parser to keep options tidy
        return cmd_parcellate(rest)
    if ns.cmd == "concat":
        PtSeriesConcatenator(PtConcatConfig(indir=Path(ns.indir), outdir=Path(ns.outdir))).run()
        return 0
    if ns.cmd == "fit":
        cfg = _HMMConfig(
            in_dir=Path(ns.in_dir),
            out_dir=Path(ns.out_dir),
            K=int(ns.K), cov=ns.cov, max_iter=ns.max_iter, tol=ns.tol, seed=ns.seed, tr_sec=ns.tr_sec,
            subjects_csv=Path(ns.subjects_csv) if ns.subjects_csv else None,
        )
        HMMRunner(cfg).fit_and_export()
        return 0
    if ns.cmd == "state-maps":
        cfg = StateMapConfig(
            dtseries_dir=Path(ns.dtseries_dir),
            states_dir=Path(ns.states_dir),
            out_dir=Path(ns.out_dir),
            K=int(ns.K),
            chunk=int(ns.chunk),
            rcond=float(ns.rcond),
        )
        StateMapEstimator(cfg).run()
        return 0
    if ns.cmd == "zscore":
        cfg = ZScoreConfig(
            dtseries_dir=Path(ns.dtseries_dir),
            betas_dir=Path(ns.betas_dir),
            K=int(ns.K),
            redo=bool(ns.redo),
        )
        ZScoreExporter(cfg).run()
        return 0
    if ns.cmd == "group-design":
        GroupDesignBuilder(GroupDesignConfig(
            subjects_csv=Path(ns.subjects_csv),
            out_dir=Path(ns.out),
            contrast=ns.contrast,
            demean=ns.demean,
        )).run()
        return 0
    if ns.cmd == "group-merge":
        GroupMerger(GroupMergeConfig(
            betas_dir=Path(ns.betas_dir),
            K=int(ns.K),
            out_dir=Path(ns.out),
            subjects_used_csv=Path(ns.subjects_used),
        )).run()
        return 0
    if ns.cmd == "check-alignment":
        ok = AlignmentChecker(AlignmentConfig(
            columns_map_csv=Path(ns.columns_map),
            subjects_used_csv=Path(ns.subjects_used),
            K=int(ns.K),
        )).check()
        return 0 if ok else 1
    if ns.cmd == "stats-rm":
        StatsRM(StatsRMConfig(
            in_csv=Path(ns.in_csv), K=int(ns.K), out_csv=Path(ns.out), n_perm=int(ns.n_perm)
        )).run()
        return 0
    if ns.cmd == "stats-between":
        StatsBetween(StatsBetweenConfig(
            in_csv=Path(ns.in_csv), out_csv=Path(ns.out), n_perm=int(ns.n_perm)
        )).run()
        return 0
    if ns.cmd == "palm":
        PalmRunner(PalmConfig(
            group_dir=Path(ns.group_dir), K=int(ns.K), n_perm=int(ns.n_perm),
            two_tailed=bool(ns.two_tailed), tfce=bool(ns.tfce), tfce2D=bool(ns.tfce2D),
            cifti=not bool(ns.no_cifti), palm_bin=str(ns.palm_bin)
        )).run()
        return 0
    if ns.cmd == "run":
        cfg = PipelineConfig.from_yaml(Path(ns.config))
        Pipeline(cfg, force=ns.force).run_all()
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
