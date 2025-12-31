#!/usr/bin/env python3
"""
Unified CLI for the HCP HMM pipeline. Run:

~>> python -m hcp_hmm.cli run --config pipeline.yaml

"""
## PROMENIO ns: u namespace:
## cfg u config
from __future__ import annotations

import argparse
from pathlib import Path

from .parcellation import ParcellationConfig, Parcellator
# from .config import PipelineConfig #prebacio dole:
from .pipeline import Pipeline, PipelineConfig
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


def _handle_parcellate(namespace: argparse.Namespace) -> int:
    """CLI handler: parcellate dtseries to ptseries using a dlabel atlas."""
    config = ParcellationConfig(
        indir=Path(namespace.indir),
        dlabel=Path(namespace.dlabel),
        outdir=Path(namespace.outdir),
        method=namespace.method,
        suffix=namespace.suffix,
        export_labels=not namespace.no_export_labels,
        redo=bool(namespace.redo),
    )
    outs = Parcellator(config).run()
    if not outs:
        print(f"[00] No .dtseries.nii found in: {config.indir}")
    else:
        print(f"[00] Done. Wrote {len(outs)} ptseries to: {config.outdir}")
    return 0


def _handle_concat(namespace: argparse.Namespace) -> int:
    """CLI handler: concatenate ptseries into training matrix/index."""
    PtSeriesConcatenator(PtConcatConfig(indir=Path(namespace.indir), outdir=Path(namespace.outdir))).run()
    return 0


def _handle_fit(namespace: argparse.Namespace) -> int:
    """CLI handler: fit HMM and export states/metrics."""
    config = _HMMConfig(
        in_dir=Path(namespace.in_dir),
        out_dir=Path(namespace.out_dir),
        K=int(namespace.K),
        cov=namespace.cov,
        max_iter=namespace.max_iter,
        tol=namespace.tol,
        seed=namespace.seed,
        backend=namespace.backend,
        tr_sec=namespace.tr_sec,
        ar_order=namespace.ar_order,
        slds_latent_dim=namespace.slds_latent_dim,
        subjects_csv=Path(namespace.subjects_csv) if namespace.subjects_csv else None,
        atlas_dlabel=Path(namespace.atlas_dlabel) if getattr(namespace, "atlas_dlabel", None) else None,
        surface_dir=Path(namespace.surface_dir) if getattr(namespace, "surface_dir", None) else None,
        surface_left=Path(namespace.surface_left) if getattr(namespace, "surface_left", None) else None,
        surface_right=Path(namespace.surface_right) if getattr(namespace, "surface_right", None) else None,
        surface_left_inflated=Path(namespace.surface_left_inflated) if getattr(namespace, "surface_left_inflated", None) else None,
        surface_right_inflated=Path(namespace.surface_right_inflated) if getattr(namespace, "surface_right_inflated", None) else None,
    )
    HMMRunner(config).fit_and_export()
    return 0


def _handle_state_maps(namespace: argparse.Namespace) -> int:
    """CLI handler: estimate subject state betas (pscalares)."""
    config = StateMapConfig(
        ptseries_dir=Path(namespace.ptseries_dir),
        states_dir=Path(namespace.states_dir),
        out_dir=Path(namespace.out_dir),
        K=int(namespace.K),
        chunk=int(namespace.chunk),
        rcond=float(namespace.rcond),
        render_brainspace=bool(getattr(namespace, "render_brainspace", False)),
        atlas_dlabel=Path(namespace.atlas_dlabel) if getattr(namespace, "atlas_dlabel", None) else None,
        surface_dir=Path(namespace.surface_dir) if getattr(namespace, "surface_dir", None) else None,
        surface_left=Path(namespace.surface_left) if getattr(namespace, "surface_left", None) else None,
        surface_right=Path(namespace.surface_right) if getattr(namespace, "surface_right", None) else None,
        surface_left_inflated=Path(namespace.surface_left_inflated) if getattr(namespace, "surface_left_inflated", None) else None,
        surface_right_inflated=Path(namespace.surface_right_inflated) if getattr(namespace, "surface_right_inflated", None) else None,
    )
    StateMapEstimator(config).run()
    return 0


def _handle_zscore(namespace: argparse.Namespace) -> int:
    """CLI handler: export z-scored betas."""
    config = ZScoreConfig(
        ptseries_dir=Path(namespace.ptseries_dir),
        betas_dir=Path(namespace.betas_dir),
        K=int(namespace.K),
        redo=bool(namespace.redo),
    )
    ZScoreExporter(config).run()
    return 0


def _handle_group_design(namespace: argparse.Namespace) -> int:
    """CLI handler: build group design matrices for FSL/PALM."""
    if getattr(namespace, "include_fd", False) and getattr(namespace, "no_fd", False):
        raise SystemExit("Cannot specify both --include-fd and --no-fd")
    demean = namespace.demean if namespace.demean else None
    include_fd = True if getattr(namespace, "include_fd", False) else (False if getattr(namespace, "no_fd", False) else None)
    GroupDesignBuilder(GroupDesignConfig(
        subjects_csv=Path(namespace.subjects_csv),
        out_dir=Path(namespace.out),
        contrast=namespace.contrast,
        demean=demean,
        include_fd=include_fd,
    )).run()
    return 0


def _handle_group_merge(namespace: argparse.Namespace) -> int:
    """CLI handler: merge subject betas and write input lists."""
    GroupMerger(GroupMergeConfig(
        betas_dir=Path(namespace.betas_dir),
        K=int(namespace.K),
        out_dir=Path(namespace.out),
        subjects_used_csv=Path(namespace.subjects_used),
        parcel_labels_nii=Path(namespace.parcel_labels_nii) if getattr(namespace, "parcel_labels_nii", None) else None,
        atlas_dlabel=Path(namespace.atlas_dlabel) if getattr(namespace, "atlas_dlabel", None) else None,
    )).run()
    return 0


def _handle_check_alignment(namespace: argparse.Namespace) -> int:
    """CLI handler: verify group stack order matches subjects used."""
    ok = AlignmentChecker(AlignmentConfig(
        columns_map_csv=Path(namespace.columns_map),
        subjects_used_csv=Path(namespace.subjects_used),
        K=int(namespace.K),
    )).check()
    return 0 if ok else 1


def _handle_stats_rm(namespace: argparse.Namespace) -> int:
    """CLI handler: run repeated-measures (statewise) stats."""
    StatsRM(StatsRMConfig(
        in_csv=Path(namespace.in_csv),
        K=int(namespace.K),
        out_csv=Path(namespace.out),
        n_perm=int(namespace.n_perm),
    )).run()
    return 0


def _handle_stats_between(namespace: argparse.Namespace) -> int:
    """CLI handler: run between-subject global stats."""
    StatsBetween(StatsBetweenConfig(
        in_csv=Path(namespace.in_csv),
        out_csv=Path(namespace.out),
        n_perm=int(namespace.n_perm),
    )).run()
    return 0


def _handle_palm(namespace: argparse.Namespace) -> int:
    """CLI handler: run PALM per state using provided group design."""
    group_dir = Path(namespace.group_dir)
    subj_mat = group_dir / "design_subjects.mat"
    subj_con = group_dir / "design_subjects.con"
    subj_grp = group_dir / "design_subjects.grp"
    for required in (subj_mat, subj_con, subj_grp):
        if not required.exists():
            raise FileNotFoundError(required)

    base_kwargs = dict(
        group_dir=group_dir,
        K=int(namespace.K),
        n_perm=int(namespace.n_perm),
        two_tailed=bool(namespace.two_tailed),
        tfce=bool(namespace.tfce),
        tfce2D=bool(namespace.tfce2D),
        cifti=not bool(namespace.no_cifti),
        palm_bin=str(namespace.palm_bin),
        subject_design_mat=subj_mat,
        subject_design_con=subj_con,
        subject_design_grp=subj_grp,
    )
    if namespace.state is not None:
        PalmRunner(PalmConfig(state=int(namespace.state), **base_kwargs)).run()
    else:
        for state in range(int(namespace.K)):
            PalmRunner(PalmConfig(state=state, **base_kwargs)).run()
    return 0


def _handle_run(namespace: argparse.Namespace) -> int:
    """CLI handler: run the full pipeline from a YAML config."""
    config = PipelineConfig.from_yaml(Path(namespace.config))
    Pipeline(config, force=namespace.force).run_all()
    return 0


def _handle_model_select(namespace: argparse.Namespace) -> int:
    """CLI handler: sweep K/seeds and write model-selection reports."""
    config = PipelineConfig.from_yaml(Path(namespace.config))
    Pipeline(config, force=namespace.force).run_model_selection()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Build the CLI, parse arguments, and dispatch to a subcommand."""
    parser = argparse.ArgumentParser(prog="hcp-hmm",
                                     description="HCP HMM modular pipeline CLI")
    subparser = parser.add_subparsers(dest="cmd", required=True)

#Parcelate
    p_par = subparser.add_parser("parcellate", help="Parcellate dtseries → ptseries using a dlabel atlas")
    p_par.add_argument("--indir", required=True, help="directory with *.dtseries.nii")
    p_par.add_argument("--dlabel", required=True, help="path to *.dlabel.nii atlas")
    p_par.add_argument("--outdir", required=True, help="output directory for *.ptseries.nii")
    p_par.add_argument("--suffix", default="REST_all_Yeo300", help="suffix to append to outputs")
    p_par.add_argument("--method", default="MEAN", help="aggregation method inside parcels (e.g., MEAN)")
    p_par.add_argument("--redo", "--force", dest="redo", action="store_true",
                       help="recompute even if the target ptseries already exists")
    p_par.add_argument("--no-export-labels", action="store_true",
                       help="do not export atlas label table alongside outputs")
    p_par.set_defaults(handler=_handle_parcellate)

#Concat
    p_concat = subparser.add_parser("concat", help="Concatenate *.ptseries.nii into train_X.npy + index")
    p_concat.add_argument("--indir", required=True)
    p_concat.add_argument("--outdir", required=True)
    p_concat.set_defaults(handler=_handle_concat)
    
#Fit HMM
    p_fit = subparser.add_parser("fit", help="Fit HMM and export states/metrics")
    p_fit.add_argument("--in-dir", required=True)
    p_fit.add_argument("--out-dir", required=True)
    p_fit.add_argument("--K", type=int, required=True)
    p_fit.add_argument("--cov", choices=["full","diag","tied"], default="diag")
    p_fit.add_argument("--max-iter", type=int, default=500)
    p_fit.add_argument("--tol", type=float, default=1e-3)
    p_fit.add_argument("--seed", type=int, default=42)
    p_fit.add_argument("--backend", choices=["dynamax_arhmm", "dynamax_slds"], default="dynamax_arhmm")
    p_fit.add_argument("--ar-order", type=int, default=1, help="ARHMM lag order (dynamax_arhmm)")
    p_fit.add_argument("--slds-latent-dim", type=int, default=4, help="SLDS latent dimension (dynamax_slds)")
    p_fit.add_argument("--tr-sec", type=float, default=0.72)
    p_fit.add_argument("--subjects-csv", default=None,
                       help="Optional CSV with subject-level covariates (sex, age, etc.)")
    # Optional surfaces for rendering betas during fit??
    p_fit.add_argument("--atlas-dlabel", default=None)
    p_fit.add_argument("--surface-dir", default=None)
    p_fit.add_argument("--surface-left", default=None)
    p_fit.add_argument("--surface-right", default=None)
    p_fit.add_argument("--surface-left-inflated", default=None)
    p_fit.add_argument("--surface-right-inflated", default=None)
    p_fit.set_defaults(handler=_handle_fit)
#state-mps
    p_maps = subparser.add_parser("state-maps", help="Compute β maps from posteriors and ptseries")
    p_maps.add_argument("--ptseries-dir", required=True)
    p_maps.add_argument("--states-dir", required=True)
    p_maps.add_argument("--out-dir", required=True)
    p_maps.add_argument("--K", type=int, required=True)
    p_maps.add_argument("--chunk", type=int, default=60000)
    p_maps.add_argument("--rcond", type=float, default=1e-6)
    p_maps.add_argument("--render-brainspace", action="store_true",
                        help="render per-subject BrainSpace figures (off by default)")
# Optional rendering
    p_maps.add_argument("--atlas-dlabel", default=None)
    p_maps.add_argument("--surface-dir", default=None)
    p_maps.add_argument("--surface-left", default=None)
    p_maps.add_argument("--surface-right", default=None)
    p_maps.add_argument("--surface-left-inflated", default=None)
    p_maps.add_argument("--surface-right-inflated", default=None)
    p_maps.set_defaults(handler=_handle_state_maps)
#z-score
    p_z = subparser.add_parser("zscore", help="Z-score β maps and export pscalars")
    p_z.add_argument("--ptseries-dir", required=True)
    p_z.add_argument("--betas-dir", required=True)
    p_z.add_argument("--K", type=int, required=True)
    p_z.add_argument("--redo", action="store_true")
    p_z.set_defaults(handler=_handle_zscore)
#group-design
    p_gd = subparser.add_parser("group-design", help="Write FSL design files and subjects_used.csv")
    p_gd.add_argument("--subjects-csv", required=True)
    p_gd.add_argument("--out", required=True)
    p_gd.add_argument("--contrast", default=None, help="Contrast name to write (e.g., 'sex' or an age dummy)")
    p_gd.add_argument("--demean", nargs="*", default=None, help="Columns to demean (e.g., 'sex', 'fd')")
    p_gd.add_argument("--include-fd", action="store_true", help="Force inclusion of mean FD if available.")
    p_gd.add_argument("--no-fd", action="store_true", help="Force exclusion of mean FD even if available.")
    p_gd.set_defaults(handler=_handle_group_design)
#group-merge?
    p_gm = subparser.add_parser("group-merge", help="Merge z-scored subject pscalars into per-state group pscalars")
    p_gm.add_argument("--betas-dir", required=True)
    p_gm.add_argument("--K", type=int, required=True)
    p_gm.add_argument("--out", required=True)
    p_gm.add_argument("--subjects-used", required=True)
    p_gm.add_argument("--parcel-labels-nii", default=None, help="Optional volumetric labels NIfTI (codes 1..P)")
    p_gm.add_argument("--atlas-dlabel", default=None, help="Optional atlas .dlabel.nii to paint dense dscalars")
    p_gm.set_defaults(handler=_handle_group_merge)
#Check-aligmenet
    p_align = subparser.add_parser("check-alignment", help="Verify merged group order matches subjects order")
    p_align.add_argument("--columns-map", required=True)
    p_align.add_argument("--subjects-used", required=True)
    p_align.add_argument("--K", type=int, required=True)
    p_align.set_defaults(handler=_handle_check_alignment)
#stats-repeated-measure
    p_rm = subparser.add_parser("stats-rm", help="Repeated-measures statewise stats")
    p_rm.add_argument("--in-csv", required=True)
    p_rm.add_argument("--K", type=int, required=True)
    p_rm.add_argument("--out", required=True)
    p_rm.add_argument("--n-perm", type=int, default=5000)
    p_rm.set_defaults(handler=_handle_stats_rm)
#Stats-between
    p_b = subparser.add_parser("stats-between", help="Between-sub_handle_runject global stats")
    p_b.add_argument("--in-csv", required=True)
    p_b.add_argument("--out", required=True)
    p_b.add_argument("--n-perm", type=int, default=5000)
    p_b.set_defaults(handler=_handle_stats_between)
#PALM
    p_palm = subparser.add_parser("palm", help="Run PALM on the merged group pscalar and design files")
    p_palm.add_argument("--group-dir", required=True, help="Directory with design.mat/.con/.grp and merged pscalar")
    p_palm.add_argument("--K", type=int, required=True)
    p_palm.add_argument("--n-perm", type=int, default=5000)
    p_palm.add_argument("--two-tailed", action="store_true")
    p_palm.add_argument("--tfce", action="store_true")
    p_palm.add_argument("--tfce2D", action="store_true")
    p_palm.add_argument("--no-cifti", action="store_true")
    p_palm.add_argument("--palm-bin", default="palm")
    p_palm.add_argument("--state", type=int, default=None,
                        help="State index to analyse (omit to run all states)")
    p_palm.set_defaults(handler=_handle_palm)
#Model-select
    p_ms = subparser.add_parser("model-select", help="Sweep K/seed combinations and write model-selection report")
    p_ms.add_argument("--config", required=True, help="YAML with paths + evaluation settings")
    p_ms.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    p_ms.set_defaults(handler=_handle_model_select)

#RUN!!!!
    p_run = subparser.add_parser("run", help="Run the full pipeline from config YAML")
    p_run.add_argument("--config", required=True, help="YAML with paths and parameters")
    p_run.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    p_run.set_defaults(handler=_handle_run)

#HANDLER
    namespace = parser.parse_args(argv)
    handler = getattr(namespace, "handler", None)
    if handler is None:
        parser.error("unknown command")
    return handler(namespace)


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:])) #if 0 success, 1 errors

"""
->it runs main(argv) and then exits the process 
with the code that main returns.
Why this pattern?
-> Ensures proper exit status: 
    main returns an int; 
    raising SystemExit(code) sets the process exit code. 
    Useful for CI/shell scripts.
-> Clean separation: main(argv) is testable and can be called programmatically without exiting the interpreter.

"""
