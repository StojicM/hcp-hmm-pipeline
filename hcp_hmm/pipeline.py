#!/usr/bin/env python3
"""
MAIN: One-command pipeline to run all steps, gets initiated by cli.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import argparse
from dataclasses import dataclass, field #dodao field
from pathlib import Path
from typing import List

# from .config import PipelineConfig #zamenio za ovo (jer treba PipelineConfig-u):
# from .config import HMMParams, ParcellateParams, PalmParams, StatsParams
# from .config import LoggingParams, GroupDesignParams, Paths
from .config import * #treba PipelineConfig-u svih 7 klassa
 
from .parcellation import ParcellationConfig, Parcellator
from .ptseries import PtConcatConfig, PtSeriesConcatenator
from .hmm_fit import HMMConfig as _HMMConfig, HMMRunner
from .state_maps import StateMapConfig, StateMapEstimator
from .zscore import ZScoreConfig, ZScoreExporter
from .logger import get_logger, configure_logging
from .group_design import GroupDesignConfig, GroupDesignBuilder
from .group_merge import GroupMergeConfig, GroupMerger
from .alignment import AlignmentConfig, AlignmentChecker
from .stats_rm import StatsRMConfig, StatsRM
from .stats_between import StatsBetweenConfig, StatsBetween
from .palm import PalmConfig, PalmRunner
from .qc import QCConfig, QCReporter
from .state_parcel import ParcellateStatesConfig, StateParcellator
from .parcel_stats import ParcelStatsConfig, ParcelStatsRunner

#dodao:
import yaml

log = get_logger(__name__)

""" Nema potrebe, #ToDel
def run(cmd: List[str], cwd: Path | None = None) -> None:
    ###Run a shell command with logging and `check=True`.
    ###Used for rare shell-outs where a Python wrapper is not available.
    log.info("run_shell", extra={"cmd": cmd, "cwd": str(cwd) if cwd else None})
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)
""" 

@dataclass
class PipelineConfig:
    paths: Paths
    hmm: HMMParams = field(default_factory=HMMParams)
    parcellate: ParcellateParams = field(default_factory=ParcellateParams)
    palm: Optional["PalmParams"] = None
    stats: Optional["StatsParams"] = None
    logging: Optional["LoggingParams"] = None
    group_design: Optional["GroupDesignParams"] = None
    evaluation: Optional["EvaluationParams"] = None

    @staticmethod
    def from_yaml(path: Path) -> "PipelineConfig":
        """Parse YAML file into a `PipelineConfig`.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # def P(p: str | None) -> Optional[Path]:
        #     if p is None:
        #         return None
        #     return Path(p)
        # Bolje zameni sa:
        P = lambda p: Path(p) if p is not None else None

        paths = Paths(
            raw_dtseries_dir=P(data["paths"]["raw_dtseries_dir"]),
            atlas_dlabel=P(data["paths"]["atlas_dlabel"]),
            ptseries_dir=P(data["paths"]["ptseries_dir"]),
            hmm_dir=P(data["paths"]["hmm_dir"]),
            betas_dir=P(data["paths"]["betas_dir"]),
            parcel_labels_dlabel=P(data["paths"].get("parcel_labels_dlabel", data["paths"]["atlas_dlabel"])) ,
            subjects_csv=P(data["paths"]["subjects_csv"]),
            fd_csv=P(data["paths"].get("fd_csv")),
            parcel_labels_nii=P(data["paths"].get("parcel_labels_nii")),
            surface_dir=P(data["paths"].get("surface_dir")),
            surface_left=P(data["paths"].get("surface_left")),
            surface_right=P(data["paths"].get("surface_right")),
            surface_left_inflated=P(data["paths"].get("surface_left_inflated")),
            surface_right_inflated=P(data["paths"].get("surface_right_inflated")),
        )

        hmm = HMMParams(**data.get("hmm", {}))
        parc = ParcellateParams(**data.get("parcellate", {}))
        palm_cfg = data.get("palm")
        palm = PalmParams(**palm_cfg) if isinstance(palm_cfg, dict) else None
        stats_cfg = data.get("stats")
        stats = StatsParams(**stats_cfg) if isinstance(stats_cfg, dict) else None
        log_cfg = data.get("logging")
        logging_params = LoggingParams(**log_cfg) if isinstance(log_cfg, dict) else None
        gd_cfg = data.get("group_design")
        if isinstance(gd_cfg, dict):
            gd_cfg = dict(gd_cfg)
            if "subject_order_file" in gd_cfg and gd_cfg["subject_order_file"] is not None:
                gd_cfg["subject_order_file"] = P(gd_cfg["subject_order_file"])
            group_design = GroupDesignParams(**gd_cfg)
        else:
            group_design = None
        eval_cfg = data.get("evaluation")
        evaluation = None
        if isinstance(eval_cfg, dict):
            ec = dict(eval_cfg)
            out_dir = P(ec.get("out_dir")) if ec.get("out_dir") is not None else None
            K_values = ec.get("K_values", [])
            if isinstance(K_values, int):
                K_values = [K_values]
            seeds = ec.get("seeds", [])
            if isinstance(seeds, int):
                seeds = [seeds]
            junk_cfg = ec.get("junk")
            indecision_cfg = ec.get("indecision")
            clone_cfg = ec.get("clone")
            reliability_cfg = ec.get("reliability")
            evaluation = EvaluationParams(
                enabled=bool(ec.get("enabled", False)),
                K_values=[int(v) for v in (K_values or [])],
                seeds=[int(v) for v in (seeds or [])],
                out_dir=out_dir,
                junk=EvalJunkParams(**junk_cfg) if isinstance(junk_cfg, dict) else EvalJunkParams(),
                indecision=EvalIndecisionParams(**indecision_cfg) if isinstance(indecision_cfg, dict) else EvalIndecisionParams(),
                clone=EvalCloneParams(**clone_cfg) if isinstance(clone_cfg, dict) else EvalCloneParams(),
                reliability=EvalReliabilityParams(**reliability_cfg) if isinstance(reliability_cfg, dict) else EvalReliabilityParams(),
            )
        elif isinstance(eval_cfg, bool):
            evaluation = EvaluationParams(enabled=bool(eval_cfg))

        return PipelineConfig(
            paths=paths,
            hmm=hmm,
            parcellate=parc,
            palm=palm,
            stats=stats,
            logging=logging_params,
            group_design=group_design,
            evaluation=evaluation,
        )


@dataclass
class Pipeline:
    configs: PipelineConfig
    force: bool = False #default value; True if you want to overwrite everything and start over

    def parcellate_dtseries(self): #OK
        """Parcellate dtseries to ptseries using the configured atlas."""
        out = self.configs.paths.ptseries_dir
        dt = list(Path(self.configs.paths.raw_dtseries_dir).glob("*.dtseries.nii"))
        pt = list(out.glob(f"*_{self.configs.parcellate.suffix}.ptseries.nii")) if out.exists() else []
        if (not self.force) and out.exists() and len(pt) >= len(dt) and len(dt) > 0:
            log.info("skip_parcellate", extra={"ptseries": len(pt), "dtseries": len(dt)})
            return
        px = Parcellator(ParcellationConfig(
            indir=self.configs.paths.raw_dtseries_dir,
            dlabel=self.configs.paths.atlas_dlabel,
            outdir=self.configs.paths.ptseries_dir,
            method=self.configs.parcellate.method,
            suffix=self.configs.parcellate.suffix,
            export_labels=True,
            redo=self.force,
        ))
        outs = px.run()
        log.info("parcellate_done", extra={"n": len(outs), "out": str(out)})


    def concat_ptseries(self):
        """Concatenate ptseries to train_X.npy and write subjects_index.csv."""
        hmm_dir = self.configs.paths.hmm_dir
        Xnpy = hmm_dir / "train_X.npy"
        if Xnpy.exists() and not self.force:
            log.info("skip_concat", extra={"path": str(Xnpy)})
            return
        conc = PtSeriesConcatenator(PtConcatConfig(
            indir=self.configs.paths.ptseries_dir,
            outdir=hmm_dir,
        ))
        conc.run()


    def fit_hmm(self):
        """Fit the HMM and export model, per-subject states, and metrics."""
        out = self.configs.paths.hmm_dir / "model.joblib"
        if out.exists() and not self.force:
            log.info("skip_fit", extra={"path": str(out)})
            return
        Hconfigs = _HMMConfig(
            in_dir=self.configs.paths.hmm_dir,
            out_dir=self.configs.paths.hmm_dir,
            K=self.configs.hmm.K,
            cov=self.configs.hmm.cov,
            max_iter=self.configs.hmm.max_iter,
            tol=self.configs.hmm.tol,
            seed=self.configs.hmm.seed,
            backend=getattr(self.configs.hmm, "backend", "dynamax_arhmm"),
            tr_sec=self.configs.hmm.tr_sec,
            ar_order=getattr(self.configs.hmm, "ar_order", 1),
            slds_latent_dim=getattr(self.configs.hmm, "slds_latent_dim", 4),
            subjects_csv=self.configs.paths.subjects_csv,
            atlas_dlabel=self.configs.paths.parcel_labels_dlabel,
            surface_dir=self.configs.paths.surface_dir,
            surface_left=self.configs.paths.surface_left,
            surface_right=self.configs.paths.surface_right,
            surface_left_inflated=self.configs.paths.surface_left_inflated,
            surface_right_inflated=self.configs.paths.surface_right_inflated,
        )
        HMMRunner(Hconfigs).fit_and_export()

    def model_selection(self):
        """Sweep K/seed combinations and write model-selection reports (optional)."""
        eval_cfg = getattr(self.configs, "evaluation", None)
        if not eval_cfg or not getattr(eval_cfg, "enabled", False):
            log.info("skip_model_selection", extra={"reason": "disabled"})
            return

        out_dir = getattr(eval_cfg, "out_dir", None) or (self.configs.paths.hmm_dir / "model_selection")
        from .model_selection import ModelSelectionConfig, ModelSelectionRunner

        ModelSelectionRunner(ModelSelectionConfig(
            in_dir=self.configs.paths.hmm_dir,
            out_dir=Path(out_dir),
            hmm=self.configs.hmm,
            evaluation=eval_cfg,
            subjects_csv=self.configs.paths.subjects_csv,
            atlas_dlabel=self.configs.paths.parcel_labels_dlabel,
            surface_dir=self.configs.paths.surface_dir,
            surface_left=self.configs.paths.surface_left,
            surface_right=self.configs.paths.surface_right,
            surface_left_inflated=self.configs.paths.surface_left_inflated,
            surface_right_inflated=self.configs.paths.surface_right_inflated,
            force=self.force,
        )).run()

    def run_model_selection(self):
        """Run minimal preprocessing and then K/seed model selection."""
        self.parcellate_dtseries()
        self.concat_ptseries()
        self.model_selection()


    def qc(self):
        """Build a lightweight QC report if missing."""
        qc_dir = self.configs.paths.hmm_dir / "qc"
        rep = qc_dir / f"report_{self.configs.hmm.K}S.html"
        if rep.exists() and not self.force:
            log.info("skip_qc", extra={"path": str(rep)})
            return
        # Look for a simple labels TSV next to ptseries_dir parent; otherwise skip fingerprints
        atlas_stem = self.configs.paths.atlas_dlabel.name.replace(".dlabel.nii", "")
        labels_tsv = self.configs.paths.ptseries_dir.parent / f"{atlas_stem}_labels.tsv"
        labels_path = labels_tsv if labels_tsv.exists() else None
        fd_csv = self.configs.paths.fd_csv if (self.configs.paths.fd_csv and Path(self.configs.paths.fd_csv).exists()) else None
        QCReporter(QCConfig(
            hmm_dir=self.configs.paths.hmm_dir,
            K=self.configs.hmm.K,
            atlas_labels=labels_path,
            fd_csv=fd_csv,
        )).run()


    def state_maps(self):
        """Estimate subject state betas (pscalares) and save them to disk."""
        out_dir = self.configs.paths.betas_dir
        probe = list(out_dir.glob(f"*_state_betas_{self.configs.hmm.K}S.pscalar.nii"))
        if probe and not self.force:
            log.info("skip_state_maps", extra={"count": len(probe), "dir": str(out_dir)})
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        configs = StateMapConfig(
            ptseries_dir=self.configs.paths.ptseries_dir,
            states_dir=self.configs.paths.hmm_dir / "per_subject_states",
            out_dir=out_dir,
            K=self.configs.hmm.K,
        )
        StateMapEstimator(configs).run()


    def zscore_export(self):
        """Export z-scored versions of subject betas."""
        # Makes *_zscored.pscalar.nii
        probe = list(self.configs.paths.betas_dir.glob(f"*_state_betas_{self.configs.hmm.K}S_zscored.pscalar.nii"))
        if probe and not self.force:
            log.info("skip_zscore", extra={"count": len(probe)})
            return
        configs = ZScoreConfig(
            ptseries_dir=self.configs.paths.ptseries_dir,
            betas_dir=self.configs.paths.betas_dir,
            K=self.configs.hmm.K,
        )
        ZScoreExporter(configs).run()


    def group_design(self):
        """Create FSL design matrices for group analyses (PALM)."""
        out_dir = self.configs.paths.betas_dir / "group"
        if (out_dir / "design.mat").exists() and not self.force:
            log.info("skip_design", extra={"path": str(out_dir / 'design.mat')})
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        gd = getattr(self.configs, 'group_design', None)
        demean = (gd.demean if gd and getattr(gd, 'demean', None) else None)
        contrast = (gd.contrast if gd and getattr(gd, 'contrast', None) else None)
        stacking = getattr(gd, 'stacking', "subject-major") if gd else "subject-major"
        subject_order = getattr(gd, 'subject_order_file', None) if gd else None
        include_fd = getattr(gd, 'include_fd', None) if gd else None

        GroupDesignBuilder(GroupDesignConfig(
            subjects_csv=self.configs.paths.subjects_csv,
            out_dir=out_dir,
            demean=demean,
            contrast=contrast,
            K=self.configs.hmm.K,
            stacking=stacking,
            subject_order_file=subject_order,
            include_fd=include_fd,
        )).run()


    def group_merge(self):
        """Merge subject betas into group stacks and write input lists."""
        out_dir = self.configs.paths.betas_dir / "group"
        Ktag = f"{self.configs.hmm.K}S"
        expected = [out_dir / f"allsubs_state{s}_{Ktag}_zscored.pscalar.nii" for s in range(self.configs.hmm.K)]
        inputs_list = out_dir / f"inputs_{Ktag}_pscalar.txt"
        stack_path = out_dir / f"allsubs_states_{Ktag}_zscored.pscalar.nii"
        if (
            not self.force
            and expected
            and all(p.exists() for p in expected)
            and inputs_list.exists()
            and stack_path.exists()
        ):
            log.info(
                "skip_group_merge",
                extra={
                    "paths": [str(p) for p in expected],
                    "inputs": str(inputs_list),
                    "stack": str(stack_path),
                },
            )
            return
        GroupMerger(GroupMergeConfig(
            betas_dir=self.configs.paths.betas_dir,
            K=self.configs.hmm.K,
            out_dir=out_dir,
            subjects_used_csv=out_dir / "subjects_used.csv",
            parcel_labels_nii=getattr(self.configs.paths, 'parcel_labels_nii', None),
            atlas_dlabel=self.configs.paths.parcel_labels_dlabel,
        )).run()


    def check_alignment(self):
        """Verify the merged group order matches the subjects used list."""
        out_dir = self.configs.paths.betas_dir / "group"
        mapcsv = out_dir / "columns_map.csv"
        if not mapcsv.exists():
            log.info("skip_check_alignment", extra={"reason": "no columns_map.csv"})
            return
        ok = AlignmentChecker(AlignmentConfig(
            columns_map_csv=mapcsv,
            subjects_used_csv=out_dir / "subjects_used.csv",
            K=self.configs.hmm.K,
        )).check()
        if not ok:
            raise SystemExit("Alignment check failed; see logs for details.")


    def parcellate_states(self):
        """Map state betas to parcel label tables (tabular summaries)."""
        out_dir = self.configs.paths.betas_dir / "parcel"
        probe = list(out_dir.glob("*_z.parc.tsv"))
        if probe and not self.force:
            log.info("skip_parcellate_states", extra={"count": len(probe), "dir": str(out_dir)})
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        configs = ParcellateStatesConfig(
            in_dir=self.configs.paths.betas_dir,
            out_dir=out_dir,
            labels_dlabel=self.configs.paths.parcel_labels_dlabel,
            K=self.configs.hmm.K,
        )
        StateParcellator(configs).run()


    def parcel_stats(self):
        """Compute simple parcel-wise statistics over state betas."""
        out_dir = self.configs.paths.betas_dir / "parcel" / "stats"
        # Skip if outputs already present (only if dir exists)
        if out_dir.exists() and any(out_dir.iterdir()) and not self.force:
            log.info("skip_parcel_stats", extra={"dir": str(out_dir)})
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        labels_tsv = (self.configs.paths.betas_dir / "parcel" / "parcel_labels.tsv")
        ParcelStatsRunner(ParcelStatsConfig(
            parcel_dir=self.configs.paths.betas_dir / "parcel",
            subjects_used_csv=self.configs.paths.betas_dir / "group" / "subjects_used.csv",
            K=self.configs.hmm.K,
            labels_tsv=labels_tsv,
            out_dir=out_dir,
        )).run()


    def stats_rm_and_between(self):
        """Run repeated-measures (statewise) and between-subject (global) stats."""
        # Repeated measures (statewise)
        Ktag = f"{self.configs.hmm.K}S"
        mdir = self.configs.paths.hmm_dir / "metrics"
        in_state = mdir / f"metrics_state_{Ktag}.csv"
        if not in_state.exists():
            raise FileNotFoundError(in_state)
        out_state = self.configs.paths.hmm_dir / "metrics" / f"stats_state_{self.configs.hmm.K}S_rm.csv"
        if (not out_state.exists()) or self.force:
            n_perm_rm = getattr(self.configs, 'stats', None).n_perm_rm if getattr(self.configs, 'stats', None) else 5000
            StatsRM(StatsRMConfig(in_csv=in_state, K=self.configs.hmm.K, out_csv=out_state, n_perm=int(n_perm_rm))).run()
        else:
            log.info("skip_stats_state_rm", extra={"path": str(out_state)})

        # Between-subject (global)
        in_global = mdir / f"metrics_global_{Ktag}.csv"
        if not in_global.exists():
            raise FileNotFoundError(in_global)
        out_global = self.configs.paths.hmm_dir / "metrics" / f"stats_global_{self.configs.hmm.K}S.csv"
        if (not out_global.exists()) or self.force:
            n_perm_between = getattr(self.configs, 'stats', None).n_perm_between if getattr(self.configs, 'stats', None) else 5000
            StatsBetween(StatsBetweenConfig(in_csv=in_global, out_csv=out_global, n_perm=int(n_perm_between))).run()
        else:
            log.info("skip_stats_global_between", extra={"path": str(out_global)})

    def palm(self):
        """Run PALM per state if enabled in config."""
        params = self.configs.palm
        assert params is not None
        group_dir = self.configs.paths.betas_dir / "group"
        palm_dir = group_dir / "palm"
        Ktag = f"{self.configs.hmm.K}S"
        subj_mat = group_dir / "design_subjects.mat"
        subj_con = group_dir / "design_subjects.con"
        subj_grp = group_dir / "design_subjects.grp"
        for required in (subj_mat, subj_con, subj_grp):
            if not required.exists():
                raise FileNotFoundError(required)

        ran_any = False
        for state in range(self.configs.hmm.K):
            sentinel = palm_dir / f"palm_{Ktag}_state{state}.ok"
            if sentinel.exists() and not self.force:
                log.info("skip_palm_state", extra={"state": state, "sentinel": str(sentinel)})
                continue
            PalmRunner(PalmConfig(
                group_dir=group_dir,
                K=self.configs.hmm.K,
                n_perm=params.n_perm,
                two_tailed=params.two_tailed,
                ise=getattr(params, 'ise', False),
                use_zscored=getattr(params, 'use_zscored', True),
                tfce=params.tfce,
                tfce2D=params.tfce2D,
                cifti=params.cifti,
                palm_bin=params.palm_bin,
                state=state,
                subject_design_mat=subj_mat,
                subject_design_con=subj_con,
                subject_design_grp=subj_grp,
            )).run()
            ran_any = True
        if not ran_any and not self.force:
            log.info("skip_palm", extra={"reason": "all state analyses already completed", "dir": str(palm_dir)})

    def run_all(self):
        """Run all pipeline stages in order, respecting `force` and config."""
        # Apply logging preferences from config (if any)
        if getattr(self.configs, 'logging', None):
            log_format = getattr(self.configs.logging, 'format', None)
            log_level = getattr(self.configs.logging, 'level', None)
            configure_logging(log_format, log_level)
        #the whole pipeline steps 
        ############# OVO DA NE PONAVLJA PARCELACIJU ####################
        self.parcellate_dtseries() #Uses Parcellator from the parcellation.py
        ############# UNCOMMENT AKO HOĆEŠ PARCELACIJU ###################
        self.concat_ptseries() #Uses PtSeriesConcatenator from ptseries.py
        self.fit_hmm() #Uses HMMRunner(HMMConfig) from hmm_fit.py 
        self.model_selection()
        self.qc() #Uses QCReporter from the qc.py
        self.state_maps() #StateMapEstimator(StateMapConfig)
        self.zscore_export() #<
        self.group_design() #<
        self.group_merge() #<
        self.check_alignment() #<
        self.stats_rm_and_between() #<
        self.parcellate_states() #
        self.parcel_stats() #
        
        ## Optional: 
        #1 Run PALM with this script or run it manually in MatLab/Octave
        if getattr(self.configs, 'palm', None) and self.configs.palm.enabled:
            self.palm()
        
""" Nepotrebno jer cli.py calls run_all() so keep it simple by removing the option
to run pipeline.py as a script since cli.py preps and initiates the pipeline
#ToDel"""
if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--config", required=True)
    argument_parser.add_argument("--force", action="store_true")
    arguments = argument_parser.parse_args()
    configs = PipelineConfig.from_yaml(Path(arguments.config))
    pipe = Pipeline(configs, force=arguments.force)
    pipe.run_all()
