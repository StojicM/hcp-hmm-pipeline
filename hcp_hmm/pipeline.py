#!/usr/bin/env python3
"""
One-command pipeline runner orchestrating all steps.

It uses the new OOP parcellation and shells out to the existing
scripts for subsequent steps to minimize code churn while
standardizing inputs/outputs and avoiding duplication.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from .config import PipelineConfig
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
from .merge_covars import merge_covars
from .state_parcel import ParcellateStatesConfig, StateParcellator
from .parcel_stats import ParcelStatsConfig, ParcelStatsRunner


log = get_logger(__name__)


def run(cmd: List[str], cwd: Path | None = None) -> None:
    log.info("run_shell", extra={"cmd": cmd, "cwd": str(cwd) if cwd else None})
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


@dataclass
class Pipeline:
    configs: PipelineConfig
    force: bool = False

    def parcellate_dtseries(self):
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
        ))
        outs = px.run()
        log.info("parcellate_done", extra={"n": len(outs), "out": str(out)})


    def concat_ptseries(self):
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
            tr_sec=self.configs.hmm.tr_sec,
            subjects_csv=self.configs.paths.subjects_csv,
        )
        HMMRunner(Hconfigs).fit_and_export()

    def qc(self):
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
        out_dir = self.configs.paths.betas_dir
        probe = list(out_dir.glob(f"*_state_betas_{self.configs.hmm.K}S.dscalar.nii"))
        if probe and not self.force:
            log.info("skip_state_maps", extra={"count": len(probe), "dir": str(out_dir)})
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        configs = StateMapConfig(
            dtseries_dir=self.configs.paths.raw_dtseries_dir,
            states_dir=self.configs.paths.hmm_dir / "per_subject_states",
            out_dir=out_dir,
            K=self.configs.hmm.K,
        )
        StateMapEstimator(configs).run()

    def zscore_export(self):
        # Produces *_zscored.dscalar.nii
        probe = list(self.configs.paths.betas_dir.glob(f"*_state_betas_{self.configs.hmm.K}S_zscored.dscalar.nii"))
        if probe and not self.force:
            log.info("skip_zscore", extra={"count": len(probe)})
            return
        configs = ZScoreConfig(
            dtseries_dir=self.configs.paths.raw_dtseries_dir,
            betas_dir=self.configs.paths.betas_dir,
            K=self.configs.hmm.K,
        )
        ZScoreExporter(configs).run()


    def group_design(self):
        out_dir = self.configs.paths.betas_dir / "group"
        if (out_dir / "design.mat").exists() and not self.force:
            log.info("skip_design", extra={"path": str(out_dir / 'design.mat')})
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        gd = getattr(self.configs, 'group_design', None)
        GroupDesignBuilder(GroupDesignConfig(
            subjects_csv=self.configs.paths.subjects_csv,
            out_dir=out_dir,
            demean=(gd.demean if gd and getattr(gd, 'demean', None) else None),
            contrast=(gd.contrast if gd and getattr(gd, 'contrast', None) else None),
        )).run()


    def group_merge(self):
        out_dir = self.configs.paths.betas_dir / "group"
        merged = out_dir / f"allsubs_{self.configs.hmm.K}S_zscored.dscalar.nii"
        if merged.exists() and not self.force:
            log.info("skip_group_merge", extra={"path": str(merged)})
            return
        GroupMerger(GroupMergeConfig(
            betas_dir=self.configs.paths.betas_dir,
            K=self.configs.hmm.K,
            out_dir=out_dir,
            subjects_used_csv=out_dir / "subjects_used.csv",
        )).run()


    def check_alignment(self):
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


    def metrics_with_covars(self):
        mdir = self.configs.paths.hmm_dir / "metrics"
        Ktag = f"{self.configs.hmm.K}S"
        base_state = mdir / f"metrics_state_{Ktag}.csv"
        base_global = mdir / f"metrics_global_{Ktag}.csv"
        self._metrics_state_with_covars = base_state
        self._metrics_global_with_covars = base_global

        def _has_covars(path: Path) -> bool:
            if not path.exists():
                return False
            try:
                sample = pd.read_csv(path, nrows=1)
            except Exception as e:
                log.warning("metrics_read_failed", extra={"path": str(path), "err": str(e)})
                return False
            need = {"Subject", "Sex", "AgeGroup"}
            return need.issubset(sample.columns)

        if _has_covars(base_state) and _has_covars(base_global):
            log.info("skip_merge_covars", extra={"reason": "covariates embedded in metrics"})
            return

        out_state, out_glob = merge_covars(
            self.configs.paths.hmm_dir,
            self.configs.paths.subjects_csv,
            self.configs.hmm.K,
            self.configs.paths.fd_csv,
        )
        self._metrics_state_with_covars = out_state
        self._metrics_global_with_covars = out_glob


    def stats_rm_and_between(self):
        # Repeated measures (statewise)
        Ktag = f"{self.configs.hmm.K}S"
        in_state = Path(getattr(
            self,
            "_metrics_state_with_covars",
            self.configs.paths.hmm_dir / "metrics" / f"metrics_state_{self.configs.hmm.K}S_with_covars.csv"
        ))
        if not in_state.exists():
            alt_state = self.configs.paths.hmm_dir / "metrics" / f"metrics_state_{Ktag}.csv"
            if alt_state.exists():
                in_state = alt_state
        out_state = self.configs.paths.hmm_dir / "metrics" / f"stats_state_{self.configs.hmm.K}S_rm.csv"
        if (not out_state.exists()) or self.force:
            StatsRM(StatsRMConfig(in_csv=in_state, K=self.configs.hmm.K, out_csv=out_state)).run()
        else:
            log.info("skip_stats_state_rm", extra={"path": str(out_state)})

        # Between-subject (global)
        in_global = Path(getattr(
            self,
            "_metrics_global_with_covars",
            self.configs.paths.hmm_dir / "metrics" / f"metrics_global_{self.configs.hmm.K}S_with_covars.csv"
        ))
        if not in_global.exists():
            alt_global = self.configs.paths.hmm_dir / "metrics" / f"metrics_global_{Ktag}.csv"
            if alt_global.exists():
                in_global = alt_global
        out_global = self.configs.paths.hmm_dir / "metrics" / f"stats_global_{self.configs.hmm.K}S.csv"
        if (not out_global.exists()) or self.force:
            StatsBetween(StatsBetweenConfig(in_csv=in_global, out_csv=out_global)).run()
        else:
            log.info("skip_stats_global_between", extra={"path": str(out_global)})
    

    def palm(self):
        out_dir = self.configs.paths.betas_dir / "group" / "palm"
        # If a success sentinel exists and not forced, skip
        sentinel = out_dir / f"palm_{self.configs.hmm.K}S.ok"
        if sentinel.exists() and not self.force:
            log.info("skip_palm", extra={"dir": str(out_dir)})
            return
        params = self.configs.palm
        assert params is not None
        PalmRunner(PalmConfig(
            group_dir=self.configs.paths.betas_dir / "group",
            K=self.configs.hmm.K,
            n_perm=params.n_perm,
            two_tailed=params.two_tailed,
            tfce=params.tfce,
            tfce2D=params.tfce2D,
            cifti=params.cifti,
            palm_bin=params.palm_bin,
        )).run()


    def run_all(self):
        # Apply logging preferences from config (if any)
        if getattr(self.configs, 'logging', None):
            log_format = getattr(self.configs.logging, 'format', None)
            log_level = getattr(self.configs.logging, 'level', None)
            configure_logging(log_format, log_level)
        #the whole pipeline steps 
        self.parcellate_dtseries() #Uses Parcellator from the parcellation.py
        self.concat_ptseries() #Uses PtSeriesConcatenator from ptseries.py
        self.fit_hmm() #Uses HMMRunner(HMMConfig) from hmm_fit.py 
        self.qc() #Uses QCReporter from the qc.py
        self.state_maps() #StateMapEstimator(StateMapEstimator)
        self.zscore_export()
        self.group_design()
        self.group_merge()
        self.check_alignment()
        self.metrics_with_covars()
        self.stats_rm_and_between()
        self.parcellate_states()
        self.parcel_stats()
        """
        Optional PALM, requires Octave/Matlab with PALM toolbox;
        With this script you don't go through truble of opening Octave or Matlab
        just having them installed in enough
        """
        if getattr(self.configs, 'palm', None) and self.configs.palm.enabled:
            self.palm()




if __name__ == "__main__":    
    # from .config import PipelineConfig
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    configs = PipelineConfig.from_yaml(Path(args.config))
    pipe = Pipeline(configs, force=args.force)
    pipe.run_all()
