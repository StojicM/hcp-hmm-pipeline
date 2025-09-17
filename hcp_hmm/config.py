#!/usr/bin/env python3
"""
Configuration helpers for the unified HCP-HMM pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Paths:
    raw_dtseries_dir: Path
    atlas_dlabel: Path
    ptseries_dir: Path
    hmm_dir: Path
    betas_dir: Path
    parcel_labels_dlabel: Path
    subjects_csv: Path
    fd_csv: Optional[Path] = None


@dataclass
class HMMParams:
    K: int = 6
    cov: str = "diag"
    max_iter: int = 500
    tol: float = 1e-3
    seed: int = 42
    tr_sec: float = 0.72


@dataclass
class ParcellateParams:
    method: str = "MEAN"
    suffix: str = "REST_all_Yeo300"


@dataclass
class PipelineConfig:
    paths: Paths
    hmm: HMMParams = field(default_factory=HMMParams)
    parcellate: ParcellateParams = field(default_factory=ParcellateParams)
    palm: Optional["PalmParams"] = None
    logging: Optional["LoggingParams"] = None
    group_design: Optional["GroupDesignParams"] = None

    @staticmethod
    def from_yaml(path: Path) -> "PipelineConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        def P(p: str | None) -> Optional[Path]:
            if p is None:
                return None
            return Path(p)

        paths = Paths(
            raw_dtseries_dir=P(data["paths"]["raw_dtseries_dir"]),
            atlas_dlabel=P(data["paths"]["atlas_dlabel"]),
            ptseries_dir=P(data["paths"]["ptseries_dir"]),
            hmm_dir=P(data["paths"]["hmm_dir"]),
            betas_dir=P(data["paths"]["betas_dir"]),
            parcel_labels_dlabel=P(data["paths"].get("parcel_labels_dlabel", data["paths"]["atlas_dlabel"])) ,
            subjects_csv=P(data["paths"]["subjects_csv"]),
            fd_csv=P(data["paths"].get("fd_csv")),
        )

        hmm = HMMParams(**data.get("hmm", {}))
        parc = ParcellateParams(**data.get("parcellate", {}))
        palm_cfg = data.get("palm")
        palm = PalmParams(**palm_cfg) if isinstance(palm_cfg, dict) else None
        log_cfg = data.get("logging")
        logging_params = LoggingParams(**log_cfg) if isinstance(log_cfg, dict) else None
        gd_cfg = data.get("group_design")
        group_design = GroupDesignParams(**gd_cfg) if isinstance(gd_cfg, dict) else None
        return PipelineConfig(paths=paths, hmm=hmm, parcellate=parc, palm=palm, logging=logging_params, group_design=group_design)


@dataclass
class PalmParams:
    enabled: bool = False
    n_perm: int = 5000
    two_tailed: bool = True
    tfce: bool = False
    tfce2D: bool = False
    cifti: bool = True
    palm_bin: str = "palm"


@dataclass
class LoggingParams:
    format: str = "json"   # "json" or "plain"
    level: str = "INFO"     # e.g., DEBUG, INFO, WARNING


@dataclass
class GroupDesignParams:
    contrast: Optional[str] = None   # e.g., 'sex' or an age dummy label
    demean: list[str] = field(default_factory=list)
