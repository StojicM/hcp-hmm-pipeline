#!/usr/bin/env python3
"""Configuration dataclasses and YAML loader for the pipeline.

This module defines typed containers for paths and parameters and a
`PipelineConfig.from_yaml` helper that parses `pipeline.yaml` into those
structures, performing light path normalization (string → `Path`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# import yaml #Prebacio u pipeline.py


@dataclass  # Instead of writing the whole class, use a wrapper...
class Paths:
    """File system locations for inputs/outputs used across the pipeline."""
    raw_dtseries_dir: Path
    atlas_dlabel: Path
    ptseries_dir: Path
    hmm_dir: Path
    betas_dir: Path
    parcel_labels_dlabel: Path
    subjects_csv: Path
    fd_csv: Optional[Path] = None
    # Optional volumetric labels NIfTI with integer codes 1..P matching parcel order
    parcel_labels_nii: Optional[Path] = None
    surface_dir: Optional[Path] = None
    surface_left: Optional[Path] = None
    surface_right: Optional[Path] = None
    surface_left_inflated: Optional[Path] = None
    surface_right_inflated: Optional[Path] = None


@dataclass
class HMMParams:
    """Hyperparameters for HMM fitting.

    `K` states, covariance type, iteration cap, convergence tolerance,
    RNG seed, and TR in seconds used for time-derived summaries.
    """
    K: int = 6
    cov: str = "diag"
    max_iter: int = 500
    tol: float = 1e-3
    seed: int = 42
    tr_sec: float = 0.72


@dataclass
class ParcellateParams:
    """Parcellation options (aggregation method and output suffix)."""
    method: str = "MEAN"
    suffix: str = "REST_all_Yeo300"
 

# @dataclass
# class PipelineConfig:
#     """Top-level configuration object combining all sub-configs."""
#     paths: Paths
#     hmm: HMMParams = field(default_factory=HMMParams)
#     parcellate: ParcellateParams = field(default_factory=ParcellateParams)
#     palm: Optional["PalmParams"] = None
#     stats: Optional["StatsParams"] = None
#     logging: Optional["LoggingParams"] = None
#     group_design: Optional["GroupDesignParams"] = None

#     @staticmethod
#     def from_yaml(path: Path) -> "PipelineConfig":
#         """Parse YAML file into a `PipelineConfig`.

#         Performs minimal type normalization for paths and optional sections.
#         """
#         with open(path, "r") as f:
#             data = yaml.safe_load(f)

#         def P(p: str | None) -> Optional[Path]:
#             if p is None:
#                 return None
#             return Path(p)

#         paths = Paths(
#             raw_dtseries_dir=P(data["paths"]["raw_dtseries_dir"]),
#             atlas_dlabel=P(data["paths"]["atlas_dlabel"]),
#             ptseries_dir=P(data["paths"]["ptseries_dir"]),
#             hmm_dir=P(data["paths"]["hmm_dir"]),
#             betas_dir=P(data["paths"]["betas_dir"]),
#             parcel_labels_dlabel=P(data["paths"].get("parcel_labels_dlabel", data["paths"]["atlas_dlabel"])) ,
#             subjects_csv=P(data["paths"]["subjects_csv"]),
#             fd_csv=P(data["paths"].get("fd_csv")),
#             parcel_labels_nii=P(data["paths"].get("parcel_labels_nii")),
#             surface_dir=P(data["paths"].get("surface_dir")),
#             surface_left=P(data["paths"].get("surface_left")),
#             surface_right=P(data["paths"].get("surface_right")),
#             surface_left_inflated=P(data["paths"].get("surface_left_inflated")),
#             surface_right_inflated=P(data["paths"].get("surface_right_inflated")),
#         )

#         hmm = HMMParams(**data.get("hmm", {}))
#         parc = ParcellateParams(**data.get("parcellate", {}))
#         palm_cfg = data.get("palm")
#         palm = PalmParams(**palm_cfg) if isinstance(palm_cfg, dict) else None
#         stats_cfg = data.get("stats")
#         stats = StatsParams(**stats_cfg) if isinstance(stats_cfg, dict) else None
#         log_cfg = data.get("logging")
#         logging_params = LoggingParams(**log_cfg) if isinstance(log_cfg, dict) else None
#         gd_cfg = data.get("group_design")
#         if isinstance(gd_cfg, dict):
#             gd_cfg = dict(gd_cfg)
#             if "subject_order_file" in gd_cfg and gd_cfg["subject_order_file"] is not None:
#                 gd_cfg["subject_order_file"] = P(gd_cfg["subject_order_file"])
#             group_design = GroupDesignParams(**gd_cfg)
#         else:
#             group_design = None
#         return PipelineConfig(paths=paths, hmm=hmm, parcellate=parc, palm=palm, stats=stats, logging=logging_params, group_design=group_design)


@dataclass
class PalmParams:
    """Parameters for optional PALM group analysis."""
    enabled: bool = False
    n_perm: int = 5000
    two_tailed: bool = True
    ise: bool = False
    tfce: bool = False
    tfce2D: bool = False
    cifti: bool = True
    palm_bin: str = "palm"
    use_zscored: bool = True


@dataclass
class StatsParams:
    """Permutation counts for repeated-measures and global stats."""
    # Number of permutations for stats modules
    n_perm_rm: int = 5000        # repeated-measures statewise tests
    n_perm_between: int = 5000   # between-subject global tests


@dataclass
class LoggingParams:
    """Logging preferences used by the pipeline runner."""
    format: str = "plain"   # "json" or "plain"
    level: str = "INFO"     # e.g., DEBUG, INFO, WARNING


@dataclass
class GroupDesignParams:
    """Group design settings for building FSL matrices and stacks."""
    contrast: Optional[str] = None   # e.g., 'sex' or an age dummy label
    demean: list[str] = field(default_factory=list)
    include_fd: Optional[bool] = None
    stacking: str = "subject-major"
    subject_order_file: Optional[Path] = None


@dataclass
class SummaryParams:
    """Toggle for building the final summary."""
    build_summary: bool = False
    # Optional centroid overlay on parcel 2D plots: none | signed | abs | positive
    layout_centroid_mode: str = "none"
