#!/usr/bin/env python3
"""
Parcellation utilities wrapping Connectome Workbench `wb_command`.
"""

from __future__ import annotations

from dataclasses import dataclass #auto-generate boilerplate methods (init, repr, eq) for classes
from pathlib import Path #object-oriented filesystem paths (better than os.path)
from typing import Iterable #type hints for variables, functions, generics (List, Dict, etc.)

from .logger import get_logger
from .workbench import ensure_workbench, export_label_table, run_wb


log = get_logger(__name__)


@dataclass #decorator from dataclasses
class ParcellationConfig:
    indir: Path
    dlabel: Path
    outdir: Path
    method: str = "MEAN"        # MEAN|MEDIAN|etc (as supported by Workbench)
    #ToDo: check suffix:
    suffix: str = "REST_all_Yeo300" #Yeo300.ptseries.nii
    export_labels: bool = True   # export label table next to outdir
    redo: bool = False           # recompute even if output exists


class Parcellator:
    """
    Run `wb_command -cifti-parcellate` across a directory of dtseries, 
    and getting .ptseries.nii per subject, saves it in 
    ptseries_dir: data/derivatives/ptseries (can be set in pipeline.yaml)
    """
    def __init__(self, config: ParcellationConfig):
        self.config = config

    def check_dependencies(self) -> None:
        ensure_workbench()

    def export_labels(self) -> None:
        """Export the full label table once for the chosen atlas (if requested)."""
        if not self.config.export_labels:
            return
        outdir = self.config.outdir
        atlas_stem = self.config.dlabel.name.replace(".dlabel.nii", "")
        labels_dir = outdir.parent
        labels_dir.mkdir(parents=True, exist_ok=True)
        label_table = labels_dir / f"{atlas_stem}_labeltable.tsv"
        export_label_table(self.config.dlabel, label_table)
            
    def iter_dtseries(self) -> Iterable[Path]:
        return sorted(self.config.indir.glob("*.dtseries.nii"))

    def out_path(self, dt_file: Path) -> Path:
        stem = dt_file.name.replace(".dtseries.nii", "")
        return self.config.outdir / f"{stem}_{self.config.suffix}.ptseries.nii"

    def parcellate_one(self, dt_file: Path) -> Path:
        out_pt = self.out_path(dt_file)
        out_pt.parent.mkdir(parents=True, exist_ok=True)
        if out_pt.exists() and not self.config.redo:
            sub_id = dt_file.name.split("_")[0]
            log.info(f"skip_parcellate_existing sub_id={sub_id}", extra={"sub_id": sub_id, "output": str(out_pt)})
            return out_pt
        run_wb([
            "-cifti-parcellate",
            str(dt_file),
            str(self.config.dlabel),
            "COLUMN",
            str(out_pt),
            "-method",
            self.config.method,
        ])
        return out_pt

    def run(self) -> list[Path]:
        self.check_dependencies() #Do you have wb_command
        self.config.outdir.mkdir(parents=True, exist_ok=True)
        self.export_labels() #optional

        files = list(self.iter_dtseries())
        if not files:
            return []

        outputs: list[Path] = []
        n_skipped = 0
        for f in files:
            #ToDo: standardize file labeling and add to docs
            sub_id = f.name.split("_")[0]
            # Only log a full parcellation when we actually run it; otherwise skip is logged in parcellate_one
            if not self.out_path(f).exists() or self.config.redo:
                log.info(f"parcellate sub_id={sub_id}", extra={"input": str(f), "sub_id": sub_id})
            out_pt = self.parcellate_one(f)
            if out_pt.exists() and not self.config.redo:
                # Count as skipped if it pre-existed and we didn't rerun
                dt_exists = f.exists()
                if dt_exists and (self.out_path(f).exists()):
                    n_skipped += 1
            outputs.append(out_pt)
        if n_skipped:
            log.info(f"parcellate_summary skipped_existing={n_skipped} total={len(files)}",
                     extra={"skipped_existing": n_skipped, "total": len(files)})
        return outputs
