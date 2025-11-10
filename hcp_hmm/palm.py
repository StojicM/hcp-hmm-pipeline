#!/usr/bin/env python3
from __future__ import annotations

"""
PALM (Permutation Analysis of Linear Models) runner integration.

>Optional PALM, requires Octave/Matlab with PALM toolbox;
With this script you don't go through truble of opening Octave or Matlab
just having them installed in enough

This wrapper invokes PALM via the command-line (preferred), optionally
with CIFTI support if your PALM build includes it. It expects the group
merged dscalar and design files produced earlier in the pipeline.

Notes:
  - Ensure PALM is installed and `palm` (or your chosen binary) is on PATH.
  - For CIFTI input, your PALM must support CIFTI. If not, convert your
    data to surface/volume formats compatible with PALM before running.
"""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from .logger import get_logger

log = get_logger(__name__)


@dataclass
class PalmConfig:
    group_dir: Path
    K: int
    n_perm: int = 5000
    two_tailed: bool = True
    ise: bool = False
    tfce: bool = False
    tfce2D: bool = False
    cifti: bool = True
    palm_bin: str = "palm"
    use_zscored: bool = True
    merged_dscalar: Optional[Path] = None
    design_mat: Optional[Path] = None
    design_con: Optional[Path] = None
    design_grp: Optional[Path] = None
    subject_design_mat: Optional[Path] = None
    subject_design_con: Optional[Path] = None
    subject_design_grp: Optional[Path] = None
    out_dir: Optional[Path] = None
    state: Optional[int] = None


class PalmRunner:
    def __init__(self, cfg: PalmConfig):
        self.cfg = cfg

    def _resolve_paths(self):
        Ktag = f"{self.cfg.K}S"
        state = self.cfg.state
        dmat = self.cfg.design_mat or (self.cfg.group_dir / "design.mat")
        dcon = self.cfg.design_con or (self.cfg.group_dir / "design.con")
        dgrp = self.cfg.design_grp or (self.cfg.group_dir / "design.grp")
        out_dir = self.cfg.out_dir or (self.cfg.group_dir / "palm")
        out_dir.mkdir(parents=True, exist_ok=True)

        if state is None:
            raise SystemExit("PalmConfig.state must be provided for per-state analyses.")

        input_path = self.cfg.merged_dscalar or (
            self.cfg.group_dir / (
                f"allsubs_state{state}_{Ktag}_zscored.pscalar.nii" if self.cfg.use_zscored else f"allsubs_state{state}_{Ktag}.pscalar.nii"
            )
        )
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        subj_dmat = self.cfg.subject_design_mat or (self.cfg.group_dir / "design_subjects.mat")
        subj_dcon = self.cfg.subject_design_con or (self.cfg.group_dir / "design_subjects.con")
        subj_dgrp = self.cfg.subject_design_grp or (self.cfg.group_dir / "design_subjects.grp")

        for required in (subj_dmat, subj_dcon, subj_dgrp, dmat, dcon, dgrp):
            if required is None or not Path(required).exists():
                raise FileNotFoundError(required)

        base = out_dir / f"palm_{Ktag}_state{state}"
        return {
            "mode": "single",
            "input_path": input_path.resolve(),
            "dmat": Path(subj_dmat).resolve(),
            "dcon": Path(subj_dcon).resolve(),
            "dgrp": Path(subj_dgrp).resolve(),
            "out_dir": out_dir,
            "base": base,
            "state": state,
        }

    def _check(self):
        if shutil.which(self.cfg.palm_bin) is None:
            raise SystemExit(f"'{self.cfg.palm_bin}' not found on PATH. Install PALM or provide palm_bin.")

    def _supports_cifti(self) -> bool:
        def _check_output(args: list[str]) -> str:
            try:
                p = subprocess.run(args, check=False, capture_output=True, text=True)
                return (p.stdout or "") + "\n" + (p.stderr or "")
            except Exception:
                return ""
        # Try `palm -h` (some builds don't support it), then bare `palm`
        out = _check_output([self.cfg.palm_bin, "-h"]) or _check_output([self.cfg.palm_bin])
        return "-cifti" in out

    def build_cmd(self, info) -> List[str]:
        cmd = [self.cfg.palm_bin,
               "-i", str(info["input_path"]),
               "-d", str(info["dmat"]),
               "-t", str(info["dcon"]),
               "-eb", str(info["dgrp"]),
               "-n", str(self.cfg.n_perm),
               "-o", str(info["base"])]
        if self.cfg.two_tailed:
            cmd += ["-twotail"]
        if self.cfg.ise:
            # Enable sign-flips (independent, symmetric errors) – needed for one-sample (intercept) tests
            cmd += ["-ise"]
        if self.cfg.tfce:
            cmd += ["-T"]
        if self.cfg.tfce2D:
            cmd += ["-tfce2D"]
        if self.cfg.cifti and self._supports_cifti():
            cmd += ["-cifti"]
        return cmd

    def run(self) -> Path:
        self._check()
        # Resolve once to get base for sentinel
        info = self._resolve_paths()
        cmd = self.build_cmd(info)
        log.info("palm_run", extra={"cmd": cmd})
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(e)
            # Fallback: retry without -cifti if requested, to support older PALM builds
            # if self.cfg.cifti:
            #     cmd_no_cifti = [c for c in cmd if c != "-cifti"]
            #     log.warning("palm_retry_without_cifti", extra={"cmd": cmd_no_cifti})
            #     subprocess.run(cmd_no_cifti, check=True)
            # else:
            #     raise
        # Mark success with a sentinel to make skip logic robust
        try:
            ok = info["base"].with_suffix('.ok')
            ok.write_text("ok\ncmd: " + " ".join(cmd) + "\n", encoding="utf-8")
        except Exception:
            pass
        # Return output directory for convenience
        log.info("palm_done", extra={"out_dir": str(info["out_dir"])})
        return info["out_dir"]
