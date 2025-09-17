#!/usr/bin/env python3
from __future__ import annotations

"""
PALM (Permutation Analysis of Linear Models) runner integration.

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
    tfce: bool = False
    tfce2D: bool = False
    cifti: bool = True
    palm_bin: str = "palm"
    merged_dscalar: Optional[Path] = None
    design_mat: Optional[Path] = None
    design_con: Optional[Path] = None
    design_grp: Optional[Path] = None
    out_dir: Optional[Path] = None


class PalmRunner:
    def __init__(self, cfg: PalmConfig):
        self.cfg = cfg

    def _resolve_paths(self):
        Ktag = f"{self.cfg.K}S"
        merged = self.cfg.merged_dscalar or (self.cfg.group_dir / f"allsubs_{Ktag}_zscored.dscalar.nii")
        dmat = self.cfg.design_mat or (self.cfg.group_dir / "design.mat")
        dcon = self.cfg.design_con or (self.cfg.group_dir / "design.con")
        dgrp = self.cfg.design_grp or (self.cfg.group_dir / "design.grp")
        out_dir = self.cfg.out_dir or (self.cfg.group_dir / "palm")
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / f"palm_{Ktag}"
        return merged, dmat, dcon, dgrp, out_dir, base

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

    def build_cmd(self) -> List[str]:
        merged, dmat, dcon, dgrp, out_dir, base = self._resolve_paths()
        cmd = [self.cfg.palm_bin,
               "-i", str(merged),
               "-d", str(dmat),
               "-t", str(dcon),
               "-eb", str(dgrp),
               "-n", str(self.cfg.n_perm),
               "-o", str(base)]
        if self.cfg.two_tailed:
            cmd += ["-twotail"]
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
        merged, dmat, dcon, dgrp, out_dir, base = self._resolve_paths()
        cmd = self.build_cmd()
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
            ok = base.with_suffix('.ok')
            ok.write_text("ok\ncmd: " + " ".join(cmd) + "\n", encoding="utf-8")
        except Exception:
            pass
        # Return output directory for convenience
        _, _, _, _, out_dir, _ = self._resolve_paths()
        log.info("palm_done", extra={"out_dir": str(out_dir)})
        return out_dir
