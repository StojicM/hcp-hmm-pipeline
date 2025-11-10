#!/usr/bin/env python3
"""Shared helpers for invoking Connectome Workbench (`wb_command`)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import shutil
import subprocess

from .logger import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def ensure_workbench() -> str:
    """Return the resolved path to `wb_command`, caching the lookup."""
    exe = shutil.which("wb_command")
    if not exe:
        raise RuntimeError(
            "wb_command not found in PATH. Install/activate Connectome Workbench."
        )
    return exe


def run_wb(args: Sequence[str], *, check: bool = True, capture_output: bool = False, text: bool = True) -> subprocess.CompletedProcess | None:
    """Invoke `wb_command` with the provided arguments.

    Parameters
    ----------
    args : Sequence[str]
        Arguments passed to `wb_command`.
    check : bool, optional
        Whether to raise on non-zero exit (default True).
    capture_output : bool, optional
        Whether to capture stdout/stderr (default False for streaming).
    text : bool, optional
        Decode output as text when capturing (default True).
    """

    exe = ensure_workbench()
    cmd = [exe, *map(str, args)]
    log.debug("wb_command", extra={"cmd": cmd})
    if capture_output:
        return subprocess.run(cmd, check=check, capture_output=True, text=text)
    subprocess.run(cmd, check=check)
    return None


def export_label_table(dlabel: Path, out_table: Path) -> Path:
    """Export the label table for an atlas, creating parent dirs as needed."""

    out_table = Path(out_table)
    if out_table.exists():
        return out_table
    out_table.parent.mkdir(parents=True, exist_ok=True)
    log.info("exporting_label_table", extra={"dlabel": str(dlabel), "out": str(out_table)})
    run_wb(["-cifti-label-export-table", str(dlabel), "1", str(out_table)])
    return out_table


def merge_cifti(output: Path, cifti_paths: Iterable[Path]) -> None:
    """Merge multiple CIFTI scalar images into a multi-map dscalar."""

    output = Path(output)
    cmd: list[str] = ["-cifti-merge", str(output)]
    for path in cifti_paths:
        cmd.extend(["-cifti", str(path)])
    run_wb(cmd)
