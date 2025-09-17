#!/usr/bin/env python3
"""Helpers for indexing dtseries files by subject."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Sequence

from .logger import get_logger

log = get_logger(__name__)

_PREFERRED_KEYWORDS: Sequence[str] = ("rest", "REST")


def _choose_preferred(paths: Sequence[Path]) -> Path:
    if len(paths) == 1:
        return paths[0]
    for key in _PREFERRED_KEYWORDS:
        key_lower = key.lower()
        for path in paths:
            if key_lower in path.name.lower():
                return path
    return sorted(paths)[0]


@lru_cache(maxsize=32)
def _index_cached(dtseries_dir: str) -> Dict[str, Path]:
    dt_dir = Path(dtseries_dir)
    if not dt_dir.exists():
        log.warning("dtseries_dir_missing", extra={"path": dtseries_dir})
        return {}

    grouped: Dict[str, list[Path]] = {}
    for path in sorted(dt_dir.glob("*.dtseries.nii")):
        sid = path.name.split("_")[0]
        grouped.setdefault(sid, []).append(path)

    best = {sid: _choose_preferred(paths) for sid, paths in grouped.items()}
    return best


def index_dtseries(dt_dir: Path) -> Dict[str, Path]:
    """Return a mapping Subject → dtseries path (cached per directory)."""

    return dict(_index_cached(str(Path(dt_dir))))


def get_dt_path(dt_dir: Path, subject: str) -> Path | None:
    """Fetch the dtseries path for a subject, if present."""

    if not subject:
        return None
    lookup = _index_cached(str(Path(dt_dir)))
    return lookup.get(subject)

