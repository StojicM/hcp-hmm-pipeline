#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from nibabel.cifti2.cifti2_axes import ParcelsAxis, ScalarAxis

from .logger import get_logger

log = get_logger(__name__)


def _sniff_delim(path: Path) -> str:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")[:2048]
    return "\t" if txt.count("\t") >= txt.count(",") else ","


def _read_sids(path: Path) -> List[str]:
    delim = _sniff_delim(path)
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter=delim)
        keys = list(r.fieldnames or [])
        sid_key = next((k for k in ("sid", "Subject", "subject", "participant", "participant_id", "id") if k in keys), None)
        if sid_key is None:
            sid_key = keys[0] if keys else None
        if sid_key is None:
            return []
        sids = [str((row.get(sid_key) or "").strip()) for row in r]
    seen = set()
    uniq = []
    for x in sids:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


@dataclass
class GroupMergeConfig:
    betas_dir: Path
    K: int
    out_dir: Path
    subjects_used_csv: Path
    subject_out_dir: Optional[Path] = None
    inputs_list_path: Optional[Path] = None
    rename_maps: bool = True
    parcel_labels_nii: Optional[Path] = None  # optional volumetric label image (codes 1..P)
    atlas_dlabel: Optional[Path] = None       # optional CIFTI dlabel to build dense dscalar


class GroupMerger:
    def __init__(self, cfg: GroupMergeConfig):
        self.cfg = cfg

    def run(self) -> List[Path]:
        out_dir = self.cfg.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        sids = _read_sids(self.cfg.subjects_used_csv)
        if not sids:
            raise SystemExit(f"No subject IDs found in {self.cfg.subjects_used_csv}")

        Ktag = f"{self.cfg.K}S"
        mapcsv = out_dir / "columns_map.csv"

        # Load z-scored subject maps (primary path used so far)
        entries, parcels_axis, state_names = self._load_subject_maps(sids, Ktag, zscored=True)
        subject_names = [sid for sid, _, _, _ in entries]
        subject_axis = ScalarAxis(subject_names)

        merged_outputs: List[Path] = []
        map_rows: List[Tuple[str, int, str, str]] = []
        # Accumulators for group means/SDs per state
        group_means = []  # list of arrays length K, each shape (P,)
        group_sds = []    # list of arrays length K, each shape (P,)

        for state_idx in range(self.cfg.K):
            merged = out_dir / f"allsubs_state{state_idx}_{Ktag}_zscored.pscalar.nii"
            stack = np.stack([data[state_idx, :] for _, _, data, _ in entries], axis=0)
            header = nib.cifti2.Cifti2Header.from_axes([subject_axis, parcels_axis])
            img = nib.Cifti2Image(stack.astype(np.float32, copy=False), header=header)
            img.to_filename(str(merged))
            merged_outputs.append(merged)

            # Group summary maps (mean and SD across subjects)
            m = stack.mean(axis=0, dtype=np.float64)
            sd = stack.std(axis=0, ddof=1, dtype=np.float64)
            group_means.append(m.astype(np.float32, copy=False))
            group_sds.append(sd.astype(np.float32, copy=False))

            stats_header = nib.cifti2.Cifti2Header.from_axes([ScalarAxis(["mean", "sd"]), parcels_axis])
            stats_img = nib.Cifti2Image(np.vstack([m, sd]).astype(np.float32, copy=False), header=stats_header)
            stats_path = out_dir / f"group_state{state_idx}_{Ktag}_zscored_stats.pscalar.nii"
            stats_img.to_filename(str(stats_path))

            for col_index, sid in enumerate(subject_names, start=1):
                map_rows.append((merged.name, col_index, sid, f"state{state_idx}"))

            log.info("state_group_merged", extra={"state": state_idx, "out": str(merged), "subjects": len(subject_names)})

        if not merged_outputs:
            raise SystemExit("No group pscalars were produced. Ensure subjects have state maps available.")

        with open(mapcsv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["merged_file", "col_index", "sid", "state"])
            for merged_name, col_index, sid, state in map_rows:
                w.writerow([merged_name, col_index, sid, state])

        list_path = self.cfg.inputs_list_path or (self.cfg.out_dir / f"inputs_{Ktag}_pscalar.txt")
        list_path = Path(list_path)
        list_path.parent.mkdir(parents=True, exist_ok=True)
        list_path.write_text("\n".join(str(path.resolve()) for _, path, _, _ in entries) + "\n", encoding="utf-8")
        log.info("palm_inputs_list", extra={"path": str(list_path), "n": len(entries)})

        # Build a subject-major stack (rows = subject×state) for PALM/FSL-style inputs
        stack_path = self.cfg.out_dir / f"allsubs_states_{Ktag}_zscored.pscalar.nii"
        self._write_subject_major_stack(entries, parcels_axis, state_names, Ktag, stack_path)

        # Also write per-state group mean (and SD) stacks for convenient viewing
        try:
            if group_means:
                mean_stack = np.stack(group_means, axis=0)  # K × P
                sd_stack = np.stack(group_sds, axis=0)
                mean_header = nib.cifti2.Cifti2Header.from_axes([ScalarAxis(state_names if state_names else [f"State{i+1}" for i in range(self.cfg.K)]), parcels_axis])
                sd_header = mean_header
                mean_img = nib.Cifti2Image(mean_stack.astype(np.float32, copy=False), header=mean_header)
                sd_img = nib.Cifti2Image(sd_stack.astype(np.float32, copy=False), header=sd_header)
                mean_path = out_dir / f"group_mean_states_{Ktag}_zscored.pscalar.nii"
                sd_path = out_dir / f"group_sd_states_{Ktag}_zscored.pscalar.nii"
                mean_img.to_filename(str(mean_path))
                sd_img.to_filename(str(sd_path))
                log.info("state_group_means_written", extra={"mean": str(mean_path), "sd": str(sd_path)})

                # Optional volumetric export using a parcel label NIfTI (codes 1..P)
                if self.cfg.parcel_labels_nii and Path(self.cfg.parcel_labels_nii).exists():
                    try:
                        lbl_img = nib.load(str(self.cfg.parcel_labels_nii))
                        lbl = np.asarray(lbl_img.get_fdata(), dtype=np.int32)
                        P = mean_stack.shape[1]
                        # Build 4D volumes: (X, Y, Z, K)
                        vol_shape = lbl.shape + (self.cfg.K,)
                        mean_vol = np.zeros(vol_shape, dtype=np.float32)
                        sd_vol = np.zeros(vol_shape, dtype=np.float32)
                        # Assume labels are 1..P in the same parcel order as the pscalar axis
                        for j in range(P):
                            mask = (lbl == (j + 1))
                            if not mask.any():
                                continue
                            vals_m = mean_stack[:, j]  # K
                            vals_sd = sd_stack[:, j]
                            mean_vol[mask, :] = vals_m
                            sd_vol[mask, :] = vals_sd
                        # Write out
                        mean_vol_img = nib.Nifti1Image(mean_vol, affine=lbl_img.affine, header=lbl_img.header)
                        sd_vol_img = nib.Nifti1Image(sd_vol, affine=lbl_img.affine, header=lbl_img.header)
                        mean_vol_path = out_dir / f"group_mean_states_{Ktag}_zscored_vol.nii.gz"
                        sd_vol_path = out_dir / f"group_sd_states_{Ktag}_zscored_vol.nii.gz"
                        nib.save(mean_vol_img, str(mean_vol_path))
                        nib.save(sd_vol_img, str(sd_vol_path))
                        log.info("state_group_volumes_written", extra={"mean_vol": str(mean_vol_path), "sd_vol": str(sd_vol_path)})
                    except Exception as e:
                        log.warning("state_group_volumes_failed", extra={"err": str(e)})

                # Optional dense CIFTI (dscalar) export using atlas dlabel to paint parcels
                if self.cfg.atlas_dlabel and Path(self.cfg.atlas_dlabel).exists():
                    try:
                        dlab = nib.load(str(self.cfg.atlas_dlabel))
                        # Find BrainModel axis
                        ax0 = dlab.header.get_axis(0)
                        ax1 = dlab.header.get_axis(1) if dlab.ndim > 1 else None
                        from nibabel.cifti2.cifti2_axes import BrainModelAxis
                        bm = ax0 if isinstance(ax0, BrainModelAxis) else (ax1 if isinstance(ax1, BrainModelAxis) else None)
                        if bm is None:
                            raise RuntimeError("Could not locate BrainModelAxis in atlas dlabel")
                        # Label values for each brain element
                        lab_vals = np.asarray(dlab.get_fdata(), dtype=np.int32)
                        lab_vals = lab_vals.reshape(-1) if lab_vals.ndim > 1 else lab_vals
                        P = mean_stack.shape[1]
                        # Build dense K×N
                        N = lab_vals.size
                        dense_mean = np.zeros((self.cfg.K, N), dtype=np.float32)
                        dense_sd = np.zeros((self.cfg.K, N), dtype=np.float32)
                        # Map labels 1..P to parcel columns 0..P-1
                        for j in range(P):
                            mask = (lab_vals == (j + 1))
                            if not mask.any():
                                continue
                            dense_mean[:, mask] = mean_stack[:, j:j+1]
                            dense_sd[:, mask] = sd_stack[:, j:j+1]
                        state_axis = ScalarAxis(state_names if state_names else [f"State{i+1}" for i in range(self.cfg.K)])
                        dmean_hdr = nib.cifti2.Cifti2Header.from_axes([state_axis, bm])
                        dsd_hdr = nib.cifti2.Cifti2Header.from_axes([state_axis, bm])
                        dmean_img = nib.Cifti2Image(dense_mean, header=dmean_hdr)
                        dsd_img = nib.Cifti2Image(dense_sd, header=dsd_hdr)
                        dmean_path = out_dir / f"group_mean_states_{Ktag}_zscored.dscalar.nii"
                        dsd_path = out_dir / f"group_sd_states_{Ktag}_zscored.dscalar.nii"
                        dmean_img.to_filename(str(dmean_path))
                        dsd_img.to_filename(str(dsd_path))
                        log.info("state_group_dscalar_written", extra={"mean": str(dmean_path), "sd": str(dsd_path)})
                    except Exception as e:
                        log.warning("state_group_dscalar_failed", extra={"err": str(e)})
        except Exception as e:
            log.warning("state_group_means_failed", extra={"err": str(e)})

        subject_out_dir = Path(self.cfg.subject_out_dir) if self.cfg.subject_out_dir else None
        if subject_out_dir and subject_out_dir.resolve() != self.cfg.betas_dir.resolve():
            self._build_subject_stacks(entries, parcels_axis, state_names, Ktag, subject_out_dir)

        # Additionally, create RAW (non-z) merged pscalars per-state to support PALM on raw betas (optional)
        try:
            entries_raw, parcels_axis_raw, state_names_raw = self._load_subject_maps(sids, Ktag, zscored=False)
            if entries_raw:
                subject_names_raw = [sid for sid, _, _, _ in entries_raw]
                subject_axis_raw = ScalarAxis(subject_names_raw)
                for state_idx in range(self.cfg.K):
                    merged_raw = out_dir / f"allsubs_state{state_idx}_{Ktag}.pscalar.nii"
                    stack_raw = np.stack([data[state_idx, :] for _, _, data, _ in entries_raw], axis=0)
                    header_raw = nib.cifti2.Cifti2Header.from_axes([subject_axis_raw, parcels_axis_raw])
                    img_raw = nib.Cifti2Image(stack_raw.astype(np.float32, copy=False), header=header_raw)
                    img_raw.to_filename(str(merged_raw))
                    log.info("state_group_merged_raw", extra={"state": state_idx, "out": str(merged_raw), "subjects": len(subject_names_raw)})
        except Exception as e:
            log.info("state_group_raw_merge_skipped", extra={"err": str(e)})

        return merged_outputs

    def _load_subject_maps(
        self, sids: Sequence[str], Ktag: str, zscored: bool = True
    ) -> Tuple[List[Tuple[str, Path, np.ndarray, nib.cifti2.Cifti2Image]], ParcelsAxis, List[str]]:
        entries: List[Tuple[str, Path, np.ndarray, nib.cifti2.Cifti2Image]] = []
        parcels_axis: Optional[ParcelsAxis] = None
        state_names: List[str] = []

        for sid in sids:
            path = self.cfg.betas_dir / (
                f"{sid}_state_betas_{Ktag}_zscored.pscalar.nii" if zscored else f"{sid}_state_betas_{Ktag}.pscalar.nii"
            )
            if not path.exists():
                log.warning("missing_subject_pscalar", extra={"sid": sid, "path": str(path)})
                continue
            img = nib.load(str(path))
            data = np.asarray(img.get_fdata(dtype=np.float32))
            if data.ndim != 2:
                raise SystemExit(f"{path.name}: expected 2D array, got {data.shape}")
            if data.shape[0] != self.cfg.K:
                raise SystemExit(f"{path.name}: expected K={self.cfg.K} rows, got {data.shape[0]}")

            parc_axis = img.header.get_axis(1)
            if parcels_axis is None:
                parcels_axis = parc_axis
            elif len(parcels_axis) != len(parc_axis):
                raise SystemExit(f"Parcel mismatch for {path.name}; expected {len(parcels_axis)}, got {len(parc_axis)}")

            if not state_names:
                state_axis = img.header.get_axis(0)
                state_names = [str(state_axis[i][0]) for i in range(len(state_axis))]

            entries.append((sid, path, data.astype(np.float32, copy=False), img))

        if not entries:
            raise SystemExit("No subject pscalars were found for merging. Check betas_dir and subjects list.")
        assert parcels_axis is not None
        return entries, parcels_axis, state_names

    def _build_subject_stacks(
        self,
        entries: List[Tuple[str, Path, np.ndarray, nib.cifti2.Cifti2Image]],
        parcels_axis: ParcelsAxis,
        state_names: List[str],
        Ktag: str,
        subject_out_dir: Path,
    ) -> List[Path]:
        subject_out_dir = Path(subject_out_dir)
        subject_out_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg.rename_maps or not state_names:
            state_names_out = [f"State{i + 1}" for i in range(self.cfg.K)]
        else:
            state_names_out = state_names[:self.cfg.K]
        state_axis = ScalarAxis(state_names_out)

        outputs: List[Path] = []
        for sid, _, data, _ in entries:
            out_path = subject_out_dir / f"{sid}_state_betas_{Ktag}_zscored.pscalar.nii"
            img = nib.Cifti2Image(data.astype(np.float32, copy=False), header=nib.cifti2.Cifti2Header.from_axes([state_axis, parcels_axis]))
            img.to_filename(str(out_path))
            outputs.append(out_path)
            log.info("subject_states_written", extra={"sid": sid, "out": str(out_path)})

        return outputs

    def _write_subject_major_stack(
        self,
        entries: List[Tuple[str, Path, np.ndarray, nib.cifti2.Cifti2Image]],
        parcels_axis: ParcelsAxis,
        state_names: List[str],
        Ktag: str,
        out_path: Path,
    ) -> None:
        if not entries:
            return

        state_names_out = state_names if state_names else [f"State{i+1}" for i in range(self.cfg.K)]
        obs_rows = []
        obs_labels: List[str] = []
        for sid, _path, data, _img in entries:
            obs_rows.append(data)
            for idx, label in enumerate(state_names_out, start=1):
                obs_labels.append(f"{sid}_{label}")

        stack = np.vstack(obs_rows).astype(np.float32, copy=False)
        obs_axis = ScalarAxis(obs_labels)
        header = nib.cifti2.Cifti2Header.from_axes([obs_axis, parcels_axis])
        img = nib.Cifti2Image(stack, header=header)
        img.to_filename(str(out_path))
        log.info(
            "state_group_stack",
            extra={"path": str(out_path), "rows": int(stack.shape[0]), "subjects": len(entries)},
        )
