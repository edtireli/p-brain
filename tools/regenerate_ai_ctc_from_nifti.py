#!/usr/bin/env python3
"""Regenerate AI-saved ITC/CTC for a slice directly from NIfTI + ROI voxels.

This is a debugging/repair tool for the "after AI finder" step:
- Loads the ROI voxels and (optionally) frame index from Analysis
- Loads the DCE NIfTI
- Auto-detects whether the DCE volume must be rotated in-plane to match
  the ROI voxel coordinate convention (compares against saved ITC)
- Runs `plot_time_intensity_curves_and_CTC_AI` to re-save the CTC using the
  current code (including TurboFLASH case12/method1 behavior).

It is safe to run repeatedly; it overwrites the existing slice files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import nibabel as nib


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = int(min(a.size, b.size))
    if n == 0:
        return float("inf")
    a = a[:n]
    b = b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return float("inf")
    d = a[m] - b[m]
    return float(np.sqrt(np.mean(d * d)))


def _max_voxel_itc(data: np.ndarray, roi_voxels: np.ndarray, slice_index: int) -> np.ndarray:
    roi_voxels = np.asarray(roi_voxels)
    xs = roi_voxels[:, 0].astype(int)
    ys = roi_voxels[:, 1].astype(int)

    best = None
    best_peak = -np.inf
    for x, y in zip(xs, ys):
        tc = np.asarray(data[int(x), int(y), int(slice_index), :], dtype=float)
        peak = float(np.nanmax(tc))
        if peak > best_peak:
            best_peak = peak
            best = tc

    if best is None:
        raise ValueError("Empty ROI voxels")
    return np.asarray(best, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=Path, required=True)
    ap.add_argument("--nifti-dir", type=Path, required=True)
    ap.add_argument("--dce", type=str, required=True, help="DCE NIfTI filename, e.g. WIPhperf120long.nii")

    ap.add_argument("--type", type=str, required=True, help="Artery or Vein (folder name under Analysis/* Data)")
    ap.add_argument("--subtype", type=str, required=True, help="Subtype folder name, e.g. Right Interior Carotid")
    ap.add_argument("--slice", type=int, required=True, help="Slice number from filename, e.g. 2 for *_slice_2")

    ap.add_argument("--rotate", choices=["auto", "true", "false"], default="auto")
    args = ap.parse_args()

    analysis_dir = args.analysis_dir
    nifti_dir = args.nifti_dir

    slice_index = int(args.slice) - 1
    if slice_index < 0:
        raise SystemExit("--slice must be >= 1")

    roi_path = analysis_dir / "ROI Data" / args.type / args.subtype / f"ROI_voxels_slice_{args.slice}.npy"
    itc_path = analysis_dir / "ITC Data" / args.type / args.subtype / f"ITC_slice_{args.slice}.npy"
    frame_path = analysis_dir / "Frame Data" / args.type / args.subtype / f"frame_index_slice_{args.slice}.npy"

    if not roi_path.exists():
        raise SystemExit(f"Missing ROI file: {roi_path}")

    roi_voxels = np.load(roi_path)
    saved_itc = np.load(itc_path) if itc_path.exists() else None
    frame_index = int(np.load(frame_path)) if frame_path.exists() else 0

    dce_path = nifti_dir / args.dce
    img = nib.load(str(dce_path))
    data_raw = np.asarray(img.get_fdata(), dtype=float)
    if data_raw.ndim != 4:
        raise SystemExit(f"Expected 4D DCE NIfTI; got shape {data_raw.shape}")

    # Try to detect whether ROI voxels are in rotated coordinates.
    candidates: list[tuple[str, np.ndarray, bool]] = [
        ("raw", data_raw, False),
        ("rot90-1", np.rot90(data_raw, -1, axes=(0, 1)), True),
    ]

    if args.rotate != "auto":
        want_rot = args.rotate == "true"
        candidates = [c for c in candidates if c[2] == want_rot]

    best = None
    for name, data_cand, rotate_maps in candidates:
        try:
            itc_cand = _max_voxel_itc(data_cand, roi_voxels, slice_index)
        except Exception:
            continue
        score = _rmse(itc_cand, saved_itc) if saved_itc is not None else 0.0
        if best is None or score < best[0]:
            best = (score, name, data_cand, rotate_maps)

    if best is None:
        raise SystemExit("Could not compute ITC candidates")

    score, name, data_sel, rotate_maps = best
    print(f"Selected DCE orientation: {name} (rotate_maps={rotate_maps})")
    if saved_itc is not None:
        print(f"RMSE(max-voxel ITC vs saved ITC): {score:.6g}")

    # Build minimal filenames tuple; only dce_filename is used.
    filenames = (
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        args.dce,
    )

    # Import late so users can run this without full p-brain import at arg-parse time.
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from utils.plotting import plot_time_intensity_curves_AI, plot_time_intensity_curves_and_CTC_AI

    time_points_s = np.arange(int(data_sel.shape[3]), dtype=float)
    max_intensity_frame = int(np.nanargmax(np.nanmean(data_sel[:, :, slice_index, :], axis=(0, 1))))

    image_dir = analysis_dir.parent / "Images"
    os.makedirs(image_dir, exist_ok=True)

    # First, re-save the selected ITC using current settings.
    plot_time_intensity_curves_AI(
        data_sel,
        roi_voxels=roi_voxels,
        slice_index=slice_index,
        frame_index=frame_index,
        time_points_s=time_points_s,
        analysis_directory=str(analysis_dir),
        image_directory=str(image_dir),
        type=args.type,
        subtype=args.subtype,
    )

    # Then, compute and save the selected CTC.
    plot_time_intensity_curves_and_CTC_AI(
        data_sel,
        max_intensity_frame=max_intensity_frame,
        roi_voxels=roi_voxels,
        slice_index=slice_index,
        frame_index=frame_index,
        time_points_s=time_points_s,
        analysis_directory=str(analysis_dir),
        image_directory=str(image_dir),
        nifti_directory=str(nifti_dir),
        type=args.type,
        subtype=args.subtype,
        IsVFA=False,
        filenames=filenames,
        rotate_ac=rotate_maps,
    )

    print("Done. Recomputed CTC saved under Analysis/CTC Data/...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
