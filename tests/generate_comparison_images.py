#!/usr/bin/env python3
"""Generate per-dataset overlay PNGs comparing seg-atlas deterministic vs CNN ROIs.

Usage:
    cd /Users/edt/Desktop/p-brain
    source .venv/bin/activate
    python tests/generate_comparison_images.py [--ids 20221003x1 20221004x1 ...] [--max 10]

Produces one PNG per dataset in tests/comparison_images_seg/
"""
from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np

PBRAIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PBRAIN_ROOT)

import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from tests.test_deterministic_vs_cnn import (
    load_atlas, extract_rois_deterministic, find_dce_nifti,
    centroid_dist, dice_2d,
)
from tools.build_seg_atlas import load_cnn_rois

OUT_DIR = os.path.join(PBRAIN_ROOT, "tests", "comparison_images_seg")


def make_overlay_image(ds_path: str, atlas: dict) -> str | None:
    """Generate a comparison overlay PNG for one dataset. Returns output path."""
    nifti_dir = os.path.join(ds_path, "NIfTI")
    analysis_dir = os.path.join(ds_path, "Analysis")
    if not os.path.isdir(nifti_dir) or not os.path.isdir(analysis_dir):
        return None

    cnn_rois = load_cnn_rois(analysis_dir)
    if not cnn_rois:
        return None

    dce_fname = find_dce_nifti(nifti_dir)
    if not dce_fname:
        return None

    img = nib.load(os.path.join(nifti_dir, dce_fname))
    dce4d = np.asarray(img.dataobj, dtype=np.float32)
    if dce4d.ndim != 4:
        return None

    seg_path = os.path.join(nifti_dir, "segmentation", "segmentation",
                            "mri", "aparc.DKTatlas+aseg.deep_in_DCE.nii.gz")
    seg_data = None
    if os.path.exists(seg_path):
        try:
            seg_data = np.asarray(nib.load(seg_path).dataobj)
        except Exception:
            pass

    det_rois = extract_rois_deterministic(dce4d, img.affine, atlas=atlas,
                                          seg_data=seg_data)

    ds_name = os.path.basename(ds_path)
    shape2d = (dce4d.shape[0], dce4d.shape[1])

    # Group CNN/DET by (type, z)
    cnn_by_tz = {}
    for (vtype, sub, z), vox in cnn_rois.items():
        st = "artery" if vtype == "artery" else "vein"
        cnn_by_tz[(st, z)] = vox
    det_by_tz = {}
    for (vtype, sub, z), vox in det_rois.items():
        st = "artery" if vtype == "artery" else "vein"
        det_by_tz[(st, z)] = vox

    # Pick best matched slice per vessel type (lowest centroid distance)
    best = {}
    for stype in ("vein", "artery"):
        candidates = []
        for (t, z), cnn_vox in cnn_by_tz.items():
            if t != stype:
                continue
            det_vox = det_by_tz.get((stype, z))
            if det_vox is not None:
                cd = centroid_dist(cnn_vox, det_vox)
                candidates.append((z, cnn_vox, det_vox, cd))
        if candidates:
            # Pick median centroid distance (representative, not cherry-picked)
            candidates.sort(key=lambda x: x[3])
            idx = len(candidates) // 2
            best[stype] = candidates[idx]

    if not best:
        return None

    n_panels = len(best)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # Background: mean of peak enhancement frames
    baseline = dce4d[..., :10].mean(axis=-1)
    peak_frame = int(np.argmax(dce4d.mean(axis=(0, 1, 2))))
    bg_vol = dce4d[..., peak_frame] - baseline

    for ax, (stype, (z, cnn_vox, det_vox, cd)) in zip(axes, sorted(best.items())):
        # z is 1-indexed in our convention
        bg_raw = bg_vol[:, :, z - 1]

        # --- Reorient to standard radiological axial view ---
        # NIfTI LAS: axis0=R→L (rows), axis1=P→A (cols)
        # Radiological: Anterior at top, patient-Right at screen-left
        # Transform: transpose (rows↔cols) then flip rows (so A at top)
        bg = bg_raw.T[::-1, :]
        disp_shape = bg.shape  # (cols_orig, rows_orig) after transpose
        n1 = bg_raw.shape[1]  # original axis-1 size for coord transform

        def to_display(vox):
            """Transform (row_orig, col_orig) → display coords."""
            return np.column_stack([n1 - 1 - vox[:, 1], vox[:, 0]])

        vmin, vmax = np.percentile(bg[bg > 0], [2, 99]) if (bg > 0).any() else (0, 1)
        ax.imshow(bg, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")

        # CNN = green overlay
        cnn_disp = to_display(cnn_vox)
        mask_cnn = np.zeros(disp_shape, dtype=bool)
        mask_cnn[cnn_disp[:, 0], cnn_disp[:, 1]] = True
        cnn_rgba = np.zeros((*disp_shape, 4), dtype=np.float32)
        cnn_rgba[mask_cnn] = [0, 1, 0, 0.5]
        ax.imshow(cnn_rgba, origin="upper")

        # DET = red overlay
        det_disp = to_display(det_vox)
        mask_det = np.zeros(disp_shape, dtype=bool)
        mask_det[det_disp[:, 0], det_disp[:, 1]] = True
        det_rgba = np.zeros((*disp_shape, 4), dtype=np.float32)
        det_rgba[mask_det] = [1, 0, 0, 0.5]
        ax.imshow(det_rgba, origin="upper")

        label = "VEIN" if stype == "vein" else "ART"
        d = dice_2d(cnn_vox, det_vox, shape2d)
        ax.set_title(f"{label} z={z} cd={cd:.0f}px dice={d:.2f}", fontsize=10)
        ax.axis("off")

        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="green", alpha=0.5, label="CNN"),
                           Patch(facecolor="red", alpha=0.5, label="DET")],
                  loc="upper right", fontsize=7, framealpha=0.7)

    has_seg = seg_data is not None
    fig.suptitle(f"{ds_name} (seg={has_seg})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{ds_name}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/Volumes/T5_EVO_EDT/data")
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()

    atlas = load_atlas()
    if atlas is None:
        print("ERROR: No seg atlas found"); return

    if args.ids:
        datasets = [os.path.join(args.data_dir, d) for d in args.ids]
    else:
        datasets = sorted([
            os.path.join(args.data_dir, d)
            for d in os.listdir(args.data_dir)
            if d.startswith("20") and os.path.isdir(os.path.join(args.data_dir, d))
        ])
    if args.max:
        datasets = datasets[:args.max]

    print(f"Generating overlays for {len(datasets)} datasets → {OUT_DIR}")
    for ds in datasets:
        ds_name = os.path.basename(ds)
        try:
            out = make_overlay_image(ds, atlas)
            if out:
                print(f"  {ds_name} → saved")
            else:
                print(f"  {ds_name} → skipped (no data)")
        except Exception as e:
            print(f"  {ds_name} → ERROR: {e}")


if __name__ == "__main__":
    main()
