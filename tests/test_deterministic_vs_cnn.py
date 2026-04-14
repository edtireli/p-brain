#!/usr/bin/env python3
"""Compare deterministic ROI extraction against CNN ground truth.

Usage:
    cd /Users/edt/Desktop/p-brain
    source .venv/bin/activate
    python tests/test_deterministic_vs_cnn.py [--data-dir /Volumes/T5_EVO_EDT/data] [--ids 20221003x1 ...]

For each dataset that has CNN-produced ROI voxels, this script:
  1. Loads the DCE 4D volume + segmentation (brain bbox).
  2. Runs the atlas-guided deterministic vessel detection (no TensorFlow).
  3. Compares artery and vein ROIs against CNN ground truth:
       - Dice overlap per slice
       - Centroid distance (pixels)
       - CTC curve correlation (Pearson r)
  4. Produces a summary CSV and per-dataset overlay PNGs.
"""

from __future__ import annotations
import argparse
import glob
import os
import sys
import warnings

import numpy as np

# Add p-brain root to path
PBRAIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PBRAIN_ROOT)

import nibabel as nib
from nibabel import orientations as nio
from scipy import ndimage

# Import seg-atlas geometry and shared utilities from the builder
from tools.build_seg_atlas import (
    compute_slice_geometry, compute_volume_geometries,
    voxel_to_norm_coords, norm_to_grid,
    COORD_RANGE, ATLAS_GRID, load_cnn_rois,
)


# ---------------------------------------------------------------------------
# Helpers: load CNN CTC curves
# ---------------------------------------------------------------------------

def load_cnn_ctc(analysis_dir: str) -> dict:
    """Return {('artery', subtype, z): 1D array, ...}."""
    curves = {}
    for vessel in ("Artery", "Vein"):
        vdir = os.path.join(analysis_dir, "CTC Data", vessel)
        if not os.path.isdir(vdir):
            continue
        for subtype in os.listdir(vdir):
            sdir = os.path.join(vdir, subtype)
            if not os.path.isdir(sdir):
                continue
            for f in sorted(os.listdir(sdir)):
                if f.startswith("._") or not f.endswith(".npy"):
                    continue
                if "shifted" in f:
                    continue
                z = int(f.replace("CTC_slice_", "").replace(".npy", ""))
                arr = np.load(os.path.join(sdir, f))
                curves[(vessel.lower(), subtype, z)] = arr
    return curves


# ---------------------------------------------------------------------------
# Atlas loading
# ---------------------------------------------------------------------------

def load_atlas(atlas_path: str | None = None) -> dict | None:
    """Load the segmentation-based ROI probability atlas."""
    if atlas_path is None:
        atlas_path = os.path.join(PBRAIN_ROOT, "data", "seg_roi_atlas.npz")
    if not os.path.exists(atlas_path):
        return None
    d = np.load(atlas_path)
    return {
        "art_maps": d["art_maps"],       # (z_bins, G, G) float32
        "vein_maps": d["vein_maps"],      # (z_bins, G, G) float32
        "grid_size": int(d["grid_size"]),
        "z_bins": int(d["z_bins"]),
        "coord_range": float(d["coord_range"]),
    }


def atlas_prior_for_slice(atlas: dict, vessel: str, z: int, n_slices: int,
                          geom: dict, img_shape: tuple,
                          smooth_sigma: float | None = None) -> np.ndarray:
    """Map seg-atlas probability to a 2D image-space prior for one slice.

    Uses the segmentation-derived geometry (centroid, principal axes) to
    transform atlas grid coordinates back to image space. This is
    rotation-invariant because the geometry adapts to the brain orientation.

    Args:
        atlas: loaded seg atlas dict
        vessel: "artery" or "vein"
        z: 0-indexed slice
        n_slices: total slices
        geom: output of compute_slice_geometry() for this slice
        img_shape: (rows, cols) of the image
        smooth_sigma: Gaussian smoothing sigma (default: 3.0 for artery,
                      12.0 for vein to cover SSS positional uncertainty)

    Returns:
        2D float32 array (rows, cols) with probability values.
    """
    if smooth_sigma is None:
        smooth_sigma = 10.0 if vessel == "vein" else 3.0
    maps = atlas["art_maps"] if vessel == "artery" else atlas["vein_maps"]
    G = atlas["grid_size"]
    Z = atlas["z_bins"]
    R = atlas["coord_range"]

    z_frac = z / max(1, n_slices - 1)
    z_bin = min(Z - 1, int(z_frac * Z))

    prob_grid = maps[z_bin]  # (G, G)
    if prob_grid.max() == 0:
        return np.zeros(img_shape[:2], dtype=np.float32)

    # Normalize to [0, 1]
    prob_norm = prob_grid / prob_grid.max()

    # Map from atlas grid back to image coordinates using geometry
    centroid = geom["centroid"]
    lr_axis = geom["lr_axis"]
    si_axis = geom["si_axis"]
    half_lr = geom["half_lr"]
    half_si = geom["half_si"]

    prior = np.zeros(img_shape[:2], dtype=np.float32)

    for gi in range(G):
        for gj in range(G):
            if prob_norm[gi, gj] < 0.01:
                continue
            # Grid cell center → normalized coordinates
            lr_norm = (gi + 0.5) / G * 2 * R - R
            si_norm = (gj + 0.5) / G * 2 * R - R
            # Normalized → image coordinates
            offset = lr_norm * half_lr * lr_axis + si_norm * half_si * si_axis
            r = int(centroid[0] + offset[0])
            c = int(centroid[1] + offset[1])
            # Map grid cell to a small patch (accounting for resolution)
            patch_r = max(1, int(half_lr * 2 * R / G))
            patch_c = max(1, int(half_si * 2 * R / G))
            r0 = max(0, r - patch_r // 2)
            r1 = min(img_shape[0], r + patch_r // 2 + 1)
            c0 = max(0, c - patch_c // 2)
            c1 = min(img_shape[1], c + patch_c // 2 + 1)
            if r0 < r1 and c0 < c1:
                prior[r0:r1, c0:c1] = np.maximum(prior[r0:r1, c0:c1],
                                                   prob_norm[gi, gj])

    # Smooth to avoid blocky edges
    prior = ndimage.gaussian_filter(prior, sigma=smooth_sigma)
    mx = prior.max()
    if mx > 0:
        prior /= mx

    return prior


# ---------------------------------------------------------------------------
# Deterministic vessel extraction (standalone, no settings import)
# ---------------------------------------------------------------------------

def find_dce_nifti(nifti_dir: str) -> str | None:
    """Find the DCE 4D NIfTI in a dataset's NIfTI directory."""
    candidates = []
    for f in os.listdir(nifti_dir):
        if f.startswith("._") or not (f.endswith(".nii") or f.endswith(".nii.gz")):
            continue
        path = os.path.join(nifti_dir, f)
        try:
            img = nib.load(path)
            if img.ndim == 4 and img.shape[3] > 50:
                candidates.append((f, img.shape[3]))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[1], reverse=True)
    return candidates[0][0]


def get_brain_bbox_per_slice(seg_or_mask: np.ndarray) -> dict:
    """Return {z: (rmin, rmax, cmin, cmax)} for each slice with brain."""
    brain = seg_or_mask > 0
    bboxes = {}
    for z in range(seg_or_mask.shape[2]):
        m = brain[:, :, z]
        if not m.any():
            continue
        rows, cols = np.where(m)
        bboxes[z] = (int(rows.min()), int(rows.max()),
                      int(cols.min()), int(cols.max()))
    return bboxes


def extract_rois_deterministic(
    dce4d: np.ndarray,
    affine: np.ndarray,
    atlas: dict | None = None,
    seg_data: np.ndarray | None = None,
    baseline_frames: int = 10,
    n_art_slices: int = 3,
    n_vein_slices: int = 5,
) -> dict:
    """Atlas-mask deterministic AIF/VIF extraction.

    Requires segmentation — cannot run without it.
    The thresholded atlas heatmap IS the ROI mask directly. No nudging,
    no peak-guided shifting. The atlas encodes where vessels are across
    the population; we just place it via the segmentation geometry.

    Returns dict of {('artery'|'vein', subtype, z): (N,2) int array of (row,col)}.
    """
    xdim, ydim, zdim, tdim = dce4d.shape

    if atlas is None or seg_data is None:
        return {}

    if seg_data.shape[:3] != dce4d.shape[:3]:
        return {}

    # --- Per-slice geometry from segmentation (rotation-invariant) ---
    slice_geometries = compute_volume_geometries(seg_data)
    if not slice_geometries:
        return {}

    # --- Peak amplitude (only used for slice ranking) ---
    baseline = dce4d[..., :baseline_frames].mean(axis=-1)
    peak_amp = np.clip(dce4d.max(axis=-1) - baseline, 0, None).astype(np.float32)

    # Atlas prior thresholds — mask size ≈ CNN ROI size
    ART_PRIOR_THR = 0.25
    VEIN_PRIOR_THR = 0.35

    results = {}

    # --- ARTERY: bottom 35% of slices ---
    art_z_end = max(2, int(zdim * 0.35))
    for z in range(0, art_z_end):
        geom = slice_geometries.get(z)
        if geom is None:
            continue
        prior = atlas_prior_for_slice(atlas, "artery", z, zdim, geom,
                                      (xdim, ydim))
        if prior.max() == 0:
            continue
        mask = prior >= ART_PRIOR_THR
        if mask.sum() < 3:
            continue
        coords = np.argwhere(mask)
        score = float(peak_amp[coords[:, 0], coords[:, 1], z].mean())
        results[("artery", "Right Interior Carotid", z + 1)] = (coords, score)

    # --- VEIN (SSS): slices 20%–90% ---
    sss_z_start = max(1, int(zdim * 0.2))
    sss_z_end = min(zdim, int(zdim * 0.9))
    for z in range(sss_z_start, sss_z_end):
        geom = slice_geometries.get(z)
        if geom is None:
            continue
        prior = atlas_prior_for_slice(atlas, "vein", z, zdim, geom,
                                      (xdim, ydim))
        if prior.max() == 0:
            continue
        mask = prior >= VEIN_PRIOR_THR
        if mask.sum() < 3:
            continue
        coords = np.argwhere(mask)
        score = float(peak_amp[coords[:, 0], coords[:, 1], z].mean())
        results[("vein", "Sinus Sagittalis", z + 1)] = (coords, score)

    # --- Keep top-N slices per vessel by score ---
    art_items = [(k, v, s) for k, (v, s) in results.items() if k[0] == "artery"]
    vein_items = [(k, v, s) for k, (v, s) in results.items() if k[0] == "vein"]
    art_items.sort(key=lambda t: t[2], reverse=True)
    vein_items.sort(key=lambda t: t[2], reverse=True)

    final = {}
    for k, v, s in art_items[:n_art_slices]:
        final[k] = v
    for k, v, s in vein_items[:n_vein_slices]:
        final[k] = v

    return final


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------

def dice_2d(vox_a: np.ndarray, vox_b: np.ndarray, shape: tuple[int, int]) -> float:
    """Dice coefficient between two sets of (row, col) voxels."""
    mask_a = np.zeros(shape, dtype=bool)
    mask_b = np.zeros(shape, dtype=bool)
    if vox_a.size:
        mask_a[vox_a[:, 0], vox_a[:, 1]] = True
    if vox_b.size:
        mask_b[vox_b[:, 0], vox_b[:, 1]] = True
    inter = (mask_a & mask_b).sum()
    total = mask_a.sum() + mask_b.sum()
    if total == 0:
        return 0.0
    return 2.0 * inter / total


def centroid_dist(vox_a: np.ndarray, vox_b: np.ndarray) -> float:
    if vox_a.size == 0 or vox_b.size == 0:
        return float("inf")
    ca = vox_a.mean(axis=0)
    cb = vox_b.mean(axis=0)
    return float(np.linalg.norm(ca - cb))


def curve_correlation(dce4d, vox_a, vox_b, z, baseline_frames=10):
    """Pearson r between mean timecourses of two ROIs in the same slice."""
    if vox_a.size == 0 or vox_b.size == 0:
        return float("nan")
    tc_a = dce4d[vox_a[:, 0], vox_a[:, 1], z, :].mean(axis=0)
    tc_b = dce4d[vox_b[:, 0], vox_b[:, 1], z, :].mean(axis=0)
    if tc_a.std() == 0 or tc_b.std() == 0:
        return float("nan")
    return float(np.corrcoef(tc_a, tc_b)[0, 1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_dataset(ds_path: str, atlas: dict | None = None) -> dict | None:
    """Run deterministic extraction and compare to CNN for one dataset."""
    nifti_dir = os.path.join(ds_path, "NIfTI")
    analysis_dir = os.path.join(ds_path, "Analysis")

    if not os.path.isdir(nifti_dir) or not os.path.isdir(analysis_dir):
        return None

    # Load CNN ground truth
    cnn_rois = load_cnn_rois(analysis_dir)
    if not cnn_rois:
        return None

    # Find DCE
    dce_fname = find_dce_nifti(nifti_dir)
    if not dce_fname:
        return None

    dce_path = os.path.join(nifti_dir, dce_fname)
    img = nib.load(dce_path)
    dce4d = np.asarray(img.dataobj, dtype=np.float32)
    affine = img.affine

    if dce4d.ndim != 4:
        return None

    # Load segmentation in DCE space if available
    seg_path = os.path.join(nifti_dir, "segmentation", "segmentation",
                            "mri", "aparc.DKTatlas+aseg.deep_in_DCE.nii.gz")
    seg_data = None
    if os.path.exists(seg_path):
        try:
            seg_data = np.asarray(nib.load(seg_path).dataobj)
        except Exception:
            pass

    ds_name = os.path.basename(ds_path)
    print(f"  [{ds_name}] DCE shape={dce4d.shape}, file={dce_fname}, seg={'yes' if seg_data is not None else 'no'}")

    # Run atlas-guided deterministic extraction
    det_rois = extract_rois_deterministic(dce4d, affine, atlas=atlas,
                                          seg_data=seg_data)

    # Compare
    shape2d = (dce4d.shape[0], dce4d.shape[1])
    comparisons = []

    # Map CNN keys to simple (type, z) for matching
    cnn_by_tz = {}
    for (vtype, subtype, z), vox in cnn_rois.items():
        simple_type = "artery" if vtype == "artery" else "vein"
        cnn_by_tz[(simple_type, z)] = (subtype, vox)

    det_by_tz = {}
    for (vtype, subtype, z), vox in det_rois.items():
        simple_type = "artery" if vtype == "artery" else "vein"
        det_by_tz[(simple_type, z)] = (subtype, vox)

    # For each CNN slice, find if deterministic also found that slice
    for (stype, z), (cnn_sub, cnn_vox) in cnn_by_tz.items():
        det_entry = det_by_tz.get((stype, z))
        if det_entry is not None:
            det_sub, det_vox = det_entry
            d = dice_2d(cnn_vox, det_vox, shape2d)
            cd = centroid_dist(cnn_vox, det_vox)
            cr = curve_correlation(dce4d, cnn_vox, det_vox, z - 1)  # z is 1-indexed
            comparisons.append({
                "type": stype,
                "z": z,
                "dice": d,
                "centroid_dist": cd,
                "curve_corr": cr,
                "cnn_nvox": cnn_vox.shape[0],
                "det_nvox": det_vox.shape[0],
                "matched": True,
            })
        else:
            comparisons.append({
                "type": stype,
                "z": z,
                "dice": 0.0,
                "centroid_dist": float("inf"),
                "curve_corr": float("nan"),
                "cnn_nvox": cnn_vox.shape[0],
                "det_nvox": 0,
                "matched": False,
            })

    # Deterministic slices not in CNN
    for (stype, z), (det_sub, det_vox) in det_by_tz.items():
        if (stype, z) not in cnn_by_tz:
            comparisons.append({
                "type": stype,
                "z": z,
                "dice": 0.0,
                "centroid_dist": float("inf"),
                "curve_corr": float("nan"),
                "cnn_nvox": 0,
                "det_nvox": det_vox.shape[0],
                "matched": False,
            })

    return {
        "dataset": ds_name,
        "dce_shape": dce4d.shape,
        "cnn_slices": len(cnn_by_tz),
        "det_slices": len(det_by_tz),
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare deterministic vs CNN ROIs")
    parser.add_argument("--data-dir", default="/Volumes/T5_EVO_EDT/data",
                        help="Root data directory")
    parser.add_argument("--ids", nargs="*", default=None,
                        help="Specific dataset IDs (default: all)")
    parser.add_argument("--max", type=int, default=None,
                        help="Max datasets to process")
    parser.add_argument("--atlas", default=None,
                        help="Path to roi_atlas.npz (default: data/roi_atlas.npz)")
    args = parser.parse_args()

    # Load atlas
    atlas = load_atlas(args.atlas)
    if atlas is not None:
        print(f"Atlas loaded: grid={atlas['grid_size']}, z_bins={atlas['z_bins']}")
    else:
        print("WARNING: No atlas found, running without spatial prior")

    data_dir = args.data_dir
    if args.ids:
        datasets = [os.path.join(data_dir, d) for d in args.ids]
    else:
        datasets = sorted([
            os.path.join(data_dir, d)
            for d in os.listdir(data_dir)
            if d.startswith("20") and os.path.isdir(os.path.join(data_dir, d))
        ])

    if args.max:
        datasets = datasets[:args.max]

    print(f"Evaluating {len(datasets)} datasets...")
    all_results = []

    for ds_path in datasets:
        try:
            result = evaluate_dataset(ds_path, atlas=atlas)
            if result:
                all_results.append(result)
                # Quick per-dataset summary
                comps = result["comparisons"]
                art_comps = [c for c in comps if c["type"] == "artery" and c["matched"]]
                vein_comps = [c for c in comps if c["type"] == "vein" and c["matched"]]
                art_dice = np.mean([c["dice"] for c in art_comps]) if art_comps else 0.0
                art_cd = np.mean([c["centroid_dist"] for c in art_comps]) if art_comps else float("inf")
                art_cr = np.nanmean([c["curve_corr"] for c in art_comps]) if art_comps else float("nan")
                vein_dice = np.mean([c["dice"] for c in vein_comps]) if vein_comps else 0.0
                vein_cd = np.mean([c["centroid_dist"] for c in vein_comps]) if vein_comps else float("inf")
                vein_cr = np.nanmean([c["curve_corr"] for c in vein_comps]) if vein_comps else float("nan")

                n_cnn = result["cnn_slices"]
                n_det = result["det_slices"]
                n_match = len(art_comps) + len(vein_comps)
                print(f"  → CNN:{n_cnn} DET:{n_det} match:{n_match} | "
                      f"ART dice={art_dice:.3f} cd={art_cd:.1f}px r={art_cr:.3f} | "
                      f"VEIN dice={vein_dice:.3f} cd={vein_cd:.1f}px r={vein_cr:.3f}")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    # Global summary
    if not all_results:
        print("No results.")
        return

    all_art = []
    all_vein = []
    for r in all_results:
        for c in r["comparisons"]:
            if c["matched"]:
                if c["type"] == "artery":
                    all_art.append(c)
                else:
                    all_vein.append(c)

    print("\n" + "=" * 70)
    print(f"SUMMARY across {len(all_results)} datasets")
    print("=" * 70)

    if all_art:
        print(f"\nARTERY ({len(all_art)} matched slices):")
        print(f"  Dice:     {np.mean([c['dice'] for c in all_art]):.3f} ± {np.std([c['dice'] for c in all_art]):.3f}")
        print(f"  Centroid: {np.mean([c['centroid_dist'] for c in all_art]):.1f} ± {np.std([c['centroid_dist'] for c in all_art]):.1f} px")
        corrs = [c["curve_corr"] for c in all_art if np.isfinite(c["curve_corr"])]
        if corrs:
            print(f"  Curve r:  {np.mean(corrs):.3f} ± {np.std(corrs):.3f}")

    if all_vein:
        print(f"\nVEIN ({len(all_vein)} matched slices):")
        print(f"  Dice:     {np.mean([c['dice'] for c in all_vein]):.3f} ± {np.std([c['dice'] for c in all_vein]):.3f}")
        print(f"  Centroid: {np.mean([c['centroid_dist'] for c in all_vein]):.1f} ± {np.std([c['centroid_dist'] for c in all_vein]):.1f} px")
        corrs = [c["curve_corr"] for c in all_vein if np.isfinite(c["curve_corr"])]
        if corrs:
            print(f"  Curve r:  {np.mean(corrs):.3f} ± {np.std(corrs):.3f}")

    # Slice detection rates
    total_cnn = sum(r["cnn_slices"] for r in all_results)
    total_matched = len(all_art) + len(all_vein)
    print(f"\nSlice detection rate: {total_matched}/{total_cnn} = {100*total_matched/max(1,total_cnn):.1f}%")


if __name__ == "__main__":
    main()
