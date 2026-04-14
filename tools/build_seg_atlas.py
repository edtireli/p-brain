#!/usr/bin/env python3
"""Build a segmentation-based ROI atlas from CNN ground truth.

Instead of a bounding-box grid (rotation-sensitive), this atlas uses the
brain segmentation labels to define a rotation-invariant coordinate frame:

  1. LEFT hemisphere labels → L centroid
  2. RIGHT hemisphere labels → R centroid
  3. LR axis = R_centroid - L_centroid (points from patient L to R)
  4. SI axis = perpendicular to LR, oriented superiorly
  5. Brain centroid = origin
  6. Coordinates normalized by brain half-extent along each axis

This handles arbitrary head rotation because the axes are defined by anatomy.

Usage:
    cd /Users/edt/Desktop/p-brain
    source .venv/bin/activate
    python tools/build_seg_atlas.py [--data-dir /Volumes/T5_EVO_EDT/data]
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import nibabel as nib
except ImportError:
    sys.exit("nibabel not found. pip install nibabel")

# Atlas parameters
ATLAS_GRID = 64       # grid resolution per axis (64×64)
COORD_RANGE = 2.0     # normalized coordinates span [-COORD_RANGE, +COORD_RANGE]
                       # This covers the SSS which is outside the brain (norm > 1)
Z_BINS = 10           # number of z-fraction bins (same as number of slices)

# FreeSurfer label sets
LEFT_LABELS = set(list(range(2, 29)) + list(range(1000, 1036)))  # left hemisphere
RIGHT_LABELS = set(list(range(41, 61)) + list(range(2000, 2036)))  # right hemisphere
MIDLINE_LABELS = {14, 15, 16, 24}  # 3rd ventricle, 4th ventricle, brainstem, CSF

# DKT cortical labels for consistent AP axis orientation
# Frontal: caudalmiddlefrontal, lateralorbitofrontal, medialorbitofrontal,
#   parsopercularis, parsorbitalis, parstriangularis, precentral,
#   rostralmiddlefrontal, superiorfrontal, frontalpole
FRONTAL_LABELS = {1003, 1012, 1014, 1018, 1019, 1020, 1024, 1027, 1028, 1032,
                  2003, 2012, 2014, 2018, 2019, 2020, 2024, 2027, 2028, 2032}
# Occipital: cuneus, lateraloccipital, lingual, pericalcarine
OCCIPITAL_LABELS = {1005, 1011, 1013, 1021,
                    2005, 2011, 2013, 2021}


def cnn_roi_to_nifti(vox: np.ndarray, M: int = 256) -> np.ndarray:
    """Convert CNN ROI voxel coords from rot90(k=-1) model space to NIfTI space.

    The CNN pipeline applies np.rot90(k=-1) before inference, so saved ROI
    coordinates are in that rotated space.  The inverse is rot90(k=+1):
        nifti_row = M - 1 - rot90_col
        nifti_col = rot90_row
    """
    rot_r = vox[:, 0].copy()
    rot_c = vox[:, 1].copy()
    out = np.empty_like(vox)
    out[:, 0] = M - 1 - rot_c
    out[:, 1] = rot_r
    return out


def load_cnn_rois(analysis_dir: str, img_size: int = 256) -> dict:
    """Return {('artery'|'vein', subtype, z_1indexed): (N,2) array}.

    Coordinates are converted from the CNN's rot90(k=-1) model space
    back to original NIfTI voxel space.
    """
    rois = {}
    roi_base = os.path.join(analysis_dir, "ROI Data")
    if not os.path.isdir(roi_base):
        return rois
    for vessel in ("Artery", "Vein"):
        vdir = os.path.join(roi_base, vessel)
        if not os.path.isdir(vdir):
            continue
        for subtype in os.listdir(vdir):
            sdir = os.path.join(vdir, subtype)
            if not os.path.isdir(sdir):
                continue
            for f in sorted(os.listdir(sdir)):
                if f.startswith("._") or not f.endswith(".npy"):
                    continue
                z = int(f.replace("ROI_voxels_slice_", "").replace(".npy", ""))
                arr = np.load(os.path.join(sdir, f))
                rois[(vessel.lower(), subtype, z)] = cnn_roi_to_nifti(arr, img_size)
    return rois


def compute_slice_geometry(seg_slice: np.ndarray, ap_reference: np.ndarray | None = None) -> dict | None:
    """Compute rotation-invariant coordinate frame for one axial slice.

    Args:
        seg_slice: 2D segmentation array for one slice
        ap_reference: if provided, a unit vector pointing toward anterior.
            Used to consistently orient the perpendicular axis across slices.

    Returns dict with:
      centroid: (r, c) brain centroid
      lr_axis: unit vector from patient Left to Right hemisphere
      si_axis: unit vector toward anterior (positive = frontal, despite name)
      half_lr: half-extent of brain along LR axis (pixels)
      half_si: half-extent of brain along perp axis (pixels)

    Returns None if insufficient segmentation.
    """
    brain = seg_slice > 0
    if brain.sum() < 100:
        return None

    # Separate left and right hemisphere voxels
    left_mask = np.isin(seg_slice, list(LEFT_LABELS))
    right_mask = np.isin(seg_slice, list(RIGHT_LABELS))

    # Need both hemispheres for axis definition
    if left_mask.sum() < 50 or right_mask.sum() < 50:
        return None  # Can't define axes without hemisphere labels

    # Compute hemisphere centroids
    lr, lc = np.where(left_mask)
    rr, rc = np.where(right_mask)
    l_centroid = np.array([lr.mean(), lc.mean()])
    r_centroid = np.array([rr.mean(), rc.mean()])

    # Brain centroid (all brain voxels)
    br, bc = np.where(brain)
    centroid = np.array([br.mean(), bc.mean()])

    # LR axis: from Left centroid to Right centroid
    lr_vec = r_centroid - l_centroid
    lr_len = np.linalg.norm(lr_vec)
    if lr_len < 5:  # hemispheres too close
        return None
    lr_axis = lr_vec / lr_len

    # Perpendicular axis (in-plane, orthogonal to LR)
    perp_axis = np.array([-lr_axis[1], lr_axis[0]])

    # Orient consistently using the AP reference direction
    if ap_reference is not None:
        if np.dot(perp_axis, ap_reference) < 0:
            perp_axis = -perp_axis

    # Half-extents: spread of brain voxels along each axis
    lr_proj = (np.stack([br - centroid[0], bc - centroid[1]], axis=1) @ lr_axis)
    ap_proj = (np.stack([br - centroid[0], bc - centroid[1]], axis=1) @ perp_axis)
    half_lr = max(float(np.percentile(np.abs(lr_proj), 95)), 5.0)
    half_ap = max(float(np.percentile(np.abs(ap_proj), 95)), 5.0)

    return {
        "centroid": centroid,
        "lr_axis": lr_axis,
        "si_axis": perp_axis,   # positive = anterior
        "half_lr": half_lr,
        "half_si": half_ap,
        "has_hemispheres": True,
    }


def compute_volume_geometries(seg_data: np.ndarray) -> dict:
    """Compute per-slice geometry with globally consistent AP orientation.

    Determines the anterior direction ONCE from the slice with the best
    frontal+occipital cortical label coverage, then applies it to all slices.

    Args:
        seg_data: 3D segmentation volume (rows, cols, slices)

    Returns:
        {z_0indexed: geometry_dict} for each slice with valid geometry
    """
    zdim = seg_data.shape[2]

    # Step 1: find the best slice for determining AP direction
    # (slice with most frontal+occipital cortical label voxels)
    best_z = None
    best_score = 0
    for z in range(zdim):
        seg_slice = seg_data[:, :, z]
        frontal_count = np.isin(seg_slice, list(FRONTAL_LABELS)).sum()
        occipital_count = np.isin(seg_slice, list(OCCIPITAL_LABELS)).sum()
        score = min(frontal_count, occipital_count)
        if score > best_score:
            best_score = score
            best_z = z

    # Step 2: compute AP reference direction from best slice
    ap_reference = None
    if best_z is not None and best_score > 50:
        seg_slice = seg_data[:, :, best_z]
        # Compute raw geometry (no AP reference yet)
        geom = compute_slice_geometry(seg_slice)
        if geom is not None:
            perp = geom["si_axis"]
            centroid = geom["centroid"]
            fr, fc = np.where(np.isin(seg_slice, list(FRONTAL_LABELS)))
            frontal_proj = float(np.mean(
                np.stack([fr - centroid[0], fc - centroid[1]], axis=1) @ perp
            ))
            ocr, occ = np.where(np.isin(seg_slice, list(OCCIPITAL_LABELS)))
            occipital_proj = float(np.mean(
                np.stack([ocr - centroid[0], occ - centroid[1]], axis=1) @ perp
            ))
            # Orient: frontal should have positive projection
            if frontal_proj < occipital_proj:
                ap_reference = -perp
            else:
                ap_reference = perp.copy()

    # Step 3: compute all slice geometries with consistent AP orientation
    geometries = {}
    for z in range(zdim):
        geom = compute_slice_geometry(seg_data[:, :, z], ap_reference=ap_reference)
        if geom is not None:
            geometries[z] = geom

    return geometries


def voxel_to_norm_coords(r, c, geom):
    """Transform image (row, col) to normalized brain coordinates.

    Returns (lr_norm, si_norm) each in [-COORD_RANGE, +COORD_RANGE].
    """
    offset = np.array([r - geom["centroid"][0], c - geom["centroid"][1]])
    lr = float(offset @ geom["lr_axis"]) / geom["half_lr"]
    si = float(offset @ geom["si_axis"]) / geom["half_si"]
    return lr, si


def norm_to_grid(lr, si, G=ATLAS_GRID, R=COORD_RANGE):
    """Map normalized (lr, si) to atlas grid indices."""
    gi = int((lr + R) / (2 * R) * G)
    gj = int((si + R) / (2 * R) * G)
    return max(0, min(G - 1, gi)), max(0, min(G - 1, gj))


def process_dataset(ds_path, data_accum):
    """Process one dataset: extract CNN ROI positions in seg-based coordinates."""
    nifti_dir = os.path.join(ds_path, "NIfTI")
    analysis_dir = os.path.join(ds_path, "Analysis")

    if not os.path.isdir(nifti_dir) or not os.path.isdir(analysis_dir):
        return False

    seg_path = os.path.join(
        nifti_dir, "segmentation/segmentation/mri/aparc.DKTatlas+aseg.deep_in_DCE.nii.gz"
    )
    if not os.path.exists(seg_path):
        return False

    cnn_rois = load_cnn_rois(analysis_dir)
    if not cnn_rois:
        return False

    seg = np.asarray(nib.load(seg_path).dataobj)
    n_slices = seg.shape[2]

    # Pre-compute geometry for all slices with globally consistent AP
    geometries = compute_volume_geometries(seg)

    if not geometries:
        return False

    ds_name = os.path.basename(ds_path)
    processed_any = False

    for (vtype, subtype, z1), vox in cnn_rois.items():
        z0 = z1 - 1
        if z0 not in geometries:
            continue

        geom = geometries[z0]
        z_frac = z0 / max(1, n_slices - 1)
        z_bin = min(Z_BINS - 1, int(z_frac * Z_BINS))

        # Transform ROI centroid to normalized coordinates
        roi_r = float(vox[:, 0].mean())
        roi_c = float(vox[:, 1].mean())
        lr_norm, si_norm = voxel_to_norm_coords(roi_r, roi_c, geom)

        vessel = "art" if vtype == "artery" else "vein"
        data_accum.setdefault(vessel, []).append({
            "ds": ds_name,
            "z": z1,
            "z_bin": z_bin,
            "z_frac": z_frac,
            "lr_norm": lr_norm,
            "si_norm": si_norm,
            "n_vox": len(vox),
            "subtype": subtype,
            "has_hemi": geom["has_hemispheres"],
        })

        # Also add individual voxels to density map (sample up to 100)
        for vi in range(min(vox.shape[0], 100)):
            r, c = float(vox[vi, 0]), float(vox[vi, 1])
            lr_v, si_v = voxel_to_norm_coords(r, c, geom)
            gi, gj = norm_to_grid(lr_v, si_v)
            data_accum.setdefault(f"{vessel}_grid", np.zeros((Z_BINS, ATLAS_GRID, ATLAS_GRID), dtype=np.float32))
            data_accum[f"{vessel}_grid"][z_bin, gi, gj] += 1.0

        processed_any = True

    return processed_any


def build_atlas(data_dir: str, output_path: str):
    data_accum = {
        "art_grid": np.zeros((Z_BINS, ATLAS_GRID, ATLAS_GRID), dtype=np.float32),
        "vein_grid": np.zeros((Z_BINS, ATLAS_GRID, ATLAS_GRID), dtype=np.float32),
    }

    datasets = sorted([
        d for d in os.listdir(data_dir)
        if d.startswith("20") and os.path.isdir(os.path.join(data_dir, d))
    ])

    n_ok = 0
    for i, ds_name in enumerate(datasets):
        ds_path = os.path.join(data_dir, ds_name)
        ok = process_dataset(ds_path, data_accum)
        if ok:
            n_ok += 1
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(datasets)} datasets ({n_ok} usable)...")

    print(f"\nProcessed {n_ok}/{len(datasets)} datasets with CNN ROIs + segmentation")

    art_entries = data_accum.get("art", [])
    vein_entries = data_accum.get("vein", [])
    print(f"Artery entries: {len(art_entries)}, Vein entries: {len(vein_entries)}")

    # Print coordinate statistics
    if art_entries:
        lr = [e["lr_norm"] for e in art_entries]
        si = [e["si_norm"] for e in art_entries]
        print(f"\nArtery normalized coords:")
        print(f"  LR: {np.mean(lr):.3f} +/- {np.std(lr):.3f} (range {np.min(lr):.3f} to {np.max(lr):.3f})")
        print(f"  SI: {np.mean(si):.3f} +/- {np.std(si):.3f} (range {np.min(si):.3f} to {np.max(si):.3f})")
        subs = {}
        for e in art_entries:
            subs.setdefault(e["subtype"], []).append(e)
        for sub, entries in sorted(subs.items()):
            lr_s = [e["lr_norm"] for e in entries]
            si_s = [e["si_norm"] for e in entries]
            print(f"    {sub}: n={len(entries)}, LR={np.mean(lr_s):.3f}+/-{np.std(lr_s):.3f}, SI={np.mean(si_s):.3f}+/-{np.std(si_s):.3f}")

    if vein_entries:
        lr = [e["lr_norm"] for e in vein_entries]
        si = [e["si_norm"] for e in vein_entries]
        print(f"\nVein normalized coords:")
        print(f"  LR: {np.mean(lr):.3f} +/- {np.std(lr):.3f} (range {np.min(lr):.3f} to {np.max(lr):.3f})")
        print(f"  SI: {np.mean(si):.3f} +/- {np.std(si):.3f} (range {np.min(si):.3f} to {np.max(si):.3f})")

    # Print grid peak info
    art_grid = data_accum["art_grid"]
    vein_grid = data_accum["vein_grid"]
    print(f"\nArt grid peak values per z-bin:")
    for zb in range(Z_BINS):
        mx = art_grid[zb].max()
        if mx > 0:
            idx = np.unravel_index(art_grid[zb].argmax(), art_grid[zb].shape)
            print(f"  z_bin {zb}: peak={mx:.1f} at grid ({idx[0]}, {idx[1]})")

    print(f"Vein grid peak values per z-bin:")
    for zb in range(Z_BINS):
        mx = vein_grid[zb].max()
        if mx > 0:
            idx = np.unravel_index(vein_grid[zb].argmax(), vein_grid[zb].shape)
            print(f"  z_bin {zb}: peak={mx:.1f} at grid ({idx[0]}, {idx[1]})")

    # Save atlas
    np.savez_compressed(
        output_path,
        art_maps=art_grid,
        vein_maps=vein_grid,
        grid_size=ATLAS_GRID,
        z_bins=Z_BINS,
        coord_range=COORD_RANGE,
    )
    fsize = os.path.getsize(output_path) / 1024
    print(f"\nSaved atlas to {output_path} ({fsize:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/Volumes/T5_EVO_EDT/data")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "seg_roi_atlas.npz"
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    build_atlas(args.data_dir, args.output)
