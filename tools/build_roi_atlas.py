#!/usr/bin/env python3
"""Build a probabilistic ROI atlas from CNN ground truth across all datasets.

For each dataset:
  1. Load segmentation in DCE space (brain mask + bounding box)
  2. Load CNN ROI voxels per slice
  3. Normalize ROI positions to brain-relative fractional coordinates
  4. Accumulate into per-slice probability maps (in normalized brain space)

Output: an atlas .npz with probability maps and statistics that can be used
as a spatial prior for deterministic AIF/VIF extraction.

Usage:
    cd /Users/edt/Desktop/p-brain
    source .venv/bin/activate
    python tools/build_roi_atlas.py [--data-dir /Volumes/T5_EVO_EDT/data]
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import nibabel as nib
except ImportError:
    sys.exit("nibabel not found. pip install nibabel")


ATLAS_GRID = 64  # normalized brain space resolution (64x64 per slice)


def load_cnn_rois(analysis_dir: str) -> dict:
    """Return {(vessel_type, subtype, z_1indexed): (N,2) array}."""
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
                if not f.startswith("ROI_voxels_slice_"):
                    continue
                z = int(f.replace("ROI_voxels_slice_", "").replace(".npy", ""))
                arr = np.load(os.path.join(sdir, f))
                rois[(vessel.lower(), subtype, z)] = arr
    return rois


def get_brain_bbox_per_slice(seg_data: np.ndarray) -> dict:
    """Compute brain bounding box per slice.
    
    Returns {z: (row_min, row_max, col_min, col_max)} for slices with brain.
    """
    bboxes = {}
    brain = seg_data > 0
    for z in range(seg_data.shape[2]):
        mask = brain[:, :, z]
        if not mask.any():
            continue
        rows, cols = np.where(mask)
        bboxes[z] = (int(rows.min()), int(rows.max()),
                      int(cols.min()), int(cols.max()))
    return bboxes


def normalize_roi_to_brain(voxels: np.ndarray, bbox: tuple,
                           grid_size: int = ATLAS_GRID) -> np.ndarray:
    """Convert voxel coords (N,2) to brain-relative grid coords.
    
    bbox = (rmin, rmax, cmin, cmax)
    Returns (N, 2) integer coords in [0, grid_size-1].
    """
    rmin, rmax, cmin, cmax = bbox
    rspan = max(1, rmax - rmin)
    cspan = max(1, cmax - cmin)
    
    r_frac = (voxels[:, 0].astype(np.float64) - rmin) / rspan
    c_frac = (voxels[:, 1].astype(np.float64) - cmin) / cspan
    
    r_grid = np.clip((r_frac * (grid_size - 1)).astype(int), 0, grid_size - 1)
    c_grid = np.clip((c_frac * (grid_size - 1)).astype(int), 0, grid_size - 1)
    
    return np.column_stack([r_grid, c_grid])


def process_dataset(ds_path: str) -> dict | None:
    """Extract normalized ROI positions from one dataset."""
    analysis_dir = os.path.join(ds_path, "Analysis")
    seg_path = os.path.join(ds_path, "NIfTI", "segmentation", "segmentation",
                            "mri", "aparc.DKTatlas+aseg.deep_in_DCE.nii.gz")
    
    if not os.path.isdir(analysis_dir):
        return None
    
    # Load CNN ROIs
    rois = load_cnn_rois(analysis_dir)
    if not rois:
        return None
    
    # Try to load segmentation for brain bbox; fall back to DCE mean intensity
    seg_data = None
    if os.path.exists(seg_path):
        try:
            seg_data = np.asarray(nib.load(seg_path).dataobj)
        except Exception:
            pass
    
    # If no segmentation, try to derive brain mask from DCE
    if seg_data is None:
        nifti_dir = os.path.join(ds_path, "NIfTI")
        dce_path = None
        if os.path.isdir(nifti_dir):
            for f in os.listdir(nifti_dir):
                if f.startswith("WIHp") or f.startswith("WIPhperf"):
                    if f.endswith(".nii") and "_real" not in f and "_imag" not in f:
                        dce_path = os.path.join(nifti_dir, f)
                        break
        if dce_path is None:
            return None
        try:
            dce = np.asarray(nib.load(dce_path).dataobj, dtype=np.float32)
            seg_data = (dce.mean(axis=-1) > np.percentile(dce.mean(axis=-1), 50)).astype(float)
        except Exception:
            return None
    
    # Dual bounding boxes: brain-only for arteries (precision), expanded for veins (SSS coverage)
    from scipy import ndimage
    brain_mask = seg_data > 0
    brain_bboxes = get_brain_bbox_per_slice(brain_mask.astype(float))
    expanded = ndimage.binary_dilation(brain_mask, iterations=30).astype(bool)
    expanded_bboxes = get_brain_bbox_per_slice(expanded.astype(float))
    
    result = {
        "dataset": os.path.basename(ds_path),
        "shape": seg_data.shape[:3],
        "has_seg": os.path.exists(seg_path),
        "entries": [],
    }
    
    for (vtype, subtype, z), voxels in rois.items():
        if voxels.size == 0:
            continue
        # z is 1-indexed in filenames, convert to 0-indexed for bbox lookup
        z0 = z - 1
        # Use brain bbox for arteries, expanded bbox for veins
        if vtype == "artery":
            if z0 not in brain_bboxes:
                continue
            bbox = brain_bboxes[z0]
        else:
            if z0 not in expanded_bboxes:
                continue
            bbox = expanded_bboxes[z0]
        
        norm_vox = normalize_roi_to_brain(voxels, bbox)
        centroid_norm = norm_vox.mean(axis=0)
        
        # Also compute z as fraction of total slices
        n_slices = seg_data.shape[2]
        z_frac = z0 / max(1, n_slices - 1)
        
        result["entries"].append({
            "type": vtype,
            "subtype": subtype,
            "z": z,
            "z_frac": float(z_frac),
            "n_voxels": voxels.shape[0],
            "centroid_abs": voxels.mean(axis=0).tolist(),
            "centroid_norm": centroid_norm.tolist(),
            "bbox": bbox,
            "norm_voxels": norm_vox,
        })
    
    return result


def build_atlas(all_results: list[dict]) -> dict:
    """Build the probability atlas from all dataset results."""
    G = ATLAS_GRID
    MAX_Z = 15  # max slices we'll support
    
    # Accumulate probability maps per (type, z_index)
    # We bin z into fractional ranges
    Z_BINS = 10  # 10 fractional z bins (0.0-0.1, 0.1-0.2, ...)
    
    art_maps = np.zeros((Z_BINS, G, G), dtype=np.float64)
    vein_maps = np.zeros((Z_BINS, G, G), dtype=np.float64)
    art_counts = np.zeros(Z_BINS, dtype=int)
    vein_counts = np.zeros(Z_BINS, dtype=int)
    
    # Also collect per-entry statistics
    art_centroids = []  # [(z_frac, r_norm, c_norm, n_vox)]
    vein_centroids = []
    art_subtypes = {}
    
    for result in all_results:
        for entry in result["entries"]:
            z_frac = entry["z_frac"]
            z_bin = min(Z_BINS - 1, int(z_frac * Z_BINS))
            cr, cc = entry["centroid_norm"]
            nvox = entry["n_voxels"]
            norm_vox = entry["norm_voxels"]
            
            if entry["type"] == "artery":
                for r, c in norm_vox:
                    art_maps[z_bin, r, c] += 1.0
                art_counts[z_bin] += 1
                art_centroids.append((z_frac, cr, cc, nvox, entry["subtype"]))
                art_subtypes[entry["subtype"]] = art_subtypes.get(entry["subtype"], 0) + 1
            else:
                for r, c in norm_vox:
                    vein_maps[z_bin, r, c] += 1.0
                vein_counts[z_bin] += 1
                vein_centroids.append((z_frac, cr, cc, nvox, entry["subtype"]))
    
    # Normalize maps to probabilities
    for zb in range(Z_BINS):
        if art_counts[zb] > 0:
            art_maps[zb] /= art_counts[zb]
        if vein_counts[zb] > 0:
            vein_maps[zb] /= vein_counts[zb]
    
    # Compute summary statistics
    art_arr = np.array([(a[0], a[1], a[2], a[3]) for a in art_centroids]) if art_centroids else np.zeros((0, 4))
    vein_arr = np.array([(v[0], v[1], v[2], v[3]) for v in vein_centroids]) if vein_centroids else np.zeros((0, 4))
    
    return {
        "art_maps": art_maps,
        "vein_maps": vein_maps,
        "art_counts": art_counts,
        "vein_counts": vein_counts,
        "art_centroids": art_arr,
        "vein_centroids": vein_arr,
        "art_subtypes": art_subtypes,
        "grid_size": G,
        "z_bins": Z_BINS,
    }


def main():
    parser = argparse.ArgumentParser(description="Build ROI probability atlas")
    parser.add_argument("--data-dir", default="/Volumes/T5_EVO_EDT/data")
    parser.add_argument("--output", default=None,
                        help="Output .npz path (default: data/roi_atlas.npz)")
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()
    
    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "roi_atlas.npz"
        )
    
    data_dir = args.data_dir
    datasets = sorted([
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if d.startswith("20") and os.path.isdir(os.path.join(data_dir, d))
    ])
    if args.max:
        datasets = datasets[:args.max]
    
    print(f"Processing {len(datasets)} datasets...")
    all_results = []
    n_with_seg = 0
    n_with_roi = 0
    
    for i, ds_path in enumerate(datasets):
        ds_name = os.path.basename(ds_path)
        try:
            result = process_dataset(ds_path)
            if result:
                all_results.append(result)
                n_with_roi += 1
                if result["has_seg"]:
                    n_with_seg += 1
                n = len(result["entries"])
                if (i + 1) % 20 == 0 or i == 0:
                    print(f"  [{i+1}/{len(datasets)}] {ds_name}: {n} ROI entries, seg={result['has_seg']}")
        except Exception as e:
            print(f"  [{i+1}] {ds_name}: ERROR {e}")
    
    print(f"\nProcessed: {len(all_results)}/{len(datasets)} datasets with ROIs "
          f"({n_with_seg} with segmentation)")
    
    if not all_results:
        print("No results. Exiting.")
        return
    
    atlas = build_atlas(all_results)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ATLAS SUMMARY (grid={atlas['grid_size']}x{atlas['grid_size']}, "
          f"z_bins={atlas['z_bins']})")
    print(f"{'='*60}")
    
    print(f"\nArtery entries: {atlas['art_centroids'].shape[0]}")
    print(f"  Subtypes: {atlas['art_subtypes']}")
    if atlas['art_centroids'].shape[0] > 0:
        a = atlas['art_centroids']
        print(f"  z_frac:   {a[:,0].mean():.3f} ± {a[:,0].std():.3f} "
              f"(range {a[:,0].min():.2f}-{a[:,0].max():.2f})")
        print(f"  r_norm:   {a[:,1].mean():.3f} ± {a[:,1].std():.3f}")
        print(f"  c_norm:   {a[:,2].mean():.3f} ± {a[:,2].std():.3f}")
        print(f"  n_vox:    {a[:,3].mean():.0f} ± {a[:,3].std():.0f}")
    
    print(f"\nVein entries: {atlas['vein_centroids'].shape[0]}")
    if atlas['vein_centroids'].shape[0] > 0:
        v = atlas['vein_centroids']
        print(f"  z_frac:   {v[:,0].mean():.3f} ± {v[:,0].std():.3f} "
              f"(range {v[:,0].min():.2f}-{v[:,0].max():.2f})")
        print(f"  r_norm:   {v[:,1].mean():.3f} ± {v[:,1].std():.3f}")
        print(f"  c_norm:   {v[:,2].mean():.3f} ± {v[:,2].std():.3f}")
        print(f"  n_vox:    {v[:,3].mean():.0f} ± {v[:,3].std():.0f}")
    
    print(f"\nSlice distribution:")
    for zb in range(atlas['z_bins']):
        a = atlas['art_counts'][zb]
        v = atlas['vein_counts'][zb]
        if a > 0 or v > 0:
            print(f"  z_bin {zb} ({zb/atlas['z_bins']:.1f}-{(zb+1)/atlas['z_bins']:.1f}): "
                  f"art={a}, vein={v}")
    
    # Peak probability per z-bin
    print(f"\nPeak probability per z-bin:")
    for zb in range(atlas['z_bins']):
        ap = atlas['art_maps'][zb].max()
        vp = atlas['vein_maps'][zb].max()
        if ap > 0 or vp > 0:
            print(f"  z_bin {zb}: art_peak={ap:.3f}, vein_peak={vp:.3f}")
    
    # Save atlas
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        art_maps=atlas["art_maps"].astype(np.float32),
        vein_maps=atlas["vein_maps"].astype(np.float32),
        art_counts=atlas["art_counts"],
        vein_counts=atlas["vein_counts"],
        art_centroids=atlas["art_centroids"].astype(np.float32),
        vein_centroids=atlas["vein_centroids"].astype(np.float32),
        grid_size=np.array(atlas["grid_size"]),
        z_bins=np.array(atlas["z_bins"]),
    )
    print(f"\nAtlas saved to: {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
