#!/usr/bin/env python3
"""Compare NIfTI maps against reference MATLAB maps.

Usage example:
  python tools/compare_maps_vs_mat.py \
    --mat /Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/CBF_maps_tik_PaReLLeL_offset200_3DGauss0mm_frames250_slice1-10_MR\
 contrast\ agent_Lambda_values_AUTO_TCBF_\ 22_ml_mg_min_.mat \
    --cbf /Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/Analysis/CBF_per_voxel_tikhonov.nii.gz \
    --mtt /Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/Analysis/mtt_map.nii.gz \
    --cth /Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/Analysis/cth_map.nii.gz \
    --ki  /Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/Analysis/Ki_per_voxel.nii.gz \
    --vp  /Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/Analysis/vp_per_voxel.nii.gz

The script searches the MATLAB file for variables matching each metric
(e.g., keys containing "cbf", "mtt", "cth", "ki", "vp"). Only metrics with
both NIfTI and MATLAB sources are compared.
"""

import argparse
import sys
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import imageio.v2 as iio

import numpy as np
import nibabel as nib
from scipy.io import loadmat


def _load_nifti_map(path: str) -> np.ndarray:
    img = nib.load(path)
    data = np.asarray(img.get_fdata(), dtype=float)
    if data.ndim > 3:
        data = data[..., 0]
    return data


def _select_mat_array(
    d: Dict[str, object],
    keywords: Iterable[str],
    *,
    prefer: Optional[str] = None,
) -> Optional[Tuple[str, np.ndarray]]:
    if prefer:
        v = d.get(prefer)
        if isinstance(v, np.ndarray) and v.size:
            return prefer, v
    kw = [k.lower() for k in keywords]
    best_key = None
    best_val: Optional[np.ndarray] = None
    for key, val in d.items():
        if key.startswith("__"):
            continue
        if not isinstance(val, np.ndarray):
            continue
        if not np.issubdtype(val.dtype, np.number):
            continue
        key_l = key.lower()
        if not any(k in key_l for k in kw):
            continue
        if val.size == 0:
            continue
        # Prefer higher-dimensional maps (3D over 2D).
        if best_val is None or val.ndim > best_val.ndim or val.size > best_val.size:
            best_key = key
            best_val = val
    if best_key is None or best_val is None:
        return None
    return best_key, best_val


def _align_shapes(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    shape = tuple(min(x, y) for x, y in zip(a.shape, b.shape))
    if len(shape) < 3:
        raise ValueError(f"Incompatible shapes for comparison: {a.shape} vs {b.shape}")
    sx, sy, sz = shape[:3]
    return a[:sx, :sy, :sz], b[:sx, :sy, :sz]


def _try_transforms(ref: np.ndarray, mat: np.ndarray) -> Tuple[np.ndarray, str, Dict[str, float]]:
    """Search simple flips/axis permutations to maximise correlation with ref."""

    def variants(arr: np.ndarray):
        axes = [(0, 1, 2), (1, 0, 2)]  # identity, swap xy
        flips = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
        for ax in axes:
            perm = np.transpose(arr, axes=ax)
            for f in flips:
                out = perm
                name = f"perm{ax}_flip{''.join(str(x) for x in f) or 'none'}"
                for ff in f:
                    out = np.flip(out, axis=ff)
                yield name, out

    best_corr = -np.inf
    best_name = "identity"
    best_stats: Dict[str, float] = {"corr": -np.inf}
    best_arr = mat
    for name, cand in variants(mat):
        a_al, b_al = _align_shapes(ref, cand)
        stats = _metrics(a_al, b_al)
        corr = stats.get("corr", np.nan)
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_name = name
            best_stats = stats
            best_arr = cand
    return best_arr, best_name, best_stats


def _metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "mean_diff": np.nan, "max_abs": np.nan, "corr": np.nan}
    diff = a[m] - b[m]
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    mean_diff = float(np.mean(diff))
    max_abs = float(np.max(np.abs(diff)))
    corr = float(np.corrcoef(a[m], b[m])[0, 1]) if diff.size > 1 else np.nan
    return {
        "n": int(diff.size),
        "rmse": rmse,
        "mae": mae,
        "mean_diff": mean_diff,
        "max_abs": max_abs,
        "corr": corr,
    }


def _best_scale(a: np.ndarray, b: np.ndarray) -> float:
    """Least-squares scalar to multiply b so it matches a (on finite mask)."""

    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return 1.0
    am = a[m]
    bm = b[m]
    denom = float(np.dot(bm, bm))
    if denom == 0.0:
        return 1.0
    return float(np.dot(am, bm) / denom)


def _maybe_mask_positive(a: np.ndarray, b: np.ndarray, enable: bool) -> Tuple[np.ndarray, np.ndarray]:
    if not enable:
        return a, b
    m = (np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0))
    if not np.any(m):
        return a, b
    return a[m], b[m]


def _compare_one(
    name: str,
    nifti_path: Optional[str],
    mat_vars: Dict[str, object],
    keywords: Iterable[str],
    *,
    prefer_key: Optional[str] = None,
    scale: float = 1.0,
    fit_scale: bool = False,
    mask_positive: bool = False,
    save_dir: Optional[str] = None,
    cmap: str = "magma",
    force_compare: bool = False,
    skip_if_constant_mat: bool = False,
) -> None:
    if nifti_path is None:
        return
    print(f"\n=== {name} ===")
    try:
        nifti_map = _load_nifti_map(nifti_path)
    except Exception as exc:
        print(f"[!] Failed to load NIfTI '{nifti_path}': {exc}")
        return

    selected = _select_mat_array(mat_vars, keywords, prefer=prefer_key)
    if selected is None:
        print(f"[!] No MATLAB variable found matching keywords {list(keywords)}; skipping")
        return

    mat_key, mat_arr = selected

    if skip_if_constant_mat and not force_compare:
        arr = np.asarray(mat_arr)
        if np.issubdtype(arr.dtype, np.number) and arr.size:
            finite = np.isfinite(arr)
            if np.any(finite):
                mn = float(arr[finite].min())
                mx = float(arr[finite].max())
                if mn == mx:
                    print(f"[!] MATLAB '{mat_key}' is constant (min=max={mn:.6g}); skipping (use --force-cth-compare to override)")
                    return

    if name.upper() == "CTH" and not force_compare:
        key_l = mat_key.lower()
        if key_l in {"capil_sd", "capil_mtt", "capil_mintrans"}:
            print(
                f"[!] Refusing to compare CTH against MATLAB '{mat_key}' by default. "
                f"(This often isn't a CTH map, and in 'tikf' it can be all zeros.) "
                f"Use --force-cth-compare to override or pick a true CTH key."
            )
            return

    mat_arr = np.asarray(mat_arr, dtype=float)

    # Try best orientation match via simple flips/permutations.
    aligned_mat, best_name, best_stats = _try_transforms(nifti_map, mat_arr)
    try:
        a_raw, b_raw = _align_shapes(nifti_map, aligned_mat)
    except Exception as exc:
        print(f"[!] Shape alignment failed: {exc}; NIfTI shape={nifti_map.shape}, MAT shape={aligned_mat.shape}")
        return

    # Optional fixed scale, with optional fitted scale.
    b_scaled = b_raw * float(scale)
    if fit_scale:
        a_fit, b_fit = _maybe_mask_positive(a_raw, b_scaled, mask_positive)
        lam = _best_scale(a_fit, b_fit)
        b_scaled = b_scaled * lam
        fitted_note = f" fitted_scale={lam:.6g}"
    else:
        lam = 1.0
        fitted_note = ""

    # Preserve pre-mask arrays for plotting.
    plot_a = a_raw
    plot_b = b_scaled

    # Optional mask to positives only for metrics.
    a, b = _maybe_mask_positive(a_raw, b_scaled, mask_positive)

    stats = _metrics(a, b)
    orient_note = "" if best_stats.get("corr", np.nan) <= stats.get("corr", np.nan) else f" (best orient={best_name} corr={best_stats.get('corr'):.6g})"
    print(
        f"count={stats['n']} rmse={stats['rmse']:.6g} mae={stats['mae']:.6g} "
        f"mean_diff={stats['mean_diff']:.6g} max_abs={stats['max_abs']:.6g} corr={stats['corr']:.6g}{orient_note}{fitted_note}"
    )

    if save_dir:
        try:
            import os

            os.makedirs(save_dir, exist_ok=True)
            # Decide display window from combined finite values.
            finite = np.isfinite(a) & np.isfinite(b)
            if np.any(finite):
                combined = np.concatenate([a[finite], b[finite]])
                vmin = float(np.quantile(combined, 0.02))
                vmax = float(np.quantile(combined, 0.98))
            else:
                vmin, vmax = np.nanmin(a), np.nanmax(a)

            # Ensure 3D for slicing.
            if plot_a.ndim < 3 or plot_b.ndim < 3:
                print(f"[plot] skip {name}: not 3D (a.shape={plot_a.shape}, b.shape={plot_b.shape})")
                return
            nz = min(plot_a.shape[2], plot_b.shape[2])

            def _scale(arr: np.ndarray) -> np.ndarray:
                with np.errstate(invalid="ignore"):
                    scaled = (np.clip(arr, vmin, vmax) - vmin) / (vmax - vmin + 1e-12)
                scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
                return (scaled * 255.0).astype(np.uint8)

            rows = []
            for z in range(nz):
                left = _scale(plot_a[:, :, z])
                right = _scale(plot_b[:, :, z])
                row = np.concatenate([left, right], axis=1)
                rows.append(row)
            grid = np.concatenate(rows, axis=0)
            out_path = os.path.join(save_dir, f"{name.lower()}_slices.png")
            iio.imwrite(out_path, grid)
            print(f"[plot] wrote {out_path}")
        except Exception as exc:
            print(f"[!] Failed to save slice plot for {name}: {exc}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare NIfTI maps to MATLAB reference maps")
    ap.add_argument("--mat", required=True, help="Path to MATLAB .mat containing reference maps")
    ap.add_argument("--cbf", help="Path to NIfTI CBF map")
    ap.add_argument("--mat-key-cbf", help="Preferred MATLAB variable name for CBF")
    ap.add_argument("--mtt", help="Path to NIfTI MTT map")
    ap.add_argument("--mat-key-mtt", help="Preferred MATLAB variable name for MTT")
    ap.add_argument("--cth", help="Path to NIfTI CTH map")
    ap.add_argument("--mat-key-cth", help="Preferred MATLAB variable name for CTH")
    ap.add_argument(
        "--force-cth-compare",
        action="store_true",
        help="Allow comparing CTH even if the MATLAB key looks suspicious/constant (e.g., capil_SD in tikf)",
    )
    ap.add_argument("--ki", help="Path to NIfTI Ki map")
    ap.add_argument("--mat-key-ki", help="Preferred MATLAB variable name for Ki")
    ap.add_argument("--vp", help="Path to NIfTI vp map")
    ap.add_argument("--mat-key-vp", help="Preferred MATLAB variable name for vp (e.g., lambda_Tik)")
    ap.add_argument("--mask-positive", action="store_true", help="Compare only on voxels where both maps are > 0")
    ap.add_argument("--fit-scale", action="store_true", help="Fit a scalar to multiply the MATLAB map to minimise least-squares error vs NIfTI")
    ap.add_argument("--scale-cbf", type=float, default=1.0, help="Fixed scale factor for CBF MATLAB map")
    ap.add_argument("--scale-mtt", type=float, default=1.0, help="Fixed scale factor for MTT MATLAB map")
    ap.add_argument("--scale-cth", type=float, default=1.0, help="Fixed scale factor for CTH MATLAB map")
    ap.add_argument("--scale-ki", type=float, default=1.0, help="Fixed scale factor for Ki MATLAB map")
    ap.add_argument("--scale-vp", type=float, default=1.0, help="Fixed scale factor for vp MATLAB map")
    ap.add_argument("--save-dir", help="If set, write slice-wise side-by-side PNGs for each map into this directory")
    ap.add_argument("--cmap", default="magma", help="Matplotlib colormap for slice plots (default: magma)")
    args = ap.parse_args(argv)

    try:
        mat_vars = loadmat(args.mat, squeeze_me=True, struct_as_record=False)
    except Exception as exc:
        print(f"Failed to load MATLAB file '{args.mat}': {exc}")
        return 1

    _compare_one(
        "CBF",
        args.cbf,
        mat_vars,
        keywords=("cbf", "tcbf"),
        prefer_key=args.mat_key_cbf,
        scale=args.scale_cbf,
        fit_scale=args.fit_scale,
        mask_positive=args.mask_positive,
        save_dir=args.save_dir,
        cmap=args.cmap,
    )
    _compare_one(
        "MTT",
        args.mtt,
        mat_vars,
        keywords=("mtt",),
        prefer_key=args.mat_key_mtt,
        scale=args.scale_mtt,
        fit_scale=args.fit_scale,
        mask_positive=args.mask_positive,
        save_dir=args.save_dir,
        cmap=args.cmap,
    )
    _compare_one(
        "CTH",
        args.cth,
        mat_vars,
        keywords=("cth",),
        prefer_key=args.mat_key_cth,
        scale=args.scale_cth,
        fit_scale=args.fit_scale,
        mask_positive=args.mask_positive,
        save_dir=args.save_dir,
        cmap=args.cmap,
        force_compare=args.force_cth_compare,
        skip_if_constant_mat=True,
    )
    _compare_one(
        "Ki",
        args.ki,
        mat_vars,
        keywords=("ki",),
        prefer_key=args.mat_key_ki,
        scale=args.scale_ki,
        fit_scale=args.fit_scale,
        mask_positive=args.mask_positive,
        save_dir=args.save_dir,
        cmap=args.cmap,
    )
    _compare_one(
        "vp",
        args.vp,
        mat_vars,
        keywords=("vp", "lambda"),
        prefer_key=args.mat_key_vp,
        scale=args.scale_vp,
        fit_scale=args.fit_scale,
        mask_positive=args.mask_positive,
        save_dir=args.save_dir,
        cmap=args.cmap,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
