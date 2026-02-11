#!/usr/bin/env python3
"""Recompute MATLAB menu_5 case '12' method 1 (TurboFLASH closed form).

Given a MATLAB `conc_methodT1_map_input_MRsignal...` .mat file containing
`s_input` and ROI info, plus a T1/M0 map .mat file (with `m0_map` and `r1_map`),
this script replicates the case12/method1 conversion:

    c = (-1/(beta*TI_dyn)) * (log(1 - s_input/(m0*sin(alpha))) + TI_dyn*r1_pre)
    c -= mean(c[2:baseline_frames])
    c[0:baseline_frames] = 0

Defaults match the MATLAB prompts: baseline_frames=20, beta=4 1/(s·mM),
TI_dyn=0.12 s, flip_angle=30 deg unless overridden.

It prints RMSE/corr against the MATLAB `c_input` if present and saves a PNG.
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Canonical validated conversion (single source of truth).
from utils.plotting import turboflash


def _extract_scalar(mat: dict, key: str, default: float | None = None) -> float | None:
    if key not in mat:
        return default
    arr = np.asarray(mat[key]).squeeze()
    if arr.size == 0:
        return default
    try:
        return float(arr.flat[0])
    except Exception:
        return default


def _load_matlab_vec(mat: dict, key: str) -> np.ndarray | None:
    if key not in mat:
        return None
    arr = np.asarray(mat[key]).squeeze()
    if arr.ndim == 1:
        return np.asarray(arr, dtype=float)
    return None


def _roi_mask(mat_obj: dict) -> tuple[np.ndarray, int]:
    """Return ROI mask (2D) and zero-based slice index."""
    mask = None
    if "BW_input" in mat_obj:
        m = np.asarray(mat_obj["BW_input"])
        if m.ndim == 2:
            mask = m.astype(bool)
        elif m.ndim == 3:
            mask = np.asarray(m[..., 0], dtype=bool)
    slice_idx = int(_extract_scalar(mat_obj, "slice_c_input", 1) - 1)
    return mask, slice_idx


def _orientation_variants(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Generate common rotation/flip variants for 2D slices."""

    variants: dict[str, np.ndarray] = {}
    for k in range(4):
        variants[f"rot{k*90}"] = np.rot90(arr, k)
        variants[f"rot{k*90}_fliplr"] = np.fliplr(np.rot90(arr, k))
        variants[f"rot{k*90}_flipud"] = np.flipud(np.rot90(arr, k))
    return variants


def turboflash_case12(
    s_input: np.ndarray,
    m0_input: float,
    r1_input_pre: float,
    *,
    flip_angle_deg: float = 30.0,
    ti_dyn_s: float = 0.12,
    beta: float = 4.0,
    baseline_frames: int = 20,
) -> np.ndarray:
    # Canonical validated conversion expects T1 in ms and r1 in 1/(s*mM) scaled by 1000.
    t1_ms = (1000.0 / float(r1_input_pre)) if np.isfinite(r1_input_pre) and float(r1_input_pre) > 0 else np.nan
    r1_scaled = float(beta) * 1000.0
    return np.asarray(
        turboflash(
            np.asarray(s_input, dtype=float),
            t1_ms,
            TD=float(ti_dyn_s) * 1000.0,
            r1=r1_scaled,
            m0=float(m0_input),
            flip_angle_deg=float(flip_angle_deg),
            ctc_model="turboflash",
            baseline_frames=int(baseline_frames),
            prints=False,
        ),
        dtype=float,
    )


def _infer_baseline_frames_from_c(c_input: np.ndarray | None, eps: float = 1e-12) -> int | None:
    """Infer MATLAB length_of_baseline from saved c_input by finding first nonzero frame."""

    if c_input is None:
        return None
    c = np.asarray(c_input, dtype=float).reshape(-1)
    nz = np.flatnonzero(np.isfinite(c) & (np.abs(c) > eps))
    if nz.size == 0:
        return None
    return int(nz[0])  # MATLAB zeroing uses 1:length_of_baseline => first nonzero at index baseline (0-based)


def _load_maps(
    mat_path: str | Path | None,
    *,
    t1_pkl: str | None = None,
    m0_pkl: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load m0_map and r1_map.

    Priority:
      1) If both pickle paths are provided, load them (expects arrays in DCE grid; T1 in ms -> r1 in 1/s).
      2) Otherwise load MATLAB mat file containing `m0_map` and `r1_map`.
    """

    if t1_pkl and m0_pkl:
        with open(t1_pkl, "rb") as f:
            t1_map = np.asarray(pickle.load(f), dtype=float)
        with open(m0_pkl, "rb") as f:
            m0_map = np.asarray(pickle.load(f), dtype=float)
        r1_map = np.where(np.isfinite(t1_map) & (t1_map > 0), 1000.0 / t1_map, np.nan)
        return m0_map, r1_map

    if mat_path is None:
        raise SystemExit("No map source provided")

    t1m0 = loadmat(mat_path)
    m0_map = np.asarray(t1m0.get("m0_map"))
    r1_map = np.asarray(t1m0.get("r1_map"))
    if m0_map is None or r1_map is None:
        raise SystemExit("m0_map or r1_map missing in T1/M0 MAT")
    return m0_map, r1_map


def main():
    ap = argparse.ArgumentParser(description="Recompute MATLAB case12/method1 from s_input and T1/M0 maps.")
    ap.add_argument("mat_path", help="MAT file with s_input and ROI (conc_methodT1_map_input_MRsignal_...)" )
    ap.add_argument("t1m0_mat", help="MAT file with m0_map and r1_map (e.g., T1_M0_plusError_maps_.mat). Use '-' when supplying --maps-pkl-*.")
    ap.add_argument("--flip-angle", type=float, default=30.0, help="Flip angle in degrees (default 30)")
    ap.add_argument("--ti", type=float, default=0.12, help="TI_dyn in seconds (default 0.12)")
    ap.add_argument("--beta", type=float, default=4.0, help="Relaxivity beta in 1/(s·mM) (default 4)")
    ap.add_argument("--baseline", type=int, default=None, help="Baseline frames (default 20; use --auto-baseline to infer from c_input if present)")
    ap.add_argument("--auto-baseline", action="store_true", help="Infer baseline frames from the first nonzero entry in c_input")
    ap.add_argument("--maps-pkl-t1", help="Path to voxel_T1_matrix.pkl (T1 in ms)")
    ap.add_argument("--maps-pkl-m0", help="Path to voxel_M0_matrix.pkl")
    ap.add_argument("--auto-rotate", action="store_true", help="Try common rotations/flips of the maps slice to match the ROI; picks lowest RMSE vs MATLAB c_input if available")
    ap.add_argument("--output", default="output_case12_compare.png", help="Output PNG path")
    args = ap.parse_args()

    mat = loadmat(args.mat_path)
    s_input = _load_matlab_vec(mat, "s_input")
    c_input = _load_matlab_vec(mat, "c_input")
    time = _load_matlab_vec(mat, "time")
    if s_input is None:
        raise SystemExit("s_input not found in MAT file")
    mask, slice_idx = _roi_mask(mat)

    m0_map, r1_map = _load_maps(
        None if args.t1m0_mat == "-" else args.t1m0_mat,
        t1_pkl=args.maps_pkl_t1,
        m0_pkl=args.maps_pkl_m0,
    )

    # Compute ROI means from mask; fall back to overall mean if mask missing.
    m0_slice = np.asarray(m0_map)[..., slice_idx]
    r1_slice = np.asarray(r1_map)[..., slice_idx]

    orientations = {"identity": (m0_slice, r1_slice)}
    if args.auto_rotate:
        orientations = {
            name: (m0_variant, _orientation_variants(r1_slice)[name])
            for name, m0_variant in _orientation_variants(m0_slice).items()
            if name in _orientation_variants(r1_slice)
        }

    best_pick: tuple[str, float, float, float, float] | None = None  # (name, rmse, corr, m0_mean, r1_mean)
    for name, (m0_variant, r1_variant) in orientations.items():
        if mask is not None and mask.shape == m0_variant.shape:
            roi_mask = mask.astype(bool)
            m0_vals = m0_variant[roi_mask]
            r1_vals = r1_variant[roi_mask]
        else:
            m0_vals = m0_variant.reshape(-1)
            r1_vals = r1_variant.reshape(-1)
        m0_vals = m0_vals[np.isfinite(m0_vals)]
        r1_vals = r1_vals[np.isfinite(r1_vals)]
        if m0_vals.size == 0 or r1_vals.size == 0:
            continue
        m0_mean = float(np.mean(m0_vals))
        r1_mean = float(np.mean(r1_vals))

        baseline_frames = args.baseline
        inferred_baseline = _infer_baseline_frames_from_c(c_input) if args.auto_baseline else None
        if baseline_frames is None:
            baseline_frames = inferred_baseline if inferred_baseline is not None else 20
        elif args.auto_baseline and inferred_baseline is not None:
            baseline_frames = inferred_baseline

        c_trial = turboflash_case12(
            s_input,
            m0_mean,
            r1_mean,
            flip_angle_deg=args.flip_angle,
            ti_dyn_s=args.ti,
            beta=args.beta,
            baseline_frames=baseline_frames,
        )
        rmse_trial = corr_trial = float("nan")
        if c_input is not None and c_input.shape == c_trial.shape:
            valid = np.isfinite(c_input) & np.isfinite(c_trial)
            if valid.any():
                rmse_trial = math.sqrt(np.nanmean((c_input - c_trial) ** 2))
                corr_trial = np.corrcoef(c_input[valid], c_trial[valid])[0, 1]
        if best_pick is None or (np.isfinite(rmse_trial) and rmse_trial < best_pick[1]):
            best_pick = (name, rmse_trial, corr_trial, m0_mean, r1_mean)

    if best_pick is None:
        raise SystemExit("ROI extraction failed for all orientations")

    orient_name, rmse_orient, corr_orient, m0_input, r1_input_pre = best_pick

    baseline_frames = args.baseline
    inferred_baseline = _infer_baseline_frames_from_c(c_input) if args.auto_baseline else None
    if baseline_frames is None:
        baseline_frames = inferred_baseline if inferred_baseline is not None else 20
    elif args.auto_baseline and inferred_baseline is not None:
        baseline_frames = inferred_baseline

    c_case12 = turboflash_case12(
        s_input,
        m0_input,
        r1_input_pre,
        flip_angle_deg=args.flip_angle,
        ti_dyn_s=args.ti,
        beta=args.beta,
        baseline_frames=baseline_frames,
    )

    # Metrics if MATLAB c_input is present.
    rmse = corr = None
    if c_input is not None and c_input.shape == c_case12.shape:
        valid = np.isfinite(c_input) & np.isfinite(c_case12)
        if valid.any():
            rmse = math.sqrt(np.nanmean((c_input - c_case12) ** 2))
            corr = np.corrcoef(c_input[valid], c_case12[valid])[0, 1]

    print(f"slice_idx (0-based): {slice_idx}")
    print(f"m0_input: {m0_input:.6g}, r1_input_pre: {r1_input_pre:.6g}")
    print(f"baseline_frames: {baseline_frames}" + (" (inferred from c_input)" if inferred_baseline is not None and baseline_frames == inferred_baseline else ""))
    if args.auto_rotate:
        print(f"orientation: {orient_name}" + (" (rmse=%.6g, corr=%.6g)" % (rmse_orient, corr_orient) if np.isfinite(rmse_orient) else ""))
    if rmse is not None:
        print(f"RMSE vs MATLAB c_input: {rmse:.6g}, corr: {corr:.6g}")

    plt.figure(figsize=(11, 6))
    if c_input is not None:
        plt.plot(c_input, label="MATLAB c_input", color="k")
    plt.plot(c_case12, label="Python case12 method1", color="r", alpha=0.8)
    plt.xlabel("Frame")
    plt.ylabel("Concentration (mM)")
    plt.legend()
    title = "case12 reproduction" + (" (RMSE=%.3g)" % rmse if rmse is not None else "")
    plt.title(title)
    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
