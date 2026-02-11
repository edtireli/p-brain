#!/usr/bin/env python3
"""Compare saved ITC->CTC conversion against MATLAB menu_5 case12 method1.

This script is meant to debug p-brain-web TurboFLASH conversion.

Inputs (example from user report):
- ITC_slice_2.npy: 1D intensity curve (T timepoints)
- CTC_slice_2.npy: 1D concentration curve (T timepoints)
- ROI_voxels_slice_2.npy: (N,2) voxel coordinates in the fitted map grid
- voxel_M0_matrix.pkl / voxel_T1_matrix.pkl: (H,W,Z) fitted maps

We recompute CTC using the exact closed-form case12/method1 formula:
  c(t) = -(1/beta) * ( (1/TI_dyn)*log(1 - s(t)/(M0*sin(fa))) + r1_pre )
then apply the MATLAB baseline correction:
  c = c - mean(c[1:baseline_frames])
  c[:baseline_frames] = 0

Notes:
- slice index ambiguity (0-based vs 1-based) is handled by trying both.
- flip angle and TI_dyn can be overridden.
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def _baseline_correct_case12_method1(c_raw: np.ndarray, baseline_frames: int) -> np.ndarray:
    baseline_frames = int(max(1, baseline_frames))
    c_raw = np.asarray(c_raw, dtype=float)

    if c_raw.ndim != 1:
        raise ValueError(f"Expected 1D time-series; got shape {c_raw.shape}")

    if c_raw.size >= baseline_frames and baseline_frames >= 2:
        start = 1
        stop = int(min(baseline_frames, c_raw.size))
        baseline = float(np.nanmean(c_raw[start:stop]))
        c = c_raw - baseline
        c[:stop] = 0.0
        return c

    if c_raw.size >= 1 and baseline_frames == 1:
        c = c_raw.copy()
        c[:1] = 0.0
        return c

    # Best-effort fallback
    baseline = float(np.nanmean(c_raw)) if c_raw.size else float("nan")
    return c_raw - baseline


def _case12_method1(
    s: np.ndarray,
    *,
    m0: float,
    t1_ms: float,
    flip_angle_deg: float,
    ti_dyn_s: float,
    relaxivity_r1: float,
    baseline_frames: int,
) -> np.ndarray:
    """Compute concentration from intensity using menu_5 case12 method1."""

    s = np.asarray(s, dtype=float)
    if s.ndim != 1:
        raise ValueError(f"Expected 1D s(t); got shape {s.shape}")

    sin_th = math.sin(math.radians(float(flip_angle_deg)))
    if not np.isfinite(sin_th) or abs(sin_th) < 1e-8:
        return np.full_like(s, np.nan, dtype=float)

    m0 = float(m0)
    if not np.isfinite(m0) or m0 <= 0:
        return np.full_like(s, np.nan, dtype=float)

    t1_s = float(t1_ms) / 1000.0
    if not np.isfinite(t1_s) or t1_s <= 0:
        return np.full_like(s, np.nan, dtype=float)

    r1_pre = 1.0 / t1_s
    beta = float(relaxivity_r1)
    ti = float(ti_dyn_s)
    if not np.isfinite(beta) or beta <= 0 or not np.isfinite(ti) or ti <= 0:
        return np.full_like(s, np.nan, dtype=float)

    denom = m0 * sin_th
    ratio = s / denom
    ratio = np.clip(ratio, -np.inf, 1.0 - 1e-6)
    log_arg = 1.0 - ratio
    log_arg = np.clip(log_arg, 1e-6, None)

    c_raw = (-1.0 / beta) * ((1.0 / ti) * np.log(log_arg) + r1_pre)
    return _baseline_correct_case12_method1(c_raw, baseline_frames)


def _load_pickle_array(p: Path) -> np.ndarray:
    with p.open("rb") as f:
        obj = pickle.load(f)
    arr = np.asarray(obj, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array in {p}; got {arr.ndim}D")
    return arr


def _roi_mean_from_maps(m0_map: np.ndarray, t1_map: np.ndarray, roi_voxels: np.ndarray, z_index: int) -> Tuple[float, float]:
    if roi_voxels.ndim != 2 or roi_voxels.shape[1] != 2:
        raise ValueError(f"ROI voxels must be (N,2); got {roi_voxels.shape}")

    rows = roi_voxels[:, 0].astype(int)
    cols = roi_voxels[:, 1].astype(int)

    m0_vals = m0_map[rows, cols, int(z_index)]
    t1_vals = t1_map[rows, cols, int(z_index)]

    m0_mean = float(np.nanmean(m0_vals))
    t1_mean = float(np.nanmean(t1_vals))
    return m0_mean, t1_mean


def _metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = int(min(a.size, b.size))
    a = a[:n]
    b = b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return {"n": n, "nFinite": 0}
    d = a[mask] - b[mask]
    rmse = float(np.sqrt(np.mean(d * d)))
    mae = float(np.mean(np.abs(d)))
    max_abs = float(np.max(np.abs(d)))
    # Correlation (guard constant vectors)
    aa = a[mask]
    bb = b[mask]
    if float(np.std(aa)) > 0 and float(np.std(bb)) > 0:
        corr = float(np.corrcoef(aa, bb)[0, 1])
    else:
        corr = float("nan")
    return {
        "n": n,
        "nFinite": int(mask.sum()),
        "rmse": rmse,
        "mae": mae,
        "maxAbs": max_abs,
        "corr": corr,
    }


def _try_slice_indices(requested: int) -> Iterable[int]:
    # Common ambiguity: filenames use 1-based slice numbers.
    # We'll try both (requested) and (requested-1) if valid.
    yield int(requested)
    if requested > 0:
        yield int(requested - 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--itc", type=Path, required=True)
    ap.add_argument("--ctc", type=Path, required=True)
    ap.add_argument("--roi", type=Path, required=True)
    ap.add_argument("--m0", type=Path, required=True)
    ap.add_argument("--t1", type=Path, required=True)
    ap.add_argument("--slice", type=int, required=True, help="Slice index from filename (e.g. 2 for *_slice_2)")

    ap.add_argument("--flip-angle-deg", type=float, default=30.0)
    ap.add_argument("--ti-dyn-s", type=float, default=0.12)
    ap.add_argument("--relaxivity", type=float, default=4.0)
    ap.add_argument("--baseline-frames", type=int, default=20)

    ap.add_argument(
        "--rotate-maps-k",
        type=int,
        default=0,
        help="Optional in-plane rotation (np.rot90) to apply to M0/T1 maps before ROI indexing. Use -1 to match ROTATE_AC rot90(-1).",
    )

    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="Optional output .npz to save predicted curve and metadata")
    args = ap.parse_args()

    itc = np.load(args.itc)
    ctc = np.load(args.ctc)
    roi = np.load(args.roi)

    if itc.ndim != 1 or ctc.ndim != 1:
        raise SystemExit(f"Expected 1D itc/ctc. Got itc={itc.shape} ctc={ctc.shape}")

    m0_map = _load_pickle_array(args.m0)
    t1_map = _load_pickle_array(args.t1)

    k = int(args.rotate_maps_k)
    if k % 4 != 0:
        m0_map = np.rot90(m0_map, k, axes=(0, 1))
        t1_map = np.rot90(t1_map, k, axes=(0, 1))

    best = None
    for z in _try_slice_indices(args.slice):
        if z < 0 or z >= m0_map.shape[2] or z >= t1_map.shape[2]:
            continue
        m0_mean, t1_mean = _roi_mean_from_maps(m0_map, t1_map, roi, z)
        pred = _case12_method1(
            itc,
            m0=m0_mean,
            t1_ms=t1_mean,
            flip_angle_deg=args.flip_angle_deg,
            ti_dyn_s=args.ti_dyn_s,
            relaxivity_r1=args.relaxivity,
            baseline_frames=args.baseline_frames,
        )
        met = _metrics(pred, ctc)
        met.update({"sliceUsed": int(z), "sliceRequested": int(args.slice), "m0Mean": m0_mean, "t1MeanMs": t1_mean})
        if best is None or (met.get("rmse", float("inf")) < best[0].get("rmse", float("inf"))):
            best = (met, pred)

    if best is None:
        raise SystemExit("No valid slice index found for provided maps")

    met, pred = best
    print("--- case12 method1 comparison ---")
    print(f"slice requested={met['sliceRequested']} used={met['sliceUsed']}")
    print(f"ROI voxels: {roi.shape[0]}")
    print(f"m0Mean={met['m0Mean']:.6g}  t1MeanMs={met['t1MeanMs']:.6g}")
    print(f"flip={args.flip_angle_deg}deg  TI_dyn={args.ti_dyn_s}s  relaxivity={args.relaxivity}  baseline_frames={args.baseline_frames}")
    print(f"n={met.get('n')} finite={met.get('nFinite')} rmse={met.get('rmse')} mae={met.get('mae')} maxAbs={met.get('maxAbs')} corr={met.get('corr')}")

    if args.out:
        np.savez(
            args.out,
            itc=itc,
            ctc_saved=ctc,
            ctc_pred=pred,
            roi=roi,
            metrics=np.array([met], dtype=object),
        )
        print(f"wrote: {args.out}")

    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(ctc, label="CTC saved")
        plt.plot(pred, label="CTC predicted (case12 m1)")
        plt.xlabel("frame")
        plt.ylabel("mM")
        plt.title(f"slice {met['sliceUsed']} (req {met['sliceRequested']}) RMSE={met.get('rmse'):.4g}")
        plt.legend()
        plt.tight_layout()
        out_png = (args.out.with_suffix(".png") if args.out else Path("ctc_compare.png"))
        plt.savefig(out_png, dpi=150)
        print(f"wrote: {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
