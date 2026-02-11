#!/usr/bin/env python
"""Spot-check equivalence of validated Tikhonov implementations on real voxels.

Samples N random nonzero voxels from a 4D Ct NIfTI, runs the validated Tikhonov
solver twice (grouped vs per-voxel), and asserts identical outputs.

Example:
  python tools/tikhonov_equivalence_random_voxels.py \
    --ctc-4d "/Volumes/T5_EVO_EDT/hemisure/20240618x2_flot/Analysis/CTC Maps/ctc_4d.nii.gz" \
    --aif-npy "/Volumes/T5_EVO_EDT/hemisure/20240618x2_flot/Analysis/TSCC Data/Max/TSCC_slice_7_3.npy" \
    --n 4 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# Allow running as a standalone script from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_ctc_4d(path: str) -> np.ndarray:
    import nibabel as nib

    img = nib.load(path)
    data = img.get_fdata(dtype=np.float64)

    if data.ndim != 4:
        raise ValueError(f"Expected 4D NIfTI, got shape {data.shape}")

    # Accept either (X,Y,Z,T) or (T,X,Y,Z); normalize to (T,X,Y,Z)
    if data.shape[-1] <= 2048 and data.shape[-1] >= 10:
        # Heuristic: time is last axis in typical NIfTI
        data_txyz = np.moveaxis(data, -1, 0)
    else:
        data_txyz = data

    return np.asarray(data_txyz, dtype=np.float64)


def _infer_time_s(n_time: int, tr_s: float | None) -> np.ndarray:
    if tr_s is None:
        # Fall back: assume 1s spacing (only for equivalence, not scientific correctness)
        tr_s = 1.0
    tr_s = float(tr_s)
    if not np.isfinite(tr_s) or tr_s <= 0:
        raise ValueError("--tr-s must be a positive finite number")
    return np.arange(n_time, dtype=np.float64) * tr_s


def _choose_voxels(ctc_txyz: np.ndarray, n: int, seed: int) -> list[tuple[int, int, int]]:
    _, X, Y, Z = ctc_txyz.shape
    # Pick voxels where Ct has any nonzero samples.
    mask = np.any(ctc_txyz != 0.0, axis=0)
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("No nonzero voxels found in Ct volume")

    rng = np.random.default_rng(int(seed))
    take = min(int(n), int(coords.shape[0]))
    sel = rng.choice(coords.shape[0], size=take, replace=False)
    out = [tuple(int(v) for v in coords[i]) for i in sel]
    # coords are (x,y,z)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctc-4d", required=True, help="Path to ctc_4d.nii.gz")
    ap.add_argument("--aif-npy", required=True, help="Path to AIF .npy (TSCC)")
    ap.add_argument("--tr-s", type=float, default=None, help="Temporal resolution (seconds)")
    ap.add_argument("--n", type=int, default=4, help="Number of voxels to sample")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = ap.parse_args()

    ctc_txyz = _load_ctc_4d(args.ctc_4d)
    n_time = int(ctc_txyz.shape[0])

    time_s = _infer_time_s(n_time, args.tr_s)

    aif = np.load(args.aif_npy)
    aif = np.asarray(aif, dtype=np.float64).reshape(-1)
    if aif.size != n_time:
        raise ValueError(f"AIF length {aif.size} != n_time {n_time}")

    from models.tikhonov import build_tikhonov_solver

    solve_ct = build_tikhonov_solver(time_s=time_s, ca=aif)

    voxels = _choose_voxels(ctc_txyz, n=args.n, seed=args.seed)

    # Build Ct matrix: shape (T, N)
    Ct = np.stack([ctc_txyz[:, x, y, z] for (x, y, z) in voxels], axis=1)

    res_grouped = solve_ct(Ct, implementation="grouped")
    res_per_voxel = solve_ct(Ct, implementation="per_voxel")

    # Exact match should hold for equivalence.
    for key in ["cbf_ml_per_100g_min", "cbv_vd", "mtt_s", "cth_s", "lambda_opt"]:
        a = getattr(res_grouped, key)
        b = getattr(res_per_voxel, key)
        np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0)

    print("OK: grouped == per_voxel for", len(voxels), "voxels")
    print("voxels (x,y,z):", voxels)
    # Small numeric peek to be human-comforting
    print("cbf:", np.asarray(res_grouped.cbf_ml_per_100g_min))
    print("lambda_opt:", np.asarray(res_grouped.lambda_opt))


if __name__ == "__main__":
    main()
