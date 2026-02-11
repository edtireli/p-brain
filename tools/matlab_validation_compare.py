#!/usr/bin/env python3
"""Compare p-brain outputs against MATLAB validation reference JSON.

This is intentionally small and pragmatic: it supports the current
`*_compare.json` schema we use during validation (a short list of voxel
coordinates with reference values).

Example:
  python tools/matlab_validation_compare.py \
    --subject-dir /Volumes/T5_EVO_EDT/hemisure/20240618x2_flot \
    --compare-json cbf_voxel_tikhonov_compare.json

By default it reports:
- Direct indexing using the JSON `coord` as (x, y, z) into the NIfTI
- A MATLAB->NIfTI coordinate mapping (for datasets where the validation JSON
    coordinates are authored in MATLAB array space)
- Best simple coordinate transform (permute/flip/offset) to help diagnose
    axis conventions if results look shifted.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import nibabel as nib
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "nibabel is required for this tool. Install p-brain requirements and retry."
    ) from exc


@dataclass(frozen=True)
class Transform:
    perm: tuple[int, int, int]
    flips: tuple[bool, bool, bool]
    offset: tuple[int, int, int]


def _matlab_to_nifti_coord(
    coord_xyz: Iterable[int],
    *,
    volume_shape: tuple[int, int, int],
    mapping: str,
) -> tuple[int, int, int]:
    """Map a voxel index from MATLAB array space into NIfTI index space.

    Notes:
    - This assumes JSON coords are already 0-based (Python-style).
    - `txy_flipy` corresponds to aligning MATLAB maps to p-brain NIfTI via:
        mat_aligned = transpose_xy(mat); mat_aligned = flip_lr(mat_aligned)
      which implies:
        pb[x, y, z] ~= mat[nx-1-y, x, z]
      therefore:
        (x_pb, y_pb, z) = (y_mat, nx-1-x_mat, z)
    """

    x_m, y_m, z_m = (int(v) for v in coord_xyz)
    nx, ny, nz = (int(d) for d in volume_shape)

    if mapping == "identity":
        return (x_m, y_m, z_m)

    if mapping == "txy_flipy":
        # pb[x, y, z] = mat[nx-1-y, x, z]
        return (y_m, nx - 1 - x_m, z_m)

    raise ValueError(f"Unknown mapping: {mapping!r}")


def _load_nifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return np.asarray(img.get_fdata(), dtype=float)


def _safe_get(volume: np.ndarray, idx: tuple[int, int, int]) -> float | None:
    x, y, z = idx
    if 0 <= x < volume.shape[0] and 0 <= y < volume.shape[1] and 0 <= z < volume.shape[2]:
        val = float(volume[x, y, z])
        if np.isfinite(val):
            return val
    return None


def _apply_transform(
    coord_xyz: Iterable[int],
    *,
    transform: Transform,
    shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    c = list(coord_xyz)
    cc = [c[i] for i in transform.perm]
    dims = [shape[i] - 1 for i in range(3)]

    out: list[int] = []
    for ax, val in enumerate(cc):
        if transform.flips[ax]:
            val = dims[ax] - val
        out.append(int(val + transform.offset[ax]))

    return (out[0], out[1], out[2])


def _score_transform(
    *,
    volume: np.ndarray,
    voxels: list[dict],
    target_key: str,
    transform: Transform,
) -> tuple[float, float] | None:
    diffs = []
    for v in voxels:
        coord = v["coord"]
        target = float(v[target_key])
        if target == 0:
            return None

        idx = _apply_transform(coord, transform=transform, shape=volume.shape)
        val = _safe_get(volume, idx)
        if val is None:
            return None

        diffs.append(abs((val - target) / target * 100.0))

    return (float(np.mean(diffs)), float(np.max(diffs)))


def _find_best_transform(
    *,
    volume: np.ndarray,
    voxels: list[dict],
    target_key: str,
) -> tuple[Transform, tuple[float, float]]:
    perms = list(itertools.permutations([0, 1, 2], 3))
    flips = list(itertools.product([False, True], repeat=3))
    offsets = list(itertools.product([-1, 0, 1], repeat=3))

    best_t: Transform | None = None
    best_score: tuple[float, float] | None = None

    for perm in perms:
        for flip in flips:
            for off in offsets:
                t = Transform(perm=perm, flips=flip, offset=off)
                score = _score_transform(volume=volume, voxels=voxels, target_key=target_key, transform=t)
                if score is None:
                    continue
                if best_score is None or score < best_score:
                    best_score = score
                    best_t = t

    if best_t is None or best_score is None:
        raise RuntimeError("No valid coordinate transform found (unexpected).")

    return best_t, best_score


def _print_comparison(*, volume: np.ndarray, voxels: list[dict], target_key: str, transform: Transform) -> None:
    print("coord\tidx\tvalue\tref\t%diff")
    diffs = []
    for v in voxels:
        coord = v["coord"]
        idx = _apply_transform(coord, transform=transform, shape=volume.shape)
        val = _safe_get(volume, idx)
        ref = float(v[target_key])
        if val is None or ref == 0:
            pdiff = float("nan")
        else:
            pdiff = (val - ref) / ref * 100.0
            diffs.append(abs(pdiff))
        print(f"{coord}\t{list(idx)}\t{val if val is not None else 'NA'}\t{ref:.6f}\t{pdiff:+.3f}%")

    if diffs:
        diffs_arr = np.asarray(diffs, dtype=float)
        print(f"\nabs %diff: mean={float(diffs_arr.mean()):.3f}% max={float(diffs_arr.max()):.3f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare p-brain outputs vs MATLAB reference JSON")
    parser.add_argument(
        "--subject-dir",
        required=True,
        type=Path,
        help="Subject folder containing Analysis/ and the compare JSON.",
    )
    parser.add_argument(
        "--compare-json",
        default="cbf_voxel_tikhonov_compare.json",
        help="Compare JSON filename (relative to subject-dir) or absolute path.",
    )
    parser.add_argument(
        "--cbf-map",
        default="Analysis/CBF_per_voxel_tikhonov.nii.gz",
        help="CBF NIfTI (relative to subject-dir) or absolute path.",
    )
    parser.add_argument(
        "--target",
        choices=["cbf_ref", "cbf_ml_per_100g_min"],
        default="cbf_ref",
        help="Which reference field to compare against.",
    )
    parser.add_argument(
        "--coord-space",
        choices=["nifti", "matlab"],
        default="nifti",
        help="Whether JSON `coord` is in NIfTI index space or MATLAB array space.",
    )
    parser.add_argument(
        "--matlab-mapping",
        choices=["identity", "txy_flipy"],
        default="txy_flipy",
        help="MATLAB->NIfTI mapping to use when --coord-space=matlab.",
    )

    args = parser.parse_args()

    subject_dir: Path = args.subject_dir
    compare_path = Path(args.compare_json)
    if not compare_path.is_absolute():
        compare_path = subject_dir / compare_path

    cbf_path = Path(args.cbf_map)
    if not cbf_path.is_absolute():
        cbf_path = subject_dir / cbf_path

    if not compare_path.exists():
        raise SystemExit(f"Compare JSON not found: {compare_path}")
    if not cbf_path.exists():
        raise SystemExit(f"CBF map not found: {cbf_path}")

    compare = json.loads(compare_path.read_text())
    voxels = compare.get("voxels", [])
    if not voxels:
        raise SystemExit(f"No voxels in compare JSON: {compare_path}")

    target_key = args.target
    missing_key = [v for v in voxels if target_key not in v]
    if missing_key:
        raise SystemExit(f"Missing key '{target_key}' in one or more voxel entries")

    volume = _load_nifti(cbf_path)

    print(f"Compare: {compare_path}")
    print(f"Map:     {cbf_path}")
    if "lambda" in compare:
        print(f"Meta:    lambda={compare.get('lambda')} penalty={compare.get('penalty')}")

    naive = Transform(perm=(0, 1, 2), flips=(False, False, False), offset=(0, 0, 0))
    print("\n== Direct indexing into NIfTI (coord as x,y,z) ==")
    _print_comparison(volume=volume, voxels=voxels, target_key=target_key, transform=naive)

    if args.coord_space == "matlab":
        mapped_voxels: list[dict] = []
        for v in voxels:
            v2 = dict(v)
            coord = v["coord"]
            v2["coord"] = list(
                _matlab_to_nifti_coord(
                    coord,
                    volume_shape=tuple(int(d) for d in volume.shape[:3]),
                    mapping=str(args.matlab_mapping),
                )
            )
            mapped_voxels.append(v2)

        print("\n== MATLAB coord-space mapping -> NIfTI sampling ==")
        print(f"mapping: {args.matlab_mapping}")
        _print_comparison(volume=volume, voxels=mapped_voxels, target_key=target_key, transform=naive)

    best_t, (mean_abs, max_abs) = _find_best_transform(volume=volume, voxels=voxels, target_key=target_key)
    print("\n== Best simple transform (permute/flip/offset in [-1,0,1]) ==")
    print(f"best transform: perm={best_t.perm} flips={best_t.flips} offset={best_t.offset}")
    print(f"score: mean_abs={mean_abs:.3f}% max_abs={max_abs:.3f}%")
    _print_comparison(volume=volume, voxels=voxels, target_key=target_key, transform=best_t)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
