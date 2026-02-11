#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np

try:
    import nibabel as nib
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"nibabel is required: {exc}")


@dataclass(frozen=True)
class MapSpec:
    name: str
    nii_path: str
    mat_key: str


_DEFAULT_SPECS: list[MapSpec] = [
    MapSpec("CBF", "CBF_per_voxel_tikhonov.nii.gz", "CBF"),
    MapSpec("Ki", "Ki_per_voxel.nii.gz", "CBKi"),
    MapSpec("Ki_patlak", "Ki_per_voxel_patlak.nii.gz", "CBKi"),
    MapSpec("MTT", "mtt_map.nii.gz", "MTT"),
    MapSpec("VP", "vp_per_voxel.nii.gz", "CBV"),
    # CTH doesn't appear in the provided MAT keys; keep it NIfTI-only unless user maps it.
    # MapSpec("CTH", "cth_map.nii.gz", "CTH"),
]


def _load_mat(mat_path: str) -> dict:
    # v7.3 is HDF5; classic is loadmat. Your file is classic.
    try:
        from scipy.io import loadmat  # type: ignore

        m = loadmat(mat_path)
        return {k: v for k, v in m.items() if not k.startswith("__")}
    except Exception as exc:
        raise RuntimeError(f"Failed to load MAT file {mat_path}: {exc}")


def _as_float32(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if np.iscomplexobj(a):
        a = np.real(a)
    a = a.astype(np.float32, copy=False)
    a[~np.isfinite(a)] = np.nan
    return a


def _ensure_same_shape(
    nii_data: np.ndarray,
    mat_data: np.ndarray,
    *,
    allow_transpose_xy: bool,
    allow_flip_ud: bool,
    allow_flip_lr: bool,
    force_transform: str | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Attempt trivial shape fixes (transpose/flip) if needed."""

    def apply_named_transform(name: str, a: np.ndarray) -> np.ndarray:
        if name == "none":
            return a
        if name == "transpose_xy":
            return np.transpose(a, (1, 0, 2))
        if name == "flip_ud":
            return a[::-1, :, :]
        if name == "flip_lr":
            return a[:, ::-1, :]
        if name == "flip_ud+flip_lr":
            return a[::-1, ::-1, :]
        if name == "transpose_xy+flip_ud":
            return np.transpose(a, (1, 0, 2))[::-1, :, :]
        if name == "transpose_xy+flip_lr":
            return np.transpose(a, (1, 0, 2))[:, ::-1, :]
        if name == "transpose_xy+flip_ud+flip_lr":
            return np.transpose(a, (1, 0, 2))[::-1, ::-1, :]
        raise ValueError(
            f"Unknown transform {name!r}. Expected one of: none, transpose_xy, flip_ud, flip_lr, flip_ud+flip_lr, transpose_xy+flip_ud, transpose_xy+flip_lr, transpose_xy+flip_ud+flip_lr"
        )

    if force_transform:
        cand = apply_named_transform(str(force_transform), mat_data)
        if cand.shape != nii_data.shape:
            raise ValueError(
                f"Forced transform {force_transform!r} produced shape {cand.shape}, expected {nii_data.shape}."
            )
        return nii_data, cand, {"transform": str(force_transform)}

    info: dict = {"transform": "none"}
    if nii_data.shape == mat_data.shape:
        return nii_data, mat_data, info

    candidates: list[tuple[str, np.ndarray]] = [("none", mat_data)]

    if allow_transpose_xy and mat_data.ndim == 3:
        candidates.append(("transpose_xy", apply_named_transform("transpose_xy", mat_data)))

    # flips operate in x/y plane
    if allow_flip_ud and mat_data.ndim == 3:
        candidates.append(("flip_ud", apply_named_transform("flip_ud", mat_data)))
        if allow_transpose_xy:
            candidates.append(("transpose_xy+flip_ud", apply_named_transform("transpose_xy+flip_ud", mat_data)))

    if allow_flip_lr and mat_data.ndim == 3:
        candidates.append(("flip_lr", apply_named_transform("flip_lr", mat_data)))
        if allow_transpose_xy:
            candidates.append(("transpose_xy+flip_lr", apply_named_transform("transpose_xy+flip_lr", mat_data)))

    if allow_flip_ud and allow_flip_lr and mat_data.ndim == 3:
        candidates.append(("flip_ud+flip_lr", apply_named_transform("flip_ud+flip_lr", mat_data)))
        if allow_transpose_xy:
            candidates.append(
                ("transpose_xy+flip_ud+flip_lr", apply_named_transform("transpose_xy+flip_ud+flip_lr", mat_data))
            )

    for name, cand in candidates:
        if cand.shape == nii_data.shape:
            return nii_data, cand, {"transform": name}

    raise ValueError(f"Shape mismatch: NIfTI {nii_data.shape} vs MAT {mat_data.shape}.")


def _stats(diff: np.ndarray) -> dict:
    d = np.asarray(diff)
    finite = np.isfinite(d)
    if not finite.any():
        return {"finite": 0}
    v = d[finite]
    return {
        "finite": int(v.size),
        "min": float(np.min(v)),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "max": float(np.max(v)),
        "p01": float(np.quantile(v, 0.01)),
        "p99": float(np.quantile(v, 0.99)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Write diff maps between p-brain NIfTI outputs and MATLAB reference .mat maps")
    ap.add_argument("--analysis-dir", required=True, help="p-brain Analysis/ directory (contains the NIfTI maps)")
    ap.add_argument("--mat", required=True, help="MATLAB reference .mat file")
    ap.add_argument("--out-dir", required=True, help="Output directory for diff NIfTIs + summary JSON")
    ap.add_argument(
        "--no-default-maps",
        action="store_true",
        help="Do not include built-in default map specs; use only --map entries.",
    )
    ap.add_argument(
        "--map",
        action="append",
        default=[],
        help="Override/add mapping as NAME:NIFTI_FILENAME:MAT_KEY (repeatable)",
    )
    ap.add_argument("--allow-transpose-xy", action="store_true", help="If shapes mismatch, allow swapping x/y on MAT arrays")
    ap.add_argument("--allow-flip-ud", action="store_true", help="If shapes mismatch, allow flipping x axis on MAT arrays")
    ap.add_argument("--allow-flip-lr", action="store_true", help="If shapes mismatch, allow flipping y axis on MAT arrays")
    ap.add_argument(
        "--force-transform",
        default=None,
        help=(
            "Force a MAT transform even if shapes already match. "
            "One of: none, transpose_xy, flip_ud, flip_lr, flip_ud+flip_lr, "
            "transpose_xy+flip_ud, transpose_xy+flip_lr, transpose_xy+flip_ud+flip_lr"
        ),
    )
    ap.add_argument(
        "--mat-scale",
        type=float,
        default=1.0,
        help="Multiply MATLAB arrays by this scalar before diffing (e.g., unit conversion).",
    )

    args = ap.parse_args()

    analysis_dir = os.path.abspath(args.analysis_dir)
    mat_path = os.path.abspath(args.mat)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    specs: list[MapSpec] = [] if args.no_default_maps else list(_DEFAULT_SPECS)

    # Custom mappings.
    for raw in args.map:
        parts = str(raw).split(":")
        if len(parts) != 3:
            raise SystemExit(f"Invalid --map {raw!r}; expected NAME:NIFTI_FILENAME:MAT_KEY")
        name, nii_fn, mat_key = parts
        specs.append(MapSpec(str(name), str(nii_fn), str(mat_key)))

    mat = _load_mat(mat_path)

    summary: dict = {
        "analysis_dir": analysis_dir,
        "mat": mat_path,
        "out_dir": out_dir,
        "maps": {},
    }

    for spec in specs:
        nii_path = os.path.join(analysis_dir, spec.nii_path)
        if not os.path.isfile(nii_path):
            summary["maps"][spec.name] = {"error": f"missing NIfTI: {nii_path}"}
            continue
        if spec.mat_key not in mat:
            summary["maps"][spec.name] = {"error": f"missing MAT key: {spec.mat_key}"}
            continue

        nii_img = nib.load(nii_path)
        nii_data = _as_float32(nii_img.get_fdata())
        mat_data = _as_float32(mat[spec.mat_key])
        if float(args.mat_scale) != 1.0:
            mat_data = (mat_data * float(args.mat_scale)).astype(np.float32, copy=False)

        try:
            nii_data2, mat_data2, xform = _ensure_same_shape(
                nii_data,
                mat_data,
                allow_transpose_xy=bool(args.allow_transpose_xy),
                allow_flip_ud=bool(args.allow_flip_ud),
                allow_flip_lr=bool(args.allow_flip_lr),
                force_transform=(str(args.force_transform) if args.force_transform else None),
            )
        except Exception as exc:
            summary["maps"][spec.name] = {
                "error": str(exc),
                "nii_shape": list(nii_data.shape),
                "mat_shape": list(mat_data.shape),
            }
            continue

        diff = nii_data2 - mat_data2
        out_path = os.path.join(out_dir, f"diff_{spec.name}_pbrain_minus_matlab.nii.gz")
        nib.save(nib.Nifti1Image(diff.astype(np.float32, copy=False), nii_img.affine, nii_img.header), out_path)

        summary["maps"][spec.name] = {
            "nii": nii_path,
            "mat_key": spec.mat_key,
            "transform": xform.get("transform", "none"),
            "out": out_path,
            "stats": _stats(diff),
        }

    with open(os.path.join(out_dir, "diff_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"Wrote outputs to: {out_dir}")
    print(f"Summary: {os.path.join(out_dir, 'diff_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
