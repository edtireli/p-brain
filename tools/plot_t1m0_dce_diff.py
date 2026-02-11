#!/usr/bin/env python3
"""Plot native vs DCE-grid T1/M0 maps and their differences.

Loads the native T1/M0 maps and the resampled DCE-grid versions, computes simple
resampled diffs (native -> DCE shape), and saves a 2x3 montage:
- T1 native, T1 in DCE grid, T1 difference (resampled native minus DCE)
- M0 native, M0 in DCE grid, M0 difference (resampled native minus DCE)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import nibabel as nib
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom
except Exception as exc:  # pragma: no cover
    sys.stderr.write(f"Missing dependency: {exc}\n")
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot T1/M0 native vs DCE-grid maps and differences.")
    subj = p.add_mutually_exclusive_group(required=True)
    subj.add_argument("--subject-root", help="Path to subject dir containing Analysis/ and Images/.")
    subj.add_argument("--id", help="Subject ID (e.g. 20240618x2_flot) when using --data-dir.")
    p.add_argument("--data-dir", help="Data root holding subject dir; required with --id.")
    p.add_argument("--out", help="Output PNG path (default: Images/Fit/Compare_T1_M0_DCE.png).")
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.subject_root:
        subject_root = Path(args.subject_root).expanduser().resolve()
    else:
        if not args.data_dir:
            raise SystemExit("--data-dir is required when using --id")
        subject_root = Path(args.data_dir).expanduser().resolve() / args.id

    default_out = subject_root / "Images" / "Fit" / "Compare_T1_M0_DCE.png"
    out_path = Path(args.out).expanduser().resolve() if args.out else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return subject_root, out_path


def _load(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asanyarray(nib.load(path).dataobj, dtype=float)


def _resample_to_shape(arr: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    if arr.shape == target_shape:
        return np.asarray(arr, dtype=float)
    factors = [t / s for t, s in zip(target_shape, arr.shape)]
    return zoom(arr, factors, order=1)


def _center_slice(vol: np.ndarray) -> np.ndarray:
    z = int(vol.shape[2] // 2)
    return np.rot90(np.asarray(vol[:, :, z], dtype=float))


def main() -> int:
    args = _parse_args()
    subject_root, out_png = _resolve_paths(args)
    fit_dir = subject_root / "Analysis" / "Fitting"

    t1_native_p = fit_dir / "t1_map.nii.gz"
    t1_dce_p = fit_dir / "t1_map_in_dce.nii.gz"
    m0_native_p = fit_dir / "m0_map.nii.gz"
    m0_dce_p = fit_dir / "m0_map_in_dce.nii.gz"

    try:
        t1_native = _load(t1_native_p)
        t1_dce = _load(t1_dce_p)
        m0_native = _load(m0_native_p)
        m0_dce = _load(m0_dce_p)
    except Exception as exc:
        sys.stderr.write(f"Failed to load maps: {exc}\n")
        return 1

    t1_native_rs = _resample_to_shape(t1_native, t1_dce.shape)
    m0_native_rs = _resample_to_shape(m0_native, m0_dce.shape)

    t1_diff = t1_native_rs - t1_dce
    m0_diff = m0_native_rs - m0_dce

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax in axes.ravel():
        ax.axis("off")

    im = axes[0, 0].imshow(_center_slice(t1_native_rs), cmap="viridis")
    axes[0, 0].set_title("T1 native (resampled)")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im = axes[0, 1].imshow(_center_slice(t1_dce), cmap="viridis")
    axes[0, 1].set_title("T1 in DCE grid")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im = axes[0, 2].imshow(_center_slice(t1_diff), cmap="coolwarm")
    axes[0, 2].set_title("T1 diff (native-DCE)")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im = axes[1, 0].imshow(_center_slice(m0_native_rs), cmap="magma")
    axes[1, 0].set_title("M0 native (resampled)")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(_center_slice(m0_dce), cmap="magma")
    axes[1, 1].set_title("M0 in DCE grid")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im = axes[1, 2].imshow(_center_slice(m0_diff), cmap="coolwarm")
    axes[1, 2].set_title("M0 diff (native-DCE)")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle("T1/M0 native vs DCE grid (center slice)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    sys.stdout.write(f"Wrote {out_png}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
