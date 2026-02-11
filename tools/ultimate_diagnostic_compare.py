"""Ultimate diagnostic comparison grid for p-brain vs MATLAB.

Creates a single figure (slice 5) containing:
1) T1 and M0: p-brain vs MATLAB + absolute difference
2) Concentration curves: p-brain TSCC (SSS->RICA shifted) vs MATLAB LICA slice2 scaled
3) CBF maps: multiple MATLAB CBF variants vs p-brain CBF (scaled to ml/100g/min)
4) Patlak Ki and vp: p-brain maps (scaled to ml/100g/min and %)

This script is meant to be a stable, repeatable diagnostic artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.io as sio


@dataclass(frozen=True)
class MapSpec:
    title: str
    data: np.ndarray
    vmin: float | None = None
    vmax: float | None = None


def _load_mat(path: Path) -> dict:
    return sio.loadmat(path)


def _mat_arr(mat: dict, key: str) -> np.ndarray:
    if key not in mat:
        raise KeyError(f"MAT key '{key}' not found. Available: {[k for k in mat.keys() if not k.startswith('__')][:50]}")
    arr = np.asarray(mat[key])
    return arr


def _load_pkl_array(path: Path) -> np.ndarray:
    # p-brain stores pickled numpy arrays for voxel matrices.
    import pickle

    with open(path, "rb") as f:
        obj = pickle.load(f)
    return np.asarray(obj)


def _center_crop_to(a: np.ndarray, shape_xy: tuple[int, int]) -> np.ndarray:
    ax, ay = a.shape[:2]
    tx, ty = shape_xy
    if ax == tx and ay == ty:
        return a
    sx = max(0, (ax - tx) // 2)
    sy = max(0, (ay - ty) // 2)
    return a[sx : sx + tx, sy : sy + ty, ...]


def _slice2d(vol: np.ndarray, slice_index_1based: int) -> np.ndarray:
    sl = int(slice_index_1based) - 1
    if vol.ndim == 2:
        return vol
    if vol.ndim < 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
    if sl < 0 or sl >= vol.shape[2]:
        raise IndexError(f"Slice {slice_index_1based} out of range for shape {vol.shape}")
    return np.asarray(vol[:, :, sl], dtype=float)


def _rotate_matlab_clockwise(img2d: np.ndarray) -> np.ndarray:
    """Rotate a 2D map 90 degrees clockwise.

    MATLAB-exported maps for this project are often rotated relative to the
    p-brain/NIfTI orientation.
    """

    return np.rot90(np.asarray(img2d), k=-1)


def _t1_to_r1(t1: np.ndarray) -> tuple[np.ndarray, str]:
    """Convert T1 map to R1 (1/s).

    Heuristic:
    - If T1 median > 20, assume milliseconds => R1 = 1000/T1.
    - Else assume seconds => R1 = 1/T1.
    """

    t1 = np.asarray(t1, dtype=float)
    finite = t1[np.isfinite(t1) & (t1 > 0)]
    med = float(np.nanmedian(finite)) if finite.size else float("nan")
    if np.isfinite(med) and med > 20.0:
        r1 = 1000.0 / t1
        return r1, "ms"
    r1 = 1.0 / t1
    return r1, "s"


def _finite_quantiles(x: np.ndarray, qlo=0.02, qhi=0.98) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (0.0, 1.0)
    return (float(np.quantile(x, qlo)), float(np.quantile(x, qhi)))


def _imshow(ax, img2d: np.ndarray, title: str, vmin=None, vmax=None, cmap="viridis"):
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    im = ax.imshow(img2d.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    return im


def _imshow_with_cbar(fig, ax, img2d: np.ndarray, title: str, *, vmin=None, vmax=None, cmap="viridis"):
    im = _imshow(ax, img2d, title, vmin=vmin, vmax=vmax, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.04)
    fig.colorbar(im, cax=cax)
    return im


def _resolve_tscc_curve(tscc_root: Path, slice_index_1based: int) -> tuple[Path, np.ndarray]:
    # Expected file pattern: TSCC_slice_{slice}_<artery_id>.npy (artery_id varies by dataset).
    # Prefer Max/ if present.
    candidates = list(tscc_root.rglob(f"TSCC_slice_{slice_index_1based}_*.npy"))
    if not candidates:
        raise FileNotFoundError(f"No TSCC_slice_{slice_index_1based}_*.npy under {tscc_root}")
    candidates = [p for p in candidates if p.is_file() and not p.name.startswith("._")]
    if not candidates:
        raise FileNotFoundError(f"No TSCC_slice_{slice_index_1based}_*.npy under {tscc_root} (after filtering)")
    candidates = sorted(candidates, key=lambda p: ("/Max/" not in str(p).replace("\\", "/"), str(p)))
    path = candidates[0]
    arr = np.load(path)
    return path, np.asarray(arr, dtype=float).reshape(-1)


def _resolve_mat_curve(mat: dict) -> tuple[np.ndarray, np.ndarray, str]:
    # For the LICA scaled MAT file we expect `time` and `c_input`.
    time = _mat_arr(mat, "time").reshape(-1)
    c_input = _mat_arr(mat, "c_input").reshape(-1)
    label = "MATLAB c_input"
    return time.astype(float), c_input.astype(float), label


def _choose_mat_key(mat: dict, preferred: str, *, candidates: list[str], label: str) -> str:
    if preferred in mat:
        return preferred
    for k in candidates:
        if k in mat:
            return k
    available = [k for k in mat.keys() if not k.startswith("__")]
    raise KeyError(f"Missing {label} key. Tried: {[preferred] + candidates}. Available: {available}")


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--out", required=True, type=Path, help="Output PNG path")
    ap.add_argument("--slice", default=5, type=int, help="Slice index (1-based). Default 5")

    ap.add_argument("--mat-t1m0", required=True, type=Path, help="MAT file: T1_M0_plusError_maps_.mat")
    ap.add_argument("--pbrain-t1", required=True, type=Path, help="p-brain voxel_T1_matrix.pkl")
    ap.add_argument("--pbrain-m0", required=True, type=Path, help="p-brain voxel_M0_matrix.pkl")

    ap.add_argument("--mat-curve", required=True, type=Path, help="MAT file with time + c_input (LICA scaled)")
    ap.add_argument("--pbrain-tscc-root", required=True, type=Path, help="p-brain Analysis/TSCC Data root")
    ap.add_argument(
        "--pbrain-tscc-file",
        default=None,
        type=Path,
        help="Optional explicit TSCC .npy file to plot (overrides --pbrain-tscc-root+--slice).",
    )
    ap.add_argument("--pbrain-time", required=True, type=Path, help="p-brain Analysis/Fitting/time_points_s.npy")

    ap.add_argument("--mat-tik", required=True, type=Path, help="MAT file with tik maps (CBF/CBV/MTT)")
    ap.add_argument("--pbrain-cbf", required=True, type=Path, help="p-brain CBF NIfTI (ml/100g/min)")
    ap.add_argument("--pbrain-mtt", required=True, type=Path, help="p-brain MTT NIfTI (seconds)")
    ap.add_argument(
        "--pbrain-cbv",
        default=None,
        type=Path,
        help="Optional p-brain CBV NIfTI (ml/100g). If omitted, derived as CBF*(MTT/60).",
    )

    ap.add_argument("--pbrain-ki", required=True, type=Path, help="p-brain Patlak Ki NIfTI")
    ap.add_argument("--pbrain-vp", required=True, type=Path, help="p-brain Patlak vp NIfTI")

    # Scaling: default matches earlier convention.
    ap.add_argument("--scale-cbf", default=1.0, type=float, help="Multiply p-brain CBF by this to get ml/100g/min")
    ap.add_argument("--scale-ki", default=6000.0, type=float, help="Multiply Ki by this to get ml/100g/min")
    ap.add_argument("--scale-vp", default=100.0, type=float, help="Multiply vp by this to get percent")

    ap.add_argument("--mat-key-cbf", default="CBF_Tik", help="MATLAB key for CBF map (typically CBF_Tik; stored in 1/s)")
    ap.add_argument("--mat-key-cbv", default="Vd", help="MATLAB key for CBV map (typically Vd)")
    ap.add_argument("--mat-key-mtt", default="MTT", help="MATLAB key for MTT map (typically MTT; seconds)")
    ap.add_argument("--mat-scale-cbf", default=6000.0, type=float, help="Multiply MATLAB CBF by this to get ml/100g/min")
    ap.add_argument(
        "--mat-scale-cbv",
        default=None,
        type=float,
        help="Multiply MATLAB CBV by this to get ml/100g (default: 100 for Vd, otherwise 1)",
    )

    # Optional: additional MATLAB CBF variants to show (besides mat-key-cbf).
    ap.add_argument(
        "--mat-cbf-keys-extra",
        nargs="+",
        default=["CBF_Monoexp"],
        help="Extra MAT keys for CBF variants to show alongside mat-key-cbf (e.g. CBF_Monoexp)",
    )

    args = ap.parse_args()

    sl = int(args.slice)

    # --- R1/M0 maps ---
    mat_t1m0 = _load_mat(args.mat_t1m0)
    r1_ref = _mat_arr(mat_t1m0, "r1_map")
    m0_ref = _mat_arr(mat_t1m0, "m0_map")

    t1_pb = _load_pkl_array(args.pbrain_t1)
    m0_pb = _load_pkl_array(args.pbrain_m0)

    r1_pb, t1_units = _t1_to_r1(t1_pb)

    # Align shapes if needed (common issue: p-brain saved in DCE space while MATLAB is 256x256).
    if r1_pb.shape[:2] != r1_ref.shape[:2]:
        r1_pb = _center_crop_to(r1_pb, r1_ref.shape[:2])
    if m0_pb.shape[:2] != m0_ref.shape[:2]:
        m0_pb = _center_crop_to(m0_pb, m0_ref.shape[:2])

    r1_ref_s = _rotate_matlab_clockwise(_slice2d(r1_ref, sl))
    r1_pb_s = _slice2d(r1_pb, sl)
    m0_ref_s = _rotate_matlab_clockwise(_slice2d(m0_ref, sl))
    m0_pb_s = _slice2d(m0_pb, sl)

    r1_diff = np.abs(r1_pb_s - r1_ref_s)
    m0_diff = np.abs(m0_pb_s - m0_ref_s)

    # --- Curve comparison: p-brain TSCC vs MATLAB c_input ---
    pb_time = np.load(args.pbrain_time).reshape(-1).astype(float)
    if args.pbrain_tscc_file is not None:
        tscc_path = Path(args.pbrain_tscc_file)
        tscc_curve = np.asarray(np.load(tscc_path), dtype=float).reshape(-1)
    else:
        tscc_path, tscc_curve = _resolve_tscc_curve(args.pbrain_tscc_root, sl)
    mat_curve = _load_mat(args.mat_curve)
    mat_time, mat_c_input, mat_label = _resolve_mat_curve(mat_curve)
    try:
        mat_label = f"{mat_label} ({Path(args.mat_curve).name})"
    except Exception:
        pass

    # --- CBF maps ---
    cbf_pb_img = nib.load(str(args.pbrain_cbf))
    cbf_pb = np.asarray(cbf_pb_img.dataobj, dtype=float)
    cbf_pb_s = _slice2d(cbf_pb, sl) * float(args.scale_cbf)

    tik_mat = _load_mat(args.mat_tik)

    cbf_key = _choose_mat_key(
        tik_mat,
        str(args.mat_key_cbf),
        candidates=["CBF_Tik", "CBF", "CBF_WE", "CBF_Monoexp"],
        label="CBF",
    )
    cbv_key = _choose_mat_key(
        tik_mat,
        str(args.mat_key_cbv),
        candidates=["Vd", "CBV", "CBV_p", "CBV_TissueUptake"],
        label="CBV",
    )
    mtt_key = _choose_mat_key(
        tik_mat,
        str(args.mat_key_mtt),
        candidates=["MTT"],
        label="MTT",
    )

    mat_scale_cbv = args.mat_scale_cbv
    if mat_scale_cbv is None:
        mat_scale_cbv = 100.0 if cbv_key == "Vd" else 1.0

    # Primary MATLAB tik maps (rotated to match p-brain orientation).
    cbf_mat_s = _rotate_matlab_clockwise(_slice2d(_mat_arr(tik_mat, cbf_key), sl)) * float(args.mat_scale_cbf)
    cbv_mat_s = _rotate_matlab_clockwise(_slice2d(_mat_arr(tik_mat, cbv_key), sl)) * float(mat_scale_cbv)
    mtt_mat_s = _rotate_matlab_clockwise(_slice2d(_mat_arr(tik_mat, mtt_key), sl))

    # Optional MATLAB extra CBF variants.
    mat_cbf_specs: list[MapSpec] = []
    for k in [cbf_key] + list(args.mat_cbf_keys_extra or []):
        if k not in tik_mat:
            continue
        arr = _mat_arr(tik_mat, k)
        arr_s = _rotate_matlab_clockwise(_slice2d(arr, sl))
        arr_disp = arr_s * float(args.mat_scale_cbf)
        mat_cbf_specs.append(MapSpec(title=f"MATLAB {k} (ml/100g/min) [rot cw]", data=arr_disp))

    # --- Patlak Ki/vp maps ---
    ki_img = nib.load(str(args.pbrain_ki))
    vp_img = nib.load(str(args.pbrain_vp))
    ki = np.asarray(ki_img.dataobj, dtype=float)
    vp = np.asarray(vp_img.dataobj, dtype=float)
    ki_s = _slice2d(ki, sl) * float(args.scale_ki)
    vp_s = _slice2d(vp, sl) * float(args.scale_vp)

    # --- p-brain CBV/MTT (for dedicated comparison rows) ---
    mtt_pb = np.asarray(nib.load(str(args.pbrain_mtt)).dataobj, dtype=float)
    mtt_pb_s = _slice2d(mtt_pb, sl)

    cbv_pb_s = None
    if args.pbrain_cbv is not None:
        cbv_pb = np.asarray(nib.load(str(args.pbrain_cbv)).dataobj, dtype=float)
        cbv_pb_s = _slice2d(cbv_pb, sl)
    else:
        # Derive CBV (ml/100g) from CBF (ml/100g/min) and MTT (s): CBV = CBF * (MTT/60)
        cbv_pb_s = cbf_pb_s * (mtt_pb_s / 60.0)

    # --- Layout ---
    # Row 0: R1 ref | R1 pbrain | |diff| | M0 ref | M0 pbrain | |diff|
    # Row 1: Curve plot spans full width
    # Row 2: CBF compare row (p-brain | MATLAB | |diff|)
    # Row 3: CBV compare row (p-brain | MATLAB | |diff|)
    # Row 4: MTT compare row (p-brain | MATLAB | |diff|)
    # Row 5: Patlak Ki | vp

    ncols = 6
    fig = plt.figure(figsize=(2.4 * ncols, 13.0))
    gs = fig.add_gridspec(6, ncols, height_ratios=[1.0, 0.9, 1.0, 1.0, 1.0, 0.8])

    # Row 1 (use first 6 cols)
    vmin_r1, vmax_r1 = _finite_quantiles(np.stack([r1_ref_s, r1_pb_s], axis=0))
    vmin_m0, vmax_m0 = _finite_quantiles(np.stack([m0_ref_s, m0_pb_s], axis=0))

    ax = fig.add_subplot(gs[0, 0])
    im1 = _imshow(ax, r1_ref_s, f"R1 MATLAB ref (slice {sl}) [rot cw]", vmin_r1, vmax_r1)
    ax = fig.add_subplot(gs[0, 1])
    im2 = _imshow(ax, r1_pb_s, f"R1 p-brain (from T1 {t1_units})", vmin_r1, vmax_r1)
    ax = fig.add_subplot(gs[0, 2])
    im3 = _imshow(ax, r1_diff, "|R1 diff|", *_finite_quantiles(r1_diff), cmap="magma")

    ax = fig.add_subplot(gs[0, 3])
    im4 = _imshow(ax, m0_ref_s, f"M0 MATLAB ref (slice {sl}) [rot cw]", vmin_m0, vmax_m0)
    ax = fig.add_subplot(gs[0, 4])
    im5 = _imshow(ax, m0_pb_s, "M0 p-brain", vmin_m0, vmax_m0)
    ax = fig.add_subplot(gs[0, 5])
    im6 = _imshow(ax, m0_diff, "|M0 diff|", *_finite_quantiles(m0_diff), cmap="magma")

    # Colorbars for row 0 (compact)
    cax1 = fig.add_axes([0.92, 0.79, 0.01, 0.14])
    fig.colorbar(im1, cax=cax1)
    cax2 = fig.add_axes([0.92, 0.62, 0.01, 0.14])
    fig.colorbar(im4, cax=cax2)

    # Row 2 curve plot spans all columns
    axc = fig.add_subplot(gs[1, :])
    axc.plot(pb_time[: tscc_curve.size], tscc_curve, "-b", lw=1.5, label=f"p-brain TSCC (SSS-shifted from RICA, slice {sl})\n{tscc_path.parent.name}")
    axc.plot(mat_time[: mat_c_input.size], mat_c_input, "-r", lw=1.2, label=mat_label)
    axc.set_title("Input/TSCC concentration comparison")
    axc.set_xlabel("time (s)")
    axc.set_ylabel("concentration")
    axc.grid(True, alpha=0.2)
    axc.legend(loc="upper right", fontsize=8)

    # Dedicated comparison rows (CBF/CBV/MTT): p-brain | MATLAB | abs diff.
    def add_compare_row(row_idx: int, label: str, pb2d: np.ndarray, mat2d: np.ndarray, *, cmap="viridis"):
        sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[row_idx, :], wspace=0.25)
        vmin, vmax = _finite_quantiles(np.stack([pb2d, mat2d], axis=0))
        diff = np.abs(pb2d - mat2d)
        dvmin, dvmax = _finite_quantiles(diff)

        ax0 = fig.add_subplot(sub[0, 0])
        _imshow_with_cbar(fig, ax0, pb2d, f"p-brain {label}", vmin=vmin, vmax=vmax, cmap=cmap)

        ax1 = fig.add_subplot(sub[0, 1])
        _imshow_with_cbar(fig, ax1, mat2d, f"MATLAB {label} [rot cw]", vmin=vmin, vmax=vmax, cmap=cmap)

        ax2 = fig.add_subplot(sub[0, 2])
        _imshow_with_cbar(fig, ax2, diff, f"|diff| {label}", vmin=dvmin, vmax=dvmax, cmap="inferno")

    add_compare_row(2, "CBF (ml/100g/min)", cbf_pb_s, cbf_mat_s)
    add_compare_row(3, "CBV (ml/100g)", cbv_pb_s, cbv_mat_s)
    add_compare_row(4, "MTT (s)", mtt_pb_s, mtt_mat_s)

    # Row 5 Ki/vp (left two columns)
    vmin_ki, vmax_ki = _finite_quantiles(ki_s)
    vmin_vp, vmax_vp = _finite_quantiles(vp_s)

    ax = fig.add_subplot(gs[5, 0])
    _imshow(ax, ki_s, f"p-brain Patlak Ki (scaled x{args.scale_ki:g})", vmin_ki, vmax_ki)
    ax = fig.add_subplot(gs[5, 1])
    _imshow(ax, vp_s, f"p-brain Patlak vp (scaled x{args.scale_vp:g})", vmin_vp, vmax_vp)

    for j in range(2, ncols):
        ax = fig.add_subplot(gs[5, j])
        ax.axis("off")

    fig.suptitle(f"Ultimate diagnostic comparison (slice {sl})", fontsize=12)
    fig.tight_layout(rect=[0.0, 0.0, 0.9, 0.96])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
