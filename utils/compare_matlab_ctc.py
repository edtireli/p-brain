from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import nibabel as nib
from scipy.io import loadmat

from utils import settings
from utils.loading import load_dce_4d
from utils.plotting import turboflash


@dataclass(frozen=True)
class MatCtcRef:
    path: str
    s_input: np.ndarray
    c_input: np.ndarray
    time_s: np.ndarray
    beta_input: float
    r1_input_pre: float
    bw_input: np.ndarray | None
    slice_c_input: int | None
    pixels_input: int | None
    x_roi_input: np.ndarray | None
    y_roi_input: np.ndarray | None


def _load_mat_ctc_ref(path: str | Path) -> MatCtcRef:
    p = Path(path).expanduser().resolve()
    d = loadmat(str(p), squeeze_me=True, struct_as_record=False)

    def req(name: str) -> Any:
        if name not in d:
            raise KeyError(f"Missing '{name}' in mat file: {p}")
        return d[name]

    s_input = np.asarray(req("s_input"), dtype=float).reshape(-1)
    c_input = np.asarray(req("c_input"), dtype=float).reshape(-1)
    time_s = np.asarray(req("time"), dtype=float).reshape(-1)

    beta_input = float(np.asarray(req("beta_input")).reshape(-1)[0])
    r1_input_pre = float(np.asarray(req("r1_input_pre")).reshape(-1)[0])

    bw_input = None
    if "BW_input" in d:
        bw_raw = np.asarray(d["BW_input"]).squeeze()
        if bw_raw.ndim == 2:
            bw_input = bw_raw.astype(bool)

    pixels_input = None
    if "pixels_input" in d:
        try:
            pixels_input = int(np.asarray(d["pixels_input"]).reshape(-1)[0])
        except Exception:
            pixels_input = None

    x_roi_input = None
    y_roi_input = None
    if "x_roi_input" in d and "y_roi_input" in d:
        try:
            x_roi_input = np.asarray(d["x_roi_input"], dtype=float).reshape(-1)
            y_roi_input = np.asarray(d["y_roi_input"], dtype=float).reshape(-1)
            if x_roi_input.size < 3 or y_roi_input.size != x_roi_input.size:
                x_roi_input = None
                y_roi_input = None
        except Exception:
            x_roi_input = None
            y_roi_input = None

    slice_c_input = None
    if "slice_c_input" in d:
        try:
            slice_c_input = int(np.asarray(d["slice_c_input"]).reshape(-1)[0])
        except Exception:
            slice_c_input = None

    if s_input.shape != c_input.shape or s_input.shape != time_s.shape:
        raise ValueError(
            f"Shape mismatch in mat ref: s_input={s_input.shape}, c_input={c_input.shape}, time={time_s.shape}"
        )

    return MatCtcRef(
        path=str(p),
        s_input=s_input,
        c_input=c_input,
        time_s=time_s,
        beta_input=beta_input,
        r1_input_pre=r1_input_pre,
        bw_input=bw_input,
        slice_c_input=slice_c_input,
        pixels_input=pixels_input,
        x_roi_input=x_roi_input,
        y_roi_input=y_roi_input,
    )


def _poly2mask_from_matlab_xy(
    x: np.ndarray,
    y: np.ndarray,
    shape_yx: tuple[int, int],
) -> np.ndarray:
    """Rasterize MATLAB-style polygon vertices into a boolean mask.

    MATLAB convention for ROI vertices typically uses x=column, y=row in 1-based
    pixel-center coordinates (imshow). A pixel at (row=r, col=c) has center at
    (x=c+1, y=r+1) in 1-based coords.
    """

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    h, w = int(shape_yx[0]), int(shape_yx[1])
    if h <= 0 or w <= 0 or x.size < 3 or y.size != x.size:
        return np.zeros((h, w), dtype=bool)

    # Pixel centers in MATLAB 1-based coordinates.
    rr, cc = np.mgrid[0:h, 0:w]
    pts = np.column_stack((cc.reshape(-1) + 1.0, rr.reshape(-1) + 1.0))

    try:
        from matplotlib.path import Path as MplPath  # type: ignore

        poly = np.column_stack((x, y))
        inside = MplPath(poly, closed=True).contains_points(pts)
        return inside.reshape(h, w)
    except Exception:
        # Fallback for very simple axis-aligned rectangles.
        xmin = float(np.nanmin(x))
        xmax = float(np.nanmax(x))
        ymin = float(np.nanmin(y))
        ymax = float(np.nanmax(y))
        # Convert bounds to 0-based pixel indices (inclusive) where center is within bounds.
        c0 = int(np.ceil(xmin - 1.0))
        c1 = int(np.floor(xmax - 1.0))
        r0 = int(np.ceil(ymin - 1.0))
        r1 = int(np.floor(ymax - 1.0))
        c0 = max(0, min(w - 1, c0))
        c1 = max(0, min(w - 1, c1))
        r0 = max(0, min(h - 1, r0))
        r1 = max(0, min(h - 1, r1))
        m = np.zeros((h, w), dtype=bool)
        if r1 >= r0 and c1 >= c0:
            m[r0 : r1 + 1, c0 : c1 + 1] = True
        return m


def _disk_mask_center_count(
    center_rc0: tuple[int, int],
    shape_yx: tuple[int, int],
    target_pixels: int,
) -> np.ndarray:
    """Create a disk-like mask centered at (r,c) with ~target_pixels voxels."""

    h, w = int(shape_yx[0]), int(shape_yx[1])
    r0, c0 = int(center_rc0[0]), int(center_rc0[1])
    if h <= 0 or w <= 0 or target_pixels <= 0:
        return np.zeros((h, w), dtype=bool)
    r0 = max(0, min(h - 1, r0))
    c0 = max(0, min(w - 1, c0))

    # Initial radius guess.
    r = float(np.sqrt(float(target_pixels) / np.pi))
    r_int = max(0, int(round(r)))

    yy, xx = np.ogrid[:h, :w]

    best = None
    best_mask = None
    for rad in range(max(0, r_int - 5), r_int + 6):
        m = (yy - r0) ** 2 + (xx - c0) ** 2 <= float(rad * rad)
        n = int(np.count_nonzero(m))
        cand = (abs(n - int(target_pixels)), rad)
        if best is None or cand < best:
            best = cand
            best_mask = m
    return np.asarray(best_mask, dtype=bool) if best_mask is not None else np.zeros((h, w), dtype=bool)


def _infer_baseline_frames_from_c_input(c_input: np.ndarray, *, eps: float = 1e-12) -> int:
    c = np.asarray(c_input, dtype=float).reshape(-1)
    idx = np.flatnonzero(np.isfinite(c) & (np.abs(c) > eps))
    if idx.size == 0:
        return int(getattr(settings, "TURBOFLASH_BASELINE_FRAMES", 10) or 10)
    return int(idx[0])


def _nifti_validator_orient_info() -> dict[str, Any]:
    try:
        k = int((os.environ.get("P_BRAIN_VALIDATOR_NIFTI_ROT90_K") or "1").strip())
    except Exception:
        k = 1
    flip_lr = (os.environ.get("P_BRAIN_VALIDATOR_NIFTI_FLIP_LR") or "0").strip().lower() in {"1", "true", "yes", "on"}
    flip_ud = (os.environ.get("P_BRAIN_VALIDATOR_NIFTI_FLIP_UD") or "0").strip().lower() in {"1", "true", "yes", "on"}
    return {"rot90_k": int(k), "flip_lr": bool(flip_lr), "flip_ud": bool(flip_ud)}


def _apply_validator_inplane_ops(arr: np.ndarray) -> np.ndarray:
    ops = _nifti_validator_orient_info()
    out = np.asarray(arr)
    k = int(ops["rot90_k"]) % 4
    if k:
        out = np.rot90(out, k=k, axes=(0, 1))
    if bool(ops["flip_ud"]):
        out = out[::-1, ...]
    if bool(ops["flip_lr"]):
        out = out[:, ::-1, ...]
    return out


def _best_align_dce_slice_to_bw(
    dce_slice_xt: np.ndarray,
    bw: np.ndarray,
    s_ref: np.ndarray,
    *,
    max_shift_px: int = 0,
    reduce: str = "mean",
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """Return (roi_signal, alignment_info) minimizing RMSE vs s_ref.

    This is used only for MATLAB parity validation to robustly map BW_input onto
    the DCE NIfTI slice (different converters can introduce in-plane flips).
    """

    sl = np.asarray(dce_slice_xt)
    if sl.ndim != 3:
        return None, None

    bw0 = np.asarray(bw, dtype=bool)
    if bw0.ndim != 2 or not np.any(bw0):
        return None, None

    s_ref = np.asarray(s_ref, dtype=float).reshape(-1)
    best = None
    best_sig = None
    best_info = None

    max_shift_px = int(max(0, max_shift_px))
    reduce = str(reduce or "mean").strip().lower()
    if reduce not in {"mean", "max"}:
        reduce = "mean"

    for transpose_mask in (False, True):
        bw_use = bw0.T if transpose_mask else bw0
        for k in (0, 1, 2, 3):
            s2 = np.rot90(sl, k=k, axes=(0, 1)) if k else sl
            for flip_ud in (False, True):
                s3 = s2[::-1, ...] if flip_ud else s2
                for flip_lr in (False, True):
                    s4 = s3[:, ::-1, ...] if flip_lr else s3
                    if s4.shape[:2] != bw_use.shape:
                        continue

                    # Optional small translation search (mask shift) to handle off-by-one
                    # differences between MATLAB and NIfTI in-plane mapping.
                    shifts = range(-max_shift_px, max_shift_px + 1) if max_shift_px else (0,)
                    for dy in shifts:
                        for dx in shifts:
                            if dx == 0 and dy == 0:
                                msk = bw_use
                            else:
                                msk = np.zeros_like(bw_use)
                                y0 = max(0, dy)
                                y1 = min(bw_use.shape[0], bw_use.shape[0] + dy)
                                x0 = max(0, dx)
                                x1 = min(bw_use.shape[1], bw_use.shape[1] + dx)
                                if y1 <= y0 or x1 <= x0:
                                    continue
                                src_y0 = max(0, -dy)
                                src_x0 = max(0, -dx)
                                h = y1 - y0
                                w = x1 - x0
                                msk[y0:y1, x0:x1] = bw_use[src_y0 : src_y0 + h, src_x0 : src_x0 + w]

                            if not np.any(msk):
                                continue

                            if reduce == "max":
                                roi = np.asarray(np.nanmax(s4[msk, :], axis=0), dtype=float).reshape(-1)
                            else:
                                roi = np.asarray(np.nanmean(s4[msk, :], axis=0), dtype=float).reshape(-1)

                            n = min(roi.size, s_ref.size)
                            if n <= 3:
                                continue
                            a = roi[:n]
                            b = s_ref[:n]
                            m = np.isfinite(a) & np.isfinite(b)
                            if not np.any(m):
                                continue
                            d = a[m] - b[m]
                            rmse = float(np.sqrt(np.mean(d * d)))
                            cand = (
                                rmse,
                                -float(np.corrcoef(a[m], b[m])[0, 1])
                                if np.std(a[m]) > 0 and np.std(b[m]) > 0
                                else 0.0,
                            )
                            if best is None or cand < best:
                                best = cand
                                best_sig = roi
                                best_info = {
                                    "rot90_k": int(k),
                                    "flip_lr": bool(flip_lr),
                                    "flip_ud": bool(flip_ud),
                                    "transpose_mask": bool(transpose_mask),
                                    "dx": int(dx),
                                    "dy": int(dy),
                                    "reduce": str(reduce),
                                    "rmse": rmse,
                                }

    return best_sig, best_info


def _metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float | int | None]:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = int(min(a.size, b.size))
    a = a[:n]
    b = b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    if not bool(np.any(m)):
        return {"n": int(n), "n_finite": 0, "rmse": None, "mae": None, "max_abs": None, "corr": None}
    d = a[m] - b[m]
    rmse = float(np.sqrt(np.mean(d * d)))
    mae = float(np.mean(np.abs(d)))
    max_abs = float(np.max(np.abs(d)))
    # Correlation (guard constant vectors)
    aa = a[m]
    bb = b[m]
    if float(np.std(aa)) > 0 and float(np.std(bb)) > 0:
        corr = float(np.corrcoef(aa, bb)[0, 1])
    else:
        corr = None
    return {
        "n": int(n),
        "n_finite": int(np.count_nonzero(m)),
        "rmse": rmse,
        "mae": mae,
        "max_abs": max_abs,
        "corr": corr,
    }


def compare_matlab_ctc(
    subject_root: str | Path,
    *,
    mat_ctc_path: str | Path,
    dce_nifti_path: str | Path | None = None,
    flip_angle_deg: float | None = None,
    td_ms: float = 120.0,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Compare TurboFLASH CTC curve against MATLAB reference .mat.

    This compares the *curve* conversion using the canonical `turboflash()` path
    and uses MATLAB's ROI mask (BW_input) + slice index to extract ROI M0/T1 from
    p-brain fitted maps.

    Outputs:
    - `Analysis/Fitting/compare_matlab_ctc.json`
    - `Images/Fit/ctc_compare_slice<z>.png`

    Returns the JSON payload dict.
    """

    subject_p = Path(subject_root).expanduser().resolve()
    ref = _load_mat_ctc_ref(mat_ctc_path)

    if ref.bw_input is None or ref.slice_c_input is None:
        raise ValueError("MATLAB CTC reference is missing BW_input and/or slice_c_input")

    bw = np.asarray(ref.bw_input, dtype=bool)
    z1 = int(ref.slice_c_input)
    if z1 <= 0:
        raise ValueError(f"Invalid slice_c_input in ref: {z1}")

    # Infer baseline frames from the saved MATLAB/reference curve.
    baseline_frames = _infer_baseline_frames_from_c_input(ref.c_input)

    # Locate fitted maps.
    t1_map_p = subject_p / "Analysis" / "Fitting" / "t1_map.nii.gz"
    m0_map_p = subject_p / "Analysis" / "Fitting" / "m0_map.nii.gz"
    if not t1_map_p.exists() or not m0_map_p.exists():
        raise FileNotFoundError("Missing fitted maps. Expected T1/M0 under Analysis/Fitting.")

    t1_vol = np.asarray(nib.load(str(t1_map_p)).get_fdata(), dtype=float)
    m0_vol = np.asarray(nib.load(str(m0_map_p)).get_fdata(), dtype=float)

    t1_vol = _apply_validator_inplane_ops(t1_vol)
    m0_vol = _apply_validator_inplane_ops(m0_vol)

    z0 = int(z1 - 1)
    if z0 < 0 or z0 >= int(t1_vol.shape[2]):
        raise ValueError(f"slice_c_input={z1} out of range for t1_map z={t1_vol.shape[2]}")

    t1_roi_ms = float(np.nanmean(t1_vol[:, :, z0][bw]))
    m0_roi = float(np.nanmean(m0_vol[:, :, z0][bw]))

    if flip_angle_deg is None:
        # Prefer JSON sidecar if available.
        if dce_nifti_path is None:
            dce_nifti_path = subject_p / "NIfTI" / "WIPhperf120long.nii"
        dce_json = Path(str(dce_nifti_path)).with_suffix(".json")
        if dce_json.exists():
            try:
                flip_angle_deg = float(json.loads(dce_json.read_text(encoding="utf-8")).get("FlipAngle"))
            except Exception:
                flip_angle_deg = None
        if flip_angle_deg is None:
            flip_angle_deg = 30.0

    # Compute curve using production TurboFLASH conversion.
    old_baseline = int(getattr(settings, "TURBOFLASH_BASELINE_FRAMES", 10) or 10)
    try:
        settings.TURBOFLASH_BASELINE_FRAMES = int(baseline_frames)
        c_est = turboflash(
            ref.s_input,
            t1_roi_ms,
            TD=float(td_ms),
            r1=float(ref.beta_input) * 1000.0,
            m0=m0_roi,
            flip_angle_deg=float(flip_angle_deg),
            prints=False,
        )
    finally:
        settings.TURBOFLASH_BASELINE_FRAMES = old_baseline

    c_met = _metrics(c_est, ref.c_input)

    # Optional: also compare extracted ROI signal from the DCE NIfTI at the same BW mask.
    s_from_nifti = None
    s_met = None
    c_from_nifti = None
    c_from_nifti_met = None
    dce_bw_alignment = None
    dce_mask_selection = None
    if dce_nifti_path is None:
        dce_nifti_path = subject_p / "NIfTI" / "WIPhperf120long.nii"
    dce_p = Path(dce_nifti_path)
    if dce_p.exists():
        try:
            _ref_img, dce = load_dce_4d(str(dce_p), prefer_complex_mag=True, dtype=np.float32)
            dce = np.asarray(dce, dtype=float)
            if dce.ndim == 4 and z0 < dce.shape[2]:
                sl = dce[:, :, z0, :]
                # Auto-align DCE slice to MATLAB ROI definition for robust parity.
                try:
                    max_shift_px = int((os.environ.get("P_BRAIN_MATLAB_BW_MAX_SHIFT") or "2").strip())
                except Exception:
                    max_shift_px = 2

                # Candidate masks:
                # 1) BW_input as provided.
                masks: list[tuple[str, np.ndarray]] = [("BW_input", np.asarray(bw, dtype=bool))]

                # 2) Polygon mask from x/y vertices if present.
                if ref.x_roi_input is not None and ref.y_roi_input is not None:
                    poly_mask = _poly2mask_from_matlab_xy(ref.x_roi_input, ref.y_roi_input, bw.shape)
                    if poly_mask.shape == bw.shape and bool(np.any(poly_mask)):
                        masks.append(("poly2mask(x_roi_input,y_roi_input)", poly_mask))

                # 3) If MATLAB stored pixels_input, try a disk-like ROI around the BW centroid.
                if ref.pixels_input is not None and int(ref.pixels_input) > 1:
                    ys, xs = np.nonzero(bw)
                    if ys.size > 0:
                        cy = int(np.round(float(np.mean(ys))))
                        cx = int(np.round(float(np.mean(xs))))
                        disk = _disk_mask_center_count((cy, cx), bw.shape, int(ref.pixels_input))
                        if bool(np.any(disk)):
                            masks.append((f"disk(center=BW_centroid, n≈{int(ref.pixels_input)})", disk))

                # Try mean and max reduction (user suspicion: MATLAB may pick max voxel).
                best_overall = None
                best_sig = None
                best_align = None
                best_mask_name = None
                best_reduce = None
                for mask_name, mask in masks:
                    for reduce in ("mean", "max"):
                        sig, info = _best_align_dce_slice_to_bw(
                            sl,
                            mask,
                            ref.s_input,
                            max_shift_px=max_shift_px,
                            reduce=reduce,
                        )
                        if sig is None or info is None:
                            continue
                        met = _metrics(sig, ref.s_input)
                        rmse = met.get("rmse")
                        corr = met.get("corr")
                        if rmse is None:
                            continue
                        cand = (float(rmse), -float(corr) if corr is not None else 0.0)
                        if best_overall is None or cand < best_overall:
                            best_overall = cand
                            best_sig = sig
                            best_align = info
                            best_mask_name = mask_name
                            best_reduce = reduce

                s_from_nifti = best_sig
                dce_bw_alignment = best_align
                if best_mask_name is not None:
                    dce_mask_selection = {"mask": str(best_mask_name), "reduce": str(best_reduce)}

                if s_from_nifti is not None:
                    s_met = _metrics(s_from_nifti, ref.s_input)

                    # End-to-end: compute concentration from the NIfTI-extracted ROI signal.
                    try:
                        old_baseline = int(getattr(settings, "TURBOFLASH_BASELINE_FRAMES", 10) or 10)
                        try:
                            settings.TURBOFLASH_BASELINE_FRAMES = int(baseline_frames)
                            c_from_nifti = turboflash(
                                s_from_nifti,
                                t1_roi_ms,
                                TD=float(td_ms),
                                r1=float(ref.beta_input) * 1000.0,
                                m0=m0_roi,
                                flip_angle_deg=float(flip_angle_deg),
                                prints=False,
                            )
                        finally:
                            settings.TURBOFLASH_BASELINE_FRAMES = old_baseline
                        c_from_nifti_met = _metrics(c_from_nifti, ref.c_input)
                    except Exception:
                        c_from_nifti = None
                        c_from_nifti_met = None
        except Exception:
            pass

    payload: dict[str, Any] = {
        "mat_path": str(ref.path),
        "subject_root": str(subject_p),
        "nifti_orientation": _nifti_validator_orient_info(),
        "slice": {"z1": int(z1), "z0": int(z0), "roi_voxels": int(np.count_nonzero(bw))},
        "params": {
            "beta_input": float(ref.beta_input),
            "r1_input_pre": float(ref.r1_input_pre),
            "td_ms": float(td_ms),
            "flip_angle_deg": float(flip_angle_deg),
            "baseline_frames": int(baseline_frames),
        },
        "mat_roi": {
            "pixels_input": ref.pixels_input,
            "x_roi_input": ref.x_roi_input.tolist() if ref.x_roi_input is not None else None,
            "y_roi_input": ref.y_roi_input.tolist() if ref.y_roi_input is not None else None,
        },
        "roi": {"t1_ms": float(t1_roi_ms), "m0": float(m0_roi)},
        "metrics": {
            "c": c_met,
            "s_from_nifti": s_met,
            "c_from_nifti": c_from_nifti_met,
        },
        "dce_bw_alignment": dce_bw_alignment,
        "dce_mask_selection": dce_mask_selection,
    }

    if write_outputs:
        fitting_dir = subject_p / "Analysis" / "Fitting"
        fitting_dir.mkdir(parents=True, exist_ok=True)
        (fitting_dir / "compare_matlab_ctc.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        try:
            import matplotlib.pyplot as plt  # type: ignore

            out_dir = subject_p / "Images" / "Fit"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_png = out_dir / f"ctc_compare_slice{int(z1)}.png"

            fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=300, constrained_layout=True)

            ax = axes[0]
            ax.plot(ref.time_s, ref.s_input, color="black", linestyle="-.", label="s_input (MATLAB)")
            if s_from_nifti is not None:
                ax.plot(
                    ref.time_s[: s_from_nifti.size],
                    s_from_nifti,
                    color="red",
                    linestyle="-.",
                    label="s_input (NIfTI ROI)",
                )
            ax.set_title(f"ROI Signal (slice {int(z1)})")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("signal")
            ax.grid(True, which="major", alpha=0.35)
            ax.legend(loc="best")

            ax = axes[1]
            ax.plot(ref.time_s, ref.c_input, color="black", linestyle="-.", label="c_input (MATLAB)")
            ax.plot(
                ref.time_s[: c_est.size],
                c_est,
                color="red",
                linestyle="-.",
                label="c_est (p-brain turboflash)",
            )
            if c_from_nifti is not None:
                ax.plot(
                    ref.time_s[: c_from_nifti.size],
                    c_from_nifti,
                    color="tab:blue",
                    linestyle=":",
                    label="c_from_nifti (ROI signal)",
                )
            ax.set_title(
                f"ROI Concentration  rmse={payload['metrics']['c'].get('rmse')}  max_abs={payload['metrics']['c'].get('max_abs')}"
            )
            ax.set_xlabel("time (s)")
            ax.set_ylabel("mM")
            ax.grid(True, which="major", alpha=0.35)
            ax.legend(loc="best")

            fig.savefig(out_png, dpi=300)
            plt.close(fig)
        except Exception:
            pass

    return payload
