#!/usr/bin/env python3

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class MatConc:
    s_input: np.ndarray
    c_input: np.ndarray
    time_s: np.ndarray
    beta_input: float
    r1_input_pre: float
    r1_input: Optional[np.ndarray]
    bw_input: Optional[np.ndarray]
    slice_c_input: Optional[int]


def _load_mat_conc(path: str) -> MatConc:
    d = loadmat(path, squeeze_me=True, struct_as_record=False)

    def req(name: str):
        if name not in d:
            raise KeyError(f"Missing '{name}' in mat file")
        return d[name]

    s_input = np.asarray(req("s_input"), dtype=float).reshape(-1)
    c_input = np.asarray(req("c_input"), dtype=float).reshape(-1)
    time_s = np.asarray(req("time"), dtype=float).reshape(-1)

    beta_input = float(req("beta_input"))
    r1_input_pre = float(req("r1_input_pre"))

    r1_input = None
    if "r1_input" in d:
        try:
            r1_input = np.asarray(d["r1_input"], dtype=float).reshape(-1)
            if r1_input.shape != s_input.shape:
                r1_input = None
        except Exception:
            r1_input = None

    bw_input = None
    if "BW_input" in d:
        bw_raw = np.asarray(d["BW_input"]).squeeze()
        if bw_raw.ndim == 2:
            bw_input = bw_raw.astype(bool)

    slice_c_input = None
    if "slice_c_input" in d:
        try:
            slice_c_input = int(np.asarray(d["slice_c_input"]).reshape(-1)[0])
        except Exception:
            slice_c_input = None

    if s_input.shape != c_input.shape or s_input.shape != time_s.shape:
        raise ValueError(
            f"Shape mismatch: s_input={s_input.shape}, c_input={c_input.shape}, time={time_s.shape}"
        )

    return MatConc(
        s_input=s_input,
        c_input=c_input,
        time_s=time_s,
        beta_input=beta_input,
        r1_input_pre=r1_input_pre,
        r1_input=r1_input,
        bw_input=bw_input,
        slice_c_input=slice_c_input,
    )


def _load_mat_maps(path: str):
    d = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "m0_map" not in d or "r1_map" not in d:
        raise KeyError("Missing 'm0_map' and/or 'r1_map' in maps mat file")
    m0_map = np.asarray(d["m0_map"], dtype=float)
    r1_map = np.asarray(d["r1_map"], dtype=float)
    if m0_map.ndim != 3 or r1_map.ndim != 3:
        raise ValueError(f"Expected 3D maps; got m0_map={m0_map.shape}, r1_map={r1_map.shape}")
    if m0_map.shape != r1_map.shape:
        raise ValueError(f"Map shape mismatch: m0_map={m0_map.shape}, r1_map={r1_map.shape}")
    return m0_map, r1_map


def _roi_mean_from_bw(map2d: np.ndarray, bw: np.ndarray) -> float:
    if bw is None:
        raise ValueError("BW_input ROI mask not available")
    bw = np.asarray(bw, dtype=bool)
    if bw.shape != map2d.shape:
        raise ValueError(f"BW_input shape {bw.shape} does not match map shape {map2d.shape}")
    vals = np.asarray(map2d, dtype=float)[bw]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("ROI mask selects no finite voxels")
    return float(np.mean(vals))


def _guess_baseline_frames_from_c_input(c_input: np.ndarray, *, eps: float = 1e-12) -> int:
    c = np.asarray(c_input, dtype=float).reshape(-1)
    idx = np.flatnonzero(np.isfinite(c) & (np.abs(c) > eps))
    if idx.size == 0:
        return 20
    # MATLAB sets c_input(1:length_of_baseline)=0, so first nonzero is at index length_of_baseline+1 (1-based)
    # => in 0-based Python, it's at index length_of_baseline.
    return int(idx[0])


def _metrics(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not bool(np.any(mask)):
        return {
            "n": 0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "max_abs": float("nan"),
            "mean": float("nan"),
        }
    diff = a[mask] - b[mask]
    return {
        "n": int(diff.size),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "mean": float(np.mean(diff)),
    }


def _best_integer_lag_rmse(
    ref: np.ndarray,
    other: np.ndarray,
    *,
    max_abs_lag: int = 10,
) -> tuple[int, float]:
    """Find lag that best aligns `other` to `ref` by minimizing RMSE.

    Lag definition: compare ref[t] with other[t - lag].
      - lag > 0 means `other` is shifted RIGHT (delayed) relative to `ref`.
      - lag < 0 means `other` is shifted LEFT (advanced) relative to `ref`.
    """

    ref = np.asarray(ref, dtype=float).reshape(-1)
    other = np.asarray(other, dtype=float).reshape(-1)
    if ref.shape != other.shape:
        raise ValueError(f"Lag compare requires equal lengths; got {ref.shape} vs {other.shape}")

    best_lag = 0
    best_rmse = float("inf")

    for lag in range(-int(max_abs_lag), int(max_abs_lag) + 1):
        if lag < 0:
            a = ref[: lag]  # drop last -lag
            b = other[-lag:]
        elif lag > 0:
            a = ref[lag:]
            b = other[: -lag]
        else:
            a = ref
            b = other

        m = np.isfinite(a) & np.isfinite(b)
        if not bool(np.any(m)):
            continue
        diff = a[m] - b[m]
        rmse = float(np.sqrt(np.mean(diff * diff)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_lag = lag

    return best_lag, best_rmse


def main() -> int:
    # Ensure imports work when executed as a standalone script.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser(
        description="Compare MATLAB menu_5 case 12 concentration vs p-brain compute_CTC TurboFLASH case12"
    )
    ap.add_argument(
        "mat_path",
        help="Path to .mat file containing s_input, c_input, time, beta_input, r1_input_pre",
    )
    ap.add_argument(
        "--maps-mat",
        default=None,
        help="Optional .mat containing r1_map and m0_map (e.g. T1_M0_plusError_maps_.mat) to derive ROI M0/T1 automatically.",
    )
    ap.add_argument(
        "--prefer-map-r1-pre",
        action="store_true",
        help="If set and --maps-mat is provided, use ROI r1_map for r1_input_pre instead of r1_input_pre from the concentration .mat.",
    )
    ap.add_argument(
        "--slice",
        type=int,
        default=None,
        help="Slice number (1-based) to use for ROI extraction from maps. Default: slice_c_input from concentration .mat.",
    )
    ap.add_argument(
        "--method",
        default="turboflash_advanced",
        choices=["turboflash_advanced"],
        help="TurboFLASH conversion mode to use (MATLAB menu_5 case12 method1)",
    )
    ap.add_argument(
        "--baseline-frames",
        type=int,
        default=None,
        help="Baseline frame count (MATLAB length_of_baseline). Default: auto from c_input first nonzero.",
    )
    ap.add_argument(
        "--ti-ms",
        type=float,
        default=120.0,
        help="TI_dyn in ms (MATLAB TI_dyn). Default: 120.",
    )
    ap.add_argument(
        "--flip-angle-deg",
        type=float,
        default=10.0,
        help="Flip angle during bolus passage (degrees). Default: 10.",
    )
    ap.add_argument(
        "--m0",
        type=float,
        default=None,
        help="M0 (from MATLAB m0_map ROI). Required.",
    )
    ap.add_argument(
        "--auto-shift",
        action="store_true",
        help="Also report best integer lag (±10 frames) aligning curves; useful when the .mat file has been time-shifted.",
    )
    ap.add_argument(
        "--apply-best-shift",
        action="store_true",
        help="When used with --auto-shift, re-compute metrics after shifting c_est by the best lag vs MATLAB c_input.",
    )
    args = ap.parse_args()

    mat = _load_mat_conc(args.mat_path)

    baseline_frames = args.baseline_frames
    if baseline_frames is None:
        baseline_frames = _guess_baseline_frames_from_c_input(mat.c_input)

    derived_m0 = None
    derived_r1_pre = None
    if args.maps_mat is not None:
        if mat.bw_input is None:
            raise SystemExit(
                "--maps-mat was provided but BW_input is missing/unusable in the concentration .mat; can't locate ROI."
            )
        slice_1b = args.slice if args.slice is not None else mat.slice_c_input
        if slice_1b is None:
            raise SystemExit("Missing slice number: provide --slice or ensure slice_c_input exists in the .mat")
        if slice_1b <= 0:
            raise SystemExit(f"Invalid slice number (1-based): {slice_1b}")
        slice_0b = int(slice_1b) - 1

        m0_map, r1_map = _load_mat_maps(args.maps_mat)
        if slice_0b >= m0_map.shape[2]:
            raise SystemExit(
                f"Slice {slice_1b} out of range for maps with {m0_map.shape[2]} slices"
            )
        derived_m0 = _roi_mean_from_bw(m0_map[:, :, slice_0b], mat.bw_input)
        derived_r1_pre = _roi_mean_from_bw(r1_map[:, :, slice_0b], mat.bw_input)

        print(f"maps_mat: {args.maps_mat}")
        print(f"roi_slice: {slice_1b} (1-based)")
        print(f"roi_m0_mean: {derived_m0:.6g}")
        print(f"roi_r1_pre_mean: {derived_r1_pre:.6g}  (=> T1_ms={1000.0/derived_r1_pre if derived_r1_pre>0 else float('nan'):.6f})")

    m0 = args.m0
    if args.method == "turboflash_advanced" and m0 is None and derived_m0 is not None:
        m0 = derived_m0

    if args.method == "turboflash_advanced" and m0 is None:
        raise SystemExit("--m0 is required for turboflash_advanced; pass --m0 or --maps-mat")

    r1_input_pre = mat.r1_input_pre
    if args.prefer_map_r1_pre and derived_r1_pre is not None and np.isfinite(derived_r1_pre) and derived_r1_pre > 0:
        r1_input_pre = float(derived_r1_pre)

    # Convert r1_input_pre (1/s) -> T1_ms for turboflash
    t1_ms = 1000.0 / float(r1_input_pre)

    # beta_input (s^-1 mM^-1) -> turboflash expects r1 in per-1000 (historical): r1_s = r1/1000
    r1_per_1000 = mat.beta_input * 1000.0

    os.environ["P_BRAIN_TURBOFLASH_CTC_METHOD"] = args.method
    os.environ["P_BRAIN_TURBOFLASH_BASELINE_FRAMES"] = str(int(baseline_frames))

    from utils.plotting import turboflash

    # Quick sanity: ensure log argument stays positive.
    if args.method == "turboflash_advanced":
        denom = float(m0) * float(np.sin(np.radians(float(args.flip_angle_deg))))
        if denom > 0:
            ratio = np.asarray(mat.s_input, dtype=float) / denom
            frac_nonpos = float(np.mean((1.0 - ratio) <= 0.0))
            if frac_nonpos > 0.0:
                max_ratio = float(np.nanmax(ratio))
                raw_ratio = np.asarray(mat.s_input, dtype=float) / float(m0)
                raw_max = float(np.nanmax(raw_ratio))
                # Need sin(alpha) > max(s/m0) for 1 - s/(m0*sin(alpha)) > 0.
                need = min(1.0, raw_max)
                alpha_min_deg = float(np.degrees(np.arcsin(need))) if need > 0 else 0.0
                print(
                    f"warning: method1 log arg <=0 for {100*frac_nonpos:.1f}% samples; "
                    f"max(s/(m0*sin(alpha)))={max_ratio:.3f}; minimal alpha≈{alpha_min_deg:.2f}° (given m0)."
                )

    c_est = turboflash(
        mat.s_input,
        t1_ms,
        TD=float(args.ti_ms),
        r1=float(r1_per_1000),
        m0=(1.0 if m0 is None else float(m0)),
        flip_angle_deg=float(args.flip_angle_deg),
        ctc_model="turboflash",
    )

    m_all = _metrics(c_est, mat.c_input)

    # Also report post-bolus region only.
    start = int(min(max(baseline_frames, 0), mat.c_input.shape[0]))
    m_post = _metrics(c_est[start:], mat.c_input[start:])

    if args.auto_shift:
        lag, lag_rmse = _best_integer_lag_rmse(mat.c_input, c_est, max_abs_lag=10)
        print(f"best_lag_cest_vs_cmat: {lag}  (rmse={lag_rmse:.6g})")
        if mat.r1_input is not None:
            try:
                lag2, lag2_rmse = _best_integer_lag_rmse(mat.c_input, mat.r1_input / float(mat.beta_input), max_abs_lag=10)
                print(f"best_lag_r1divbeta_vs_cmat: {lag2}  (rmse={lag2_rmse:.6g})")
            except Exception:
                pass

        if args.apply_best_shift:
            # Shift c_est by `lag` to best align with mat.c_input.
            if lag < 0:
                c_shift = np.concatenate([c_est[-lag:], np.full((-lag,), np.nan)])
            elif lag > 0:
                c_shift = np.concatenate([np.full((lag,), np.nan), c_est[:-lag]])
            else:
                c_shift = c_est
            m_all_shift = _metrics(c_shift, mat.c_input)
            m_post_shift = _metrics(c_shift[start:], mat.c_input[start:])
            print("metrics_all_shifted:", m_all_shift)
            print("metrics_post_bolus_shifted:", m_post_shift)

    print(f"file: {args.mat_path}")
    print(f"method: {args.method}")
    print(f"baseline_frames: {baseline_frames}")
    print(f"ti_ms: {args.ti_ms}")
    print(f"flip_angle_deg: {args.flip_angle_deg}")
    print(f"beta_input: {mat.beta_input}  (=> r1_per_1000={r1_per_1000})")
    print(f"r1_input_pre: {r1_input_pre}  (=> T1_ms={t1_ms:.6f})")
    if m0 is not None:
        print(f"m0: {m0}")
    print("metrics_all:", m_all)
    print("metrics_post_bolus:", m_post)

    # Print a few sample points for quick sanity.
    idxs = [0, max(0, baseline_frames - 1), baseline_frames, min(mat.c_input.size - 1, baseline_frames + 5)]
    idxs = [i for i in dict.fromkeys(idxs) if 0 <= i < mat.c_input.size]
    for i in idxs:
        print(
            f"i={i:3d} t={mat.time_s[i]:9.3f}s  c_mat={mat.c_input[i]: .6e}  c_est={c_est[i]: .6e}  diff={c_est[i]-mat.c_input[i]: .6e}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
