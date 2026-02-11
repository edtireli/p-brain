"""Time-shifted AIF (TSCC) generation.

This module is intentionally lightweight (no TensorFlow) so both the AI ROI
pipeline and the deterministic ROI pipeline can generate TSCC curves.

It consumes per-slice CTC curves from:
- Analysis/CTC Data/Vein/Sinus Sagittalis/CTC[_shifted]_slice_*.npy
- Analysis/CTC Data/Artery/<ArterySubtype>/CTC[_shifted]_slice_*.npy

and writes time-shifted+rescaled SSS-derived curves to:
- Analysis/TSCC Data/<ArterySubtype>/TSCC_slice_<venous>_<arterial>.npy
- Analysis/TSCC Data/Max/TSCC_slice_<venous>_<arterial>.npy (best)

"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from scipy.signal import correlate, find_peaks
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

from utils.cli_logging import auto_logging_enabled, log_process_end, log_process_start
from utils.fonts import blaa, blaa1, prop, roed
from utils.loading import load_curves
from utils.plotting import butter_lowpass_filter
import utils.settings as settings


_ARTERY_CODE_TO_SUBTYPE = {
    "RICA": "Right Interior Carotid",
    "LICA": "Left Interior Carotid",
}


def _read_input_function_forces(analysis_directory: str) -> tuple[dict | None, dict | None]:
    """Read per-subject forced AIF/VIF selection (if present).

    Expected file: Analysis/input_function_forces.json
    {
      "forcedAif": {"roiType":"Artery","roiSubType":"Right Interior Carotid","sliceIndex":12} | null,
      "forcedVif": {"roiType":"Vein","roiSubType":"Sinus Sagittalis","sliceIndex":34} | null
    }
    """

    path = os.path.join(str(analysis_directory), "input_function_forces.json")
    try:
        if not os.path.isfile(path):
            return None, None
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return None, None

        forced_aif = raw.get("forcedAif")
        forced_vif = raw.get("forcedVif")

        def _norm(d: object) -> dict | None:
            if not isinstance(d, dict):
                return None
            roi_type = str(d.get("roiType") or "").strip()
            roi_sub = str(d.get("roiSubType") or "").strip()
            try:
                slice_index = int(d.get("sliceIndex"))
            except Exception:
                return None
            if not roi_type or not roi_sub:
                return None
            if slice_index < 0:
                return None
            return {"roiType": roi_type, "roiSubType": roi_sub, "sliceIndex": int(slice_index)}

        return _norm(forced_aif), _norm(forced_vif)
    except Exception:
        return None, None


def _shift_pchip_padded(time_s: np.ndarray, curve: np.ndarray, deltaT_s: float) -> np.ndarray:
    """Shift a curve by Δt using PCHIP interpolation with padded endpoints.

    Padding matches the legacy/reference implementation:
    - Δt > 0: pre-pad with zeros
    - Δt < 0: post-pad with the last value
    """

    time = np.asarray(time_s, dtype=float).reshape(-1)
    Ca = np.asarray(curve, dtype=float).reshape(-1)
    if time.size == 0:
        return np.asarray([], dtype=np.float32)
    if Ca.size == 0:
        return np.zeros_like(time, dtype=np.float32)
    if Ca.size != time.size:
        # Match existing pipeline behaviour: work on the overlapping prefix.
        n = int(min(Ca.size, time.size))
        Ca = Ca[:n]
        time = time[:n]

    deltaT = float(deltaT_s)
    if deltaT == 0.0:
        return Ca.astype(np.float32, copy=True)

    # Reference uses timeunit = time[2]-time[1]; be defensive for short arrays.
    if time.size >= 3:
        timeunit = float(time[2] - time[1])
    elif time.size >= 2:
        timeunit = float(time[1] - time[0])
    else:
        timeunit = 0.0
    if not np.isfinite(timeunit) or timeunit == 0.0:
        diffs = np.diff(time)
        diffs = diffs[np.isfinite(diffs)]
        timeunit = float(np.median(diffs)) if diffs.size else 1.0
    if timeunit <= 0:
        timeunit = 1.0

    numberobs = int(Ca.size)
    notimeunit = int(np.ceil(abs(deltaT) / timeunit))

    if deltaT > 0:
        start = (time[0] + deltaT) - (notimeunit * timeunit)
        end = (time[numberobs - 1] + deltaT)
        timetemp = np.arange(start, end + 0.5 * timeunit, timeunit, dtype=float)
        sCa = np.zeros(int(notimeunit + numberobs), dtype=float)
        sCa[int(notimeunit) : int(notimeunit + numberobs)] += Ca[:numberobs]
        if timetemp.size != sCa.size:
            # Guard against floating drift causing off-by-one.
            n = int(min(timetemp.size, sCa.size))
            timetemp = timetemp[:n]
            sCa = sCa[:n]
        interp = PchipInterpolator(timetemp, sCa, extrapolate=True)
        return np.asarray(interp(time), dtype=np.float32)

    # deltaT < 0
    start = (time[0] + deltaT)
    end = (time[numberobs - 1] + deltaT + (notimeunit * timeunit))
    timetemp = np.arange(start, end + 0.5 * timeunit, timeunit, dtype=float)
    sCa = np.empty(int(notimeunit + numberobs), dtype=float)
    sCa[:numberobs] = Ca[:numberobs]
    sCa[numberobs:] = Ca[numberobs - 1]
    if timetemp.size != sCa.size:
        n = int(min(timetemp.size, sCa.size))
        timetemp = timetemp[:n]
        sCa = sCa[:n]
    interp = PchipInterpolator(timetemp, sCa, extrapolate=True)
    return np.asarray(interp(time), dtype=np.float32)


def _baseline_to_zero(curve: np.ndarray, baseline_frames: int) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32).reshape(-1)
    if curve.size == 0:
        return curve
    baseline_frames = max(1, int(baseline_frames))
    b = min(int(curve.size), baseline_frames)
    base = float(np.nanmean(curve[:b])) if b > 0 else 0.0
    if not np.isfinite(base):
        base = 0.0
    out = curve.astype(np.float32, copy=True) - base
    if b > 0:
        out[:b] = 0.0
    return out


def _fit_tscc_pchip_fit(
    *,
    time_points_s: np.ndarray,
    artery_curve: np.ndarray,
    vein_curve: np.ndarray,
    rescale: bool,
    baseline_frames: int,
) -> tuple[np.ndarray, float, float]:
    """Fit Δt+scale by minimizing SSE between scaled artery and shifted vein.

    Objective: SSE(scale*artery - shift_pchip_padded(time, vein, Δt)).

    Returns:
      - tscc_curve: shifted(+optionally rescaled) venous curve in *arterial* scale
      - scale: fitted scale applied to artery in the objective
      - deltaT: fitted time shift applied to vein (seconds)
    """

    time = np.asarray(time_points_s, dtype=float).reshape(-1)
    art0 = _baseline_to_zero(artery_curve, baseline_frames)
    ve0 = _baseline_to_zero(vein_curve, baseline_frames)

    if time.size == 0 or art0.size == 0 or ve0.size == 0:
        return np.asarray([], dtype=np.float32), 1.0, 0.0

    n = int(min(time.size, art0.size, ve0.size))
    time = time[:n]
    art0 = art0[:n]
    ve0 = ve0[:n]

    def sse(deltaT: float, scale: float) -> float:
        ve_shift = _shift_pchip_padded(time, ve0, float(deltaT))
        diff = (art0.astype(np.float64) * float(scale)) - ve_shift.astype(np.float64)
        val = float(np.dot(diff, diff))
        return val if np.isfinite(val) else float("inf")

    if rescale:
        def obj(x):
            return sse(float(x[0]), float(x[1]))

        x0 = np.asarray([0.0, 1.0], dtype=float)
        res = minimize(
            obj,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-3, "fatol": 1e-3},
        )
        deltaT = float(res.x[0]) if getattr(res, "x", None) is not None else 0.0
        scale = float(res.x[1]) if getattr(res, "x", None) is not None else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
    else:
        def obj1(x):
            return sse(float(x[0]), 1.0)

        x0 = np.asarray([0.0], dtype=float)
        res = minimize(
            obj1,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-3, "fatol": 1e-3},
        )
        deltaT = float(res.x[0]) if getattr(res, "x", None) is not None else 0.0
        scale = 1.0

    # Policy: never scale VIF down.
    # In this formulation, `tscc = ve_shift / scale`.
    # - scale < 1  => TSCC (VIF) scales UP (allowed)
    # - scale > 1  => TSCC (VIF) scales DOWN (disallowed)
    # When a down-scale would be applied, we instead reset the shift to the
    # start (deltaT=0) and keep scale=1.
    if rescale:
        if not np.isfinite(deltaT):
            deltaT = 0.0
        if float(scale) > 1.0:
            deltaT = 0.0
            scale = 1.0

    ve_shift = _shift_pchip_padded(time, ve0, deltaT)

    # Convert venous curve into arterial scale.
    tscc = ve_shift / float(scale) if rescale else ve_shift
    return tscc.astype(np.float32, copy=False), float(scale), float(deltaT)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_available_slices(analysis_directory: str, structure_type: str, structure_subtype: str) -> list[int]:
    available_slices: list[int] = []
    base = os.path.join(analysis_directory, "CTC Data", structure_type, structure_subtype)
    if not os.path.isdir(base):
        return available_slices

    # Legacy code only checked slices 1..10, which breaks most real datasets.
    # We now discover all available slice indices by scanning the directory.
    pat = re.compile(r"^CTC_(?:shifted_)?slice_(\d+)\.npy$", re.IGNORECASE)
    found: set[int] = set()
    for fname in os.listdir(base):
        match = pat.match(fname)
        if match:
            try:
                found.add(int(match.group(1)))
            except Exception:
                continue
    return sorted(found)


def align_first_peaks(
    vein_curve: np.ndarray,
    artery_curve: np.ndarray,
    *,
    radius: int = 10,
    double_peak_radius: int = 3,
    rescale: bool = True,
    rescale_method: str = "peak",
    num_peaks: int | None = None,
) -> tuple[np.ndarray, list[int], float, int]:
    vein_curve = np.asarray(vein_curve, dtype=np.float32).reshape(-1)
    artery_curve = np.asarray(artery_curve, dtype=np.float32).reshape(-1)
    if vein_curve.size == 0 or artery_curve.size == 0:
        return np.asarray([], dtype=np.float32), [], 1.0, 0

    if num_peaks is None:
        num_peaks = int(getattr(settings, "NUMBER_OF_PEAKS", 2))
    num_peaks = max(1, int(num_peaks))

    baseline_frames = int(getattr(settings, "ROI_DCE_BASELINE_FRAMES", 5))
    baseline_frames = max(1, baseline_frames)
    normalize_curves = bool(getattr(settings, "ROI_NORMALIZE_CURVES", True))

    def _baseline_mean(curve: np.ndarray) -> float:
        if curve.size == 0:
            return 0.0
        b = min(int(curve.size), baseline_frames)
        base = float(np.nanmean(curve[:b])) if b > 0 else 0.0
        return base if np.isfinite(base) else 0.0

    def _baseline_to_zero(curve: np.ndarray) -> np.ndarray:
        if curve.size == 0:
            return curve
        b = min(int(curve.size), baseline_frames)
        base = _baseline_mean(curve)
        out = curve.astype(np.float32, copy=True) - base
        if b > 0:
            out[:b] = 0.0
        return out

    def _shift_with_zeros(curve: np.ndarray, shift_samples: int, *, target_len: int) -> np.ndarray:
        if target_len <= 0:
            return np.asarray([], dtype=np.float32)
        curve = np.asarray(curve, dtype=np.float32).reshape(-1)
        if curve.size == 0:
            return np.zeros(int(target_len), dtype=np.float32)

        shift_samples = int(shift_samples)
        if shift_samples > 0:
            if shift_samples >= curve.size:
                shifted = np.zeros(int(target_len), dtype=np.float32)
            else:
                shifted = np.concatenate(
                    [curve[shift_samples:], np.zeros(int(shift_samples), dtype=np.float32)]
                )
        elif shift_samples < 0:
            s = int(-shift_samples)
            if s >= curve.size:
                shifted = np.zeros(int(target_len), dtype=np.float32)
            else:
                shifted = np.concatenate(
                    [np.zeros(int(s), dtype=np.float32), curve[: curve.size - s]]
                )
        else:
            shifted = curve

        if shifted.size < target_len:
            shifted = np.pad(shifted, (0, int(target_len - shifted.size)), mode="constant")
        elif shifted.size > target_len:
            shifted = shifted[:target_len]
        return shifted.astype(np.float32, copy=False)

    def _top_peaks_by_height(curve: np.ndarray, *, n_peaks: int, separation: int) -> list[int]:
        if curve.size == 0:
            return []
        peaks, _ = find_peaks(curve)
        if len(peaks) == 0:
            return []
        candidates = [int(p) for p in peaks if np.isfinite(curve[int(p)]) and float(curve[int(p)]) > 0]
        if not candidates:
            return []
        sorted_peaks = sorted(candidates, key=lambda x: float(curve[int(x)]), reverse=True)
        chosen: list[int] = []
        for p in sorted_peaks:
            if all(abs(int(p) - cp) > separation for cp in chosen):
                chosen.append(int(p))
                if len(chosen) >= int(n_peaks):
                    break
        return chosen

    # Baseline alignment: match venous baseline mean to arterial baseline mean.
    base_a = _baseline_mean(artery_curve)
    base_v = _baseline_mean(vein_curve)
    vein_baseline_matched = vein_curve - float(base_v - base_a)

    # Work in baseline-to-zero space for peak detection, shifting, and scaling.
    artery_zero = _baseline_to_zero(artery_curve)
    vein_zero = _baseline_to_zero(vein_baseline_matched)

    vein_top_peaks = _top_peaks_by_height(vein_zero, n_peaks=num_peaks, separation=int(radius))
    artery_top_peaks = _top_peaks_by_height(artery_zero, n_peaks=num_peaks, separation=int(radius))

    k = min(len(vein_top_peaks), len(artery_top_peaks), int(num_peaks))
    if k > 0:
        diffs = [int(vein_top_peaks[i]) - int(artery_top_peaks[i]) for i in range(k)]
        shift = int(np.round(float(np.median(np.asarray(diffs, dtype=float)))))
    else:
        # Fallback: broad cross-correlation (less peak-focused, but works when peak finding fails).
        cross_corr = correlate(vein_zero, artery_zero)
        shift = int(np.argmax(cross_corr) - len(vein_zero) + 1)

    target_len = int(artery_zero.size)
    aligned_vein_curve = _shift_with_zeros(vein_zero, shift, target_len=target_len)

    # Return peak indices in aligned curve coordinates.
    aligned_peaks: list[int] = []
    for p in vein_top_peaks:
        pp = int(p) - int(shift)
        if 0 <= pp < target_len:
            aligned_peaks.append(pp)

    rescaled = 1.0
    rescale_method = (str(rescale_method) or "peak").strip().lower()
    if rescale and rescale_method == "peak":
        ratios: list[float] = []
        for i in range(k):
            v_peak = int(vein_top_peaks[i]) - int(shift)
            a_peak = int(artery_top_peaks[i])
            if not (0 <= v_peak < aligned_vein_curve.size and 0 <= a_peak < artery_zero.size):
                continue
            v = float(aligned_vein_curve[v_peak])
            a = float(artery_zero[a_peak])
            if not np.isfinite(v) or not np.isfinite(a) or abs(v) <= 1e-8:
                continue
            ratios.append(a / v)
        if ratios:
            scaling_factor = float(np.median(np.asarray(ratios, dtype=float)))
            if np.isfinite(scaling_factor) and scaling_factor > 0:
                # Policy: never scale VIF down. If VIF would be downscaled
                # (scaling_factor < 1), reset to no-shift and no-rescale.
                if float(scaling_factor) < 1.0:
                    shift = 0
                    aligned_vein_curve = _shift_with_zeros(vein_zero, shift, target_len=target_len)
                    aligned_peaks = [int(p) for p in vein_top_peaks if 0 <= int(p) < target_len]
                    rescaled = 1.0
                else:
                    aligned_vein_curve = aligned_vein_curve * scaling_factor
                    rescaled = scaling_factor
    elif rescale and rescale_method == "auc":
        def _maybe_normalize_for_auc(curve: np.ndarray) -> np.ndarray:
            curve = np.asarray(curve, dtype=np.float32)
            if not normalize_curves:
                return curve
            denom = float(np.percentile(np.abs(curve), 95)) if curve.size else 1.0
            if not np.isfinite(denom) or denom <= 1e-6:
                denom = 1.0
            return curve / denom

        v_auc_curve = _maybe_normalize_for_auc(aligned_vein_curve)
        a_auc_curve = _maybe_normalize_for_auc(artery_zero)
        v_auc = float(np.trapz(np.clip(v_auc_curve, 0.0, None))) if v_auc_curve.size else 0.0
        a_auc = float(np.trapz(np.clip(a_auc_curve, 0.0, None))) if a_auc_curve.size else 0.0
        if np.isfinite(v_auc) and np.isfinite(a_auc) and v_auc > 1e-8 and a_auc > 1e-8:
            scaling_factor = float(a_auc / v_auc)
            if np.isfinite(scaling_factor) and scaling_factor > 0:
                # Policy: never scale VIF down. If VIF would be downscaled
                # (scaling_factor < 1), reset to no-shift and no-rescale.
                if float(scaling_factor) < 1.0:
                    shift = 0
                    aligned_vein_curve = _shift_with_zeros(vein_zero, shift, target_len=target_len)
                    aligned_peaks = [int(p) for p in vein_top_peaks if 0 <= int(p) < target_len]
                    rescaled = 1.0
                else:
                    aligned_vein_curve = aligned_vein_curve * scaling_factor
                    rescaled = scaling_factor

    # Keep legacy double-peak knob as a no-op for now (retained for API compat).
    _ = int(double_peak_radius)

    return aligned_vein_curve, aligned_peaks, float(rescaled), int(shift)


def _load_ctc_curve(
    *,
    analysis_directory: str,
    structure_type: str,
    structure_subtype: str,
    slice_index: int,
) -> np.ndarray | None:
    base = os.path.join(analysis_directory, "CTC Data", structure_type, structure_subtype)
    for filename in (f"CTC_shifted_slice_{slice_index}.npy", f"CTC_slice_{slice_index}.npy"):
        path = os.path.join(base, filename)
        if os.path.exists(path):
            try:
                return np.asarray(np.load(path), dtype=np.float32)
            except Exception:
                return None
    return None


def _best_slice_by_peak(
    analysis_directory: str,
    structure_type: str,
    structure_subtype: str,
    slice_candidates: list[int],
) -> tuple[int, np.ndarray] | None:
    best_i = -1
    best_curve: np.ndarray | None = None
    best_val = float("-inf")
    for si in slice_candidates:
        c = _load_ctc_curve(
            analysis_directory=analysis_directory,
            structure_type=structure_type,
            structure_subtype=structure_subtype,
            slice_index=int(si),
        )
        if c is None or c.size == 0:
            continue
        v = float(np.nanmax(c))
        if np.isfinite(v) and v > best_val:
            best_val = v
            best_i = int(si)
            best_curve = c
    if best_curve is None:
        return None
    return best_i, best_curve


def _write_tscc_shift_rescale_demo(
    *,
    analysis_directory: str,
    image_directory: str,
    time_points_s: np.ndarray,
    reference_subtype: str,
    reference_curve: np.ndarray,
    vein_curve: np.ndarray,
    tscc_curve: np.ndarray,
    tscc_shift_only_curve: np.ndarray,
    scaling: float,
    time_shift_s: float,
) -> None:
    if plt is None:
        return

    out_dir = os.path.join(image_directory, "Time Shifted Concentration Curves")
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "tscc_shift_rescale_demo.png")

    # Best-effort include both carotids if available.
    rica_sub = _ARTERY_CODE_TO_SUBTYPE.get("RICA", "Right Interior Carotid")
    lica_sub = _ARTERY_CODE_TO_SUBTYPE.get("LICA", "Left Interior Carotid")
    rica = _best_slice_by_peak(
        analysis_directory,
        "Artery",
        rica_sub,
        get_available_slices(analysis_directory, "Artery", rica_sub),
    )
    lica = _best_slice_by_peak(
        analysis_directory,
        "Artery",
        lica_sub,
        get_available_slices(analysis_directory, "Artery", lica_sub),
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.2))

    def _plot(curve: np.ndarray, *, label: str, color: str, ls: str = "-", alpha: float = 0.9, lw: float = 1.3):
        if curve is None or curve.size == 0:
            return
        x = time_points_s[: curve.size]
        ax.plot(x, curve, label=label, color=color, linestyle=ls, alpha=alpha, linewidth=lw)

    # Arterial curves (thin).
    if rica is not None:
        _, rica_curve = rica
        _plot(rica_curve, label="rICA (artery)", color="#1f77b4", alpha=0.7, lw=1.0)
    if lica is not None:
        _, lica_curve = lica
        _plot(lica_curve, label="lICA (artery)", color="#2ca02c", alpha=0.7, lw=1.0)

    # Reference artery emphasized.
    _plot(reference_curve, label=f"Reference artery ({reference_subtype})", color="#111111", alpha=0.9, lw=1.4)

    # Vein/TSCC overlays (requested style: grey dashed + red).
    _plot(vein_curve, label="SSS (raw)", color="#9e9e9e", ls="--", alpha=0.35, lw=1.2)
    _plot(
        tscc_shift_only_curve,
        label=f"SSS time-shifted ({float(time_shift_s):.2f} s)",
        color="#d62728",
        ls="--",
        alpha=0.6,
        lw=1.2,
    )
    if scaling != 1.0:
        _plot(tscc_curve, label=f"SSS time-shifted + rescaled (×{scaling:.2f})", color="#d62728", ls="-", alpha=0.95, lw=1.4)
    else:
        _plot(tscc_curve, label=f"SSS time-shifted (no rescale)", color="#d62728", ls="-", alpha=0.95, lw=1.4)

    ax.set_title("TSCC demonstration: time shifting and rescaling")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Concentration (mM)")
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def remove_trailing_zeros(arr: np.ndarray) -> np.ndarray:
    non_zero_idx = np.where(arr != 0)[0]
    if len(non_zero_idx) == 0:
        return np.array([])
    last_non_zero = int(non_zero_idx[-1])
    return arr[: last_non_zero + 1]


def shift_curve_to_zero_start(curve: np.ndarray) -> np.ndarray:
    if curve.size == 0:
        return curve
    start_value = float(curve[0])
    return curve - start_value


def plot_transformed_curves(
    shifted_vein_curve: np.ndarray,
    shifted_artery_curve: np.ndarray,
    venous_slice_index: int,
    arterial_slice_index: int,
    *,
    vein_top_peaks: list[int],
    time_points_s: np.ndarray,
    analysis_directory: str,
    image_directory: str,
    subtype: str,
    original_vein_curve: np.ndarray | None = None,
    scaling: float = 1.0,
    time_shift: float = 0.0,
) -> None:
    tscc_root = os.path.join(analysis_directory, "TSCC Data", subtype)
    img_root = os.path.join(image_directory, "Time Shifted Concentration Curves", subtype)
    _ensure_dir(tscc_root)
    _ensure_dir(img_root)

    np.save(
        os.path.join(tscc_root, f"TSCC_slice_{venous_slice_index}_{arterial_slice_index}.npy"),
        shifted_vein_curve,
    )

    if plt is None:
        return

    fs = 15
    cutoff = 4.0
    order = 3

    plt.figure(figsize=(10, 5))
    if scaling == 1:
        title = (
            f" Time-shifted Concentration Curve for {subtype} "
            f"(Veinous Slice {venous_slice_index}, Arterial Slice {arterial_slice_index})"
        )
    else:
        title = (
            f"Rescaled & Time-shifted Concentration Curve for {subtype} "
            f"(Veinous Slice {venous_slice_index}, Arterial Slice {arterial_slice_index})"
        )

    plt.title(title, fontproperties=prop, fontsize=16)

    # Original VIF placement (grey dashed).
    if original_vein_curve is not None and np.size(original_vein_curve) > 0:
        plt.plot(
            time_points_s[: len(original_vein_curve)],
            original_vein_curve,
            label="VIF (original)",
            color="#9e9e9e",
            alpha=0.45,
            linestyle="--",
            linewidth=1.2,
        )

    # New VIF (shifted/rescaled) in red.
    if scaling != 1:
        vif_label = f"VIF (shifted+rescaled ×{scaling:.2f}, Δt={float(time_shift):.2f} s)"
    else:
        vif_label = f"VIF (shifted, Δt={float(time_shift):.2f} s)"
    plt.plot(
        time_points_s[: len(shifted_vein_curve)],
        shifted_vein_curve,
        label=vif_label,
        color="#d62728",
        alpha=0.95,
        linestyle="-",
        linewidth=1.4,
    )

    # AIF (black).
    plt.plot(
        time_points_s[: len(shifted_artery_curve)],
        shifted_artery_curve,
        label="AIF",
        color="#111111",
        alpha=0.9,
        linestyle="-",
        linewidth=1.3,
    )

    plt.xlabel("Time (s)", fontproperties=prop, fontsize=14)
    plt.ylabel("Concentration (mM)", fontproperties=prop, fontsize=14)
    plt.legend()
    plt.grid(which="minor", alpha=0.25)
    plt.minorticks_on()

    plt.tight_layout()
    plt.savefig(
        os.path.join(img_root, f"TSCC_slice_{venous_slice_index}_{arterial_slice_index}.png"),
        dpi=300,
    )
    plt.close()


def plot_transformed_curves_max(
    shifted_vein_curve: np.ndarray,
    *,
    venous_slice_index: int,
    arterial_slice_index: int,
    time_points_s: np.ndarray,
    analysis_directory: str,
    image_directory: str,
    subtype: str,
    original_vein_curve: np.ndarray | None = None,
    artery_curve: np.ndarray | None = None,
) -> None:
    max_root = os.path.join(analysis_directory, "TSCC Data", "Max")
    max_img = os.path.join(image_directory, "Time Shifted Concentration Curves", "Max")
    _ensure_dir(max_root)
    _ensure_dir(max_img)

    np.save(
        os.path.join(max_root, f"TSCC_slice_{venous_slice_index}_{arterial_slice_index}.npy"),
        shifted_vein_curve,
    )

    if plt is None:
        return

    plt.figure(figsize=(10, 5))
    plt.title(
        f"Rescaled & Time-shifted Concentration Curve for {subtype} "
        f"(Veinous Slice {venous_slice_index}, Aterial Slice {arterial_slice_index})",
        fontproperties=prop,
        fontsize=16,
    )

    if original_vein_curve is not None and np.size(original_vein_curve) > 0:
        plt.plot(
            time_points_s[: len(original_vein_curve)],
            original_vein_curve,
            label="VIF (original)",
            color="#9e9e9e",
            linestyle="--",
            alpha=0.45,
            linewidth=1.2,
        )

    plt.plot(
        time_points_s[: len(shifted_vein_curve)],
        shifted_vein_curve,
        label="VIF (shifted+rescaled)",
        color="#d62728",
        alpha=0.95,
        linewidth=1.4,
    )

    if artery_curve is not None and np.size(artery_curve) > 0:
        plt.plot(
            time_points_s[: len(artery_curve)],
            artery_curve,
            label="AIF",
            color="#111111",
            alpha=0.9,
            linewidth=1.3,
        )

    plt.xlabel("Time (s)", fontproperties=prop, fontsize=14)
    plt.ylabel("Concentration (mM)", fontproperties=prop, fontsize=14)
    plt.legend()
    plt.grid(which="minor", alpha=0.25)
    plt.minorticks_on()

    plt.tight_layout()
    plt.savefig(
        os.path.join(max_img, f"TSCC_slice_{venous_slice_index}_{arterial_slice_index}.png"),
        dpi=300,
    )
    plt.close()


@dataclass
class _TsccBest:
    file_path: str = ""
    value: float = float("-inf")
    subtype: str = ""
    venous_slice: int = -1
    arterial_slice: int = -1


def _is_tscc_forked_peak(curve: np.ndarray) -> bool:
    """Detect a narrow "forked" apex near the global maximum.

    This aims to exclude unstable converted curves from being selected as the
    global Max TSCC. It is intentionally conservative to avoid false positives:
    it only triggers when two distinct near-maximum local maxima occur within
    a small window around the global peak, separated by a measurable dip.
    """

    try:
        c = np.asarray(curve, dtype=float).reshape(-1)
    except Exception:
        return False
    if c.size < 5:
        return False
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.max(c))
    min_peak = float(getattr(settings, "TSCC_FORKED_PEAK_MIN_PEAK_MM", 1.0))
    if not np.isfinite(peak) or peak <= min_peak:
        return False

    peak_frac = float(getattr(settings, "TSCC_FORKED_PEAK_NEAR_FRAC", 0.95))
    max_sep = int(getattr(settings, "TSCC_FORKED_PEAK_MAX_SEPARATION_FRAMES", 3))
    win = int(getattr(settings, "TSCC_FORKED_PEAK_WINDOW_FRAMES", 3))
    min_dip_frac = float(getattr(settings, "TSCC_FORKED_PEAK_MIN_DIP_FRAC", 0.06))
    if win < 1:
        win = 1
    if max_sep < 1:
        max_sep = 1
    if not (0.0 < peak_frac <= 1.0):
        peak_frac = 0.95
    if not (0.0 < min_dip_frac < 1.0):
        min_dip_frac = 0.06

    i = int(np.argmax(c))
    lo = max(1, i - win)
    hi = min(int(c.size) - 2, i + win)
    if hi <= lo:
        return False

    cand = []
    thr = peak_frac * peak
    for k in range(lo, hi + 1):
        v = float(c[k])
        if v < thr:
            continue
        if v >= float(c[k - 1]) and v >= float(c[k + 1]):
            cand.append(k)
    if len(cand) < 2:
        return False

    cand = sorted(set(cand))
    groups: list[list[int]] = []
    g = [cand[0]]
    for k in cand[1:]:
        if k == g[-1] + 1:
            g.append(k)
        else:
            groups.append(g)
            g = [k]
    groups.append(g)

    reps: list[int] = [int(gr[len(gr) // 2]) for gr in groups]
    if len(reps) < 2:
        return False

    for a in range(len(reps)):
        for b in range(a + 1, len(reps)):
            p1 = int(reps[a])
            p2 = int(reps[b])
            sep = abs(p2 - p1)
            if sep < 1 or sep > max_sep:
                continue
            l = min(p1, p2)
            r = max(p1, p2)
            valley = float(np.min(c[l : r + 1]))
            if valley <= (1.0 - min_dip_frac) * peak:
                return True
    return False


def find_max_npy_file(
    analysis_directory: str,
    *,
    num_peaks: int | None = None,
    peak_tolerance: float = 0.20,
    peak_separation: int = 10,
    allowed_subtypes: list[str] | None = None,
) -> _TsccBest:
    tscc_root = os.path.join(analysis_directory, "TSCC Data")
    if not os.path.isdir(tscc_root):
        return _TsccBest()

    if num_peaks is None:
        num_peaks = int(getattr(settings, "NUMBER_OF_PEAKS", 2))

    if allowed_subtypes is None:
        allowed_subtypes = [
            d
            for d in os.listdir(tscc_root)
            if os.path.isdir(os.path.join(tscc_root, d)) and d != "Max"
        ]

    def _top_peaks(curve: np.ndarray, n_peaks: int, separation: int) -> list[int]:
        peaks, _ = find_peaks(curve)
        if len(peaks) == 0:
            return []
        sorted_peaks = sorted(peaks, key=lambda x: float(curve[x]), reverse=True)
        chosen: list[int] = []
        for p in sorted_peaks:
            v = float(curve[p])
            if not np.isfinite(v) or v <= 0:
                continue
            if all(abs(int(p) - cp) > separation for cp in chosen):
                chosen.append(int(p))
                if len(chosen) >= n_peaks:
                    break
        return chosen

    def _load_artery_curve(artery_subtype: str, arterial_slice_index: int) -> np.ndarray | None:
        artery_dir = os.path.join(analysis_directory, "CTC Data", "Artery", artery_subtype)
        for filename in (
            f"CTC_shifted_slice_{arterial_slice_index}.npy",
            f"CTC_slice_{arterial_slice_index}.npy",
        ):
            path = os.path.join(artery_dir, filename)
            if os.path.exists(path):
                return np.load(path)
        return None

    def _peaks_consistent(
        vein_curve: np.ndarray,
        artery_curve: np.ndarray | None,
        n_peaks: int,
        tolerance: float,
        separation: int,
    ) -> bool:
        if artery_curve is None or vein_curve.size == 0 or artery_curve.size == 0:
            return False

        vein_peaks = _top_peaks(vein_curve, n_peaks, separation)
        artery_peaks = _top_peaks(artery_curve, n_peaks, separation)
        if len(vein_peaks) < n_peaks or len(artery_peaks) < n_peaks:
            return False

        ratios: list[float] = []
        for v_peak, a_peak in zip(vein_peaks, artery_peaks):
            v = float(vein_curve[v_peak])
            a = float(artery_curve[a_peak])
            if not np.isfinite(v) or not np.isfinite(a) or a == 0:
                return False
            ratios.append(v / a)

        ratios_arr = np.asarray(ratios, dtype=float)
        median_ratio = float(np.median(ratios_arr))
        if not np.isfinite(median_ratio) or median_ratio == 0:
            return False

        rel_dev = np.abs(ratios_arr - median_ratio) / abs(median_ratio)
        return bool(np.all(rel_dev <= tolerance))

    best_any_all = _TsccBest()
    best_consistent_all = _TsccBest()
    best_any = _TsccBest()
    best_consistent = _TsccBest()

    skip_forked = bool(getattr(settings, "TSCC_SKIP_FORKED_PEAKS", True))

    for subtype in allowed_subtypes:
        file_paths = glob.glob(os.path.join(tscc_root, subtype, "*.npy"))
        for file_path in file_paths:
            arr = np.load(file_path)
            if arr.size == 0:
                continue
            curr_max = float(np.max(arr))

            base = os.path.basename(file_path)
            parts = base.split("_")
            if len(parts) < 4:
                continue
            venous_slice = int(parts[-2])
            arterial_slice = int(parts[-1].split(".npy")[0])

            # Unfiltered best trackers (fallback when everything is excluded).
            if curr_max > best_any_all.value:
                best_any_all = _TsccBest(
                    file_path=file_path,
                    value=curr_max,
                    subtype=subtype,
                    venous_slice=venous_slice,
                    arterial_slice=arterial_slice,
                )

            if skip_forked and _is_tscc_forked_peak(arr):
                continue

            if curr_max > best_any.value:
                best_any = _TsccBest(
                    file_path=file_path,
                    value=curr_max,
                    subtype=subtype,
                    venous_slice=venous_slice,
                    arterial_slice=arterial_slice,
                )

            artery_curve = _load_artery_curve(subtype, arterial_slice)
            if _peaks_consistent(arr, artery_curve, int(num_peaks), float(peak_tolerance), int(peak_separation)):
                if curr_max > best_consistent_all.value:
                    best_consistent_all = _TsccBest(
                        file_path=file_path,
                        value=curr_max,
                        subtype=subtype,
                        venous_slice=venous_slice,
                        arterial_slice=arterial_slice,
                    )
                if curr_max > best_consistent.value:
                    best_consistent = _TsccBest(
                        file_path=file_path,
                        value=curr_max,
                        subtype=subtype,
                        venous_slice=venous_slice,
                        arterial_slice=arterial_slice,
                    )

    # Prefer consistent-filtered, then any-filtered. If everything is filtered
    # out, fall back to the unfiltered best to avoid breaking the pipeline.
    if best_consistent.file_path:
        return best_consistent
    if best_any.file_path:
        return best_any
    return best_consistent_all if best_consistent_all.file_path else best_any_all


def time_shifting(
    analysis_directory: str,
    nifti_directory: str,
    image_directory: str,
    *,
    artery: str | None = None,
) -> None:
    """Generate TSCC curves by aligning SSS (vein) to the selected artery."""

    if auto_logging_enabled():
        log_process_start("Time shifting")

    if artery is None:
        artery = getattr(settings, "INPUT_FUNCTION_ARTERY", "RICA")

    forced_aif, forced_vif = _read_input_function_forces(analysis_directory)

    artery_norm = (str(artery) or "RICA").strip().upper()
    artery_subtype = _ARTERY_CODE_TO_SUBTYPE.get(artery_norm, artery)
    if forced_aif and str(forced_aif.get("roiType")) == "Artery":
        artery_subtype = str(forced_aif.get("roiSubType"))

    time_path = os.path.join(analysis_directory, "Fitting", "time_points_s.npy")
    if not os.path.isfile(time_path):
        # Nothing to align without timing.
        if auto_logging_enabled():
            log_process_end("Time shifting")
        return

    time_points_s = np.load(time_path)

    # Optional: import MATLAB's TSCC curve directly for strict parity.
    # This is typically a MATLAB v7.3 MAT produced by the reference pipeline
    # (e.g. Brain_AIF_time_conc_*_shifted_.mat) containing `c_input`.
    matlab_tscc = getattr(settings, "MATLAB_TSCC_MAT", None)
    if matlab_tscc:
        mat_path = str(matlab_tscc)

        def _load_matlab_c_input(path: str) -> np.ndarray:
            # Prefer v7.3/HDF5 (h5py), fall back to scipy.io.loadmat.
            try:
                import h5py  # type: ignore

                with h5py.File(path, "r") as f:
                    if "c_input" not in f:
                        raise KeyError("c_input")
                    arr = np.array(f["c_input"], dtype=np.float64)
            except Exception:
                from scipy.io import loadmat  # type: ignore

                d = loadmat(path, struct_as_record=False, squeeze_me=True)
                if "c_input" not in d:
                    raise KeyError("c_input")
                arr = np.asarray(d["c_input"], dtype=np.float64)

            arr = np.asarray(arr, dtype=np.float32).reshape(-1)
            return arr

        def _infer_slice_from_name(name: str) -> int | None:
            # Common patterns: *_slice4_* or *slice4_shifted*.
            m = re.findall(r"slice(\d+)", name, flags=re.IGNORECASE)
            if not m:
                return None
            try:
                return int(m[0])
            except Exception:
                return None

        tscc_curve = _load_matlab_c_input(mat_path)
        # Match the time axis length when possible.
        if time_points_s is not None and getattr(time_points_s, "size", 0):
            n = int(min(len(tscc_curve), int(time_points_s.size)))
            tscc_curve = tscc_curve[:n]

        arterial_slices = get_available_slices(analysis_directory, "Artery", str(artery_subtype))
        venous_slices = get_available_slices(analysis_directory, "Vein", "Sinus Sagittalis")
        if not arterial_slices or not venous_slices:
            if auto_logging_enabled():
                log_process_end("Time shifting")
            return

        venous_guess = _infer_slice_from_name(os.path.basename(mat_path))
        venous_slice = int(venous_guess) if venous_guess in venous_slices else int(venous_slices[0])
        arterial_slice = int(arterial_slices[0])

        _ensure_dir(os.path.join(analysis_directory, "TSCC Data", str(artery_subtype)))
        _ensure_dir(os.path.join(analysis_directory, "TSCC Data", "Max"))

        # Clear any prior TSCC outputs so downstream logic doesn't pick stale files.
        for f in glob.glob(os.path.join(analysis_directory, "TSCC Data", "*", "TSCC_slice_*.npy")):
            try:
                os.remove(f)
            except Exception:
                pass

        out1 = os.path.join(
            analysis_directory,
            "TSCC Data",
            str(artery_subtype),
            f"TSCC_slice_{venous_slice}_{arterial_slice}.npy",
        )
        np.save(out1, np.asarray(tscc_curve, dtype=np.float32))

        out2 = os.path.join(
            analysis_directory,
            "TSCC Data",
            "Max",
            f"TSCC_slice_{venous_slice}_{arterial_slice}.npy",
        )
        np.save(out2, np.asarray(tscc_curve, dtype=np.float32))

        with open(os.path.join(analysis_directory, "max_info.json"), "w") as f:
            json.dump([f"Max artery type: {artery_subtype}"], f)

        with open(os.path.join(analysis_directory, "tscc_selection.json"), "w") as f:
            json.dump(
                {
                    "reference_artery": str(getattr(settings, "INPUT_FUNCTION_ARTERY", "RICA")),
                    "artery_subtype": str(artery_subtype),
                    "venous_slice": int(venous_slice),
                    "arterial_slice": int(arterial_slice),
                    "tscc_method": "matlab_import",
                    "tscc_rescale": True,
                    "tscc_rescale_method": str(getattr(settings, "TSCC_RESCALE_METHOD", "peak")),
                    "num_peaks": int(getattr(settings, "NUMBER_OF_PEAKS", 2)),
                    "matlab_tscc_mat": os.path.abspath(mat_path),
                },
                f,
                indent=2,
            )

        if auto_logging_enabled():
            log_process_end("Time shifting")
        return

    arterial_slices = get_available_slices(analysis_directory, "Artery", str(artery_subtype))
    venous_slices = get_available_slices(analysis_directory, "Vein", "Sinus Sagittalis")

    if forced_aif and str(forced_aif.get("roiType")) == "Artery":
        try:
            forced_art = int(forced_aif.get("sliceIndex"))
        except Exception:
            forced_art = -1
        # Only apply the force if the curve exists.
        if forced_art >= 0 and _load_ctc_curve(
            analysis_directory=analysis_directory,
            structure_type="Artery",
            structure_subtype=str(artery_subtype),
            slice_index=int(forced_art),
        ) is not None:
            arterial_slices = [int(forced_art)]

    if forced_vif and str(forced_vif.get("roiType")) == "Vein":
        # Current pipeline supports only SSS as VIF.
        if str(forced_vif.get("roiSubType")) == "Sinus Sagittalis":
            try:
                forced_vein = int(forced_vif.get("sliceIndex"))
            except Exception:
                forced_vein = -1
            if forced_vein >= 0 and _load_ctc_curve(
                analysis_directory=analysis_directory,
                structure_type="Vein",
                structure_subtype="Sinus Sagittalis",
                slice_index=int(forced_vein),
            ) is not None:
                venous_slices = [int(forced_vein)]

    if not arterial_slices or not venous_slices:
        if auto_logging_enabled():
            log_process_end("Time shifting")
        return

    _ensure_dir(os.path.join(analysis_directory, "TSCC Data", str(artery_subtype)))
    _ensure_dir(os.path.join(analysis_directory, "TSCC Data", "Max"))
    _ensure_dir(os.path.join(image_directory, "Time Shifted Concentration Curves", str(artery_subtype)))
    _ensure_dir(os.path.join(image_directory, "Time Shifted Concentration Curves", "Max"))

    for arterial_slice in arterial_slices:
        for venous_slice in venous_slices:
            vein_curve, artery_curve = load_curves(venous_slice, arterial_slice, str(artery_subtype), analysis_directory)
            tscc_method = (str(getattr(settings, "TSCC_METHOD", "peaks")) or "peaks").strip().lower()
            rescale_flag = bool(getattr(settings, "TSCC_RESCALE", True))
            if tscc_method == "pchip_fit":
                baseline_frames = int(getattr(settings, "ROI_DCE_BASELINE_FRAMES", 5))
                aligned_vein_curve, scale, deltaT_s = _fit_tscc_pchip_fit(
                    time_points_s=np.asarray(time_points_s, dtype=float),
                    artery_curve=artery_curve,
                    vein_curve=vein_curve,
                    rescale=rescale_flag,
                    baseline_frames=baseline_frames,
                )
                peaks = []
                rescaled = (1.0 / float(scale)) if rescale_flag else 1.0
                time_shift = float(deltaT_s)
            else:
                aligned_vein_curve, peaks, rescaled, time_shift_int = align_first_peaks(
                    vein_curve,
                    artery_curve,
                    rescale=rescale_flag,
                    rescale_method=str(getattr(settings, "TSCC_RESCALE_METHOD", "peak")),
                    num_peaks=int(getattr(settings, "NUMBER_OF_PEAKS", 2)),
                )
                time_shift = float(time_shift_int)
            aligned_vein_curve_no_zeros = remove_trailing_zeros(aligned_vein_curve)
            aligned_vein_curve_no_zeros_shifted = shift_curve_to_zero_start(aligned_vein_curve_no_zeros)
            plot_transformed_curves(
                aligned_vein_curve_no_zeros_shifted,
                artery_curve,
                venous_slice,
                arterial_slice,
                vein_top_peaks=peaks,
                time_points_s=time_points_s,
                analysis_directory=analysis_directory,
                image_directory=image_directory,
                subtype=str(artery_subtype),
                original_vein_curve=vein_curve,
                scaling=rescaled,
                time_shift=time_shift,
            )

    if forced_aif and forced_vif:
        try:
            venous_slice = int(forced_vif.get("sliceIndex"))
            arterial_slice = int(forced_aif.get("sliceIndex"))
        except Exception:
            venous_slice = -1
            arterial_slice = -1
        forced_path = os.path.join(
            analysis_directory,
            "TSCC Data",
            str(artery_subtype),
            f"TSCC_slice_{venous_slice}_{arterial_slice}.npy",
        )
        if os.path.isfile(forced_path):
            try:
                arr = np.load(forced_path)
                best = _TsccBest(
                    file_path=forced_path,
                    value=float(np.max(arr)) if arr.size else float("-inf"),
                    subtype=str(artery_subtype),
                    venous_slice=int(venous_slice),
                    arterial_slice=int(arterial_slice),
                )
            except Exception:
                best = find_max_npy_file(analysis_directory)
        else:
            best = find_max_npy_file(analysis_directory)
    else:
        best = find_max_npy_file(analysis_directory)
    if not best.file_path or not os.path.isfile(best.file_path):
        if auto_logging_enabled():
            log_process_end("Time shifting")
        return

    # Refresh Max outputs.
    for f in glob.glob(os.path.join(analysis_directory, "TSCC Data", "Max", "*.npy")):
        try:
            os.remove(f)
        except Exception:
            pass

    corresponding = np.load(best.file_path)
    try:
        best_vein, best_artery = load_curves(
            int(best.venous_slice),
            int(best.arterial_slice),
            str(best.subtype),
            analysis_directory,
        )
    except Exception:
        best_vein, best_artery = None, None
    plot_transformed_curves_max(
        corresponding,
        venous_slice_index=int(best.venous_slice),
        arterial_slice_index=int(best.arterial_slice),
        time_points_s=time_points_s,
        analysis_directory=analysis_directory,
        image_directory=image_directory,
        subtype=str(best.subtype),
        original_vein_curve=best_vein,
        artery_curve=best_artery,
    )

    with open(os.path.join(analysis_directory, "max_info.json"), "w") as f:
        json.dump([f"Max artery type: {best.subtype}"], f)

    # Extra debug metadata (non-breaking; separate file).
    try:
        with open(os.path.join(analysis_directory, "tscc_selection.json"), "w") as f:
            json.dump(
                {
                    "reference_artery": str(getattr(settings, "INPUT_FUNCTION_ARTERY", "RICA")),
                    "artery_subtype": str(best.subtype),
                    "venous_slice": int(best.venous_slice),
                    "arterial_slice": int(best.arterial_slice),
                    "forced_aif": forced_aif,
                    "forced_vif": forced_vif,
                    "tscc_method": str(getattr(settings, "TSCC_METHOD", "peaks")),
                    "tscc_rescale": bool(getattr(settings, "TSCC_RESCALE", True)),
                    "tscc_rescale_method": str(getattr(settings, "TSCC_RESCALE_METHOD", "peak")),
                    "num_peaks": int(getattr(settings, "NUMBER_OF_PEAKS", 2)),
                },
                f,
                indent=2,
            )
            f.write("\n")
    except Exception:
        pass

    # One compact demo plot (raw SSS vs shifted-only vs shifted+rescaled) overlaid with arteries.
    if bool(getattr(settings, "TSCC_WRITE_DEMO_PLOT", True)):
        try:
            ref_curve = _load_ctc_curve(
                analysis_directory=analysis_directory,
                structure_type="Artery",
                structure_subtype=str(best.subtype),
                slice_index=int(best.arterial_slice),
            )
            vein_raw = _load_ctc_curve(
                analysis_directory=analysis_directory,
                structure_type="Vein",
                structure_subtype="Sinus Sagittalis",
                slice_index=int(best.venous_slice),
            )
            if ref_curve is not None and vein_raw is not None and ref_curve.size and vein_raw.size:
                tscc_method = (str(getattr(settings, "TSCC_METHOD", "peaks")) or "peaks").strip().lower()
                rescale_flag = bool(getattr(settings, "TSCC_RESCALE", True))
                if tscc_method == "pchip_fit":
                    baseline_frames = int(getattr(settings, "ROI_DCE_BASELINE_FRAMES", 5))
                    shifted_only, _, ts_only = _fit_tscc_pchip_fit(
                        time_points_s=np.asarray(time_points_s, dtype=float),
                        artery_curve=ref_curve,
                        vein_curve=vein_raw,
                        rescale=False,
                        baseline_frames=baseline_frames,
                    )
                    shifted_scaled, scale, ts = _fit_tscc_pchip_fit(
                        time_points_s=np.asarray(time_points_s, dtype=float),
                        artery_curve=ref_curve,
                        vein_curve=vein_raw,
                        rescale=rescale_flag,
                        baseline_frames=baseline_frames,
                    )
                    sc = (1.0 / float(scale)) if rescale_flag else 1.0
                else:
                    # Shift-only (no rescale).
                    shifted_only, _, _, ts_only_int = align_first_peaks(
                        vein_raw,
                        ref_curve,
                        rescale=False,
                        rescale_method=str(getattr(settings, "TSCC_RESCALE_METHOD", "peak")),
                        num_peaks=int(getattr(settings, "NUMBER_OF_PEAKS", 2)),
                    )
                    ts_only = float(ts_only_int)

                    # Shift + (optional) rescale using current settings.
                    shifted_scaled, _, sc, ts_int = align_first_peaks(
                        vein_raw,
                        ref_curve,
                        rescale=rescale_flag,
                        rescale_method=str(getattr(settings, "TSCC_RESCALE_METHOD", "peak")),
                        num_peaks=int(getattr(settings, "NUMBER_OF_PEAKS", 2)),
                    )
                    ts = float(ts_int)

                shifted_only = shift_curve_to_zero_start(remove_trailing_zeros(np.asarray(shifted_only, dtype=np.float32)))
                shifted_scaled = shift_curve_to_zero_start(remove_trailing_zeros(np.asarray(shifted_scaled, dtype=np.float32)))

                _write_tscc_shift_rescale_demo(
                    analysis_directory=analysis_directory,
                    image_directory=image_directory,
                    time_points_s=np.asarray(time_points_s, dtype=np.float32),
                    reference_subtype=str(best.subtype),
                    reference_curve=np.asarray(ref_curve, dtype=np.float32),
                    vein_curve=np.asarray(vein_raw, dtype=np.float32),
                    tscc_curve=np.asarray(shifted_scaled, dtype=np.float32),
                    tscc_shift_only_curve=np.asarray(shifted_only, dtype=np.float32),
                    scaling=float(sc),
                    time_shift_s=float(ts),
                )
        except Exception:
            pass

    if auto_logging_enabled():
        log_process_end("Time shifting")
