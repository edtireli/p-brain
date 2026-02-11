
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from matplotlib.patches import Rectangle
import os
import nibabel as nib
from nibabel.processing import resample_from_to
from utils.fonts import *
from utils.loading import *
from matplotlib.path import Path
from scipy.signal import argrelextrema
from matplotlib.widgets import Button
import utils.settings as settings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default

turbo_mode = True # doesnt show plots


def _load_map_in_dce_grid(
    analysis_directory: str,
    nifti_directory: str,
    dce_filename: str,
    *,
    base_name: str,
):
    """Load a fitted map (T1/M0) aligned to the DCE voxel grid.

    Preference order:
    1) Analysis/Fitting/<base_name>_in_dce.nii.gz (cached)
    2) Analysis/Fitting/<base_name>.nii.gz resampled into DCE grid and cached
    3) Analysis/Fitting/voxel_<BASE>_matrix.pkl (legacy; only if shape matches DCE)
    """

    dce_path = os.path.join(nifti_directory, dce_filename) if dce_filename else ""
    if not dce_path or not os.path.isfile(dce_path):
        return None

    fitting_dir = os.path.join(analysis_directory, "Fitting")
    target_img = None
    try:
        target_img = nib.load(dce_path)
    except Exception:
        return None

    in_dce_path = os.path.join(fitting_dir, f"{base_name}_in_dce.nii.gz")
    if os.path.isfile(in_dce_path):
        try:
            return np.asarray(nib.load(in_dce_path).get_fdata(), dtype=float)
        except Exception:
            pass

    base_path = os.path.join(fitting_dir, f"{base_name}.nii.gz")
    if os.path.isfile(base_path):
        try:
            src_img = nib.load(base_path)
            resampled = resample_from_to(src_img, (target_img.shape[:3], target_img.affine), order=1)
            try:
                nib.save(resampled, in_dce_path)
            except Exception:
                pass
            return np.asarray(resampled.get_fdata(), dtype=float)
        except Exception:
            pass

    # Legacy pickles (only safe if shape matches DCE).
    pkl_name = None
    if base_name.lower().startswith("t1"):
        pkl_name = "voxel_T1_matrix.pkl"
    elif base_name.lower().startswith("m0"):
        pkl_name = "voxel_M0_matrix.pkl"
    if pkl_name:
        pkl_path = os.path.join(fitting_dir, pkl_name)
        if os.path.isfile(pkl_path):
            try:
                arr = np.asarray(load_from_pickle(pkl_path), dtype=float)
                if arr.shape == tuple(target_img.shape[:3]):
                    return arr
            except Exception:
                pass

    return None


def load_from_pickle(file_path):
    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # NumPy 2.x moved several internals under `numpy._core.*`.
            # Pickles created with NumPy 2 may fail to load under NumPy 1.
            if isinstance(module, str) and module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core', 1)
            return super().find_class(module, name)

    with open(file_path, 'rb') as file:
        try:
            return pickle.load(file)
        except ModuleNotFoundError as e:
            if 'numpy._core' not in str(e):
                raise
            file.seek(0)
            return _CompatUnpickler(file).load()

def turboflash(
    S,
    T1,
    TD=120,
    r1=4000,
    m0=1,
    slice=-1,
    prints=True,
    flip_angle_deg=None,
    *,
    tr_s=None,
    nph=None,
    ctc_model=None,
    baseline_frames: int | None = None,
):
    if flip_angle_deg is None:
        # Preserve legacy behaviour (30°) when metadata is missing, but allow an
        # explicit override via settings/environment.
        if bool(getattr(settings, "STRICT_METADATA", False)) and settings.FLIP_ANGLE_DEG is None:
            raise ValueError(
                "Missing flip angle. Add FlipAngle to the JSON sidecar or set P_BRAIN_FLIP_ANGLE=<deg>. "
                "(STRICT_METADATA is enabled.)"
            )
        theta = 30.0 if settings.FLIP_ANGLE_DEG is None else float(settings.FLIP_ANGLE_DEG)
        if not getattr(turboflash, "_warned_missing_flip_angle", False):
            warnings.warn(
                "Flip angle missing; using default %.1f°. Set P_BRAIN_FLIP_ANGLE or add FlipAngle to the JSON sidecar." % theta,
                RuntimeWarning,
            )
            turboflash._warned_missing_flip_angle = True
    else:
        theta = float(flip_angle_deg)
    theta_rad = np.radians(theta)

    model = (ctc_model or getattr(settings, "CTC_MODEL", "turboflash") or "turboflash").strip().lower()
    if model in {"advanced", "method4", "validated_method4"}:
        model = "turboflash"
    if model != "turboflash":
        raise ValueError(
            f"Unsupported CTC model {model!r}. Validator parity requires CTC_MODEL=turboflash."
        )

    s_arr = np.asarray(S, dtype=float)
    t1_arr = np.asarray(T1, dtype=float)
    m0_arr = np.asarray(m0, dtype=float)

    # Unit normalization: inputs are historically in milliseconds and per-1000.
    TD_s = float(TD) / 1000.0
    r1_s = float(r1) / 1000.0
    T1_s = t1_arr / 1000.0
    valid_t1 = np.isfinite(T1_s) & (T1_s > 0)

    def _nan_like_input():
        return np.full_like(s_arr, np.nan, dtype=float)

    # Validated TurboFLASH conversion (Hemisure validator / MATLAB menu_15_5 method 4).
    # Important: do NOT clamp the log argument or apply any M0 heuristics.
    sin_th = float(np.sin(theta_rad))
    if not np.isfinite(sin_th) or abs(sin_th) < 1e-8:
        return _nan_like_input()

    c = _turboflash_raw_concentration(
        s_arr,
        T1_s,
        valid_t1,
        TD_s,
        r1_s,
        m0_arr,
        sin_th,
    )

    # Diagnostic option: allow a true "raw" curve with no baseline subtraction
    # and no forced zeroing. Used only for debugging/overlay plots.
    if baseline_frames is not None:
        try:
            if int(baseline_frames) <= 0:
                return np.nan_to_num(np.asarray(c, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass

    if baseline_frames is None:
        baseline_frames = int(getattr(settings, "TURBOFLASH_BASELINE_FRAMES", 10) or 10)
    baseline_frames = max(1, int(baseline_frames))
    c_out = np.asarray(c, dtype=float)
    if c_out.ndim >= 1 and c_out.size and c_out.shape[-1] >= baseline_frames:
        # MATLAB/validator baseline correction uses a baseline mean that skips the
        # first two frames (1-based frames 1..2) and averages frames 3..baseline.
        # This matches legacy `detect_baseline(s_input(2:50)+1)` workflows and
        # removes a constant offset vs MATLAB reference curves.
        if baseline_frames >= 3:
            baseline_region = c_out[..., 2:baseline_frames]
        else:
            baseline_region = c_out[..., :baseline_frames]
        baseline = np.nanmean(baseline_region, axis=-1, keepdims=True)
        c_out = c_out - baseline
        try:
            c_out[..., :baseline_frames] = 0.0
        except Exception:
            pass
    else:
        baseline = np.nanmean(c_out)
        c_out = c_out - baseline

    return np.nan_to_num(c_out, nan=0.0, posinf=0.0, neginf=0.0)


def _turboflash_raw_concentration(
    s_arr: np.ndarray,
    T1_s: np.ndarray,
    valid_t1: np.ndarray,
    TD_s: float,
    r1_s: float,
    m0_arr: np.ndarray,
    sin_th: float,
):
    """Raw validated TurboFLASH concentration without baseline correction.

    This is the single source of truth for the pointwise conversion math.
    Baseline handling (method 1) is applied by `turboflash()`.
    """

    inside = 1.0 - (s_arr / (m0_arr * sin_th))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        r1_pre = np.where(valid_t1, 1.0 / T1_s, np.nan)
        if np.ndim(r1_pre) > 0 and r1_pre.shape != s_arr.shape and getattr(s_arr, "ndim", 0) >= 1:
            try:
                r1_pre = np.expand_dims(r1_pre, axis=-1)
            except Exception:
                pass
        c = (-1.0 / (float(r1_s) * float(TD_s))) * (np.log(inside) + float(TD_s) * r1_pre)
        c = np.where(valid_t1, c, np.nan)
    return c


def compute_ctc_peak_and_auc_maps(
    data_4d,
    t1_ms_3d: np.ndarray,
    m0_3d: np.ndarray,
    time_points_s: np.ndarray,
    *,
    TD_ms: float,
    r1_per_1000: float = 4000.0,
    flip_angle_deg: float | None = None,
    baseline_frames: int | None = None,
    ctc_out: np.ndarray | None = None,
):
    """Compute voxelwise peak height and AUC maps of baseline-corrected CTC.

    - Uses the validated TurboFLASH math.
    - Applies the same baseline correction as `turboflash()` (method 1).
    - Memory-efficient: does not materialize the full 4D CTC array.

    Returns:
        peak_map (float32): shape (X,Y,Z)
        auc_map (float32): shape (X,Y,Z), units mM*s
    """

    t1_ms_3d = np.asarray(t1_ms_3d, dtype=float)
    m0_3d = np.asarray(m0_3d, dtype=float)
    time_points_s = np.asarray(time_points_s, dtype=float).ravel()

    if getattr(data_4d, "ndim", 4) != 4:
        raise ValueError(f"Expected 4D DCE data (X,Y,Z,T), got shape {getattr(data_4d, 'shape', None)}")
    if t1_ms_3d.shape != tuple(data_4d.shape[:3]) or m0_3d.shape != tuple(data_4d.shape[:3]):
        raise ValueError(
            f"T1/M0 shapes must match spatial DCE shape {data_4d.shape[:3]}; got T1={t1_ms_3d.shape} M0={m0_3d.shape}"
        )
    if time_points_s.size != data_4d.shape[3]:
        raise ValueError(
            f"time_points_s length must match number of dynamics {data_4d.shape[3]}; got {time_points_s.size}"
        )

    if baseline_frames is None:
        baseline_frames = int(getattr(settings, "TURBOFLASH_BASELINE_FRAMES", 10) or 10)
    baseline_frames = int(baseline_frames)
    baseline_frames = max(1, baseline_frames)

    # Flip angle handling consistent with turboflash().
    if flip_angle_deg is None:
        theta = 30.0 if settings.FLIP_ANGLE_DEG is None else float(settings.FLIP_ANGLE_DEG)
    else:
        theta = float(flip_angle_deg)
    sin_th = float(np.sin(np.radians(theta)))
    if not np.isfinite(sin_th) or abs(sin_th) < 1e-8:
        raise ValueError("Invalid flip angle (sin(theta) ~ 0)")

    # Unit normalization: ms/per-1000 -> seconds/per-second.
    TD_s = float(TD_ms) / 1000.0
    r1_s = float(r1_per_1000) / 1000.0
    T1_s = t1_ms_3d / 1000.0
    valid_t1 = np.isfinite(T1_s) & (T1_s > 0)

    # Baseline mean over frames 3..baseline (1-based) => indices [2:baseline_frames)
    if baseline_frames >= 3:
        baseline_idxs = list(range(2, min(baseline_frames, data_4d.shape[3])))
    else:
        baseline_idxs = list(range(0, min(baseline_frames, data_4d.shape[3])))

    if not baseline_idxs:
        baseline_idxs = [0]

    baseline_sum = np.zeros(data_4d.shape[:3], dtype=float)
    baseline_count = np.zeros(data_4d.shape[:3], dtype=float)
    for t in baseline_idxs:
        c_raw = _turboflash_raw_concentration(
            np.asarray(data_4d[..., t], dtype=float),
            T1_s,
            valid_t1,
            TD_s,
            r1_s,
            m0_3d,
            sin_th,
        )
        ok = np.isfinite(c_raw)
        baseline_sum[ok] += c_raw[ok]
        baseline_count[ok] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        baseline_mean = baseline_sum / np.maximum(baseline_count, 1.0)

    peak_map = np.full(data_4d.shape[:3], -np.inf, dtype=float)
    auc_map = np.zeros(data_4d.shape[:3], dtype=float)

    prev_c = None
    prev_t = None
    for t in range(data_4d.shape[3]):
        c_raw = _turboflash_raw_concentration(
            np.asarray(data_4d[..., t], dtype=float),
            T1_s,
            valid_t1,
            TD_s,
            r1_s,
            m0_3d,
            sin_th,
        )
        c_corr = c_raw - baseline_mean
        if t < baseline_frames:
            c_corr = np.zeros_like(c_corr, dtype=float)
        c_corr = np.nan_to_num(c_corr, nan=0.0, posinf=0.0, neginf=0.0)

        if ctc_out is not None:
            try:
                ctc_out[..., t] = c_corr.astype(np.float32, copy=False)
            except Exception:
                pass

        peak_map = np.maximum(peak_map, c_corr)

        t_s = float(time_points_s[t])
        if prev_c is not None and prev_t is not None:
            dt = float(t_s - prev_t)
            if dt > 0:
                auc_map += 0.5 * (prev_c + c_corr) * dt
        prev_c = c_corr
        prev_t = t_s

    peak_map[~np.isfinite(peak_map)] = 0.0
    peak_map = np.nan_to_num(peak_map, nan=0.0, posinf=0.0, neginf=0.0)
    auc_map = np.nan_to_num(auc_map, nan=0.0, posinf=0.0, neginf=0.0)

    return peak_map.astype(np.float32), auc_map.astype(np.float32)


def compute_CTC(*args, **kwargs):
    """Back-compat wrapper for the validated TurboFLASH conversion."""

    return turboflash(*args, **kwargs)

def interp_nans(arr):
    arr = np.asarray(arr, dtype=float)
    mask = np.isnan(arr)
    if not np.any(mask):
        return arr
    good = ~mask
    if not np.any(good):
        # No valid samples to interpolate from.
        return np.zeros_like(arr)
    x_good = np.flatnonzero(good)
    y_good = arr[good]
    if x_good.size == 1:
        arr[mask] = float(y_good[0])
        return arr
    arr[mask] = np.interp(np.flatnonzero(mask), x_good, y_good)
    return arr


def shifter(array):
    if np.min(array > 0):
        shifted_array = array - np.min(array)
    else:
        shifted_array = array 

        
    if np.any(array < 0):
        shifted_array = array - np.min(array)
    else:
        shifted_array = array
    return shifted_array
    
def custom_shifter(array, baseline_point):
    array = np.asarray(array, dtype=float)
    if array.size == 0:
        return array

    if baseline_point is None:
        baseline_point = 10  # Default baseline point if None
    try:
        baseline_point = int(baseline_point)
    except Exception:
        baseline_point = 10
    baseline_point = max(0, min(baseline_point, int(array.size - 1)))

    array_after_baseline = array[baseline_point:]
    
    # Apply the shifting logic
    if np.min(array_after_baseline) > 0:
        shifted_array = array_after_baseline - np.min(array_after_baseline)
    elif np.any(array_after_baseline < 0):
        shifted_array = array_after_baseline - np.min(array_after_baseline)
    else:
        shifted_array = array_after_baseline
    
    # Add the first few points again but as zeros
    return np.concatenate([np.zeros(baseline_point), shifted_array])


def find_major_peaks(gradient, radius=10, num_peaks=None):
    """
    Finds the indices of the major peaks in the given 1D array based on the gradient.

    Parameters:
        gradient (numpy.ndarray): The 1D array containing the gradient data.
        radius (int): The radius around the peaks for filtering out subdominant peaks.
        num_peaks (int, optional): Number of dominant peaks to return. Defaults to
            ``settings.NUMBER_OF_PEAKS``.

    Returns:
        list: The indices of the dominant peaks.
    """
    if num_peaks is None:
        num_peaks = settings.NUMBER_OF_PEAKS

    # Identify peaks
    peak_indices = argrelextrema(gradient, np.greater)[0]
    peak_values = gradient[peak_indices]

    # Sort peaks by value
    sorted_peak_indices = [x for _, x in sorted(zip(peak_values, peak_indices), reverse=True)]

    # Extract the major peaks based on radius
    major_peaks = []
    for peak in sorted_peak_indices:
        if all(abs(peak - mp) >= radius for mp in major_peaks):
            major_peaks.append(peak)
            if len(major_peaks) >= num_peaks:
                break
    return major_peaks


def _top_peaks_by_height(curve, *, num_peaks=2, separation=5):
    """Return up to `num_peaks` peak indices sorted by height.

    Uses scipy.signal.find_peaks when available (already a dependency here).
    """
    curve = np.asarray(curve, dtype=float)
    if curve.size < 3:
        return []
    try:
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(curve)
    except Exception:
        peaks = []

    if len(peaks) == 0:
        return []

    peaks = [int(p) for p in peaks if np.isfinite(curve[int(p)])]
    peaks.sort(key=lambda p: float(curve[p]), reverse=True)

    chosen = []
    for p in peaks:
        if float(curve[p]) <= 0:
            continue
        if all(abs(p - c) > separation for c in chosen):
            chosen.append(p)
            if len(chosen) >= int(num_peaks):
                break
    return chosen


def detect_truncated_bolus(
    curve,
    *,
    num_peaks=2,
    peak_fraction=0.99,
    plateau_min_points=3,
    plateau_slope_fraction=0.02,
    edge_window=2,
):
    """Heuristically detect truncated/clipped bolus peaks.

    Returns
    -------
    (is_truncated, details)

    Heuristics (robust, low-assumption):
    - Flat/plateau region around a dominant peak (many points near max with near-zero slope)
    - Dominant peak occurring at the very beginning/end of the series

    This aims to catch "cut-off" peaks (common with saturation/clipping or bad ROI).
    """
    arr = np.asarray(curve, dtype=float)
    details = {
        "reason": None,
        "peak_indices": [],
        "plateau_width": None,
        "peak_value": None,
    }

    if arr.size < 8:
        return False, details

    finite = np.isfinite(arr)
    if not np.any(finite):
        return False, details
    arr = arr.copy()
    arr[~finite] = np.nan

    # Smooth gently to avoid noise-driven false positives.
    try:
        # Keep existing behaviour (fs/cutoff used elsewhere in this file).
        smoothed = butter_lowpass_filter(arr, cutoff=3.0, fs=10, order=3)
    except Exception:
        smoothed = arr

    # Use only finite samples for peak logic.
    finite_s = np.isfinite(smoothed)
    if np.sum(finite_s) < 8:
        return False, details

    # Pick the top few peaks (or fall back to global max).
    peaks = _top_peaks_by_height(smoothed, num_peaks=num_peaks, separation=5)
    if not peaks:
        peak_idx = int(np.nanargmax(smoothed))
        peaks = [peak_idx]

    details["peak_indices"] = peaks

    n = int(smoothed.size)
    for peak_idx in peaks:
        peak_val = float(smoothed[peak_idx])
        if not np.isfinite(peak_val) or peak_val <= 0:
            continue

        # Edge-truncation: the maximum is at the boundary (common when capture starts/ends mid-bolus).
        if peak_idx <= int(edge_window) or peak_idx >= (n - 1 - int(edge_window)):
            details.update({"reason": "edge_peak", "peak_value": peak_val, "plateau_width": 1})
            return True, details

        thr = peak_val * float(peak_fraction)
        # Find contiguous plateau region around the peak above `thr`.
        left = peak_idx
        while left - 1 >= 0 and np.isfinite(smoothed[left - 1]) and float(smoothed[left - 1]) >= thr:
            left -= 1
        right = peak_idx
        while right + 1 < n and np.isfinite(smoothed[right + 1]) and float(smoothed[right + 1]) >= thr:
            right += 1

        width = int(right - left + 1)
        if width >= int(plateau_min_points):
            plateau = smoothed[left : right + 1]
            # Flatness: consecutive differences near 0 relative to the peak.
            diffs = np.diff(plateau)
            max_step = float(np.nanmax(np.abs(diffs))) if diffs.size else 0.0
            if max_step <= float(plateau_slope_fraction) * max(1e-9, abs(peak_val)):
                details.update({
                    "reason": "plateau_near_peak",
                    "peak_value": peak_val,
                    "plateau_width": width,
                })
                return True, details

            # Also catch numeric clipping (many values essentially equal).
            ptp = float(np.nanmax(plateau) - np.nanmin(plateau))
            if ptp <= (1e-6 + 1e-3 * abs(peak_val)):
                details.update({
                    "reason": "flat_top_quantized",
                    "peak_value": peak_val,
                    "plateau_width": width,
                })
                return True, details

    return False, details


def find_main_peaks(data, height_threshold, num_peaks=None):
    if num_peaks is None:
        num_peaks = settings.NUMBER_OF_PEAKS
    peaks, _ = find_peaks(data, height=height_threshold)
    sorted_peaks = sorted(peaks, key=lambda x: data[x], reverse=True)[:num_peaks]
    return sorted(sorted_peaks)

def find_baseline_from_minima_skip(data, first_peak, skip_points):
    if first_peak is None:
        return 0
    first_peak = int(first_peak)
    if first_peak <= 0:
        return 0
    skip_points = max(0, int(skip_points))
    # If the peak occurs very early, don't skip past it.
    if first_peak <= skip_points:
        return max(0, first_peak - 1)
    min_val = float('inf')
    min_idx = skip_points
    for i in range(skip_points, first_peak):
        if data[i] < min_val:
            min_val = data[i]
            min_idx = i
    return min_idx

def find_shifted_baseline(data, height_threshold=0.5, skip_points=10, num_peaks=None):
    data = np.asarray(data, dtype=float)
    if data.size < 2 or not np.any(np.isfinite(data)):
        return None
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None
    max_val = float(np.max(finite))
    if not np.isfinite(max_val) or max_val <= 0:
        return None

    main_peaks = find_main_peaks(data, height_threshold * max_val, num_peaks=num_peaks)
    if main_peaks:
        first_peak = main_peaks[0]
        baseline_point = find_baseline_from_minima_skip(data, first_peak, skip_points)
        return baseline_point
    else:
        # Fallback: treat the global max as the peak.
        try:
            first_peak = int(np.nanargmax(data))
        except Exception:
            return None
        return find_baseline_from_minima_skip(data, first_peak, skip_points)



def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def on_esc(event):
    if not turbo_mode:
        if event.key == 'escape':
            if not turbo_mode:
                plt.close(event.canvas.figure)

import threading


def close_plot_after_delay_plt(delay, fig=None):
    if turbo_mode:
        def close_plot():
            if fig:
                plt.close(fig)
            else:
                plt.close()
        threading.Timer(delay, close_plot).start()



def close_plot_after_delay(delay, fig):
    """
    Close the plot automatically after a delay if no interaction occurs.
    :param delay: Time in seconds to wait before closing the plot.
    :param fig: The figure object to close.
    """
    if turbo_mode:
        def close():
            plt.close(fig)

        timer = threading.Timer(delay, close)
        timer.start()

        # If there is user interaction, cancel the timer
        fig.canvas.mpl_connect('key_press_event', lambda event: timer.cancel())


def plot_time_intensity_curves(data, roi_voxels, slice_index, frame_index, time_points_s, analysis_directory, image_directory, type='test', subtype='test'):

    # This helper plots max-voxel and adaptive-max curves; treat the selected
    # curve as "max" for labeling.
    curve_method = "max"

    # Array to store the maximum intensity at each time point from potentially different voxels
    num_time_points = data.shape[3]
    adaptive_voxel_time_courses = np.zeros(num_time_points)
    
    # Loop through each time frame to find the maximum intensity voxel
    for t in range(num_time_points):
        max_intensity_at_t = -1  # Start with a very low value
        for (x, y) in roi_voxels:
            voxel_intensity_at_t = data[x, y, slice_index, t]
            if voxel_intensity_at_t > max_intensity_at_t:
                max_intensity_at_t = voxel_intensity_at_t
        adaptive_voxel_time_courses[t] = max_intensity_at_t

    # For comparison, calculate the maximum intensity voxel over all time (non-adaptive)
    max_intensity = -1
    max_voxel = None
    voxel_time_course_max = None
    for (x, y) in roi_voxels:
        voxel_time_course = data[x, y, slice_index, :]
        if voxel_time_course.max() > max_intensity:
            max_intensity = voxel_time_course.max()
            max_voxel = (x, y)
            voxel_time_course_max = voxel_time_course

    # Filter the max voxel time course for non-adaptive approach
    fs = 15
    cutoff = 4.0
    order = 3
    smoothed_values = butter_lowpass_filter(voxel_time_course_max, cutoff, fs, order)

    # Plotting both non-adaptive and adaptive results
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].scatter(time_points_s, voxel_time_course_max, label='Standard Max ITC', s=5, color='red')
    axs[0].scatter(time_points_s, adaptive_voxel_time_courses, label='Adaptive-max ITC', s=5, color='blue')
    axs[0].plot(time_points_s, smoothed_values, label='Smoothed Standard ITC', linestyle='-', alpha=0.75, color='black')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Signal Intensity (a.u.)')
    axs[0].set_title(f'Intensity-Time Curve Comparison (Slice {slice_index + 1})')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].imshow(data[:, :, slice_index, frame_index], cmap='viridis', origin='lower')
    if max_voxel:
        x, y = max_voxel
        rect = Rectangle((y-0.5, x-0.5), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
        axs[1].add_patch(rect)
    axs[1].set_title(f'Slice with brightest voxel (slice {slice_index + 1}, frame {frame_index + 1})', fontproperties=prop, fontsize=14)

    fig.tight_layout()
    plt.savefig(os.path.join(image_directory, 'Intensity Time Curves', type, subtype, f'ITC+M0_slice_{slice_index+1}.png'), dpi=300)
    if not turbo_mode:
        close_plot_after_delay_plt(3)
        plt.gcf().canvas.mpl_connect('key_press_event', on_esc)    
        plt.show()
        plt.close()

    plt.figure(figsize=(20, 6))
    plt.scatter(time_points_s, voxel_time_course_max, label='Raw ITC', s=5, color='red')
    plt.plot(time_points_s, smoothed_values, label='Smoothened ITC', linestyle='-', alpha=0.75, color='black')
    plt.xlabel('Time (s)', fontproperties=prop, fontsize=15)
    plt.ylabel('Signal intensity (a. u.)', fontproperties=prop, fontsize=15)
    title_sel = 'Intensity-Time Curve (Max voxel)'
    plt.title(f'{title_sel} ({subtype},  Slice {slice_index + 1})', fontproperties=prop, fontsize=18)
    plt.grid(which='minor', alpha=0.25)
    plt.minorticks_on()

    plt.tight_layout()
    plt.savefig(os.path.join(image_directory, 'Intensity Time Curves', type, subtype, f'ITC_slice_{slice_index+1}.png'), dpi=300)
    if not turbo_mode:
        plt.gcf().canvas.mpl_connect('key_press_event', on_esc)   
        close_plot_after_delay_plt(3) 
        plt.close()


    
    np.save(os.path.join(analysis_directory, 'ITC Data', type, subtype, f'ITC_slice_{slice_index+1}.npy'), voxel_time_course_max)
    
    np.save(os.path.join(analysis_directory, 'ROI Data', type, subtype, f'ROI_voxels_slice_{slice_index+1}.npy'), roi_voxels)
    np.save(os.path.join(analysis_directory, 'Frame Data', type, subtype, f'frame_index_slice_{slice_index+1}.npy'), frame_index)



def plot_time_intensity_curves_AI(data, roi_voxels, slice_index, frame_index, time_points_s, analysis_directory, image_directory, type='test', subtype='test'):

    curve_method = (getattr(settings, "VASCULAR_ROI_CURVE_METHOD", "max") or "max").strip().lower()
    if curve_method not in {"max", "mean", "median"}:
        curve_method = "max"

    adaptive_enabled = bool(getattr(settings, "VASCULAR_ROI_ADAPTIVE_MAX", False))

    # Array to store the maximum intensity at each time point from potentially different voxels
    num_time_points = data.shape[3]
    adaptive_voxel_time_courses = np.zeros(num_time_points)
    
    # Loop through each time frame to find the maximum intensity voxel
    for t in range(num_time_points):
        max_intensity_at_t = -1  # Start with a very low value
        for (x, y) in roi_voxels:
            voxel_intensity_at_t = data[x, y, slice_index, t]
            if voxel_intensity_at_t > max_intensity_at_t:
                max_intensity_at_t = voxel_intensity_at_t
        adaptive_voxel_time_courses[t] = max_intensity_at_t

    roi_voxels = np.asarray(roi_voxels)
    if roi_voxels.size == 0:
        return

    # For comparison, calculate the maximum intensity voxel over all time (non-adaptive)
    max_intensity = -1
    max_voxel = None
    voxel_time_course_max = None
    for (x, y) in roi_voxels:
        voxel_time_course = data[int(x), int(y), slice_index, :]
        if voxel_time_course.max() > max_intensity:
            max_intensity = voxel_time_course.max()
            max_voxel = (int(x), int(y))
            voxel_time_course_max = voxel_time_course

    # Optional: mean/median curve over ROI
    roi_tc = data[roi_voxels[:, 0].astype(int), roi_voxels[:, 1].astype(int), slice_index, :]
    mean_curve = roi_tc.mean(axis=0)
    median_curve = np.median(roi_tc, axis=0)
    if curve_method == "max" and adaptive_enabled:
        selected_itc = adaptive_voxel_time_courses
    elif curve_method == "mean":
        selected_itc = mean_curve
    elif curve_method == "median":
        selected_itc = median_curve
    else:
        selected_itc = voxel_time_course_max

    # Filter the max voxel time course for non-adaptive approach
    fs = 15
    cutoff = 4.0
    order = 3
    smoothed_values = butter_lowpass_filter(selected_itc, cutoff, fs, order)

    # Plotting both non-adaptive and adaptive results
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    if curve_method == "mean":
        sel_label = 'Selected ITC (ROI mean)'
        diag_curve = adaptive_voxel_time_courses
        diag_label = 'Adaptive-max ITC (diagnostic)'
    elif curve_method == "median":
        sel_label = 'Selected ITC (ROI median)'
        diag_curve = adaptive_voxel_time_courses
        diag_label = 'Adaptive-max ITC (diagnostic)'
    else:
        if adaptive_enabled:
            sel_label = 'Selected ITC (Adaptive max voxel)'
            diag_curve = voxel_time_course_max
            diag_label = 'Fixed max-voxel ITC (diagnostic)'
        else:
            sel_label = 'Selected ITC (Max voxel)'
            diag_curve = adaptive_voxel_time_courses
            diag_label = 'Adaptive-max ITC (diagnostic)'

    axs[0].scatter(time_points_s, selected_itc, label=sel_label, s=5, color='red')
    axs[0].scatter(time_points_s, diag_curve, label=diag_label, s=5, color='blue')
    axs[0].plot(time_points_s, smoothed_values, label='Smoothed selected ITC', linestyle='-', alpha=0.75, color='black')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Signal Intensity (a.u.)')
    axs[0].set_title(f'Intensity-Time Curve Comparison (Slice {slice_index + 1})')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].imshow(data[:, :, slice_index, frame_index], cmap='viridis', origin='lower')
    if max_voxel:
        x, y = max_voxel
        rect = Rectangle((y-0.5, x-0.5), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
        axs[1].add_patch(rect)
    axs[1].set_title(f'DCE (Slice {slice_index + 1}, frame {frame_index + 1})', fontproperties=prop, fontsize=14)

    fig.tight_layout()
    try:
        os.makedirs(os.path.join(image_directory, 'Intensity Time Curves', type, subtype), exist_ok=True)
    except Exception:
        pass
    plt.savefig(os.path.join(image_directory, 'Intensity Time Curves', type, subtype, f'ITC+DCE_slice_{slice_index+1}.png'), dpi=300)
    if not turbo_mode:
        plt.close()

    plt.figure(figsize=(20, 6))
    plt.scatter(time_points_s, selected_itc, label='Raw ITC', s=5, color='red')
    plt.plot(time_points_s, smoothed_values, label='Smoothened ITC', linestyle='-', alpha=0.75, color='black')
    plt.xlabel('Time (s)', fontproperties=prop, fontsize=15)
    plt.ylabel('Signal intensity (a. u.)', fontproperties=prop, fontsize=15)
    if curve_method == "mean":
        title_sel = 'Intensity-Time Curve (ROI mean)'
    elif adaptive_enabled:
        title_sel = 'Intensity-Time Curve (Adaptive max voxel)'
    else:
        title_sel = 'Intensity-Time Curve (Max voxel)'
    plt.title(f'{title_sel} ({subtype},  Slice {slice_index + 1})', fontproperties=prop, fontsize=18)
    plt.grid(which='minor', alpha=0.25)
    plt.minorticks_on()

    plt.tight_layout()
    try:
        os.makedirs(os.path.join(image_directory, 'Intensity Time Curves', type, subtype), exist_ok=True)
    except Exception:
        pass
    plt.savefig(os.path.join(image_directory, 'Intensity Time Curves', type, subtype, f'ITC_slice_{slice_index+1}.png'), dpi=300)
    if not turbo_mode:
        plt.close()
    try:
        os.makedirs(os.path.join(analysis_directory, 'ITC Data', type, subtype), exist_ok=True)
        os.makedirs(os.path.join(analysis_directory, 'ROI Data', type, subtype), exist_ok=True)
        os.makedirs(os.path.join(analysis_directory, 'Frame Data', type, subtype), exist_ok=True)
    except Exception:
        pass
    np.save(os.path.join(analysis_directory, 'ITC Data', type, subtype, f'ITC_slice_{slice_index+1}.npy'), selected_itc)
    np.save(os.path.join(analysis_directory, 'ROI Data', type, subtype, f'ROI_voxels_slice_{slice_index+1}.npy'), roi_voxels)
    np.save(os.path.join(analysis_directory, 'Frame Data', type, subtype, f'frame_index_slice_{slice_index+1}.npy'), frame_index)



def rescale_peak_to_four(array):
    max_peak = np.max(array)
    threshold = float(getattr(settings, "CTC_PEAK_RESCALE_THRESHOLD", 4.0))
    target_raw = getattr(settings, "CTC_PEAK_RESCALE_TARGET", None)
    try:
        target = float(target_raw) if target_raw is not None else float(threshold)
    except Exception:
        target = float(threshold)
    if max_peak > threshold:
        print(f'Rescaling unphysical concentration curve to {target:.3g} mM!')
        scaling_factor = target / max_peak
        return array * scaling_factor
    return array

def button_callback(event, C_t, type, subtype, analysis_directory, slice_index, radius=10):
    try:
        os.makedirs(os.path.join(analysis_directory, 'CTC Data', type, subtype), exist_ok=True)
    except Exception:
        pass
    np.save(os.path.join(analysis_directory, 'CTC Data', type, subtype, f'CTC_slice_{slice_index+1}.npy'), C_t)
    method_name = 'default'
    if event is not None and getattr(event, 'inaxes', None) is not None:
        method_name = event.inaxes.get_title()
    print(f"Data saved using the {method_name} method.")
    # Shift and rescale CTC before saving
    baseline_point = find_shifted_baseline(C_t, skip_points=radius)
    baseline_point = (int(baseline_point) - 1) if baseline_point is not None else int(radius)
    # Guard against negative / out-of-range baseline indices (e.g. early peak => baseline==0).
    if C_t is not None and np.size(C_t) > 0:
        peak_idx = int(np.nanargmax(np.asarray(C_t, dtype=float)))
        baseline_point = max(0, min(int(baseline_point), int(np.size(C_t) - 1)))
        if peak_idx > 0 and baseline_point >= peak_idx:
            baseline_point = max(0, peak_idx - 1)
    else:
        baseline_point = max(0, int(baseline_point))
    C_t_shifted = custom_shifter(C_t, baseline_point)
    ctc_model = (getattr(settings, "CTC_MODEL", "saturation") or "saturation").strip().lower()
    if ctc_model == "advanced":
        ctc_model = "turboflash"
    if ctc_model not in {"turboflash"} and bool(getattr(settings, "CTC_PEAK_RESCALE_ENABLED", True)):
        C_t_shifted = rescale_peak_to_four(C_t_shifted)
    
    # Save the shifted CTC data
    try:
        os.makedirs(os.path.join(analysis_directory, 'CTC Data', type, subtype), exist_ok=True)
    except Exception:
        pass
    np.save(os.path.join(analysis_directory, 'CTC Data', type, subtype, f'CTC_shifted_slice_{slice_index+1}.npy'), C_t_shifted)
    print(f"Shifted and rescaled CTC data saved using the {method_name} method.")
    if not turbo_mode:
        plt.close()

def save_plot_data_AI(C_t, type, subtype, analysis_directory, slice_index, radius=10):
    try:
        os.makedirs(os.path.join(analysis_directory, 'CTC Data', type, subtype), exist_ok=True)
    except Exception:
        pass
    np.save(os.path.join(analysis_directory, 'CTC Data', type, subtype, f'CTC_slice_{slice_index+1}.npy'), C_t)
    # Shift and rescale CTC before saving
    baseline_point = find_shifted_baseline(C_t, skip_points=radius)
    baseline_point = (int(baseline_point) - 1) if baseline_point is not None else int(radius)
    # Guard against negative / out-of-range baseline indices (e.g. early peak => baseline==0).
    if C_t is not None and np.size(C_t) > 0:
        peak_idx = int(np.nanargmax(np.asarray(C_t, dtype=float)))
        baseline_point = max(0, min(int(baseline_point), int(np.size(C_t) - 1)))
        if peak_idx > 0 and baseline_point >= peak_idx:
            baseline_point = max(0, peak_idx - 1)
    else:
        baseline_point = max(0, int(baseline_point))
    C_t_shifted = custom_shifter(C_t, baseline_point)
    ctc_model = (getattr(settings, "CTC_MODEL", "saturation") or "saturation").strip().lower()
    if ctc_model == "advanced":
        ctc_model = "turboflash"
    if ctc_model not in {"turboflash"} and bool(getattr(settings, "CTC_PEAK_RESCALE_ENABLED", True)):
        C_t_shifted = rescale_peak_to_four(C_t_shifted)
    
    # Save the shifted CTC data
    try:
        os.makedirs(os.path.join(analysis_directory, 'CTC Data', type, subtype), exist_ok=True)
    except Exception:
        pass
    np.save(os.path.join(analysis_directory, 'CTC Data', type, subtype, f'CTC_shifted_slice_{slice_index+1}.npy'), C_t_shifted)
    if not turbo_mode:
        plt.close()

def plot_time_intensity_curves_and_CTC(
    data,
    roi_voxels,
    slice_index,
    frame_index,
    time_points_s,
    analysis_directory,
    image_directory,
    nifti_directory,
    r1=4000,
    TD=120,
    type='test',
    subtype='test',
    IsVFA=False,
    filenames='filenames',
    *,
    rotate_ac: bool | None = None,
):
    
    (
        _t1_3D_filename,
        _axial_t1_3D_filename,
        _t2_3D_filename,
        _axial_t2_3D_filename,
        _flair_3D_filename,
        _axial_flair_3D_filename,
        axial_t2_2D_filename,
        _diffusion_filename,
        dce_filename,
    ) = filenames
    
    # Initialize variables
    max_intensity = -1
    max_voxel = None
    voxel_time_course_max = None
    num_time_points = data.shape[3]
    adaptive_voxel_time_courses = np.zeros(num_time_points)

    # Find maximum voxel over all time and adaptively
    for t in range(num_time_points):
        max_intensity_at_t = -1  # Start with a very low value
        for (x, y) in roi_voxels:
            voxel_intensity_at_t = data[x, y, slice_index, t]
            if voxel_intensity_at_t > max_intensity_at_t:
                max_intensity_at_t = voxel_intensity_at_t
                if max_intensity_at_t > max_intensity:
                    max_intensity = max_intensity_at_t
                    max_voxel = (x, y)
                    voxel_time_course_max = data[x, y, slice_index, :]
            adaptive_voxel_time_courses[t] = max_intensity_at_t

    # Choose which intensity-time curve (ITC) to use (automatic, no prompts).
    curve_method = (getattr(settings, "VASCULAR_ROI_CURVE_METHOD", "max") or "max").strip().lower()
    if curve_method not in {"max", "mean", "median"}:
        curve_method = "max"
    adaptive_enabled = bool(getattr(settings, "VASCULAR_ROI_ADAPTIVE_MAX", False))

    roi_voxels_arr = np.asarray(roi_voxels)
    mean_curve = None
    median_curve = None
    try:
        if roi_voxels_arr.size:
            roi_tc = data[
                roi_voxels_arr[:, 0].astype(int),
                roi_voxels_arr[:, 1].astype(int),
                slice_index,
                :,
            ]
            mean_curve = roi_tc.mean(axis=0)
            median_curve = np.median(roi_tc, axis=0)
    except Exception:
        mean_curve = None
        median_curve = None

    # Ensure this name exists for static analyzers even if ROI is empty.
    selected_itc = voxel_time_course_max

    if curve_method == "max" and adaptive_enabled:
        selected_itc = adaptive_voxel_time_courses
    elif curve_method == "mean":
        selected_itc = mean_curve if mean_curve is not None else voxel_time_course_max
    elif curve_method == "median":
        selected_itc = median_curve if median_curve is not None else voxel_time_course_max
    else:
        selected_itc = voxel_time_course_max
    
    fs = 15
    cutoff = 4.0
    order = 3

    # Compute Concentration-Time Curves
    x, y, z = max_voxel[0], max_voxel[1], slice_index
    # Default: rotate maps in-plane (k=-1) to match the ROI voxel coordinate
    # convention used throughout the vascular AI pipeline.
    # This is intentionally NOT configurable via env vars.
    if rotate_ac is None:
        rotate_ac = True

    t1_map = _load_map_in_dce_grid(analysis_directory, nifti_directory, dce_filename, base_name="t1_map")
    m0_map = _load_map_in_dce_grid(analysis_directory, nifti_directory, dce_filename, base_name="m0_map")
    if t1_map is None or m0_map is None:
        T1_matrix = np.asarray(load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_T1_matrix.pkl')), dtype=float)
        M0_matrix = np.asarray(load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_M0_matrix.pkl')), dtype=float)
    else:
        T1_matrix = np.asarray(t1_map, dtype=float)
        M0_matrix = np.asarray(m0_map, dtype=float)

    if rotate_ac:
        T1_matrix = np.rot90(T1_matrix, -1, axes=(0, 1))
        M0_matrix = np.rot90(M0_matrix, -1, axes=(0, 1))
    T1 = T1_matrix[x, y, z]
    M0 = M0_matrix[x, y, z]

    radius = 10 # radius between boluses, assuming double bolus input
    flip_angle_deg = resolve_flip_angle_deg(
        os.path.join(nifti_directory, dce_filename),
        default=None,
    )
    ctc_model = "turboflash"
    tr_s = None
    nph = None
    ti_s = resolve_turboflash_ti_s(
        os.path.join(nifti_directory, dce_filename),
        default=float(TD) * 1e-3,
    )
    TD = float(ti_s) * 1e3
    C_t_standard = np.array(
        turboflash(
            voxel_time_course_max,
            T1,
            TD,
            r1=4000,
            m0=M0,
            slice=slice_index,
            flip_angle_deg=flip_angle_deg,
            tr_s=tr_s,
            nph=nph,
            ctc_model=ctc_model,
        )
    )
    C_t_adaptive = np.array(
        turboflash(
            adaptive_voxel_time_courses,
            T1,
            TD,
            r1=4000,
            m0=M0,
            slice=slice_index,
            flip_angle_deg=flip_angle_deg,
            tr_s=tr_s,
            nph=nph,
            ctc_model=ctc_model,
        )
    )
    
    C_t_standard = interp_nans(C_t_standard)
    C_t_adaptive = interp_nans(C_t_adaptive)

    #print(f"Slice {slice_index+1}: TD: {TD}, TR: {TR}, T1: {round(T1,1)}, M0: {round(M0,1)}")
    print("")
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 1]})
    smoothed_standard = butter_lowpass_filter(C_t_standard, cutoff, fs, order)
    smoothed_adaptive = butter_lowpass_filter(C_t_adaptive, cutoff, fs, order)

    
    # Concentration-Time Curve
    axs[0].plot(time_points_s, smoothed_standard, color='red', label='Standard CTC')
    axs[0].scatter(time_points_s, C_t_standard, color='r', s=5, label='Standard Raw')
    axs[0].plot(time_points_s, smoothed_adaptive, color='blue', alpha=0.5, label='Adaptive CTC')
    axs[0].scatter(time_points_s, C_t_adaptive, color='blue',alpha=0.5, s=5, label='Adaptive Raw')
    axs[0].legend()
    axs[0].set_xlabel('Time (s)', fontproperties=prop, fontsize=12)
    axs[0].set_ylabel('Concentration (mM)', fontproperties=prop, fontsize=12)
    axs[0].set_title(f'Concentration-Time Curve (Slice {slice_index + 1})',fontproperties=prop, fontsize=14)
    axs[0].grid(which='minor', alpha=0.25)
    axs[0].minorticks_on()

    # Equilibrium Magnetisation Map
    axs[1].imshow(M0_matrix[:, :, slice_index], cmap='plasma', origin='lower')
    if max_voxel:
        x, y = max_voxel  # x, y are row, column indices in the data
        # Plot the rectangle using column, row coordinates
        rect = Rectangle((y-0.5, x-0.5), 1, 1, linewidth=1, edgecolor='g', facecolor='none')
        axs[1].add_patch(rect)
    axs[1].set_title(f'Equilibrium magnetisation map (Slice {slice_index + 1})', fontproperties=prop, fontsize=14)

    if not turbo_mode:
        # Place buttons for saving data
        ax_button_std = plt.axes([0.25, 0.05, 0.15, 0.075])
        ax_button_adp = plt.axes([0.55, 0.05, 0.15, 0.075])
        btn_standard = Button(ax_button_std, 'Save Standard', color='lightgray', hovercolor='gray')
        btn_adaptive = Button(ax_button_adp, 'Save Adaptive', color='lightgray', hovercolor='gray')

        # Callbacks for buttons
        btn_standard.on_clicked(lambda event: button_callback(event, C_t_standard, type, subtype, analysis_directory, slice_index))
        btn_adaptive.on_clicked(lambda event: button_callback(event, C_t_adaptive, type, subtype, analysis_directory, slice_index))
        close_plot_after_delay_special(3, lambda: button_callback(None, C_t_standard, type, subtype, analysis_directory, slice_index))
        plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
        plt.show()


def plot_time_intensity_curves_and_CTC_AI(
    data,
    max_intensity_frame,
    roi_voxels,
    slice_index,
    frame_index,
    time_points_s,
    analysis_directory,
    image_directory,
    nifti_directory,
    r1=4000,
    TD=120,
    type='test',
    subtype='test',
    IsVFA=False,
    filenames='filenames',
    *,
    rotate_ac: bool | None = None,
):
    
    (
        _t1_3D_filename,
        _axial_t1_3D_filename,
        _t2_3D_filename,
        _axial_t2_3D_filename,
        _flair_3D_filename,
        _axial_flair_3D_filename,
        axial_t2_2D_filename,
        _diffusion_filename,
        dce_filename,
    ) = filenames
    
    curve_method = (getattr(settings, "VASCULAR_ROI_CURVE_METHOD", "max") or "max").strip().lower()
    if curve_method not in {"max", "mean", "median"}:
        curve_method = "max"

    adaptive_enabled = bool(getattr(settings, "VASCULAR_ROI_ADAPTIVE_MAX", False))

    # Initialize variables
    max_intensity = -1
    max_voxel = None
    voxel_time_course_max = None
    num_time_points = data.shape[3]
    adaptive_voxel_time_courses = np.zeros(num_time_points)

    roi_voxels = np.asarray(roi_voxels)
    if roi_voxels.size == 0:
        return

    # Find maximum voxel over all time and adaptively
    for t in range(num_time_points):
        max_intensity_at_t = -1  # Start with a very low value
        for (x, y) in roi_voxels:
            voxel_intensity_at_t = data[int(x), int(y), slice_index, t]
            if voxel_intensity_at_t > max_intensity_at_t:
                max_intensity_at_t = voxel_intensity_at_t
                if max_intensity_at_t > max_intensity:
                    max_intensity = max_intensity_at_t
                    max_voxel = (int(x), int(y))
                    voxel_time_course_max = data[int(x), int(y), slice_index, :]
            adaptive_voxel_time_courses[t] = max_intensity_at_t
    
    fs = 15
    cutoff = 4.0
    order = 3

    # Compute Concentration-Time Curves
    x, y, z = max_voxel[0], max_voxel[1], slice_index
    # Load T1/M0 aligned to the DCE grid to ensure exact voxel correspondence.
    # Some ROI pipelines rotate the in-memory DCE array by 90° (k=-1) before
    # voxel indexing; others operate in native DCE voxel coordinates. Tie the
    # Default: rotate maps in-plane (k=-1) to match the ROI voxel coordinate
    # convention used throughout the vascular AI pipeline.
    # This is intentionally NOT configurable via env vars.
    if rotate_ac is None:
        rotate_ac = True

    dce_path = os.path.join(nifti_directory, dce_filename)
    t1_map = _load_map_in_dce_grid(analysis_directory, nifti_directory, dce_filename, base_name="t1_map")
    m0_map = _load_map_in_dce_grid(analysis_directory, nifti_directory, dce_filename, base_name="m0_map")
    if t1_map is None or m0_map is None:
        # Fall back to legacy pickles (best effort) when maps are missing.
        T1_matrix = np.asarray(load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_T1_matrix.pkl')), dtype=float)
        M0_matrix = np.asarray(load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_M0_matrix.pkl')), dtype=float)
    else:
        T1_matrix = np.asarray(t1_map, dtype=float)
        M0_matrix = np.asarray(m0_map, dtype=float)

    # Keep native (DCE-grid) versions for volume exports.
    T1_native = T1_matrix
    M0_native = M0_matrix

    if rotate_ac:
        T1_matrix = np.rot90(T1_matrix, -1, axes=(0, 1))
        M0_matrix = np.rot90(M0_matrix, -1, axes=(0, 1))
    # Parameters for the max-voxel (fixed/adaptive) curve.
    T1_max = T1_matrix[x, y, z]
    M0_max = M0_matrix[x, y, z]

    roi_tc = data[roi_voxels[:, 0].astype(int), roi_voxels[:, 1].astype(int), slice_index, :]
    mean_itc = roi_tc.mean(axis=0)
    median_itc = np.median(roi_tc, axis=0)

    # Parameters for ROI-aggregated curves.
    # Use ROI-mean parameters for mean curve, ROI-median parameters for median curve.
    T1_roi_mean = T1_max
    M0_roi_mean = M0_max
    T1_roi_median = T1_max
    M0_roi_median = M0_max
    t1_vals = T1_matrix[roi_voxels[:, 0].astype(int), roi_voxels[:, 1].astype(int), slice_index]
    m0_vals = M0_matrix[roi_voxels[:, 0].astype(int), roi_voxels[:, 1].astype(int), slice_index]
    t1_vals = t1_vals[np.isfinite(t1_vals)]
    m0_vals = m0_vals[np.isfinite(m0_vals)]
    if t1_vals.size:
        T1_roi_mean = float(np.mean(t1_vals))
        T1_roi_median = float(np.median(t1_vals))
    if m0_vals.size:
        M0_roi_mean = float(np.mean(m0_vals))
        M0_roi_median = float(np.median(m0_vals))

    radius = 10 # radius between boluses, assuming double bolus input
    flip_angle_deg = resolve_flip_angle_deg(
        os.path.join(nifti_directory, dce_filename),
        default=None,
    )
    ctc_model = "turboflash"
    tr_s = None
    nph = None
    ti_s = resolve_turboflash_ti_s(
        os.path.join(nifti_directory, dce_filename),
        default=float(TD) * 1e-3,
    )
    TD = float(ti_s) * 1e3

    # Voxelwise maps + NIfTI exports (native DCE grid): peak height (post-baseline), AUC volume,
    # and full 4D concentration. Controlled by settings (Defaults/env).
    if bool(getattr(settings, "CTC_WRITE_MAPS", True)) and not getattr(plot_time_intensity_curves_and_CTC_AI, "_ctc_maps_done", False):
        try:
            plot_time_intensity_curves_and_CTC_AI._ctc_maps_done = True
            slice_1b = int(getattr(settings, "CTC_MAP_SLICE", 5) or 5)
            # Use DCE native Z for slice selection.
            slice_0b = 0
            try:
                dce_img = nib.load(dce_path)
                dce_proxy = dce_img.dataobj
                slice_0b = max(0, min(int(slice_1b) - 1, int(dce_img.shape[2] - 1)))
            except Exception:
                dce_img = None
                dce_proxy = data
                slice_0b = max(0, min(int(slice_1b) - 1, int(data.shape[2] - 1)))

            baseline_frames = int(getattr(settings, "TURBOFLASH_BASELINE_FRAMES", 10) or 10)
            make_4d = bool(getattr(settings, "CTC_WRITE_4D", True))
            ctc_4d = None
            if make_4d:
                try:
                    ctc_4d = np.zeros(tuple(dce_proxy.shape), dtype=np.float32)
                except Exception:
                    ctc_4d = None

            peak_map, auc_map = compute_ctc_peak_and_auc_maps(
                dce_proxy,
                T1_native,
                M0_native,
                time_points_s,
                TD_ms=float(TD),
                r1_per_1000=4000.0,
                flip_angle_deg=flip_angle_deg,
                baseline_frames=baseline_frames,
                ctc_out=ctc_4d,
            )

            out_a = os.path.join(analysis_directory, "CTC Maps")
            out_i = os.path.join(image_directory, "CTC Maps")
            try:
                os.makedirs(out_a, exist_ok=True)
                os.makedirs(out_i, exist_ok=True)
            except Exception:
                pass

            np.save(os.path.join(out_a, "ctc_peak_map.npy"), peak_map)
            np.save(os.path.join(out_a, "ctc_auc_map.npy"), auc_map)

            # NIfTI exports in native DCE grid
            if dce_img is not None:
                try:
                    hdr3 = dce_img.header.copy()
                    hdr3.set_data_dtype(np.float32)
                    nib.save(
                        nib.Nifti1Image(peak_map.astype(np.float32, copy=False), dce_img.affine, header=hdr3),
                        os.path.join(out_a, "ctc_peak_map.nii.gz"),
                    )
                    nib.save(
                        nib.Nifti1Image(auc_map.astype(np.float32, copy=False), dce_img.affine, header=hdr3),
                        os.path.join(out_a, "ctc_auc_map.nii.gz"),
                    )
                except Exception:
                    pass

                if ctc_4d is not None:
                    try:
                        hdr4 = dce_img.header.copy()
                        hdr4.set_data_dtype(np.float32)
                        nib.save(
                            nib.Nifti1Image(ctc_4d, dce_img.affine, header=hdr4),
                            os.path.join(out_a, "ctc_4d.nii.gz"),
                        )
                    except Exception:
                        pass

            # Slice PNGs
            for name, vol, cmap in (
                ("ctc_peak", peak_map, "magma"),
                ("ctc_auc", auc_map, "magma"),
            ):
                figm, axm = plt.subplots(1, 1, figsize=(6.2, 5.2))
                im = axm.imshow(vol[:, :, slice_0b], origin="lower", cmap=cmap)
                axm.set_title(f"{name} map (Slice {slice_0b + 1})", fontproperties=prop, fontsize=12)
                axm.set_xticks([])
                axm.set_yticks([])
                figm.colorbar(im, ax=axm, fraction=0.046, pad=0.04)
                figm.tight_layout()
                figm.savefig(os.path.join(out_i, f"{name}_slice_{slice_0b + 1}.png"), dpi=300, bbox_inches="tight")
                plt.close(figm)
        except Exception:
            pass
    C_t_fixed_max_voxel = np.array(
        turboflash(
            voxel_time_course_max,
            T1_max,
            TD,
            r1=4000,
            m0=M0_max,
            slice=slice_index,
            flip_angle_deg=flip_angle_deg,
            tr_s=tr_s,
            nph=nph,
            ctc_model=ctc_model,
        )
    )
    C_t_adaptive_voxel = np.array(
        turboflash(
            adaptive_voxel_time_courses,
            T1_max,
            TD,
            r1=4000,
            m0=M0_max,
            slice=slice_index,
            flip_angle_deg=flip_angle_deg,
            tr_s=tr_s,
            nph=nph,
            ctc_model=ctc_model,
        )
    )
    # For compatibility with the case12 debug script: use ROI-mean params for the max-voxel ITC.
    C_t_fixed_max_roi = np.array(
        turboflash(
            voxel_time_course_max,
            T1_roi_mean,
            TD,
            r1=4000,
            m0=M0_roi_mean,
            slice=slice_index,
            flip_angle_deg=flip_angle_deg,
            tr_s=tr_s,
            nph=nph,
            ctc_model=ctc_model,
        )
    )
    C_t_adaptive_roi = np.array(
        turboflash(
            adaptive_voxel_time_courses,
            T1_roi_mean,
            TD,
            r1=4000,
            m0=M0_roi_mean,
            slice=slice_index,
            flip_angle_deg=flip_angle_deg,
            tr_s=tr_s,
            nph=nph,
            ctc_model=ctc_model,
        )
    )
    # Parameters for mean/median ROI curves (match the curve aggregation).
    if curve_method == "median":
        T1_param = T1_roi_median
        M0_param = M0_roi_median
    elif curve_method == "mean":
        T1_param = T1_roi_mean
        M0_param = M0_roi_mean
    else:
        T1_param = T1_max
        M0_param = M0_max

    C_t_mean = np.array(
        turboflash(
            mean_itc,
            T1_param,
            TD,
            r1=4000,
            m0=M0_param,
            slice=slice_index,
            flip_angle_deg=flip_angle_deg,
            tr_s=tr_s,
            nph=nph,
            ctc_model=ctc_model,
        )
    )
    C_t_median = np.array(
        turboflash(
            median_itc,
            T1_param,
            TD,
            r1=4000,
            m0=M0_param,
            slice=slice_index,
            flip_angle_deg=flip_angle_deg,
            tr_s=tr_s,
            nph=nph,
            ctc_model=ctc_model,
        )
    )
    
    C_t_fixed_max_voxel = interp_nans(C_t_fixed_max_voxel)
    C_t_adaptive_voxel = interp_nans(C_t_adaptive_voxel)
    C_t_fixed_max_roi = interp_nans(C_t_fixed_max_roi)
    C_t_adaptive_roi = interp_nans(C_t_adaptive_roi)
    C_t_mean = interp_nans(C_t_mean)
    C_t_median = interp_nans(C_t_median)

    #print(f"Slice {slice_index+1}: TD: {TD}, TR: {TR}, T1: {round(T1,1)}, M0: {round(M0,1)}")
    print("")
    
    itc_for_overlay = None
    t1_for_overlay = None
    m0_for_overlay = None

    if curve_method == "mean":
        C_t_selected = C_t_mean
        C_t_diag = C_t_adaptive_roi
        label_sel = 'Selected CTC (ROI mean)'
        label_diag = 'Adaptive-max CTC (diagnostic)'
        itc_for_overlay = mean_itc
        t1_for_overlay = T1_param if 'T1_param' in locals() else T1_roi_mean
        m0_for_overlay = M0_param if 'M0_param' in locals() else M0_roi_mean
    elif curve_method == "median":
        C_t_selected = C_t_median
        C_t_diag = C_t_adaptive_roi
        label_sel = 'Selected CTC (ROI median)'
        label_diag = 'Adaptive-max CTC (diagnostic)'
        itc_for_overlay = median_itc
        t1_for_overlay = T1_param if 'T1_param' in locals() else T1_roi_median
        m0_for_overlay = M0_param if 'M0_param' in locals() else M0_roi_median
    else:
        if adaptive_enabled:
            # Use max-voxel parameters for the selected curve to avoid ROI-mean
            # parameter bias (can push S/(M0*sinθ) toward saturation and yield
            # unrealistically large CTC values).
            C_t_selected = C_t_adaptive_voxel
            C_t_diag = C_t_adaptive_roi
            label_sel = 'Selected CTC (Adaptive max voxel)'
            label_diag = 'Adaptive-max CTC (ROI-mean params, diagnostic)'
            itc_for_overlay = adaptive_voxel_time_courses
            t1_for_overlay = T1_max
            m0_for_overlay = M0_max
        else:
            C_t_selected = C_t_fixed_max_voxel
            C_t_diag = C_t_fixed_max_roi
            label_sel = 'Selected CTC (Max voxel)'
            label_diag = 'Max-voxel CTC (ROI-mean params, diagnostic)'
            itc_for_overlay = voxel_time_course_max
            t1_for_overlay = T1_max
            m0_for_overlay = M0_max

    fig, axs = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 1]})

    # Concentration-Time Curve (validated only)
    x = time_points_s[: min(len(time_points_s), len(C_t_selected))]
    y = np.asarray(C_t_selected, dtype=float)[: x.size]
    axs[0].plot(x, y, color='red', linewidth=1.8, label='Validated CTC')
    axs[0].legend()
    axs[0].set_xlabel('Time (s)', fontproperties=prop, fontsize=12)
    axs[0].set_ylabel('Concentration (mM)', fontproperties=prop, fontsize=12)
    axs[0].set_title(f'Concentration-Time Curve (Slice {slice_index + 1})',fontproperties=prop, fontsize=14)
    axs[0].grid(which='minor', alpha=0.25)
    axs[0].minorticks_on()

    # Equilibrium Magnetisation Map
    axs[1].imshow(data[:, :, slice_index, max_intensity_frame], cmap='plasma', origin='lower')
    if max_voxel:
        x, y = max_voxel  # x, y are row, column indices in the data
        # Plot the rectangle using column, row coordinates
        rect = Rectangle((y-0.5, x-0.5), 1, 1, linewidth=1, edgecolor='g', facecolor='none')
        axs[1].add_patch(rect)
    axs[1].set_title(f'DCE (Slice {slice_index + 1}, Frame {max_intensity_frame + 1})', fontproperties=prop, fontsize=14)

    fig.tight_layout()
    out_dir = os.path.join(image_directory, 'Concentration Time Curves', type, subtype)
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    plt.savefig(os.path.join(out_dir, f'CTC+DCE_slice_{slice_index+1}.png'), dpi=300)
    if not turbo_mode:
        plt.show(block=False)  # Display the plot without blocking
        plt.pause(2)           # Pause the script to keep the plot open for 2 seconds
    # Persist the selected curve.
    save_plot_data_AI(C_t_selected, type, subtype, analysis_directory, slice_index)

    # Keep legacy filename but plot only the validated CTC curve.
    # (Historically this figure compared multiple normalization/method variants; those are removed.)
    if str(type).strip().lower() in {"artery", "vein"}:
        try:
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4.2))
            x = time_points_s[: min(len(time_points_s), len(C_t_selected))]
            y = np.asarray(C_t_selected, dtype=float)[: x.size]
            ax2.plot(
                x,
                y,
                linestyle='-',
                color='#d62728',
                linewidth=1.8,
                alpha=0.95,
                label='Validated CTC',
            )
            ax2.set_title(f'CTC (validated) (Slice {slice_index + 1})', fontproperties=prop, fontsize=14)
            ax2.set_xlabel('Time (s)', fontproperties=prop, fontsize=12)
            ax2.set_ylabel('Concentration (mM)', fontproperties=prop, fontsize=12)
            ax2.grid(True, alpha=0.2)
            ax2.legend(loc='best', frameon=False)
            fig2.tight_layout()
            fig2.savefig(
                os.path.join(out_dir, f'CTC_baseline_overlay_slice_{slice_index+1}.png'),
                dpi=300,
                bbox_inches='tight',
            )
            plt.close(fig2)
        except Exception:
            pass

        # Optional debug compare: new baseline method (validated) vs old method2 vs raw/no-baseline.
        debug_compare = (os.environ.get("P_BRAIN_DEBUG_CTC_BASELINE_COMPARE", "0") or "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if debug_compare:
            try:
                if itc_for_overlay is None or t1_for_overlay is None or m0_for_overlay is None:
                    raise ValueError("Missing ITC/T1/M0 for baseline comparison")

                # Resolve baseline length used by the validated method.
                baseline_len = int(getattr(settings, "TURBOFLASH_BASELINE_FRAMES", 10) or 10)
                baseline_len = max(1, int(baseline_len))

                # Ensure scalar T1/M0 for method2.
                t1_scalar_ms = float(np.nanmean(np.asarray(t1_for_overlay, dtype=float)))
                m0_scalar = float(np.nanmean(np.asarray(m0_for_overlay, dtype=float)))
                s_vec = np.asarray(itc_for_overlay, dtype=float).squeeze()
                if s_vec.ndim != 1:
                    # Fallback: average over any leading dims to get a single curve.
                    s_vec = np.nanmean(s_vec, axis=tuple(range(s_vec.ndim - 1)))
                    s_vec = np.asarray(s_vec, dtype=float).ravel()

                # Grey dashed: raw curve with NO baseline subtraction and NO forced zeroing.
                c_raw = np.asarray(
                    turboflash(
                        s_vec,
                        t1_scalar_ms,
                        TD,
                        r1=4000,
                        m0=m0_scalar,
                        slice=slice_index,
                        flip_angle_deg=flip_angle_deg,
                        tr_s=tr_s,
                        nph=nph,
                        ctc_model="turboflash",
                        baseline_frames=0,
                    ),
                    dtype=float,
                )

                # Red: validated method (baseline mean over frames 3..baseline and zeroing baseline frames).
                c_m1 = np.asarray(C_t_selected, dtype=float)

                # Blue: legacy Method2 (baseline-normalized S) for comparison only.
                TD_s = float(TD) / 1000.0
                r1_s = float(4000) / 1000.0
                t1_s = float(t1_scalar_ms) / 1000.0
                r1_pre = (1.0 / t1_s) if (np.isfinite(t1_s) and t1_s > 0) else float("nan")

                if baseline_len >= 3:
                    s0 = float(np.nanmean(s_vec[2:baseline_len]))
                else:
                    s0 = float(np.nanmean(s_vec[:baseline_len]))
                if not np.isfinite(s0) or s0 == 0.0:
                    s_norm = np.zeros_like(s_vec, dtype=float)
                else:
                    s_norm = (s_vec - s0) / s0
                s_norm = np.asarray(s_norm, dtype=float)
                s_norm[:baseline_len] = 0.0
                with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
                    c_m2 = (-1.0 / (r1_s * TD_s)) * np.log(1.0 - s_norm * (np.exp(TD_s * r1_pre) - 1.0))
                c_m2 = np.asarray(c_m2, dtype=float)
                # Apply the same baseline correction style as MATLAB used for method2 comparisons.
                if baseline_len > 1 and c_m2.size >= baseline_len:
                    c_m2 = c_m2 - float(np.nanmean(c_m2[1:baseline_len]))
                    c_m2[:baseline_len] = 0.0
                c_m2 = np.nan_to_num(c_m2, nan=0.0, posinf=0.0, neginf=0.0)

                fig3, ax3 = plt.subplots(1, 1, figsize=(10, 4.2))
                x = np.asarray(time_points_s, dtype=float)
                n = int(min(x.size, c_raw.size, c_m1.size, c_m2.size))
                x = x[:n]
                ax3.plot(x, c_raw[:n], linestyle='--', color='0.6', linewidth=1.3, label='Raw (no baseline)')
                ax3.plot(x, c_m1[:n], linestyle='-', color='#d62728', linewidth=1.9, label='Baseline method 1 (validated)')
                ax3.plot(x, c_m2[:n], linestyle='-', color='#1f77b4', linewidth=1.6, alpha=0.95, label='Baseline method 2 (legacy)')

                # Shade baseline region.
                try:
                    if n >= baseline_len:
                        ax3.axvspan(x[0], x[baseline_len - 1], color='0.93', zorder=-1)
                except Exception:
                    pass

                ax3.set_title(f'CTC baseline compare (Slice {slice_index + 1})', fontproperties=prop, fontsize=14)
                ax3.set_xlabel('Time (s)', fontproperties=prop, fontsize=12)
                ax3.set_ylabel('Concentration (mM)', fontproperties=prop, fontsize=12)
                ax3.grid(True, which='major', alpha=0.25)
                ax3.legend(loc='best', frameon=False)
                fig3.tight_layout()
                fig3.savefig(
                    os.path.join(out_dir, f'CTC_baseline_compare_slice_{slice_index+1}.png'),
                    dpi=300,
                    bbox_inches='tight',
                )
                plt.close(fig3)
            except Exception:
                pass
    


def close_plot_after_delay_special(delay, default_save_callback):
    """
    Close the plot automatically after a delay if no interaction occurs, and save the default data.
    :param delay: Time in seconds to wait before closing the plot.
    :param default_save_callback: Function to call for saving the default data.
    """
    if turbo_mode:
        def close():
            default_save_callback()
            plt.close(plt.gcf())  # Close the current figure

        timer = threading.Timer(delay, close)
        timer.start()

        # If there is user interaction, cancel the timer
        plt.gcf().canvas.mpl_connect('key_press_event', lambda event: timer.cancel())
    

class ConcentrationCurveEditor:
    def __init__(self, curve_path, curve2_path):
        self.data = curve_path
        self.original_data = np.copy(self.data)
        self.annotated_points = []
        self.volume_points = []
        self.selected_volume = []
        self.area_closed = False
        self.selecting_volume = True

        self.fig, self.ax = plt.subplots()
        self.redraw()
        if not turbo_mode:
            plt.show()

    def update(self, val):
        self.redraw()

    def onclick(self, event):
        if event.inaxes != self.ax: return
        x, y = int(event.xdata), event.ydata

        if event.key == 'shift' and self.selecting_volume:
            self.volume_points.append(self.volume_points[0])
            self.selected_volume = self.volume_points.copy()
            self.area_closed = True
            self.selecting_volume = False
        elif self.selecting_volume:
            self.volume_points.append((x, y))
        elif not self.selecting_volume and self.area_closed:
            self.annotated_points.append((x, y))
        self.redraw()

    def on_key(self, event):
        if event.key == 'escape':
            plt.close(self.fig)
        elif event.key == 'enter':
            self.adjust_curve()
            self.annotated_points = []
            self.volume_points = []  # Reset the volume points
            self.selected_volume = []  # Reset the selected volume
            self.area_closed = False  # Reset the area closed flag
            self.selecting_volume = True  # Reset to selecting volume
            self.redraw()

    def adjust_curve(self):
        if len(self.annotated_points) < 2 or len(self.volume_points) < 2: return

        x_volume, _ = zip(*sorted(self.volume_points))
        start_volume, end_volume = min(x_volume), max(x_volume)

        x_annot, y_annot = zip(*sorted(self.annotated_points))
        start, end = max(min(x_annot), start_volume), min(max(x_annot), end_volume)

        # Interpolate the user-drawn line within the range of interest
        user_line_segment = np.interp(range(start, end + 1), x_annot, y_annot)

        # Mean of the user-drawn line
        mean_user_line = np.mean(user_line_segment)

        # Selected data points
        selected_data = [self.original_data[i] for i in range(start, end + 1) if Path(self.volume_points).contains_point((i, self.original_data[i]))]

        # Mean of the selected data
        mean_selected_data = np.mean(selected_data)

        # Calculate the required shift to center the data around the line
        required_shift = mean_selected_data - mean_user_line

        # Apply the shift to the selected points to align the means
        for i in range(start, end + 1):
            if Path(self.volume_points).contains_point((i, self.original_data[i])):
                self.data[i] = self.original_data[i] - required_shift

        self.original_data = np.copy(self.data)

    def redraw(self):
        self.ax.clear()
        self.ax.grid(True, which='both')
        self.ax.minorticks_on()
        self.ax.grid(which='minor', alpha=0.25)

        self.ax.plot(self.original_data, 'k:', marker='o', markersize=2, label="Original Curve")
        self.ax.plot(self.data, 'k:', marker='o', markersize=2, label="Adjusted Curve")

        if self.selecting_volume and self.volume_points:
            x_vol, y_vol = zip(*self.volume_points)
            self.ax.plot(x_vol, y_vol, 'r-', linewidth=1, marker='o', markersize=2, label="Volume Points")

        if self.selected_volume:
            path = Path(self.selected_volume)
            for i, point in enumerate(self.original_data):
                if path.contains_point((i, point)):
                    self.ax.scatter(i, point, color='green', alpha=0.6)

        if self.annotated_points:
            x, y = zip(*self.annotated_points)
            self.ax.plot(x, y, 'r-', linewidth=1, marker='o', markersize=2, label="Annotated Points")

        self.ax.legend()
        self.fig.canvas.draw()

    def save_corrected_curve(self):
        save_path = os.path.splitext(self.curve_path)[0] + '.npy'
        np.save(save_path, self.data)

def plot_corrected_tissue_curve(curve_path, data2, roi_voxels_upscaled, slice_index, type='test', time_points_s=1, image_directory = 'dir', rot90 = False, final_curve_path='dir'):
    fig, axs = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 1]})

    # Load existing concentration-time curve and Butterworth low-pass filter
    avg_C_t = curve_path
    fs = 15
    cutoff = 4.0
    order = 3
    smoothed_values = butter_lowpass_filter(avg_C_t, cutoff, fs, order)

    # Concentration-Time Curve
    axs[0].plot(time_points_s, smoothed_values, color='black')
    axs[0].scatter(time_points_s, avg_C_t, color='r', s=5)
    axs[0].set_xlabel('Time (s)', fontproperties=prop, fontsize=12)
    axs[0].set_ylabel('Concentration (mM)', fontproperties=prop, fontsize=12)
    axs[0].set_title(f'Average Concentration-Time Curve (Slice {slice_index + 1})', fontproperties=prop, fontsize=14)
    axs[0].grid(which='minor', alpha=0.25)
    axs[0].minorticks_on()

    if rot90 == False:
        axs[1].imshow(data2[:, :, slice_index], cmap='magma', origin='lower')
        for x, y in roi_voxels_upscaled:
            rect = Rectangle((y, x), 1, 1, linewidth=1, edgecolor='g', facecolor='none', alpha=0.5)
            axs[1].add_patch(rect)
        axs[1].set_title(f'T2-weighted Image (Slice {slice_index + 1})', fontproperties=prop, fontsize=14)
        fig.tight_layout()
        plt.savefig(os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', type, f'CTC+ROI_slice_{slice_index+1}_corrected.png'), dpi=300)
    elif rot90:
        rect_array = np.zeros((data2.shape[0], data2.shape[1]))
        for x, y in roi_voxels_upscaled:
            rect_array[x, y] = 1
        rotated_rect_array = np.rot90(rect_array, 3)
        rotated_roi_voxels = np.array(np.where(rotated_rect_array)).T
        axs[1].imshow(np.rot90(data2[:, :, slice_index],3), cmap='magma', origin='lower')
        for x, y in rotated_roi_voxels:
            rect = Rectangle((y, x), 1, 1, linewidth=1, edgecolor='g', facecolor='none', alpha=0.5)
            axs[1].add_patch(rect)
        axs[1].set_title(f'T2-weighted Image (Slice {slice_index + 1})', fontsize=14) 
    if not turbo_mode:
        plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
        plt.show()
        plt.close()
    np.save(final_curve_path, avg_C_t)

