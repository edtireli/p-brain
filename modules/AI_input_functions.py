import os

# In noninteractive/headless runs (e.g., p-brain-web), ensure Matplotlib does not
# try to use GUI backends like TkAgg (which requires tkinter).
if os.environ.get("PBRAIN_NONINTERACTIVE") == "1" or os.environ.get("PBRAIN_TURBO") == "1":
    os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2
from termcolor import colored
from utils.fonts import *
from utils.mapping import *
from utils.plotting import *
from utils.loading import *
from matplotlib.path import Path
from utils.settings import AI_MODEL_PATHS
from utils.cli_logging import auto_logging_enabled, log_process_end, log_process_start
import glob
import json
import time

from utils.loading import build_time_points_s, resolve_dce_time_step_s

import utils.settings as settings
import modules.time_shifting as tscc


def _roi_method() -> str:
    return (getattr(settings, "ROI_METHOD", "ai") or "ai").strip().lower()

turbo_mode = True # doesnt show plots


def get_available_arteries(analysis_directory):
    artery_types = {'lica': 'Left Interior Carotid', 'rica': 'Right Interior Carotid', 'b': 'Basilar', 'lmca': 'Left Middle Cerebral', 'rmca': 'Right Middle Cerebral'}
    available_arteries = {}
    available_slices = {}
    
    for code, artery in artery_types.items():
        artery_dir = os.path.join(analysis_directory, 'CTC Data', 'Artery', artery)
        if os.path.exists(artery_dir) and any(fname.endswith('.npy') for fname in os.listdir(artery_dir)):
            available_arteries[code] = artery
            slice_indexes = []
            for i in range(1, 11):
                shifted_slice_file = os.path.join(artery_dir, f'CTC_shifted_slice_{i}.npy')
                slice_file = os.path.join(artery_dir, f'CTC_slice_{i}.npy')
                if os.path.exists(shifted_slice_file):
                    slice_indexes.append(i)
                elif os.path.exists(slice_file):
                    slice_indexes.append(i)
            available_slices[code] = slice_indexes

    return available_arteries, available_slices



def get_available_slices(analysis_directory, structure_type, structure_subtype):
    available_slices = []
    for i in range(1, 11):
        slice_file = os.path.join(analysis_directory, 'CTC Data', structure_type, structure_subtype, f'CTC_shifted_slice_{i}.npy')
        if os.path.exists(slice_file):
            available_slices.append(i)
    return available_slices


from scipy.signal import correlate, find_peaks

def align_first_peaks(vein_curve, artery_curve, radius=10, double_peak_radius=3, num_peaks=None):
    cross_corr = correlate(vein_curve, artery_curve)
    shift = np.argmax(cross_corr) - len(vein_curve) + 1

    if shift >= 0:
        aligned_vein_curve = vein_curve[shift:]
    else:
        aligned_vein_curve = np.concatenate([vein_curve[-shift:], np.zeros(-shift)])

    vein_peaks, _ = find_peaks(aligned_vein_curve)
    artery_peaks, _ = find_peaks(artery_curve)

    if num_peaks is None:
        num_peaks = int(getattr(settings, "NUMBER_OF_PEAKS", 2))

    def filter_double_peaks(peaks, curve, radius, n_peaks):
        sorted_peaks = sorted(peaks, key=lambda x: curve[x], reverse=True)
        top_peaks = []
        
        for p in sorted_peaks:
            if all(abs(p - tp) > radius for tp in top_peaks):
                top_peaks.append(p)
                if len(top_peaks) >= n_peaks:
                    break

        return top_peaks

    vein_top2_peaks = filter_double_peaks(vein_peaks, aligned_vein_curve, radius, num_peaks)
    artery_top2_peaks = filter_double_peaks(artery_peaks, artery_curve, radius, num_peaks)

    rescaled = 1

    for v_peak, a_peak in zip(vein_top2_peaks, artery_top2_peaks):
        if aligned_vein_curve[v_peak] < artery_curve[a_peak]:
            scaling_factor = artery_curve[a_peak] / aligned_vein_curve[v_peak]
            aligned_vein_curve *= scaling_factor
            rescaled = scaling_factor
            break

    is_double_peak = any(abs(vein_top2_peaks[i] - vein_top2_peaks[j]) <= double_peak_radius for i in range(len(vein_top2_peaks)) for j in range(i+1, len(vein_top2_peaks)))

    if is_double_peak:
        shift += 1
        aligned_vein_curve = aligned_vein_curve[1:]

    return aligned_vein_curve, vein_top2_peaks, rescaled, shift

def remove_trailing_zeros(arr):
    non_zero_idx = np.where(arr != 0)[0]
    if len(non_zero_idx) == 0:
        return np.array([])
    last_non_zero = non_zero_idx[-1]
    return arr[:last_non_zero+1]


def shift_curve_to_zero_start(curve):
    start_value = curve[0]
    return curve - start_value


def select_arterial_slice(available_slices):
    print("Available arterial slices:", available_slices)
    selected_slice = int(input("Select an arterial slice: "))
    if selected_slice not in available_slices:
        print("Invalid choice, exiting.")
        exit(1)
    return selected_slice

def select_venous_slice(available_slices):
    print("Available venous slices:", available_slices)
    selected_slice = int(input("Select a venous slice: "))
    if selected_slice not in available_slices:
        print("Invalid choice, exiting.")
        exit(1)
    return selected_slice



def find_max_npy_file(analysis_directory, num_peaks=None, peak_tolerance=0.20, peak_separation=10):
    subtypes = ["Left Interior Carotid", "Right Interior Carotid", "Basilar", "Left Middle Cerebral", "Right Middle Cerebral"]

    if num_peaks is None:
        num_peaks = int(getattr(settings, "NUMBER_OF_PEAKS", 2))

    def _top_peaks(curve, n_peaks, separation):
        peaks, _ = find_peaks(curve)
        if len(peaks) == 0:
            return []
        sorted_peaks = sorted(peaks, key=lambda x: curve[x], reverse=True)
        chosen = []
        for p in sorted_peaks:
            if not np.isfinite(curve[p]):
                continue
            if curve[p] <= 0:
                continue
            if all(abs(p - cp) > separation for cp in chosen):
                chosen.append(p)
                if len(chosen) >= n_peaks:
                    break
        return chosen

    def _load_artery_curve(artery_subtype, arterial_slice_index):
        artery_dir = os.path.join(analysis_directory, 'CTC Data', 'Artery', artery_subtype)
        for filename in (f'CTC_shifted_slice_{arterial_slice_index}.npy', f'CTC_slice_{arterial_slice_index}.npy'):
            path = os.path.join(artery_dir, filename)
            if os.path.exists(path):
                return np.load(path)
        return None

    def _peaks_consistent(vein_curve, artery_curve, n_peaks, tolerance, separation):
        if artery_curve is None or len(vein_curve) == 0 or len(artery_curve) == 0:
            return False

        vein_peaks = _top_peaks(vein_curve, n_peaks, separation)
        artery_peaks = _top_peaks(artery_curve, n_peaks, separation)
        if len(vein_peaks) < n_peaks or len(artery_peaks) < n_peaks:
            return False

        ratios = []
        for v_peak, a_peak in zip(vein_peaks, artery_peaks):
            v = float(vein_curve[v_peak])
            a = float(artery_curve[a_peak])
            if not np.isfinite(v) or not np.isfinite(a) or a == 0:
                return False
            ratios.append(v / a)

        ratios = np.asarray(ratios, dtype=float)
        median_ratio = float(np.median(ratios))
        if not np.isfinite(median_ratio) or median_ratio == 0:
            return False

        rel_dev = np.abs(ratios - median_ratio) / abs(median_ratio)
        return bool(np.all(rel_dev <= tolerance))

    best_any = {
        'value': float('-inf'),
        'file_path': "",
        'subtype': "",
        'slice_index': -1,
        'arterial_slice_index': -1,
    }
    best_consistent = {
        'value': float('-inf'),
        'file_path': "",
        'subtype': "",
        'slice_index': -1,
        'arterial_slice_index': -1,
    }

    for subtype in subtypes:
        file_paths = glob.glob(os.path.join(analysis_directory, 'TSCC Data', subtype, '*.npy'))
        for file_path in file_paths:
            arr = np.load(file_path)
            curr_max = float(np.max(arr))

            base = os.path.basename(file_path)
            split_filename = base.split('_')
            if len(split_filename) < 4:
                continue
            slice_index = int(split_filename[-2])
            arterial_slice_index = int(split_filename[-1].split('.npy')[0])

            if curr_max > best_any['value']:
                best_any.update({
                    'value': curr_max,
                    'file_path': file_path,
                    'subtype': subtype,
                    'slice_index': slice_index,
                    'arterial_slice_index': arterial_slice_index,
                })

            artery_curve = _load_artery_curve(subtype, arterial_slice_index)
            if _peaks_consistent(arr, artery_curve, num_peaks, peak_tolerance, peak_separation):
                if curr_max > best_consistent['value']:
                    best_consistent.update({
                        'value': curr_max,
                        'file_path': file_path,
                        'subtype': subtype,
                        'slice_index': slice_index,
                        'arterial_slice_index': arterial_slice_index,
                    })

    best = best_consistent if best_consistent['file_path'] else best_any
    return best['file_path'], best['value'], best['subtype'], best['slice_index'], best['arterial_slice_index']


def plot_transformed_curves_max(shifted_vein_curve, slice_index, artery_index, vein_top2_peaks, subtype='test', time_points_s=1, analysis_directory='dir', image_directory = 'dir'):
    def on_esc(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)

    fs = 10
    cutoff = 3.0
    order = 3
    smoothed_vein = butter_lowpass_filter(shifted_vein_curve, cutoff, fs, order)

    plt.figure(figsize=(10, 5))
    plt.title(f"Rescaled & Time-shifted Concentration Curve for {subtype} (Veinous Slice {slice_index}, Aterial Slice {artery_index})", fontproperties=prop, fontsize=16)
    plt.plot(time_points_s[0:len(shifted_vein_curve)], smoothed_vein, label=f'Time-Shifted Vein Curve', color=blaa1)
    plt.scatter(time_points_s[0:len(shifted_vein_curve)], shifted_vein_curve, label=f'Time-Shifted Vein Curve', color=roed, s=5)
    
    plt.xlabel('Time (s)', fontproperties=prop, fontsize=14)
    plt.ylabel('Concentration (mM)', fontproperties=prop, fontsize=14)
    plt.legend()
    plt.grid(which='minor', alpha=0.25)
    plt.minorticks_on()

    plt.tight_layout()
    plt.savefig(os.path.join(image_directory, 'Time Shifted Concentration Curves', 'Max', f'TSCC_slice_{slice_index}_{artery_index}.png'), dpi=300)
    np.save(os.path.join(analysis_directory, 'TSCC Data', 'Max', f'TSCC_slice_{slice_index}_{artery_index}.npy'), shifted_vein_curve)
    if not turbo_mode:
        plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
        close_plot_after_delay_plt(3)
        plt.show()



def plot_transformed_curves(shifted_vein_curve, shifted_artery_curve, slice_index, arterial_slice_index, vein_top2_peaks, time_points_s, analysis_directory, image_directory, subtype='test', scaling=1, time_shift=1):
    global turbo_mode
    # No need to modify the subtype here; it should already be correct
    def on_esc(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)

    fs = 15
    cutoff = 4.0
    order = 3
    smoothed_vein = butter_lowpass_filter(shifted_vein_curve, cutoff, fs, order)
    plt.figure(figsize=(10, 5))
    if scaling == 1:
        plt.title(f" Time-shifted Concentration Curve for {subtype} (Veinous Slice {slice_index}, Arterial Slice {arterial_slice_index})", fontproperties=prop, fontsize=16)
    elif scaling != 1:
        plt.title(f"Rescaled & Time-shifted Concentration Curve for {subtype} (Veinous Slice {slice_index}, Arterial Slice {arterial_slice_index})", fontproperties=prop, fontsize=16)
    
    if scaling != 1:
        plt.plot(time_points_s[0:len(shifted_vein_curve)], shifted_vein_curve, label=f'Rescaled ({round(scaling,1)}) & Time-Shifted ({time_shift} s) Vein Curve', color=blaa)
    else:    
        plt.plot(time_points_s[0:len(shifted_vein_curve)], shifted_vein_curve, label=f'Time-Shifted ({time_shift} s) Vein Curve', color=blaa)
    
    plt.plot(time_points_s[0:len(shifted_artery_curve)], shifted_artery_curve, label=f'Artery Curve', color=roed, alpha=0.75, linestyle='dotted')

    plt.xlabel('Time (s)', fontproperties=prop, fontsize=14)
    plt.ylabel('Concentration (mM)', fontproperties=prop, fontsize=14)
    plt.legend()
    plt.grid(which='minor', alpha=0.25)
    plt.minorticks_on()

    plt.tight_layout()
    plt.savefig(os.path.join(image_directory, 'Time Shifted Concentration Curves', subtype, f'TSCC_slice_{slice_index}_{arterial_slice_index}.png'), dpi=300)
    np.save(os.path.join(analysis_directory, 'TSCC Data', subtype, f'TSCC_slice_{slice_index}_{arterial_slice_index}.npy'), shifted_vein_curve)
    if not turbo_mode:
        plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
        close_plot_after_delay_plt(3)
        plt.show()    



def preprocess_data(filename, *, projection: str = "mean"):
    """Load and rotate the DCE volume.

    Returns:
    - normalized_projection: 3D (x,y,slice) float32 in [0,1] used by legacy 2D models
      (SSS slice classifier/ROI pipeline).
    - mri_data: 4D rotated raw data (x,y,slice,time)
    - mri_data_norm: 4D rotated data, globally min-max normalized to [0,1]
      (useful for ICA slice classification on raw frames).
    """
    mri_data = nib.load(filename).get_fdata()
    # Discard leading unstable DCE volumes when requested.
    _skip = getattr(settings, "DCE_SKIP_VOLUMES", 0)
    if _skip > 0 and mri_data.ndim >= 4 and mri_data.shape[-1] > _skip:
        mri_data = mri_data[..., _skip:]
    # Model training/labeling was performed in a fixed orientation. Historically we
    # used a single rot90 here; keep that as the default but allow override.
    try:
        rot_k = int(str(os.getenv("P_BRAIN_AI_ROT90_K", "-1")).strip())
    except Exception:
        rot_k = -1
    rot_k = rot_k % 4
    mri_data = np.rot90(mri_data, k=rot_k, axes=(0, 1))

    # Projection for the legacy 2D inference path (SSS).
    proj = None
    if str(projection).strip().lower() == "max":
        proj = np.max(mri_data, axis=-1)
    else:
        proj = np.mean(mri_data, axis=-1)
    normalized_projection = _minmax01(proj)

    # Global normalization for raw-frame ICA classification.
    mri_data_norm = _minmax01(mri_data)
    return normalized_projection, mri_data, mri_data_norm


def _minmax01(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    vmin = float(np.nanmin(img))
    vmax = float(np.nanmax(img))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - vmin) / (vmax - vmin)
    return out.astype(np.float32, copy=False)


def _normalize_to_uint8_slice(img2d: np.ndarray) -> np.ndarray:
    """Per-slice min-max normalization to uint8 (matches debug script semantics)."""
    x = np.asarray(img2d, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - mn) / (mx - mn)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)


def _center_crop_or_pad_2d(img: np.ndarray, target_hw: tuple[int, int] = (256, 256)) -> np.ndarray:
    h, w = img.shape
    th, tw = target_hw

    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    if pad_h or pad_w:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        img = np.pad(img, ((top, bottom), (left, right)), mode="constant", constant_values=0)
        h, w = img.shape

    start_h = (h - th) // 2
    start_w = (w - tw) // 2
    return img[start_h : start_h + th, start_w : start_w + tw]


def _prep_slice_float01(sl2d: np.ndarray, *, flip_lr: bool) -> np.ndarray:
    """Prepare a single 2D slice for the ICA classifier/ROI models.

    For ICA we found the rICA slice-classifier behaves well on individual DCE
    frames normalized to [0,1]. We keep this lightweight to stay fast.
    """
    # Match the working CLI debug semantics:
    # - per-slice min-max normalize to uint8
    # - center crop/pad to 256x256 (do NOT stretch via resize)
    # - scale to [0,1]
    u8 = _normalize_to_uint8_slice(sl2d)
    if flip_lr:
        u8 = np.fliplr(u8)
    if u8.shape != (256, 256):
        u8 = _center_crop_or_pad_2d(u8, (256, 256))
    return (u8.astype(np.float32) / 255.0)


def _select_best_ica_frame(
    mri_data_norm_ica: np.ndarray,
    slice_classifier,
    *,
    flip_lr: bool,
    frame_candidates: list[int],
) -> tuple[int, float]:
    """Pick a single frame index that maximizes ICA slice-classifier confidence.

    Returns (best_frame_index, best_max_probability_over_slices).
    """
    tmax = int(mri_data_norm_ica.shape[3])
    cands = [int(t) for t in frame_candidates if 0 <= int(t) < tmax]
    if not cands:
        return 0, 0.0

    zmax = int(mri_data_norm_ica.shape[2])

    # Build one big batch: (n_frames * n_slices, 256,256,1)
    batch = np.empty((len(cands) * zmax, 256, 256, 1), dtype=np.float32)
    idx = 0
    for t in cands:
        for z in range(zmax):
            sl = _prep_slice_float01(mri_data_norm_ica[:, :, z, t], flip_lr=flip_lr)
            batch[idx, :, :, 0] = sl
            idx += 1

    try:
        preds = slice_classifier.predict(batch, verbose=0)
    except TypeError:
        preds = slice_classifier.predict(batch)
    scores = np.asarray(preds, dtype=np.float32).reshape(len(cands), zmax)
    max_per_frame = np.max(scores, axis=1)
    best_i = int(np.argmax(max_per_frame))
    return int(cands[best_i]), float(max_per_frame[best_i])


def _select_best_ica_per_slice(
    mri_data_norm_ica: np.ndarray,
    slice_classifier,
    *,
    flip_lr: bool,
    frame_candidates: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """For each slice z, pick the frame t that maximizes slice-classifier score.

    Returns:
    - best_t_per_slice: (z,) int
    - best_score_per_slice: (z,) float
    """
    tmax = int(mri_data_norm_ica.shape[3])
    cands = [int(t) for t in frame_candidates if 0 <= int(t) < tmax]
    if not cands:
        zmax = int(mri_data_norm_ica.shape[2])
        return np.zeros((zmax,), dtype=np.int32), np.zeros((zmax,), dtype=np.float32)

    zmax = int(mri_data_norm_ica.shape[2])

    # Build one big batch: (n_frames * n_slices, 256,256,1)
    batch = np.empty((len(cands) * zmax, 256, 256, 1), dtype=np.float32)
    idx = 0
    for t in cands:
        for z in range(zmax):
            sl = _prep_slice_float01(mri_data_norm_ica[:, :, z, t], flip_lr=flip_lr)
            batch[idx, :, :, 0] = sl
            idx += 1

    try:
        preds = slice_classifier.predict(batch, verbose=0)
    except TypeError:
        preds = slice_classifier.predict(batch)
    scores = np.asarray(preds, dtype=np.float32).reshape(len(cands), zmax)

    best_i = np.argmax(scores, axis=0).astype(np.int32)
    best_t = np.asarray([cands[int(i)] for i in best_i], dtype=np.int32)
    best_score = scores[best_i, np.arange(zmax, dtype=np.int32)].astype(np.float32)
    return best_t, best_score


def extract_and_accumulate_rois_best_frame_per_slice(
    mri_data_norm_ica: np.ndarray,
    slice_classifier,
    roi_model,
    choice,
    *,
    frame_candidates: list[int],
    slice_relevance_thresholds: tuple[float, ...] = (0.5, 0.3),
    flip_lr: bool = False,
    max_slices: int = 3,
):
    """ICA inference that can return multiple slices.

    We score each slice across candidate frames, pick each slice's best frame,
    then select up to `max_slices` slices and segment ROIs on those.
    """
    relevant_slices: list[np.ndarray] = []
    relevant_rois: list[np.ndarray] = []
    original_slices: list[np.ndarray] = []
    slice_labels: list[int] = []
    roi_voxels: dict[int, np.ndarray] = {}

    zmax = int(mri_data_norm_ica.shape[2])
    best_t, best_score = _select_best_ica_per_slice(
        mri_data_norm_ica,
        slice_classifier,
        flip_lr=flip_lr,
        frame_candidates=frame_candidates,
    )

    # Select slices: prefer thresholds (high->low), but cap by max_slices.
    try:
        max_slices_i = int(max_slices)
    except Exception:
        max_slices_i = 3
    max_slices_i = max(1, min(max_slices_i, zmax))

    thresholds = [float(t) for t in (slice_relevance_thresholds or (0.5,))]
    thresholds = [t for t in thresholds if np.isfinite(t) and 0.0 < t <= 1.0]
    if not thresholds:
        thresholds = [0.5]

    chosen: list[int] = []
    for thr in thresholds:
        cand = [int(z) for z in range(zmax) if float(best_score[z]) > float(thr)]
        cand = sorted(cand, key=lambda z: float(best_score[z]), reverse=True)
        chosen = cand[:max_slices_i]
        if chosen:
            break
    if not chosen:
        # As a final fallback: take the top-N slices even if below threshold.
        order = np.argsort(-best_score)
        chosen = [int(z) for z in order[:max_slices_i] if float(best_score[int(z)]) > 0.0]

    if not chosen:
        return (original_slices, relevant_slices, relevant_rois, slice_labels, roi_voxels)

    # Report chosen slices.
    for z in chosen:
        try:
            print(
                f"Slice {z + 1} is classified as relevant (probability: {float(best_score[z]):.2f}, frame: {int(best_t[z])})."
            )
        except Exception:
            pass

    # ROI segmentation batch.
    roi_batch = np.empty((len(chosen), 256, 256, 1), dtype=np.float32)
    for i, z in enumerate(chosen):
        t = int(best_t[z])
        roi_batch[i, :, :, 0] = _prep_slice_float01(mri_data_norm_ica[:, :, z, t], flip_lr=flip_lr)

    try:
        roi_preds = roi_model.predict(roi_batch, verbose=0)
    except TypeError:
        roi_preds = roi_model.predict(roi_batch)
    roi_preds = np.asarray(roi_preds)

    for i, z in enumerate(chosen):
        mask = np.squeeze(roi_preds[i])
        thr = 0.5 * float(np.max(mask)) if np.size(mask) else 0.0
        binary_mask_model = (mask > thr).astype(np.uint8)
        binary_mask = np.fliplr(binary_mask_model) if flip_lr else binary_mask_model

        # Resize mask back to native slice resolution for voxel indexing.
        try:
            h, w = int(mri_data_norm_ica.shape[0]), int(mri_data_norm_ica.shape[1])
            binary_mask_native = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            binary_mask_native = (binary_mask_native > 0).astype(np.uint8)
        except Exception:
            binary_mask_native = binary_mask

        roi_coords = np.argwhere(binary_mask_native)
        roi_voxels[int(z)] = roi_coords

        t = int(best_t[z])
        selected_slice = mri_data_norm_ica[:, :, z, t]
        if flip_lr:
            selected_slice = np.fliplr(selected_slice)
        original_slices.append(selected_slice)
        relevant_slices.append(roi_batch[i, :, :, 0])
        relevant_rois.append(binary_mask)
        slice_labels.append(int(choice))

    return (original_slices, relevant_slices, relevant_rois, slice_labels, roi_voxels)


def extract_and_accumulate_rois_single_frame(
    mri_data_norm_ica: np.ndarray,
    slice_classifier,
    roi_model,
    choice,
    *,
    frame_index: int,
    slice_relevance_threshold: float = 0.5,
    flip_lr: bool = False,
):
    """Fast ICA inference on a single chosen frame (batched predict).

    Returns the same 5-tuple as other extract_* helpers.
    """
    relevant_slices = []
    relevant_rois = []
    original_slices = []
    slice_labels = []
    roi_voxels = {}

    zmax = int(mri_data_norm_ica.shape[2])
    t = int(frame_index)
    t = max(0, min(t, int(mri_data_norm_ica.shape[3]) - 1))

    batch = np.empty((zmax, 256, 256, 1), dtype=np.float32)
    for z in range(zmax):
        batch[z, :, :, 0] = _prep_slice_float01(mri_data_norm_ica[:, :, z, t], flip_lr=flip_lr)

    try:
        preds = slice_classifier.predict(batch, verbose=0)
    except TypeError:
        preds = slice_classifier.predict(batch)
    scores = np.asarray(preds, dtype=np.float32).reshape(zmax)

    relevant_idx = [int(z) for z in range(zmax) if float(scores[z]) > float(slice_relevance_threshold)]
    if relevant_idx:
        for z in relevant_idx:
            print(
                f"Slice {z + 1} is classified as relevant (probability: {float(scores[z]):.2f}, frame: {t})."
            )
    else:
        return (original_slices, relevant_slices, relevant_rois, slice_labels, roi_voxels)

    roi_batch = batch[relevant_idx, :, :, :]
    try:
        roi_preds = roi_model.predict(roi_batch, verbose=0)
    except TypeError:
        roi_preds = roi_model.predict(roi_batch)
    roi_preds = np.asarray(roi_preds)

    for j, z in enumerate(relevant_idx):
        mask = np.squeeze(roi_preds[j])
        thr = 0.5 * float(np.max(mask)) if np.size(mask) else 0.0
        binary_mask_model = (mask > thr).astype(np.uint8)
        binary_mask = np.fliplr(binary_mask_model) if flip_lr else binary_mask_model

        # Resize mask back to native slice resolution for voxel indexing.
        try:
            h, w = int(mri_data_norm_ica.shape[0]), int(mri_data_norm_ica.shape[1])
            binary_mask_native = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            binary_mask_native = (binary_mask_native > 0).astype(np.uint8)
        except Exception:
            binary_mask_native = binary_mask

        roi_coords = np.argwhere(binary_mask_native)
        roi_voxels[z] = roi_coords

        # Store for montage plotting.
        selected_slice = mri_data_norm_ica[:, :, z, t]
        if flip_lr:
            selected_slice = np.fliplr(selected_slice)
        original_slices.append(selected_slice)
        relevant_slices.append(batch[z, :, :, 0])
        relevant_rois.append(binary_mask)
        slice_labels.append(choice)

    return (original_slices, relevant_slices, relevant_rois, slice_labels, roi_voxels)


def _max_slice_relevance(rotated_data: np.ndarray, model, *, flip_lr: bool = False) -> float:
    max_p = 0.0
    for slice_index in range(rotated_data.shape[2]):
        # Keep the same normalization semantics used in the main inference path
        # (volume-level normalization from preprocess_data).
        sl = rotated_data[:, :, slice_index].astype(np.float32, copy=False)
        if flip_lr:
            sl = np.fliplr(sl)
        x = cv2.resize(sl, (256, 256), interpolation=cv2.INTER_LINEAR)
        x = np.expand_dims(x, axis=-1)
        x = np.expand_dims(x, axis=0)
        try:
            p = float(model.predict(x, verbose=0)[0][0])
        except TypeError:
            p = float(model.predict(x)[0][0])
        if p > max_p:
            max_p = p
    return float(max_p)


def _choose_best_rotation_k(
    filename: str,
    slice_classifier_rica,
    slice_classifier_ss,
    *,
    ica_flip_lr: bool,
    default_k: int,
) -> int:
    """Pick the rotation that best matches the carotid + SSS classifiers.

    This is a safety net for cases where the stored NIfTI orientation differs
    from what the artery model was trained on.
    """
    candidates = [default_k % 4] + [k for k in (0, 1, 2, 3) if k != (default_k % 4)]
    best_k = candidates[0]
    best_ica = -1.0
    best_ss = -1.0

    for k in candidates:
        try:
            os.environ["P_BRAIN_AI_ROT90_K"] = str(int(k))
            rotated_data, _ = preprocess_data(filename)
            ica_max = _max_slice_relevance(rotated_data, slice_classifier_rica, flip_lr=ica_flip_lr)
            ss_max = _max_slice_relevance(rotated_data, slice_classifier_ss, flip_lr=False)
        except Exception:
            continue

        # Primary objective: maximize ICA confidence; tie-break with SSS.
        if (ica_max > best_ica + 1e-6) or (abs(ica_max - best_ica) <= 1e-6 and ss_max > best_ss + 1e-6):
            best_k, best_ica, best_ss = int(k), float(ica_max), float(ss_max)

    # Restore default env (caller may override later).
    os.environ["P_BRAIN_AI_ROT90_K"] = str(int(default_k % 4))
    return int(best_k)

def extract_and_accumulate_rois(
    rotated_data,
    slice_classifier,
    roi_model,
    choice,
    slice_relevance_threshold=0.5,
    flip_lr: bool = False,
):
    relevant_slices = []
    relevant_rois = []
    original_slices = []
    slice_labels = []
    roi_voxels = {}
    max_relevance_seen = 0.0

    zmax = int(rotated_data.shape[2])
    batch = np.empty((zmax, 256, 256, 1), dtype=np.float32)

    for slice_index in range(zmax):
        selected_slice = rotated_data[:, :, slice_index].astype(np.float32, copy=False)
        if flip_lr:
            selected_slice = np.fliplr(selected_slice)
        if selected_slice.shape != (256, 256):
            selected_slice = cv2.resize(selected_slice, (256, 256), interpolation=cv2.INTER_LINEAR)
        batch[slice_index, :, :, 0] = selected_slice

    try:
        preds = slice_classifier.predict(batch, verbose=0)
    except TypeError:
        preds = slice_classifier.predict(batch)
    scores = np.asarray(preds, dtype=np.float32).reshape(zmax)
    if scores.size:
        max_relevance_seen = float(np.max(scores))

    relevant_idx = [int(z) for z in range(zmax) if float(scores[z]) > float(slice_relevance_threshold)]
    if relevant_idx:
        for z in relevant_idx:
            print(f"Slice {z + 1} is classified as relevant (probability: {float(scores[z]):.2f}).")

        try:
            roi_preds = roi_model.predict(batch[relevant_idx], verbose=0)
        except TypeError:
            roi_preds = roi_model.predict(batch[relevant_idx])
        roi_preds = np.asarray(roi_preds)

        for j, slice_index in enumerate(relevant_idx):
            predicted_mask = np.squeeze(roi_preds[j])
            threshold = 0.5 * float(np.max(predicted_mask)) if np.size(predicted_mask) else 0.0
            binary_mask_model = (predicted_mask > threshold).astype(np.uint8)
            binary_mask = np.fliplr(binary_mask_model) if flip_lr else binary_mask_model

            # IMPORTANT: `binary_mask` is in model/display space (256x256). Downstream
            # curve extraction indexes into `mri_data` (original resolution), so we
            # must resize the mask back to the original slice shape before extracting
            # voxel coordinates.
            try:
                h, w = int(rotated_data.shape[0]), int(rotated_data.shape[1])
                binary_mask_native = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                binary_mask_native = (binary_mask_native > 0).astype(np.uint8)
            except Exception:
                binary_mask_native = binary_mask

            roi_coords = np.argwhere(binary_mask_native)
            roi_voxels[slice_index] = roi_coords
            original_slices.append(rotated_data[:, :, slice_index])
            relevant_slices.append(batch[slice_index, :, :, 0])
            relevant_rois.append(binary_mask)
            slice_labels.append(choice)

    if not relevant_rois:
        try:
            print(
                colored(
                    f"Max slice relevance observed: {float(max_relevance_seen):.2f}",
                    "cyan",
                )
            )
        except Exception:
            pass

    return (
        original_slices,
        relevant_slices,
        relevant_rois,
        slice_labels,
        roi_voxels,
        float(max_relevance_seen),
    )


def extract_and_accumulate_rois_raw_dce(
    mri_data_norm,
    slice_classifier,
    roi_model,
    choice,
    slice_relevance_threshold=0.5,
    flip_lr: bool = False,
    frame_stride: int = 1,
):
    """ICA-specific inference using raw DCE frames.

    The ICA slice-classifier appears to be trained on single-frame anatomy, not
    on the time-averaged projection. We therefore scan frames per slice and use
    the best-scoring frame for ROI segmentation.
    """
    relevant_slices = []
    relevant_rois = []
    original_slices = []
    slice_labels = []
    roi_voxels = {}
    max_relevance_seen = 0.0

    try:
        stride = int(frame_stride)
    except Exception:
        stride = 1
    stride = max(1, stride)

    for slice_index in range(mri_data_norm.shape[2]):
        best_p = -1.0
        best_t = 0

        for t in range(0, mri_data_norm.shape[3], stride):
            selected_slice = mri_data_norm[:, :, slice_index, t]
            u8 = _normalize_to_uint8_slice(selected_slice)
            if int(choice) == 2:
                u8 = 255 - u8
            selected_slice_for_model = np.fliplr(u8) if flip_lr else u8

            resized_slice_model = cv2.resize(
                selected_slice_for_model, (256, 256), interpolation=cv2.INTER_LINEAR
            )
            resized_slice_expanded = np.expand_dims(resized_slice_model.astype(np.float32) / 255.0, axis=-1)
            resized_slice_expanded = np.expand_dims(resized_slice_expanded, axis=0)

            try:
                p = slice_classifier.predict(resized_slice_expanded, verbose=0)[0][0]
            except TypeError:
                p = slice_classifier.predict(resized_slice_expanded)[0][0]

            try:
                if float(p) > float(max_relevance_seen):
                    max_relevance_seen = float(p)
            except Exception:
                pass

            if float(p) > float(best_p):
                best_p = float(p)
                best_t = int(t)

        if best_p > slice_relevance_threshold:
            print(
                f"Slice {slice_index} is classified as relevant (probability: {best_p:.2f}, frame: {best_t})."
            )

            selected_slice = mri_data_norm[:, :, slice_index, best_t]
            u8 = _normalize_to_uint8_slice(selected_slice)
            if int(choice) == 2:
                u8 = 255 - u8
            selected_slice_for_model = np.fliplr(u8) if flip_lr else u8

            resized_slice_display = cv2.resize(u8, (256, 256), interpolation=cv2.INTER_LINEAR)
            resized_slice_model = cv2.resize(
                selected_slice_for_model, (256, 256), interpolation=cv2.INTER_LINEAR
            )
            resized_slice_expanded = np.expand_dims(resized_slice_model.astype(np.float32) / 255.0, axis=-1)
            resized_slice_expanded = np.expand_dims(resized_slice_expanded, axis=0)

            try:
                predicted_mask = roi_model.predict(resized_slice_expanded, verbose=0).squeeze()
            except TypeError:
                predicted_mask = roi_model.predict(resized_slice_expanded).squeeze()
            threshold = 0.5 * predicted_mask.max()
            binary_mask_model = (predicted_mask > threshold).astype(np.uint8)
            binary_mask = np.fliplr(binary_mask_model) if flip_lr else binary_mask_model

            # Resize mask back to native slice resolution for voxel indexing.
            try:
                h, w = int(selected_slice.shape[0]), int(selected_slice.shape[1])
                binary_mask_native = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                binary_mask_native = (binary_mask_native > 0).astype(np.uint8)
            except Exception:
                binary_mask_native = binary_mask

            roi_coords = np.argwhere(binary_mask_native)
            roi_voxels[slice_index] = roi_coords

            original_slices.append(selected_slice)
            relevant_slices.append(resized_slice_display)
            relevant_rois.append(binary_mask)
            slice_labels.append(choice)

    if not relevant_rois:
        try:
            print(colored(f"Max slice relevance observed: {float(max_relevance_seen):.2f}", "cyan"))
        except Exception:
            pass

    return (
        original_slices,
        relevant_slices,
        relevant_rois,
        slice_labels,
        roi_voxels,
        float(max_relevance_seen),
    )


def _describe_choice(choice):
    _, subtype = choice2type(choice)
    return subtype


def extract_rois_with_fallback(
    rotated_data,
    slice_classifier,
    roi_model,
    choice,
    slice_relevance_thresholds=(0.5, 0.3),
    flip_lr: bool = False,
):
    if not slice_relevance_thresholds:
        slice_relevance_thresholds = (0.5,)

    choice_description = _describe_choice(choice)
    last_result = ([], [], [], [], {})

    thresholds_to_try = list(slice_relevance_thresholds)
    tried = set()
    lower_limit = min(thresholds_to_try) if thresholds_to_try else 0.0

    idx = 0
    while idx < len(thresholds_to_try):
        threshold = float(thresholds_to_try[idx])
        if threshold in tried:
            idx += 1
            continue
        tried.add(threshold)

        result = extract_and_accumulate_rois(
            rotated_data,
            slice_classifier,
            roi_model,
            choice,
            slice_relevance_threshold=threshold,
            flip_lr=flip_lr,
        )

        result_core = result[:5]
        max_relevance_seen = float(result[5]) if len(result) > 5 else 0.0

        if result_core[2]:
            if idx > 0:
                print(
                    colored(
                        f"{choice_description} slices detected after lowering the threshold to {threshold:.2f}.",
                        "yellow",
                    )
                )
            return result_core

        last_result = result_core

        if idx < len(thresholds_to_try) - 1:
            # If even the best slice never got close, don't keep retrying.
            if max_relevance_seen < float(lower_limit) - 1e-6:
                print(
                    colored(
                        f"Max slice relevance observed ({max_relevance_seen:.2f}) is below the minimum threshold ({float(lower_limit):.2f}); not retrying.",
                        "red",
                    )
                )
                break

            next_idx = idx + 1
            next_threshold = float(thresholds_to_try[next_idx])

            # Don't invent a new threshold (e.g. 0.37). Instead, skip to the next
            # *configured* threshold that could possibly pass (<= observed max).
            if next_threshold > max_relevance_seen + 1e-9:
                for j in range(idx + 1, len(thresholds_to_try)):
                    cand = float(thresholds_to_try[j])
                    if cand <= (max_relevance_seen - 1e-6):
                        next_idx = j
                        next_threshold = cand
                        break

            print(
                colored(
                    f"No {choice_description} slices detected with threshold {threshold:.2f}. "
                    f"Retrying with {next_threshold:.2f}...",
                    "yellow",
                )
            )
            idx = next_idx
            continue

        idx += 1

    print(colored(f"No {choice_description} slices detected even after lowering the threshold.", "red"))
    return last_result


def extract_rois_with_fallback_raw_dce(
    mri_data_norm,
    slice_classifier,
    roi_model,
    choice,
    slice_relevance_thresholds=(0.5, 0.3),
    flip_lr: bool = False,
    frame_stride: int = 1,
):
    if not slice_relevance_thresholds:
        slice_relevance_thresholds = (0.5,)

    choice_description = _describe_choice(choice)
    last_result = ([], [], [], [], {})

    thresholds_to_try = list(slice_relevance_thresholds)
    tried = set()
    lower_limit = min(thresholds_to_try) if thresholds_to_try else 0.0

    idx = 0
    while idx < len(thresholds_to_try):
        threshold = float(thresholds_to_try[idx])
        if threshold in tried:
            idx += 1
            continue
        tried.add(threshold)

        result = extract_and_accumulate_rois_raw_dce(
            mri_data_norm,
            slice_classifier,
            roi_model,
            choice,
            slice_relevance_threshold=threshold,
            flip_lr=flip_lr,
            frame_stride=frame_stride,
        )

        result_core = result[:5]
        max_relevance_seen = float(result[5]) if len(result) > 5 else 0.0

        if result_core[2]:
            if idx > 0:
                print(colored(f"Detected {choice_description} slices after lowering threshold to {threshold:.2f}.", "green"))
            return result_core

        last_result = result_core

        # Keep trying lower thresholds.
        if idx < len(thresholds_to_try) - 1:
            # If even the best slice never got close, don't keep retrying.
            if max_relevance_seen < float(lower_limit) - 1e-6:
                print(
                    colored(
                        f"Max slice relevance observed ({max_relevance_seen:.2f}) is below the minimum threshold ({float(lower_limit):.2f}); not retrying.",
                        "red",
                    )
                )
                break

            next_idx = idx + 1
            next_threshold = float(thresholds_to_try[next_idx])

            # Don't invent a new threshold (e.g. 0.37). Instead, skip to the next
            # *configured* threshold that could possibly pass (<= observed max).
            if next_threshold > max_relevance_seen + 1e-9:
                for j in range(idx + 1, len(thresholds_to_try)):
                    cand = float(thresholds_to_try[j])
                    if cand <= (max_relevance_seen - 1e-6):
                        next_idx = j
                        next_threshold = cand
                        break

            print(
                colored(
                    f"No {choice_description} slices detected with threshold {threshold:.2f}. Retrying with {next_threshold:.2f}...",
                    "yellow",
                )
            )
            idx = next_idx
            continue
        else:
            print(colored(f"No {choice_description} slices detected even after lowering the threshold.", "red"))

        idx += 1

    return last_result

def plot_relevant_slices_with_rois(original_slices, relevant_slices, relevant_rois, slice_labels, image_directory):
    global turbo_mode
    num_slices = len(relevant_slices)
    if num_slices == 0:
        # Nothing to plot. Avoid matplotlib throwing when cols=0.
        return

    try:
        os.makedirs(os.path.join(image_directory, "AI"), exist_ok=True)
    except Exception:
        pass

    def _label_name(lbl) -> str:
        try:
            i = int(lbl)
        except Exception:
            return str(lbl)
        if i == 1:
            return "SSS"
        if i == 2:
            return "LICA"
        if i == 3:
            return "RICA"
        try:
            _t, subtype = choice2type(i)
            return str(subtype)
        except Exception:
            return str(i)
    rows = 2
    cols = num_slices

    fig, axes = plt.subplots(rows, cols, figsize=(7 * num_slices, 15))

    for i in range(num_slices):
        if num_slices > 1:
            ax1 = axes[0, i]
            ax2 = axes[1, i]
        else:
            ax1 = axes[0]
            ax2 = axes[1]

        # Flip the original slice along the vertical axis
        flipped_original_slice = np.flipud(original_slices[i])
        ax1.imshow(flipped_original_slice, cmap='gray')
        ax1.set_title(f"Original Slice {_label_name(slice_labels[i])}")
        ax1.axis('off')

        # Normalize and flip the relevant slice along the vertical axis
        normalized_resized_slice = cv2.normalize(relevant_slices[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        flipped_normalized_slice = np.flipud(normalized_resized_slice)

        # Flip the ROI along the vertical axis
        flipped_roi = np.flipud(relevant_rois[i])

        # Find contours on the flipped ROI and draw them on the flipped slice
        contour_image = cv2.cvtColor(flipped_normalized_slice, cv2.COLOR_GRAY2RGB)
        contours, _ = cv2.findContours(flipped_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)

        ax2.imshow(contour_image)
        ax2.set_title(f"Predicted ROI {_label_name(slice_labels[i])}")
        ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(image_directory, 'AI', f'AI_input_function_ROIs.png'), dpi=300)
    if not turbo_mode:
        plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
        close_plot_after_delay(3, fig)
        plt.show()




def run_ai_roi_extraction(filename, analysis_dir, image_dir, nifti_dir, time, IsVFA=False, filenames='filenames'):
    print(colored('Starting AI-based ROI extraction...', 'green'))

    def _normalize_artery_choice(raw: str | None) -> str | None:
        s = (raw or "").strip().upper()
        if not s:
            return None
        if s in {"BOTH", "BILATERAL", "LR", "RL", "RICA+LICA", "LICA+RICA", "RICA,LICA", "LICA,RICA"}:
            return "BOTH"
        if s in {"RICA", "RIGHT", "RIGHTICA", "R", "R-ICA", "R_ICA"}:
            return "RICA"
        if s in {"LICA", "LEFT", "LEFTICA", "L", "L-ICA", "L_ICA"}:
            return "LICA"
        return None

    def _build_slice_thresholds() -> tuple[float, ...]:
        """Return descending slice-relevance thresholds.

        Defaults to the legacy fast schedule (0.50, 0.30).

        Optional overrides:
        - P_BRAIN_AI_SLICE_CONF_THRESHOLDS="0.5,0.3,0.1"
        - P_BRAIN_AI_SLICE_CONF_START=0.5
          P_BRAIN_AI_SLICE_CONF_MIN=0.1
          P_BRAIN_AI_SLICE_CONF_STEP=0.05
        """

        raw = os.getenv("P_BRAIN_AI_SLICE_CONF_THRESHOLDS")
        if raw and str(raw).strip():
            vals = []
            for part in str(raw).split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    v = float(part)
                except Exception:
                    continue
                if np.isfinite(v) and 0.0 < v <= 1.0:
                    vals.append(float(v))
            if vals:
                vals = sorted(set(vals), reverse=True)
                return tuple(vals)

        def _f(name: str, default: float) -> float:
            try:
                v = float(os.getenv(name, str(default)))
            except Exception:
                v = float(default)
            if not np.isfinite(v):
                v = float(default)
            return float(v)

        start = _f("P_BRAIN_AI_SLICE_CONF_START", 0.50)
        min_v = _f("P_BRAIN_AI_SLICE_CONF_MIN", 0.30)
        step = _f("P_BRAIN_AI_SLICE_CONF_STEP", 0.20)

        start = max(0.01, min(1.0, start))
        min_v = max(0.01, min(1.0, min_v))
        if start < min_v:
            start, min_v = min_v, start
        if step <= 0 or not np.isfinite(step):
            step = 0.05
        step = max(0.01, min(0.50, step))

        out = []
        v = start
        # Build inclusive sequence down to min_v.
        while v >= (min_v - 1e-9):
            out.append(round(float(v), 4))
            v -= step
        # Ensure min_v is included.
        if out and out[-1] > min_v + 1e-6:
            out.append(round(float(min_v), 4))
        out = sorted(set(out), reverse=True)
        return tuple(out) if out else (0.5, 0.3)

    # Prefer env overrides (used by p-brain-web/p-brain-platform runners) but
    # fall back to legacy settings attribute.
    selected_artery = (
        _normalize_artery_choice(os.getenv("P_BRAIN_AIF_ARTERY"))
        or _normalize_artery_choice(os.getenv("P_BRAIN_INPUT_FUNCTION_ARTERY"))
        or _normalize_artery_choice(getattr(settings, "INPUT_FUNCTION_ARTERY", None))
        or "RICA"
    )
    extract_both_ica = selected_artery == "BOTH"
    preferred_artery = selected_artery
    if extract_both_ica:
        # When extracting both ICA sides, keep downstream selection behaviour
        # aligned with the legacy setting (or default to RICA).
        preferred_artery = (
            _normalize_artery_choice(getattr(settings, "INPUT_FUNCTION_ARTERY", None))
            or "RICA"
        )
        if preferred_artery == "BOTH":
            preferred_artery = "RICA"

    if preferred_artery == "LICA":
        ica_choice = 2
        ica_flip_lr = True
    else:
        # Default to rICA behaviour.
        ica_choice = 3
        ica_flip_lr = False

    used_ica_choice = int(ica_choice)
    used_ica_flip_lr = bool(ica_flip_lr)

    # Resolve model paths relative to the p-brain repo root so that callers can
    # run from any working directory (e.g. p-brain-web / tauri runners).
    from pathlib import Path as _FsPath

    _pbrain_root = _FsPath(__file__).resolve().parents[1]

    def _resolve_model_path(p: str) -> str:
        s = str(p or "").strip()
        if not s:
            return s
        if os.path.isabs(s):
            return s
        cand = _pbrain_root / s
        if cand.exists():
            return str(cand)
        cand = _pbrain_root / "AI" / s
        if cand.exists():
            return str(cand)
        return s

    slice_classifier_rica = load_model(
        _resolve_model_path(AI_MODEL_PATHS.get('slice_classifier_rica', 'AI/slice_classifier_model_rica.keras')),
        compile=False,
    )
    rica_roi_model = load_model(
        _resolve_model_path(AI_MODEL_PATHS.get('rica_roi', 'AI/rica_roi_model.keras')),
        compile=False,
    )
    slice_classifier_ss = load_model(
        _resolve_model_path(AI_MODEL_PATHS.get('slice_classifier_ss', 'AI/ss_slice_classifier.keras')),
        compile=False,
    )
    ss_roi_model = load_model(
        _resolve_model_path(AI_MODEL_PATHS.get('ss_roi', 'AI/ss_roi_model.keras')),
        compile=False,
    )

    # Use default rotation first (can be overridden by P_BRAIN_AI_ROT90_K)
    orig_rot_env = os.environ.get("P_BRAIN_AI_ROT90_K")
    try:
        default_rot_k = int(str(os.getenv("P_BRAIN_AI_ROT90_K", "-1")).strip())
    except Exception:
        default_rot_k = -1

    default_rot_k_mod = int(default_rot_k) % 4
    os.environ["P_BRAIN_AI_ROT90_K"] = str(default_rot_k_mod)

    # Keep a stable default-rotation view for SSS (and as the starting point for ICA).
    rotated_data_default, mri_data_default, mri_data_norm_default = preprocess_data(filename, projection="mean")
    rotated_data, mri_data = rotated_data_default, mri_data_default

    # Track ICA-specific rotated/mri data in case we need to try a different rotation.
    rotated_data_ica, mri_data_ica = rotated_data_default, mri_data_default
    mri_data_norm_ica = mri_data_norm_default
    rotated_data_ss, mri_data_ss = rotated_data_default, mri_data_default

    slice_thresholds = _build_slice_thresholds()

    used_artery_code = "LICA" if int(ica_choice) == 2 else "RICA"

    try:
        ica_frame_stride = int(str(os.getenv("P_BRAIN_AI_ICA_FRAME_STRIDE", "1")).strip())
    except Exception:
        ica_frame_stride = 1
    ica_frame_stride = max(1, int(ica_frame_stride))

    # First attempt: projection-based inference (matches the debug script behavior).
    # ICA (RICA/LICA): the slice-classifier is most reliable on a *single DCE frame*
    # (e.g. early post-bolus anatomy). Mean/max projections often score ~0 in
    # the packaged runtime, which used to trigger the very slow full-frame scan.
    #
    # To keep this lightning fast, we pick one good frame from a small candidate
    # set using a single batched predict(), then run batched ROI on that frame.
    tmax = int(mri_data_norm_ica.shape[3])
    # Use *all* frames for ICA slice scoring.
    # This is slower than an early-window scan but is much more robust for
    # datasets where the best anatomy/contrast frame occurs later in the series.
    frame_candidates = list(range(0, tmax, 1))

    try:
        ica_max_slices = int(str(os.getenv("P_BRAIN_AI_ICA_MAX_SLICES", "3")).strip())
    except Exception:
        ica_max_slices = 3
    ica_max_slices = max(1, min(int(ica_max_slices), int(mri_data_norm_ica.shape[2])))

    def _ica_score(res: tuple) -> tuple[int, float]:
        # Prefer more slices; then prefer larger ROI area.
        try:
            n = len(res[2])
        except Exception:
            n = 0
        try:
            area = float(sum(int(np.sum(r)) for r in (res[2] or [])))
        except Exception:
            area = 0.0
        return (int(n), float(area))

    def _extract_ica(choice: int) -> tuple[list, list, list, list, dict, bool]:
        """Extract ICA ROIs for one side.

        Returns: (orig_slices, slices, rois, labels, voxels, used_flip_lr)
        """

        if int(choice) == 2:
            # LICA: try both flips and keep the better scoring variant.
            res_flip = extract_and_accumulate_rois_best_frame_per_slice(
                mri_data_norm_ica,
                slice_classifier_rica,
                rica_roi_model,
                choice=2,
                frame_candidates=frame_candidates,
                slice_relevance_thresholds=slice_thresholds,
                flip_lr=True,
                max_slices=ica_max_slices,
            )
            res_noflip = extract_and_accumulate_rois_best_frame_per_slice(
                mri_data_norm_ica,
                slice_classifier_rica,
                rica_roi_model,
                choice=2,
                frame_candidates=frame_candidates,
                slice_relevance_thresholds=slice_thresholds,
                flip_lr=False,
                max_slices=ica_max_slices,
            )
            if _ica_score(res_noflip) > _ica_score(res_flip):
                orig_slices, slices, rois, labels, voxels = res_noflip
                used_flip = False
            else:
                orig_slices, slices, rois, labels, voxels = res_flip
                used_flip = True
            try:
                print(colored(f"ICA multi-slice (LICA): flip_lr={used_flip} slices={len(rois)}", "cyan"))
            except Exception:
                pass
            return list(orig_slices), list(slices), list(rois), list(labels), dict(voxels), bool(used_flip)

        # RICA: fixed orientation.
        orig_slices, slices, rois, labels, voxels = extract_and_accumulate_rois_best_frame_per_slice(
            mri_data_norm_ica,
            slice_classifier_rica,
            rica_roi_model,
            choice=3,
            frame_candidates=frame_candidates,
            slice_relevance_thresholds=slice_thresholds,
            flip_lr=False,
            max_slices=ica_max_slices,
        )
        try:
            print(colored(f"ICA multi-slice (RICA): slices={len(rois)}", "cyan"))
        except Exception:
            pass
        return list(orig_slices), list(slices), list(rois), list(labels), dict(voxels), False

    ica_results: dict[str, dict] = {}

    # Extract preferred side first.
    rica_orig_slices, rica_slices, rica_rois, rica_labels, rica_voxels, ica_flip_lr = _extract_ica(int(ica_choice))
    used_ica_flip_lr = bool(ica_flip_lr)
    used_ica_choice = int(ica_choice)
    used_artery_code = "LICA" if int(ica_choice) == 2 else "RICA"
    ica_results[used_artery_code] = {
        "choice": int(ica_choice),
        "flip_lr": bool(ica_flip_lr),
        "orig_slices": rica_orig_slices,
        "slices": rica_slices,
        "rois": rica_rois,
        "labels": rica_labels,
        "voxels": rica_voxels,
    }

    # If requested, also extract the other ICA side in the same run.
    if extract_both_ica:
        other_choice = 2 if int(ica_choice) == 3 else 3
        other_code = "LICA" if int(other_choice) == 2 else "RICA"
        o_orig, o_slices, o_rois, o_labels, o_voxels, o_flip = _extract_ica(int(other_choice))
        ica_results[other_code] = {
            "choice": int(other_choice),
            "flip_lr": bool(o_flip),
            "orig_slices": o_orig,
            "slices": o_slices,
            "rois": o_rois,
            "labels": o_labels,
            "voxels": o_voxels,
        }

    # If preferred side is missing (and we didn't already extract both), retry the other side.
    if (not rica_rois) and (not extract_both_ica):
        alt_choice = 2 if int(ica_choice) == 3 else 3
        alt_desc = "LICA" if int(alt_choice) == 2 else "RICA"
        print(colored(f"ICA fallback side: {alt_desc}", "yellow"))
        a_orig, a_slices, a_rois, a_labels, a_voxels, a_flip = _extract_ica(int(alt_choice))
        if a_rois:
            used_artery_code = alt_desc
            used_ica_choice = int(alt_choice)
            used_ica_flip_lr = bool(a_flip)
            rica_orig_slices, rica_slices, rica_rois, rica_labels, rica_voxels = a_orig, a_slices, a_rois, a_labels, a_voxels
            ica_results[alt_desc] = {
                "choice": int(alt_choice),
                "flip_lr": bool(a_flip),
                "orig_slices": a_orig,
                "slices": a_slices,
                "rois": a_rois,
                "labels": a_labels,
                "voxels": a_voxels,
            }

    # Final ICA safety net: if we still have no carotid ROI, retry with an explicit
    # full-series scan (all frames, stride=1) or with a wider stride when requested.
    # This is still a single batched predict over (n_frames * n_slices).
    def _widen_ica_if_missing(choice: int) -> tuple[list, list, list, list, dict, bool]:
        """Widen ICA frame search for the given side and return the extracted result."""

        full_stride = max(1, int(ica_frame_stride))
        full_candidates = list(range(0, tmax, full_stride))
        if not full_candidates or full_candidates == frame_candidates:
            return ([], [], [], [], {}, False)

        desc = "LICA" if int(choice) == 2 else "RICA"
        print(
            colored(
                f"ICA widening search ({desc}): frames=0..{tmax-1} stride={full_stride} (no ROI in default window)",
                "yellow",
            )
        )

        if int(choice) == 2:
            res_flip = extract_and_accumulate_rois_best_frame_per_slice(
                mri_data_norm_ica,
                slice_classifier_rica,
                rica_roi_model,
                choice=2,
                frame_candidates=full_candidates,
                slice_relevance_thresholds=slice_thresholds,
                flip_lr=True,
                max_slices=ica_max_slices,
            )
            res_noflip = extract_and_accumulate_rois_best_frame_per_slice(
                mri_data_norm_ica,
                slice_classifier_rica,
                rica_roi_model,
                choice=2,
                frame_candidates=full_candidates,
                slice_relevance_thresholds=slice_thresholds,
                flip_lr=False,
                max_slices=ica_max_slices,
            )
            if _ica_score(res_noflip) > _ica_score(res_flip):
                orig_slices, slices, rois, labels, voxels = res_noflip
                used_flip = False
            else:
                orig_slices, slices, rois, labels, voxels = res_flip
                used_flip = True
            try:
                print(colored(f"ICA widened (LICA): flip_lr={used_flip} slices={len(rois)}", "yellow"))
            except Exception:
                pass
            return list(orig_slices), list(slices), list(rois), list(labels), dict(voxels), bool(used_flip)

        orig_slices, slices, rois, labels, voxels = extract_and_accumulate_rois_best_frame_per_slice(
            mri_data_norm_ica,
            slice_classifier_rica,
            rica_roi_model,
            choice=3,
            frame_candidates=full_candidates,
            slice_relevance_thresholds=slice_thresholds,
            flip_lr=False,
            max_slices=ica_max_slices,
        )
        try:
            print(colored(f"ICA widened (RICA): slices={len(rois)}", "yellow"))
        except Exception:
            pass
        return list(orig_slices), list(slices), list(rois), list(labels), dict(voxels), False

    if not rica_rois:
        full_stride = max(1, int(ica_frame_stride))
        full_candidates = list(range(0, tmax, full_stride))
        if full_candidates and full_candidates != frame_candidates:
            print(
                colored(
                    f"ICA widening search: frames=0..{tmax-1} stride={full_stride} (no ROI in default window)",
                    "yellow",
                )
            )

            def _try_choice(choice: int, *, flip_lr: bool) -> tuple[int, float, tuple]:
                # We no longer use a single best frame, but keep the signature for logs.
                res = extract_and_accumulate_rois_best_frame_per_slice(
                    mri_data_norm_ica,
                    slice_classifier_rica,
                    rica_roi_model,
                    choice=choice,
                    frame_candidates=full_candidates,
                    slice_relevance_thresholds=slice_thresholds,
                    flip_lr=flip_lr,
                    max_slices=ica_max_slices,
                )
                # Best score proxy for logs: max slice relevance among selected.
                try:
                    bs = float(max(float(np.max(r)) for r in (res[2] or [])))
                except Exception:
                    bs = 0.0
                return 0, float(bs), res

            # Retry preferred side first.
            if int(ica_choice) == 2:
                # LICA: try both flips.
                bf1, bs1, res1 = _try_choice(2, flip_lr=True)
                bf0, bs0, res0 = _try_choice(2, flip_lr=False)
                bf, bs, res = (bf0, bs0, res0) if float(bs0) > float(bs1) else (bf1, bs1, res1)
                chosen_flip = False if float(bs0) > float(bs1) else True
                print(colored(f"ICA widened (LICA): flip_lr={chosen_flip} best_t={bf} (max p={bs:.2f})", "yellow"))
                if res[2]:
                    used_artery_code = "LICA"
                    used_ica_choice = 2
                    used_ica_flip_lr = bool(chosen_flip)
                    rica_orig_slices, rica_slices, rica_rois, rica_labels, rica_voxels = res
            else:
                bf, bs, res = _try_choice(3, flip_lr=False)
                print(colored(f"ICA widened (RICA): best_t={bf} (max p={bs:.2f})", "yellow"))
                if res[2]:
                    used_artery_code = "RICA"
                    used_ica_choice = 3
                    used_ica_flip_lr = False
                    rica_orig_slices, rica_slices, rica_rois, rica_labels, rica_voxels = res

            # If still missing, try the opposite side as a last resort.
            if not rica_rois:
                if int(ica_choice) == 3:
                    bf, bs, res = _try_choice(2, flip_lr=True)
                    print(colored(f"ICA widened fallback side (LICA): best_t={bf} (max p={bs:.2f})", "yellow"))
                    if res[2]:
                        used_artery_code = "LICA"
                        used_ica_choice = 2
                        used_ica_flip_lr = True
                        rica_orig_slices, rica_slices, rica_rois, rica_labels, rica_voxels = res
                else:
                    bf, bs, res = _try_choice(3, flip_lr=False)
                    print(colored(f"ICA widened fallback side (RICA): best_t={bf} (max p={bs:.2f})", "yellow"))
                    if res[2]:
                        used_artery_code = "RICA"
                        used_ica_choice = 3
                        used_ica_flip_lr = False
                        rica_orig_slices, rica_slices, rica_rois, rica_labels, rica_voxels = res

    # If extracting both sides, optionally widen the search for a missing side.
    if extract_both_ica:
        for code, choice in (("RICA", 3), ("LICA", 2)):
            existing = ica_results.get(code, {})
            if existing.get("rois"):
                continue
            w_orig, w_slices, w_rois, w_labels, w_voxels, w_flip = _widen_ica_if_missing(int(choice))
            if w_rois:
                ica_results[code] = {
                    "choice": int(choice),
                    "flip_lr": bool(w_flip),
                    "orig_slices": w_orig,
                    "slices": w_slices,
                    "rois": w_rois,
                    "labels": w_labels,
                    "voxels": w_voxels,
                }

    # NOTE: legacy brute-force rotation and full-frame scanning are intentionally
    # disabled here for speed. If ICA is missing, downstream will fall back to
    # deterministic ROI selection.

    # Restore default rotation before extracting SSS.
    os.environ["P_BRAIN_AI_ROT90_K"] = str(default_rot_k_mod)

    ss_orig_slices, ss_slices, ss_rois, ss_labels, ss_voxels = extract_rois_with_fallback(
        rotated_data_ss,
        slice_classifier_ss,
        ss_roi_model,
        choice=1,
        slice_relevance_thresholds=slice_thresholds,
    )

    # Build a montage that includes SSS + any extracted carotids.
    all_orig_slices: list = []
    all_slices: list = []
    all_rois: list = []
    all_labels: list = []

    for code in ("RICA", "LICA"):
        r = ica_results.get(code)
        if not r:
            continue
        all_orig_slices += list(r.get("orig_slices") or [])
        all_slices += list(r.get("slices") or [])
        all_rois += list(r.get("rois") or [])
        all_labels += list(r.get("labels") or [])

    all_orig_slices += list(ss_orig_slices)
    all_slices += list(ss_slices)
    all_rois += list(ss_rois)
    all_labels += list(ss_labels)
    plot_relevant_slices_with_rois(all_orig_slices, all_slices, all_rois, all_labels, image_dir)

    # Process artery and vein ROIs separately to avoid slice-index collisions and any
    # label/voxel misalignment.
    for code, r in ica_results.items():
        vox = r.get("voxels") or {}
        if not vox:
            continue
        artery_choice = int(r.get("choice") or (2 if code == "LICA" else 3))

        for slice_index, roi_voxels in vox.items():
            type, subtype = choice2type(int(artery_choice))

            curve_method = (getattr(settings, "VASCULAR_ROI_CURVE_METHOD", "max") or "max").strip().lower()
            if curve_method not in {"max", "mean", "median"}:
                curve_method = "max"
            adaptive_enabled = bool(getattr(settings, "VASCULAR_ROI_ADAPTIVE_MAX", False))

            roi_voxels_arr = np.asarray(roi_voxels)
            if roi_voxels_arr.size == 0:
                continue
            roi_tc = mri_data_ica[
                roi_voxels_arr[:, 0].astype(int),
                roi_voxels_arr[:, 1].astype(int),
                int(slice_index),
                :,
            ]
            if roi_tc.ndim != 2 or roi_tc.shape[0] == 0:
                continue

            if curve_method == "mean":
                sel_tc = roi_tc.mean(axis=0)
            elif curve_method == "median":
                sel_tc = np.median(roi_tc, axis=0)
            elif curve_method == "max" and adaptive_enabled:
                sel_tc = roi_tc.max(axis=0)
            else:
                idx = int(np.argmax(roi_tc.max(axis=1)))
                sel_tc = roi_tc[idx]

            baseline_frames = int(getattr(settings, "ROI_DCE_BASELINE_FRAMES", 5))
            baseline = sel_tc[:baseline_frames].mean() if sel_tc.size else 0.0
            max_intensity_frame = int(np.argmax(sel_tc - baseline)) if sel_tc.size else 0

            plot_time_intensity_curves_AI(
                mri_data_ica,
                roi_voxels,
                slice_index,
                max_intensity_frame,
                time,
                analysis_dir,
                image_dir,
                type=type,
                subtype=subtype,
            )
            try:
                plot_time_intensity_curves_and_CTC_AI(
                    mri_data_ica,
                    max_intensity_frame,
                    roi_voxels,
                    slice_index,
                    max_intensity_frame,
                    time,
                    analysis_dir,
                    image_dir,
                    nifti_dir,
                    type=type,
                    subtype=subtype,
                    IsVFA=IsVFA,
                    filenames=filenames,
                    rotate_ac=True,
                )
            except FileNotFoundError as e:
                # Some runners invoke input-functions before T1/CTC prerequisites exist.
                # Keep ROI extraction results and let downstream stages compute CTC later.
                try:
                    print(colored(f"[AI] Skipping CTC plot (missing prerequisite): {e}", "yellow"))
                except Exception:
                    pass
            except Exception as e:
                try:
                    print(colored(f"[AI] Skipping CTC plot (error): {type(e).__name__}: {e}", "yellow"))
                except Exception:
                    pass

    for slice_index, roi_voxels in ss_voxels.items():
        type, subtype = choice2type(1)

        curve_method = (getattr(settings, "VASCULAR_ROI_CURVE_METHOD", "max") or "max").strip().lower()
        if curve_method not in {"max", "mean", "median"}:
            curve_method = "max"
        adaptive_enabled = bool(getattr(settings, "VASCULAR_ROI_ADAPTIVE_MAX", False))

        roi_voxels_arr = np.asarray(roi_voxels)
        if roi_voxels_arr.size == 0:
            continue
        roi_tc = mri_data_ss[
            roi_voxels_arr[:, 0].astype(int),
            roi_voxels_arr[:, 1].astype(int),
            int(slice_index),
            :,
        ]
        if roi_tc.ndim != 2 or roi_tc.shape[0] == 0:
            continue

        if curve_method == "mean":
            sel_tc = roi_tc.mean(axis=0)
        elif curve_method == "median":
            sel_tc = np.median(roi_tc, axis=0)
        elif curve_method == "max" and adaptive_enabled:
            sel_tc = roi_tc.max(axis=0)
        else:
            idx = int(np.argmax(roi_tc.max(axis=1)))
            sel_tc = roi_tc[idx]

        baseline_frames = int(getattr(settings, "ROI_DCE_BASELINE_FRAMES", 5))
        baseline = sel_tc[:baseline_frames].mean() if sel_tc.size else 0.0
        max_intensity_frame = int(np.argmax(sel_tc - baseline)) if sel_tc.size else 0

        plot_time_intensity_curves_AI(
            mri_data_ss,
            roi_voxels,
            slice_index,
            max_intensity_frame,
            time,
            analysis_dir,
            image_dir,
            type=type,
            subtype=subtype,
        )
        try:
            plot_time_intensity_curves_and_CTC_AI(
                mri_data_ss,
                max_intensity_frame,
                roi_voxels,
                slice_index,
                max_intensity_frame,
                time,
                analysis_dir,
                image_dir,
                nifti_dir,
                type=type,
                subtype=subtype,
                IsVFA=IsVFA,
                filenames=filenames,
                rotate_ac=True,
            )
        except FileNotFoundError as e:
            try:
                print(colored(f"[AI] Skipping CTC plot (missing prerequisite): {e}", "yellow"))
            except Exception:
                pass
        except Exception as e:
            try:
                print(colored(f"[AI] Skipping CTC plot (error): {type(e).__name__}: {e}", "yellow"))
            except Exception:
                pass

    # Restore the caller's env (best-effort).
    if orig_rot_env is None:
        os.environ.pop("P_BRAIN_AI_ROT90_K", None)
    else:
        os.environ["P_BRAIN_AI_ROT90_K"] = orig_rot_env

    print(colored('AI-based ROI extraction completed.', 'green'))

    # Optional: TSCC time shifting (SSS -> arterial reference) for AIF.
    # If we extracted both ICA sides, generate TSCC for both; the Max selection
    # should happen after both sides exist, so the global Max considers all
    # candidates across *all* available arteries.
    if bool(getattr(settings, "INPUT_FUNCTION_USE_SSS", True)):
        if extract_both_ica:
            # Order doesn't matter for the final Max selection, but running
            # preferred first keeps logs stable.
            preferred = str(used_artery_code or getattr(settings, "INPUT_FUNCTION_ARTERY", "RICA"))
            preferred = preferred if preferred in {"RICA", "LICA"} else "RICA"
            other = "LICA" if preferred == "RICA" else "RICA"
            for artery in (preferred, other):
                try:
                    tscc.time_shifting(
                        analysis_dir,
                        nifti_dir,
                        image_dir,
                        artery=str(artery),
                        write_max=False,
                        reference_artery="BOTH",
                    )
                except TypeError:
                    # Legacy signature fallback.
                    tscc.time_shifting(analysis_dir, nifti_dir, image_dir)
            try:
                tscc.write_tscc_max_outputs(
                    analysis_dir,
                    nifti_dir,
                    image_dir,
                    reference_artery="BOTH",
                )
            except Exception as e:
                try:
                    print(colored(f"[AI] TSCC Max refresh failed: {type(e).__name__}: {e}", "yellow"))
                except Exception:
                    pass
                # Legacy fallback: a full TSCC run will still compute a global
                # Max across all candidates present on disk.
                tscc.time_shifting(analysis_dir, nifti_dir, image_dir, artery=str(preferred))
        else:
            tscc.time_shifting(
                analysis_dir,
                nifti_dir,
                image_dir,
                artery=str(used_artery_code or getattr(settings, "INPUT_FUNCTION_ARTERY", "RICA")),
            )
        print(colored('Time shifting completed.', 'green'))

def time_shifting(analysis_directory, nifti_directory, image_directory):
    if auto_logging_enabled():
        log_process_start("Time shifting")
    time_points_s = np.load(os.path.join(analysis_directory,'Fitting', 'time_points_s.npy'))

    subtype = 'Right Interior Carotid' 
    artery_choice = subtype

    arterial_slices = get_available_slices(analysis_directory, 'Artery', artery_choice)
    venous_slices = get_available_slices(analysis_directory, 'Vein', 'Sinus Sagittalis')

    for arterial_slice in arterial_slices:
        for venous_slice in venous_slices:
            vein_curve, artery_curve = load_curves(venous_slice, arterial_slice, artery_choice, analysis_directory)
            aligned_vein_curve, peaks, rescaled, time_shift = align_first_peaks(vein_curve, artery_curve)
            aligned_vein_curve_no_zeros = remove_trailing_zeros(aligned_vein_curve)
            aligned_vein_curve_no_zeros_shifted = shift_curve_to_zero_start(aligned_vein_curve_no_zeros)
            plot_transformed_curves(aligned_vein_curve_no_zeros_shifted, artery_curve, venous_slice, arterial_slice, time_points_s=time_points_s, analysis_directory=analysis_directory, image_directory=image_directory, subtype=subtype, vein_top2_peaks=peaks, scaling=rescaled, time_shift=time_shift)

    print('[!] Finding maximum')
    
    max_file_path, max_value, max_subtype, max_slice_index, max_arterial_slice_index = find_max_npy_file(analysis_directory)
    corresponding_vein_curve = np.load(os.path.join(analysis_directory, 'TSCC Data', max_subtype, f'TSCC_slice_{max_slice_index}_{max_arterial_slice_index}.npy'))
    [os.remove(f) for f in glob.glob(os.path.join(analysis_directory, 'TSCC Data', 'Max', '*.npy'))]
    plot_transformed_curves_max(corresponding_vein_curve, slice_index=max_slice_index, artery_index=max_arterial_slice_index, vein_top2_peaks=[0,0], subtype=max_subtype, time_points_s=time_points_s, analysis_directory=analysis_directory, image_directory=image_directory)
    values = [f'Max artery type: {max_subtype}']
    with open(os.path.join(analysis_directory, 'max_info.json'), 'w') as f:
        json.dump(values, f)
    if auto_logging_enabled():
        log_process_end("Time shifting")

def start_roi_selection(filename, rotate_AC=True, time=1, analysis='dir', image='dir', nifti='dir', filenames='filenames', IsVFA=False):
    run_ai_roi_extraction(filename, analysis, image, nifti, time, IsVFA=IsVFA, filenames=filenames)

def refresh_nifti_directory(nifti_directory):
    return os.listdir(nifti_directory)

def input_function_AI(analysis_directory, nifti_directory, image_directory, filenames, parameters):
    if _roi_method() == "geometry":
        from modules.deterministic_input_functions import input_function_geometry

        return input_function_geometry(
            analysis_directory,
            nifti_directory,
            image_directory,
            filenames,
            parameters,
        )

    (
        t1_3D_filename,
        axial_t1_3D_filename,
        t2_3D_filename,
        axial_t2_3D_filename,
        flair_3D_filename,
        axial_flair_3D_filename,
        axial_t2_2D_filename,
        diffusion_filename,
        dce_filename,
    ) = filenames
    refresh_nifti_directory(nifti_directory)
    
    IsVFA, IsIR, apple_metal, boundary, *_ = parameters

    filename = os.path.join(nifti_directory, dce_filename)
    nifti_img = nib.load(filename)
    num_volumes = nifti_img.shape[-1]
    dt_s = resolve_dce_time_step_s(filename, default=None)
    time_points_s = build_time_points_s(num_volumes, dt_s)
    np.save(os.path.join(analysis_directory,'Fitting', 'time_points_s.npy'), time_points_s)
    start_roi_selection(filename, rotate_AC=True, time=time_points_s, analysis=analysis_directory, image=image_directory, nifti=nifti_directory, IsVFA=IsVFA, filenames=filenames)
