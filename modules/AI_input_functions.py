import os
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

import utils.settings as settings

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



def preprocess_data(filename):
    mri_data = nib.load(filename).get_fdata()
    mri_data = np.rot90(mri_data, k=-1, axes=(0, 1))
    time_averaged_data = np.mean(mri_data, axis=-1)
    normalized_data = (time_averaged_data - time_averaged_data.min()) / (time_averaged_data.max() - time_averaged_data.min())
    return normalized_data, mri_data

def extract_and_accumulate_rois(
    rotated_data,
    slice_classifier,
    roi_model,
    choice,
    slice_relevance_threshold=0.5,
):
    relevant_slices = []
    relevant_rois = []
    original_slices = []
    slice_labels = []
    roi_voxels = {}

    for slice_index in range(rotated_data.shape[2]):
        selected_slice = rotated_data[:, :, slice_index]
        resized_slice = cv2.resize(selected_slice, (256, 256), interpolation=cv2.INTER_LINEAR)
        resized_slice_expanded = np.expand_dims(resized_slice, axis=-1)
        resized_slice_expanded = np.expand_dims(resized_slice_expanded, axis=0)

        slice_relevance = slice_classifier.predict(resized_slice_expanded)[0][0]

        if slice_relevance > slice_relevance_threshold:
            print(f"Slice {slice_index} is classified as relevant (probability: {slice_relevance:.2f}).")
            predicted_mask = roi_model.predict(resized_slice_expanded).squeeze()
            threshold = 0.5 * predicted_mask.max()
            binary_mask = (predicted_mask > threshold).astype(np.uint8)
            roi_coords = np.argwhere(binary_mask)
            roi_voxels[slice_index] = roi_coords
            original_slices.append(selected_slice)
            relevant_slices.append(resized_slice)
            relevant_rois.append(binary_mask)
            slice_labels.append(choice)

    return original_slices, relevant_slices, relevant_rois, slice_labels, roi_voxels


def _describe_choice(choice):
    _, subtype = choice2type(choice)
    return subtype


def extract_rois_with_fallback(
    rotated_data,
    slice_classifier,
    roi_model,
    choice,
    slice_relevance_thresholds=(0.5, 0.3),
):
    if not slice_relevance_thresholds:
        slice_relevance_thresholds = (0.5,)

    choice_description = _describe_choice(choice)
    last_result = ([], [], [], [], {})

    for idx, threshold in enumerate(slice_relevance_thresholds):
        result = extract_and_accumulate_rois(
            rotated_data,
            slice_classifier,
            roi_model,
            choice,
            slice_relevance_threshold=threshold,
        )

        if result[2]:
            if idx > 0:
                print(
                    colored(
                        f"{choice_description} slices detected after lowering the threshold to {threshold:.2f}.",
                        "yellow",
                    )
                )
            return result

        last_result = result

        if idx < len(slice_relevance_thresholds) - 1:
            print(
                colored(
                    f"No {choice_description} slices detected with threshold {threshold:.2f}. "
                    f"Retrying with {slice_relevance_thresholds[idx + 1]:.2f}...",
                    "yellow",
                )
            )

    print(
        colored(
            f"No {choice_description} slices detected even after lowering the threshold.",
            "red",
        )
    )

    return last_result

def plot_relevant_slices_with_rois(original_slices, relevant_slices, relevant_rois, slice_labels, image_directory):
    global turbo_mode
    num_slices = len(relevant_slices)
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
        ax1.set_title(f'Original Slice {slice_labels[i]}')
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
        ax2.set_title(f'Predicted ROI {slice_labels[i]}')
        ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(image_directory, 'AI', f'AI_input_function_ROIs.png'), dpi=300)
    if not turbo_mode:
        plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
        close_plot_after_delay(3, fig)
        plt.show()




def run_ai_roi_extraction(filename, analysis_dir, image_dir, nifti_dir, time, IsVFA=False, filenames='filenames'):
    print(colored('Starting AI-based ROI extraction...', 'green'))

    slice_classifier_rica = load_model(
        AI_MODEL_PATHS['slice_classifier_rica'], compile=False
    )
    rica_roi_model = load_model(
        AI_MODEL_PATHS['rica_roi'], compile=False
    )
    slice_classifier_ss = load_model(
        AI_MODEL_PATHS['slice_classifier_ss'], compile=False
    )
    ss_roi_model = load_model(
        AI_MODEL_PATHS['ss_roi'], compile=False
    )

    rotated_data, mri_data = preprocess_data(filename)

    slice_thresholds = (0.5, 0.3)

    rica_orig_slices, rica_slices, rica_rois, rica_labels, rica_voxels = extract_rois_with_fallback(
        rotated_data,
        slice_classifier_rica,
        rica_roi_model,
        choice=3,
        slice_relevance_thresholds=slice_thresholds,
    )

    ss_orig_slices, ss_slices, ss_rois, ss_labels, ss_voxels = extract_rois_with_fallback(
        rotated_data,
        slice_classifier_ss,
        ss_roi_model,
        choice=1,
        slice_relevance_thresholds=slice_thresholds,
    )

    all_orig_slices = rica_orig_slices + ss_orig_slices
    all_slices = rica_slices + ss_slices
    all_rois = rica_rois + ss_rois
    all_labels = rica_labels + ss_labels
    all_voxels = {**rica_voxels, **ss_voxels}

    plot_relevant_slices_with_rois(all_orig_slices, all_slices, all_rois, all_labels, image_dir)

    for idx, (slice_index, roi_voxels) in enumerate(all_voxels.items()):
        type, subtype = choice2type(all_labels[idx])

        max_intensity = -1
        max_intensity_frame = 0

        for (x, y) in roi_voxels:
            voxel_intensity = mri_data[x, y, slice_index, :]
            voxel_max_frame = np.argmax(voxel_intensity)
            voxel_max_intensity = voxel_intensity[voxel_max_frame]

            if voxel_max_intensity > max_intensity:
                max_intensity = voxel_max_intensity
                max_intensity_frame = voxel_max_frame

        plot_time_intensity_curves_AI(mri_data, roi_voxels, slice_index, max_intensity_frame, time, analysis_dir, image_dir, type=type, subtype=subtype)
        plot_time_intensity_curves_and_CTC_AI(mri_data, max_intensity_frame, roi_voxels, slice_index, max_intensity_frame, time, analysis_dir, image_dir, nifti_dir, type=type, subtype=subtype, IsVFA=IsVFA, filenames=filenames)

    print(colored('AI-based ROI extraction completed.', 'green'))

    # Now add the time-shifting functionality:
    time_shifting(analysis_dir, nifti_dir, image_dir)

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
    
    IsVFA, IsIR, apple_metal, boundary, _, _, _ = parameters

    filename = os.path.join(nifti_directory, dce_filename)
    nifti_img = nib.load(filename)
    TR = nifti_img.header.get_zooms()[-1] 
    num_volumes = nifti_img.shape[-1]
    total_scan_duration = TR * num_volumes 
    time_points_s = np.linspace(0, total_scan_duration, num_volumes)
    np.save(os.path.join(analysis_directory,'Fitting', 'time_points_s.npy'), time_points_s)
    start_roi_selection(filename, rotate_AC=True, time=time_points_s, analysis=analysis_directory, image=image_directory, nifti=nifti_directory, IsVFA=IsVFA, filenames=filenames)
