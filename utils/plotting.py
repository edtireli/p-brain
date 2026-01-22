
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from matplotlib.patches import Rectangle
import os
from utils.fonts import *
from utils.loading import *
from matplotlib.path import Path
from scipy.signal import argrelextrema
from matplotlib.widgets import Button
import utils.settings as settings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

turbo_mode = True # doesnt show plots


def load_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        matrix = pickle.load(file)
    return matrix


def _turboflash_signal_from_r1(R1_s, M0, TR_s, TI_s, alpha_rad, nph):
    """TurboFLASH signal model matching MATLAB `sigdif_turbof_r1_90_2`.

    Parameters use SI units: R1 in 1/s, TR/TI in s, alpha in rad.
    nph is 1-based index of ky=0 line (k-space center) within the readout train.
    """
    R1_s = float(R1_s)
    a = np.cos(alpha_rad) * np.exp(-TR_s * R1_s)
    b = 1.0 - np.exp(-TR_s * R1_s)
    # Guard against a==1 causing division by zero.
    denom = (1.0 - a)
    if abs(denom) < 1e-12:
        denom = 1e-12 if denom >= 0 else -1e-12
    a_pow = a ** (int(nph) - 1)
    return float(M0) * np.sin(alpha_rad) * (
        (1.0 - np.exp(-R1_s * TI_s)) * a_pow + b * (1.0 - a_pow) / denom
    )


def _invert_turboflash_r1_from_signal(S, M0, TR_s, TI_s, alpha_rad, nph):
    """Numerically invert TurboFLASH model for R1(t) given signal S(t)."""
    from scipy.optimize import brentq

    S = float(S)
    if not np.isfinite(S) or S <= 0:
        return np.nan

    # For physical signals, R1 is typically within ~[0, 10] 1/s.
    # Use a wide bracket and expand if needed.
    lo = 1e-6
    hi = 10.0

    def f(r1):
        return _turboflash_signal_from_r1(r1, M0, TR_s, TI_s, alpha_rad, nph) - S

    flo = f(lo)
    fhi = f(hi)
    if np.isnan(flo) or np.isnan(fhi):
        return np.nan

    # Expand upper bound until sign change or until it is clearly unreasonable.
    tries = 0
    while flo * fhi > 0 and tries < 12:
        hi *= 2.0
        fhi = f(hi)
        tries += 1

    if flo * fhi > 0:
        # No bracket; return NaN to be handled upstream.
        return np.nan

    return float(brentq(f, lo, hi, maxiter=200))


def compute_CTC(
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
):
    if flip_angle_deg is None:
        # Preserve legacy behaviour (30°) when metadata is missing, but allow an
        # explicit override via settings/environment.
        theta = 30.0 if settings.FLIP_ANGLE_DEG is None else float(settings.FLIP_ANGLE_DEG)
        if not getattr(compute_CTC, "_warned_missing_flip_angle", False):
            warnings.warn(
                "Flip angle missing; using default %.1f°. Set P_BRAIN_FLIP_ANGLE or add FlipAngle to the JSON sidecar." % theta,
                RuntimeWarning,
            )
            compute_CTC._warned_missing_flip_angle = True
    else:
        theta = float(flip_angle_deg)
    theta_rad = np.radians(theta)

    model = (ctc_model or getattr(settings, "CTC_MODEL", "saturation") or "saturation").strip().lower()

    # times in ms and not sec.
    TD_s = TD * 1e-3
    r1_s = r1 * 1e-3
    T1_s = T1 * 1e-3

    if model == "turboflash":
        if tr_s is None:
            raise ValueError("TurboFLASH CTC requires tr_s (seconds).")
        if nph is None:
            nph = getattr(settings, "TURBOFLASH_NPH", 1)
        tr_s = float(tr_s)
        ti_s = TD_s

        s_arr = np.asarray(S, dtype=float)
        r1_t = np.empty_like(s_arr, dtype=float)
        it = np.nditer(s_arr, flags=["multi_index"])
        while not it.finished:
            r1_t[it.multi_index] = _invert_turboflash_r1_from_signal(
                float(it[0]), m0, tr_s, ti_s, theta_rad, int(nph)
            )
            it.iternext()

        r1_0 = 1.0 / float(T1_s)
        C_t = (r1_t - r1_0) / float(r1_s)
    else:
        # Saturation-recovery closed-form inversion used by legacy p-Brain.
        C_t = -(1 / r1_s) * ((1 / TD_s) * np.log(1 - (S / (m0 * np.sin(theta_rad)))) + (1 / T1_s))

    #if prints:
        #print(f"T1: {round(T1, 3)}, M0: {round(m0, 1)}, Max Concentration: {round(np.max(C_t), 2)} mM")

    return C_t


def compute_CTC_VFA(s_tissue, m0_tissue, FA, TR, r1, beta_tissue):
    A = s_tissue / (m0_tissue * np.sin(FA)) 
    c_tissue = -r1 / beta_tissue - np.log((A - 1) / (A * np.cos(FA) - 1)) / (beta_tissue * TR)
    return c_tissue

def interp_nans(arr):
    mask = np.isnan(arr)
    arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
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
    if baseline_point is None:
        baseline_point = 10  # Default baseline point if None
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


def find_main_peaks(data, height_threshold, num_peaks=None):
    if num_peaks is None:
        num_peaks = settings.NUMBER_OF_PEAKS
    peaks, _ = find_peaks(data, height=height_threshold)
    sorted_peaks = sorted(peaks, key=lambda x: data[x], reverse=True)[:num_peaks]
    return sorted(sorted_peaks)

def find_baseline_from_minima_skip(data, first_peak, skip_points):
    min_val = float('inf')
    min_idx = skip_points
    for i in range(skip_points, first_peak):
        if data[i] < min_val:
            min_val = data[i]
            min_idx = i
    return min_idx

def find_shifted_baseline(data, height_threshold=0.5, skip_points=10, num_peaks=None):
    main_peaks = find_main_peaks(data, height_threshold * np.max(data), num_peaks=num_peaks)
    if main_peaks:
        first_peak = main_peaks[0]
        baseline_point = find_baseline_from_minima_skip(data, first_peak, skip_points)
        return baseline_point
    else:
        return None



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
    plt.title(f'Intensity-Time Curve for brightest voxel ({subtype},  Slice {slice_index + 1})', fontproperties=prop, fontsize=18)
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
    axs[1].set_title(f'DCE (Slice {slice_index + 1}, frame {frame_index + 1})', fontproperties=prop, fontsize=14)

    fig.tight_layout()
    plt.savefig(os.path.join(image_directory, 'Intensity Time Curves', type, subtype, f'ITC+DCE_slice_{slice_index+1}.png'), dpi=300)
    if not turbo_mode:
        plt.close()

    plt.figure(figsize=(20, 6))
    plt.scatter(time_points_s, voxel_time_course_max, label='Raw ITC', s=5, color='red')
    plt.plot(time_points_s, smoothed_values, label='Smoothened ITC', linestyle='-', alpha=0.75, color='black')
    plt.xlabel('Time (s)', fontproperties=prop, fontsize=15)
    plt.ylabel('Signal intensity (a. u.)', fontproperties=prop, fontsize=15)
    plt.title(f'Intensity-Time Curve for brightest voxel ({subtype},  Slice {slice_index + 1})', fontproperties=prop, fontsize=18)
    plt.grid(which='minor', alpha=0.25)
    plt.minorticks_on()

    plt.tight_layout()
    plt.savefig(os.path.join(image_directory, 'Intensity Time Curves', type, subtype, f'ITC_slice_{slice_index+1}.png'), dpi=300)
    if not turbo_mode:
        plt.close()
    np.save(os.path.join(analysis_directory, 'ITC Data', type, subtype, f'ITC_slice_{slice_index+1}.npy'), voxel_time_course_max)
    np.save(os.path.join(analysis_directory, 'ROI Data', type, subtype, f'ROI_voxels_slice_{slice_index+1}.npy'), roi_voxels)
    np.save(os.path.join(analysis_directory, 'Frame Data', type, subtype, f'frame_index_slice_{slice_index+1}.npy'), frame_index)



def rescale_peak_to_four(array):
    max_peak = np.max(array)
    if max_peak > 4:
        print('Rescaling unphysical concentration curve to 3.0 mM!')
        scaling_factor = 3 / max_peak
        return array * scaling_factor
    return array

def button_callback(event, C_t, type, subtype, analysis_directory, slice_index, radius=10):
    np.save(os.path.join(analysis_directory, 'CTC Data', type, subtype, f'CTC_slice_{slice_index+1}.npy'), C_t)
    method_name = 'default'
    if event is not None and getattr(event, 'inaxes', None) is not None:
        method_name = event.inaxes.get_title()
    print(f"Data saved using the {method_name} method.")
    # Shift and rescale CTC before saving
    baseline_point = find_shifted_baseline(C_t, skip_points=radius) - 1
    C_t_shifted = custom_shifter(C_t, baseline_point)
    C_t_shifted = rescale_peak_to_four(C_t_shifted)
    
    # Save the shifted CTC data
    np.save(os.path.join(analysis_directory, 'CTC Data', type, subtype, f'CTC_shifted_slice_{slice_index+1}.npy'), C_t_shifted)
    print(f"Shifted and rescaled CTC data saved using the {method_name} method.")
    if not turbo_mode:
        plt.close()

def save_plot_data_AI(C_t, type, subtype, analysis_directory, slice_index, radius=10):
    np.save(os.path.join(analysis_directory, 'CTC Data', type, subtype, f'CTC_slice_{slice_index+1}.npy'), C_t)
    # Shift and rescale CTC before saving
    baseline_point = find_shifted_baseline(C_t, skip_points=radius) - 1
    C_t_shifted = custom_shifter(C_t, baseline_point)
    C_t_shifted = rescale_peak_to_four(C_t_shifted)
    
    # Save the shifted CTC data
    np.save(os.path.join(analysis_directory, 'CTC Data', type, subtype, f'CTC_shifted_slice_{slice_index+1}.npy'), C_t_shifted)
    if not turbo_mode:
        plt.close()

def plot_time_intensity_curves_and_CTC(data, roi_voxels, slice_index, frame_index, time_points_s, analysis_directory, image_directory, nifti_directory, r1=4000, TD=120, type='test', subtype='test', IsVFA=False, filenames='filenames'):
    
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
    
    fs = 15
    cutoff = 4.0
    order = 3

    # Compute Concentration-Time Curves
    x, y, z = max_voxel[0], max_voxel[1], slice_index
    T1_matrix = np.rot90(load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_T1_matrix.pkl')), -1, axes=(0, 1))
    M0_matrix = np.rot90(load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_M0_matrix.pkl')), -1, axes=(0, 1))
    T1 = T1_matrix[x, y, z]
    M0 = M0_matrix[x, y, z]

    if IsVFA:
        radius = 10 # radius between boluses, assuming double bolus input
        json_filename = dce_filename.replace('.nii', '.json')
        with open(os.path.join(nifti_directory, json_filename), 'r') as file:
            json_data = json.load(file)
            FA = json_data['FlipAngle']  # Flip angle in degrees
            TR = json_data['RepetitionTimeExcitation']  # TR in seconds

            FA_rad = np.radians(FA)  # Convert flip angle to radians

        # Compute Concentration-Time Curve for VFA
        beta_tissue = 4 # relaxitivity in 1/ms
        T1 = T1 * 1e-3 # T1 from ms to sec

        C_t_standard = np.array(compute_CTC_VFA(voxel_time_course_max, M0, FA_rad, TR, r1=1/T1, beta_tissue=beta_tissue))
        C_t_adaptive = np.array(compute_CTC_VFA(adaptive_voxel_time_courses, M0, FA_rad, TR, r1=1/T1, beta_tissue=beta_tissue))
    else:  
        radius = 10 # radius between boluses, assuming double bolus input   
        flip_angle_deg = resolve_flip_angle_deg(
            os.path.join(nifti_directory, dce_filename),
            default=None,
        )
        ctc_model = (getattr(settings, "CTC_MODEL", "saturation") or "saturation").strip().lower()
        tr_s = None
        nph = None
        if ctc_model == "turboflash":
            tr_s = read_repetition_time_s_from_sidecar(os.path.join(nifti_directory, dce_filename))
            if tr_s is None:
                raise ValueError(
                    "CTC_MODEL=turboflash requires RepetitionTime in the DCE JSON sidecar; "
                    "re-run conversion with dcm2niix JSON output or set P_BRAIN_CTC_MODEL=saturation."
                )
            nph = getattr(settings, "TURBOFLASH_NPH", 1)
        C_t_standard = np.array(
            compute_CTC(
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
            compute_CTC(
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


def plot_time_intensity_curves_and_CTC_AI(data, max_intensity_frame, roi_voxels, slice_index, frame_index, time_points_s, analysis_directory, image_directory, nifti_directory, r1=4000, TD=120, type='test', subtype='test', IsVFA=False, filenames='filenames'):
    
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
    
    fs = 15
    cutoff = 4.0
    order = 3

    # Compute Concentration-Time Curves
    x, y, z = max_voxel[0], max_voxel[1], slice_index
    T1_matrix = np.rot90(load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_T1_matrix.pkl')), -1, axes=(0, 1))
    M0_matrix = np.rot90(load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_M0_matrix.pkl')), -1, axes=(0, 1))
    T1 = T1_matrix[x, y, z]
    M0 = M0_matrix[x, y, z]

    if IsVFA:
        radius = 10 # radius between boluses, assuming double bolus input
        json_filename = dce_filename.replace('.nii', '.json')
        with open(os.path.join(nifti_directory, json_filename), 'r') as file:
            json_data = json.load(file)
            FA = json_data['FlipAngle']  # Flip angle in degrees
            TR = json_data['RepetitionTimeExcitation']  # TR in seconds

            FA_rad = np.radians(FA)  # Convert flip angle to radians

        # Compute Concentration-Time Curve for VFA
        beta_tissue = 4 # relaxitivity in 1/ms
        T1 = T1 * 1e-3 # T1 from ms to sec

        C_t_standard = np.array(compute_CTC_VFA(voxel_time_course_max, M0, FA_rad, TR, r1=1/T1, beta_tissue=beta_tissue))
        C_t_adaptive = np.array(compute_CTC_VFA(adaptive_voxel_time_courses, M0, FA_rad, TR, r1=1/T1, beta_tissue=beta_tissue))
    else:  
        radius = 10 # radius between boluses, assuming double bolus input   
        flip_angle_deg = resolve_flip_angle_deg(
            os.path.join(nifti_directory, dce_filename),
            default=None,
        )
        ctc_model = (getattr(settings, "CTC_MODEL", "saturation") or "saturation").strip().lower()
        tr_s = None
        nph = None
        if ctc_model == "turboflash":
            tr_s = read_repetition_time_s_from_sidecar(os.path.join(nifti_directory, dce_filename))
            if tr_s is None:
                raise ValueError(
                    "CTC_MODEL=turboflash requires RepetitionTime in the DCE JSON sidecar; "
                    "re-run conversion with dcm2niix JSON output or set P_BRAIN_CTC_MODEL=saturation."
                )
            nph = getattr(settings, "TURBOFLASH_NPH", 1)
        C_t_standard = np.array(
            compute_CTC(
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
            compute_CTC(
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
    axs[1].imshow(data[:, :, slice_index, max_intensity_frame], cmap='plasma', origin='lower')
    if max_voxel:
        x, y = max_voxel  # x, y are row, column indices in the data
        # Plot the rectangle using column, row coordinates
        rect = Rectangle((y-0.5, x-0.5), 1, 1, linewidth=1, edgecolor='g', facecolor='none')
        axs[1].add_patch(rect)
    axs[1].set_title(f'DCE (Slice {slice_index + 1}, Frame {max_intensity_frame + 1})', fontproperties=prop, fontsize=14)

    fig.tight_layout()
    plt.savefig(os.path.join(image_directory, 'Concentration Time Curves', type, subtype, f'CTC+DCE_slice_{slice_index+1}.png'), dpi=300)
    if not turbo_mode:
        plt.show(block=False)  # Display the plot without blocking
        plt.pause(2)           # Pause the script to keep the plot open for 2 seconds
    save_plot_data_AI(C_t_adaptive, type, subtype, analysis_directory, slice_index)
    


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

