
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from matplotlib.patches import Rectangle
import os
from utils.fonts import *
from utils.loading import *



def normalize_ctc(C_t, baseline_point):
    baseline_length = baseline_point
    mean_baseline = np.mean(C_t[:baseline_length])
    C_t = (C_t - mean_baseline) / mean_baseline
    C_t[:baseline_length] = 0


def compute_CTC(S, T1, TD=120, r1=4000, m0=1, slice=-1, prints=True):
    theta = 90 #flip angle
    theta_rad = np.radians(theta)

    #times in ms and not sec. 
    TD = TD*1e-3
    r1 = r1*1e-3
    T1 = T1*1e-3

    C_t = -(1 / r1) * ((1 / TD) * np.log(1 - (S / (m0 * np.sin(theta_rad)))) + (1 / T1))

    if prints:
        print('')
        print(f"Slice {slice+1}: TD: {TD}, T1: {round(T1, 3)}, M0: {round(m0, 1)}")
        print(f"Slice {slice+1}: Max Concentration: {round(np.max(C_t), 2)} mM")

    return C_t


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


def find_main_peaks(data, height_threshold):
    peaks, _ = find_peaks(data, height=height_threshold)
    sorted_peaks = sorted(peaks, key=lambda x: data[x], reverse=True)[:2]
    return sorted(sorted_peaks)

def find_baseline_from_minima_skip(data, first_peak, skip_points):
    min_val = float('inf')
    min_idx = skip_points
    for i in range(skip_points, first_peak):
        if data[i] < min_val:
            min_val = data[i]
            min_idx = i
    return min_idx

def find_shifted_baseline(data, height_threshold=0.5, skip_points=10):
    main_peaks = find_main_peaks(data, height_threshold * np.max(data))
    if main_peaks:
        first_peak = main_peaks[0]
        baseline_point = find_baseline_from_minima_skip(data, first_peak, skip_points)
        return baseline_point
    else:
        return None



def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def plot_time_intensity_curves(data, roi_voxels, slice_index, frame_index, time_points_s, analysis_directory, image_directory, type='test', subtype='test'):
    def on_esc(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)

    N = data.shape[0]
    max_intensity = -1
    max_voxel = None
    voxel_time_course_max = None
    
    for (x, y) in roi_voxels:
        voxel_time_course = data[x, y, slice_index, :]
        max_voxel_intensity = voxel_time_course.max()
        if max_voxel_intensity > max_intensity:
            max_intensity = max_voxel_intensity
            max_voxel = (x, y)
            voxel_time_course_max = voxel_time_course

    fs = 15
    cutoff = 4.0
    order = 3
    smoothed_values = butter_lowpass_filter(voxel_time_course_max, cutoff, fs, order)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].scatter(time_points_s, voxel_time_course_max, label='Raw ITC', s=5, color='red')
    axs[0].plot(time_points_s, smoothed_values, label='Smoothened ITC', linestyle='-', alpha=0.75, color='black')
    axs[0].set_xlabel('Time (sec)', fontproperties=prop, fontsize=12)
    axs[0].set_ylabel('Signal Intensity (a. u.)', fontproperties=prop, fontsize=12)
    axs[0].set_title(f'Intensity-Time Curve for brightest voxel (Slice {slice_index + 1})', fontproperties=prop, fontsize=14)
    axs[0].grid(which='minor', alpha=0.25)
    axs[0].minorticks_on()
    axs[0].legend()
    
    axs[1].imshow(data[:, :, slice_index, frame_index], cmap='viridis', origin='lower')
    if max_voxel:
        x, y = max_voxel
        rect = Rectangle((y, x), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
        axs[1].add_patch(rect)
    axs[1].set_title(f'Slice with brightest voxel (slice {slice_index + 1}, frame {frame_index + 1})', fontproperties=prop, fontsize=14)

    plt.savefig(os.path.join(image_directory, 'Intensity Time Curves', type, subtype, f'ITC+M0_slice_{slice_index+1}.png'), dpi=200)

    plt.gcf().canvas.mpl_connect('key_press_event', on_esc)    
    plt.show()
    plt.close()

    plt.figure(figsize=(20, 6))
    plt.scatter(time_points_s, voxel_time_course_max, label='Raw ITC', s=5, color='red')
    plt.plot(time_points_s, smoothed_values, label='Smoothened ITC', linestyle='-', alpha=0.75, color='black')
    plt.xlabel('Time (sec)', fontproperties=prop, fontsize=15)
    plt.ylabel('Signal intensity (a. u.)', fontproperties=prop, fontsize=15)
    plt.title(f'Intensity-Time Curve for brightest voxel ({subtype},  Slice {slice_index + 1})', fontproperties=prop, fontsize=18)
    plt.grid(which='minor', alpha=0.25)
    plt.minorticks_on()
    
    plt.savefig(os.path.join(image_directory, 'Intensity Time Curves', type, subtype, f'ITC_slice_{slice_index+1}.png'), dpi=200)
    
    plt.gcf().canvas.mpl_connect('key_press_event', on_esc)    
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


def plot_time_intensity_curves_and_CTC(data, roi_voxels, slice_index, frame_index, time_points_s, analysis_directory, image_directory, r1=4000, TD=120, type='test', subtype='test'):
    def on_esc(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)
    N = data.shape[0]
    max_intensity = -1
    max_voxel = None
    voxel_time_course_max = None
    
    for (x, y) in roi_voxels:
        voxel_time_course = data[x, y, slice_index, :]
        max_voxel_intensity = voxel_time_course.max()
        if max_voxel_intensity > max_intensity:
            max_intensity = max_voxel_intensity
            max_voxel = (x, y)
            voxel_time_course_max = voxel_time_course
    
    fs = 15
    cutoff = 4.0
    order = 3
    
    # Compute Concentration-Time Curve (CTC)
    S0 = voxel_time_course_max[0] 
    x, y, z = max_voxel[0], max_voxel[1], slice_index
    T1_matrix = np.rot90(load_from_pickle(os.path.join(analysis_directory, 'voxel_T1_matrix.pkl')), -1, axes=(0, 1))
    M0_matrix = np.rot90(load_from_pickle(os.path.join(analysis_directory, 'voxel_M0_matrix.pkl')), -1, axes=(0, 1))
    T1 = T1_matrix[x, y, z]
    M0 = M0_matrix[x, y, z]
    C_t = np.array(compute_CTC(voxel_time_course_max, T1, TD, r1=4000, m0=M0, slice=slice_index))
    C_t = interp_nans(C_t)
    #print(f"Slice {slice_index+1}: TD: {TD}, TR: {TR}, T1: {round(T1,1)}, M0: {round(M0,1)}")
    print("")
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 1]})
    smoothed_values = butter_lowpass_filter(C_t, cutoff, fs, order)
    
    # Concentration-Time Curve
    axs[0].plot(time_points_s, smoothed_values, color='black')
    axs[0].scatter(time_points_s, C_t, color='r', s=5)
    axs[0].set_xlabel('Time (sec)', fontproperties=prop, fontsize=12)
    axs[0].set_ylabel('Concentration (mM)', fontproperties=prop, fontsize=12)
    axs[0].set_title(f'Concentration-Time Curve for brightest voxel (Slice {slice_index + 1})',fontproperties=prop, fontsize=14)
    axs[0].grid(which='minor', alpha=0.25)
    axs[0].minorticks_on()

    # Equilibrium Magnetisation Map
    axs[1].imshow(M0_matrix[:, :, slice_index], cmap='plasma', origin='lower')
    if max_voxel:
        x, y = max_voxel
        rect = Rectangle((y, x), 1, 1, linewidth=1, edgecolor='g', facecolor='none')
        axs[1].add_patch(rect)
    axs[1].set_title(f'Equilibrium magnetisation map (Slice {slice_index + 1})', fontproperties=prop, fontsize=14)
    
    plt.savefig(os.path.join(image_directory, 'Concentration Time Curves', type, subtype, f'CTC+M0_slice_{slice_index+1}.png'), dpi=200)
    
    plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
    plt.show()
    #plt.close()

    baseline_point = find_shifted_baseline(C_t)-1
    C_t_shifted = custom_shifter(C_t, baseline_point)
    C_t_shifted = rescale_peak_to_four(C_t_shifted) 
    
    smoothed_values_shifted = butter_lowpass_filter(C_t_shifted, cutoff, fs, order)
    plt.figure(figsize=(20, 6))
    plt.plot(time_points_s, smoothed_values_shifted, color='black')
    plt.scatter(time_points_s, C_t_shifted, color='r', s=5)
    plt.xlabel('Time (sec)', fontproperties=prop, fontsize=15)
    plt.ylabel('Concentration (mM)', fontproperties=prop, fontsize=15)
    plt.title(f'Concentration-Time Curve for brightest voxel (Slice {slice_index + 1})', fontproperties=prop, fontsize=18)
    plt.grid(which='minor', alpha=0.25)
    plt.minorticks_on()

    plt.savefig(os.path.join(image_directory, 'Concentration Time Curves', type, subtype, f'CTC_slice_{slice_index+1}.png'), dpi=200) 
    plt.gcf().canvas.mpl_connect('key_press_event', on_esc)      
    plt.show()
    # Save the tissue concentrations
    np.save(os.path.join(analysis_directory, 'CTC Data', type, subtype, f'CTC_slice_{slice_index+1}.npy'), C_t_shifted)

    