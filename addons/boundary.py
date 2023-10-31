import nibabel as nib
from utils.plotting import *
from utils.loading import *
from scipy.ndimage import zoom
from scipy.stats import zscore
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.ndimage import binary_dilation
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import correlate2d
from skimage.transform import resize
import subprocess
import glob
from scipy.signal import argrelextrema

def find_major_peaks(gradient, radius=10):
    """
    Finds the indices of the two major peaks in the given 1D array based on the gradient.
    
    Parameters:
        gradient (numpy.ndarray): The 1D array containing the gradient data.
        radius (int): The radius around the peaks for filtering out subdominant peaks.
        
    Returns:
        list: The indices of the two major peaks.
    """
    # Identify peaks
    peak_indices = argrelextrema(gradient, np.greater)[0]
    peak_values = gradient[peak_indices]
    
    # Sort peaks by value
    sorted_peak_indices = [x for _, x in sorted(zip(peak_values, peak_indices), reverse=True)]
    
    # Extract the two major peaks based on radius
    major_peaks = []
    for peak in sorted_peak_indices:
        if all(abs(peak - mp) >= radius for mp in major_peaks):
            major_peaks.append(peak)
            if len(major_peaks) >= 2:
                break
    return major_peaks





def find_baseline_point_advanced(y_data, fs=15, cutoff=4.0, order=3, radius=10):
    """
    Finds the baseline point in the given 1D array of y-values based on advanced filtering and gradient analysis.
    
    Parameters:
        y_data (numpy.ndarray): The 1D array containing the data.
        fs (int): Sampling frequency for the low-pass filter.
        cutoff (float): Cutoff frequency for the low-pass filter.
        order (int): Order of the low-pass filter.
        radius (int): The radius around the peaks for filtering out subdominant peaks.
        
    Returns:
        int: The index of the baseline point.
    """
    # Ignore the first point
    y_data = y_data[1:]
    
    # Apply the low-pass filter
    y_filtered = butter_lowpass_filter(y_data, cutoff, fs, order)
    
    # Compute the gradient of the filtered data
    gradient_filtered = np.gradient(y_filtered)
    
    # Find the major peaks in the gradient
    major_peaks_gradient = find_major_peaks(gradient_filtered, radius)
    
    # Find the baseline points as the points right before the major peaks in the gradient
    baseline_points_gradient = [peak - 1 for peak in major_peaks_gradient]
    
    # Select the baseline point with the smaller index
    baseline_point = min(baseline_points_gradient) if baseline_points_gradient else None
    
    # Adjust the index due to ignoring the first point
    if baseline_point is not None:
        baseline_point += 1
    
    return baseline_point


def find_best_match_for_selected_slice_partial(i, t1_slice_norm, dce_data):
    dce_slice = dce_data[:, :, i]
    dce_slice_norm = zscore(dce_slice.ravel()).reshape(dce_slice.shape)
    corr = correlate2d(t1_slice_norm, dce_slice_norm, mode='same')
    return i, np.max(corr)

def find_best_match_for_selected_slice(selected_dce_slice_idx, t1_data, dce_data, start_idx=None, end_idx=None):
    start_idx = start_idx if start_idx is not None else 0
    end_idx = end_idx if end_idx is not None else t1_data.shape[2]

    dce_slice = dce_data[:, :, selected_dce_slice_idx]
    dce_slice_norm = zscore(dce_slice.ravel()).reshape(dce_slice.shape)
    max_corr = 0
    best_t1_idx = -1

    print('------------------------------------------------')
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(find_best_match_for_selected_slice_partial, i, dce_slice_norm, t1_data): i for i in range(start_idx, end_idx)}
        for future in futures:
            i, corr_value = future.result()
            print(f'Slice: {i + 1:2d} - Max. Correlation: {corr_value}')
            if corr_value > max_corr:
                max_corr = corr_value
                best_t1_idx = i
    print('------------------------------------------------')                
    return best_t1_idx


def show_slices(dce_data):
    fig, ax = plt.subplots()
    slice_idx = [0]  # Use a list to make it mutable
    im = ax.imshow(np.rot90(dce_data[:, :, slice_idx[0]]), cmap='viridis')
    
    def on_key(event):
        if event.key == 'up':
            slice_idx[0] += 1
        elif event.key == 'down':
            slice_idx[0] -= 1
        elif event.key == 'escape':
            plt.close()
            return
        slice_idx[0] = slice_idx[0] % dce_data.shape[2]
        im.set_data(np.rot90(dce_data[:, :, slice_idx[0]]))
        plt.draw()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    return slice_idx[0]

def preprocess_t1_slices(t1_data, target_shape):
    processed_t1_data = np.zeros((target_shape[0], target_shape[1], t1_data.shape[2]))
    for i in range(t1_data.shape[2]):
        t1_slice = t1_data[:, :, i]
        processed_t1_data[:, :, i] = crop_and_rescale_slice(t1_slice, target_shape)
    return processed_t1_data

def crop_and_rescale_slice(slice_data, target_shape):
    scaling_factors = np.array(target_shape) / np.array(slice_data.shape)
    return zoom(slice_data, scaling_factors, order=3)


def show_slices_with_buttons(t1_data):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    slice_idx = [0]
    start_slice = [None]
    end_slice = [None]
    exit_flag = [False] 
    im = ax.imshow(np.rot90(t1_data[:, :, slice_idx[0]]), cmap='viridis')
    
    ax_start = plt.axes([0.2, 0.05, 0.1, 0.075])
    ax_end = plt.axes([0.6, 0.05, 0.1, 0.075])
    
    btn_start = Button(ax_start, 'Start slice')
    btn_end = Button(ax_end, 'End slice')
    
    def on_start(event):
        start_slice[0] = slice_idx[0]
    
    def on_end(event):
        end_slice[0] = slice_idx[0]
    
    btn_start.on_clicked(on_start)
    btn_end.on_clicked(on_end)
    
    def on_key(event):
        if event.key == 'up':
            slice_idx[0] += 1
        elif event.key == 'down':
            slice_idx[0] -= 1
        elif event.key == 'escape':
            plt.close()
            exit_flag[0] = True
            return
        slice_idx[0] = slice_idx[0] % t1_data.shape[2]
        im.set_data(np.rot90(t1_data[:, :, slice_idx[0]]))
        plt.draw()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    if exit_flag[0]:  # Check the flag
        return start_slice[0], end_slice[0]


def plot_correlated_slices(t1_real_slice, t1_data_slice, dce_slice, dce_real_slice):
    plt.figure(figsize=(20, 10))
    plt.tight_layout()
    plt.subplot(1, 4, 1)
    plt.imshow(np.rot90(t1_real_slice), cmap='viridis')
    plt.title('3D T1W')
    
    plt.subplot(1, 4, 2)
    plt.imshow(np.rot90(t1_data_slice), cmap='viridis')
    plt.title('3D T2W')
    
    plt.subplot(1, 4, 3)
    plt.imshow(np.rot90(dce_slice), cmap='viridis')
    plt.title('Axial T2')
    
    plt.subplot(1, 4, 4)
    plt.imshow(np.rot90(dce_real_slice), cmap='viridis')
    plt.title('DCE')
    plt.show()


def correlate_slices(filename_t1, filename_t1_real, filename_dce, filename_dce_real, analysis_directory):
    correlated_slice_path = os.path.join(analysis_directory, "correlated_t1_slice.npy")
    selected_slice_path = os.path.join(analysis_directory, "selected_t2_slice.npy")
    # Load the NIfTI images
    nifti_img_t1 = nib.load(filename_t1)
    nifti_img_t1_real = nib.load(filename_t1_real)
    nifti_img_dce = nib.load(filename_dce)
    nifti_img_dce_real = nib.load(filename_dce_real)

    # Extract the data arrays from the NIfTI images
    t1_data = nifti_img_t1.get_fdata()
    t1_data = crop_and_downscale_3d(t1_data, crop_factor = 0.89)
    t1_real_data = nifti_img_t1_real.get_fdata()
    t1_real_data = crop_and_downscale_3d(t1_real_data, crop_factor = 0.89)
    dce_data = nifti_img_dce.get_fdata()
    dce_real_data = nifti_img_dce_real.get_fdata()

    # Preprocess T1 slices to make them compatible with DCE
    target_shape = (dce_data.shape[0], dce_data.shape[1])
    processed_t1_data = preprocess_t1_slices(t1_data, target_shape)
    
    # Check if correlated slice already exists
    if os.path.exists(correlated_slice_path):
        print("Loading existing correlated T1 slice.")
        best_t1_slice_idx = np.load(correlated_slice_path)
        selected_dce_slice_idx = np.load(selected_slice_path)
    else:
        print("Finding new correlated T1 slice.")

        # Show DCE slices to the user for selection
        selected_dce_slice_idx = show_slices(dce_data)
        
        # Show T1 slices to the user for start and end selection
        start_slice, end_slice = show_slices_with_buttons(processed_t1_data)

        # Run the correlation
        best_t1_slice_idx = find_best_match_for_selected_slice(selected_dce_slice_idx, processed_t1_data, dce_data, start_slice, end_slice)

    print(f"Best match: DCE slice {selected_dce_slice_idx} with T1 slice {best_t1_slice_idx}")
    np.save(correlated_slice_path, best_t1_slice_idx)
    np.save(selected_slice_path, selected_dce_slice_idx)
    # Extract the slices for plotting
    dce_real_slice = dce_real_data[:, :, selected_dce_slice_idx, 4]
    t1_real_slice = t1_real_data[:, :, best_t1_slice_idx]
    t1_data_slice = processed_t1_data[:, :, best_t1_slice_idx]
    dce_slice = dce_data[:, :, selected_dce_slice_idx]

    # Plot the slices
    plot_correlated_slices(t1_real_slice, t1_data_slice, dce_slice, dce_real_slice)

    return selected_dce_slice_idx, best_t1_slice_idx


def find_matching_file(directory, pattern):
    regex = re.compile(pattern, re.IGNORECASE)
    for filename in os.listdir(directory):
        if regex.match(filename):
            return os.path.join(directory, filename)
    return None

def first_existing_dce_file(directory, filenames, preferred_filename='WIPDelRec-hperf120long.nii'):
    preferred_file_path = os.path.join(directory, preferred_filename)
    if os.path.exists(preferred_file_path):
        return preferred_file_path

    for fname in filenames:
        if fname == preferred_filename:
            continue
        file_path = os.path.join(directory, fname)
        if os.path.exists(file_path):
            return file_path
    return None


def nii2anat_extension(filename):
    import os

    # Extract the base name and directory from the filename
    base_name = os.path.basename(filename)
    directory = os.path.dirname(filename)

    # Remove the .nii extension and append .anat
    base_name_without_extension = os.path.splitext(base_name)[0]
    new_base_name = base_name_without_extension + ".anat"

    # Create the new directory path
    new_directory = os.path.join(directory, new_base_name)

    return new_directory


def ROI_selector(filename_t1, filename_t1_real, filename_dce, filename_dce_real, analysis_directory, nifti_directory):
    selected_dce_slice_idx, best_t1_slice_idx = correlate_slices(
        filename_t1, filename_t1_real, filename_dce, filename_dce_real, analysis_directory
    )
    
    # 1. Run fsl_anat if segmentation files don't already exist
    segment_files = ['T1_fast_pve_0.nii.gz', 'T1_fast_pve_1.nii.gz', 'T1_fast_pve_2.nii.gz']
    if not all(os.path.exists(os.path.join(nifti_directory, nii2anat_extension(filename_t1_real), f)) for f in segment_files):
        subprocess.run(["fsl_anat", "-i", filename_t1_real])
    
    # 2. Load the segmentation files
    seg_files = [nib.load(os.path.join(nifti_directory, nii2anat_extension(filename_t1_real), f)).get_fdata() for f in segment_files]
    
    # Show the segmented slices along with the t1_real slice
    t1_real_data = nib.load(filename_t1_real).get_fdata()
    t1_real_slice = t1_real_data[:, :, best_t1_slice_idx]

    dce_real_data = nib.load(filename_dce_real).get_fdata()
    dce_real_slice = dce_real_data[:, :, selected_dce_slice_idx, 4]
    
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 4, 1)
    plt.imshow(np.rot90(t1_real_slice), cmap='gray')
    plt.title("t1_real")
    
    for i, seg in enumerate(seg_files):
        plt.subplot(1, 4, i+2)
        plt.imshow(np.rot90(seg[:, :, best_t1_slice_idx]), cmap='gray')
        plt.title(f"t1_real_seg_{i}")

    plt.show()
    
    # 3. Identify the boundary between grey and white matter
    grey_matter = seg_files[1][:, :, best_t1_slice_idx]
    white_matter = seg_files[2][:, :, best_t1_slice_idx]
    
    # Dilate the grey matter by 3 pixels
    dilated_grey = binary_dilation(grey_matter, structure=np.ones((3,3)), iterations=3)
    
    # Find the boundary by logical AND with white matter
    boundary = np.logical_and(dilated_grey, white_matter)
    
    #FOV scale factor computation
    hdr_t1 = nib.load(filename_t1_real).header
    hdr_dce = nib.load(filename_dce_real).header

    # Obtain pixel dimensions (usually in hdr['pixdim'][1:4])
    pixdim_t1 = hdr_t1['pixdim'][1:4]
    pixdim_dce = hdr_dce['pixdim'][1:4]

    # Calculate FOV for each dataset
    fov_t1 = np.array(pixdim_t1[:2]) * np.array(grey_matter.shape)
    fov_dce = np.array(pixdim_dce[:2]) * np.array(dce_real_slice.shape)

    # Calculate scaling factors for FOV
    fov_scaling_factors = fov_dce[:2] / fov_t1[:2]
    # Crop the boundary based on the FOV scale factor of 0.89
    crop_factor = fov_scaling_factors[0] #assuming symmetric scaling between x and y (we take x value)
    crop_margin = int((1 - crop_factor) * 512)  # half the margin for cropping
    cropped_boundary = boundary[crop_margin:-crop_margin, crop_margin:-crop_margin]
    
    # Downscale the cropped ROI to fit DCE dimensions (256x256)
    downscaling_factor = 256 / cropped_boundary.shape[0]
    boundary_downscaled = zoom(cropped_boundary, (downscaling_factor, downscaling_factor), order=0)
    
    # 4. Load the DCE data
    dce_real_data = nib.load(filename_dce_real).get_fdata()
    dce_real_slice = dce_real_data[:, :, selected_dce_slice_idx, 4]
    
    # Plot the downscaled boundary and the DCE slice
    plt.figure()
    plt.imshow(np.rot90(dce_real_slice), cmap='gray')
    plt.imshow(np.rot90(boundary_downscaled), cmap='jet', alpha=0.5)
    plt.show()
    
    # Identify the voxels in the downscaled boundary
    downscaled_boundary_voxels = np.array(np.where(boundary_downscaled)).T
    
    
    return downscaled_boundary_voxels, selected_dce_slice_idx, best_t1_slice_idx





def plot_time_intensity_curves_and_CTC_boundary(data, data2, data3, roi_voxels, roi_voxels_upscaled, slice_index, slice_index2, r1=4000, TD=120, type='test', subtype='test', skipshift=False, time_points_s=1, analysis_directory='dir', image_directory='dir'):
    def on_esc(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)
    N = data.shape[0]
    all_C_t = []
    all_unnormalized_C_t = []
    T1_matrix = load_from_pickle(os.path.join(analysis_directory, 'voxel_T1_matrix.pkl'))
    M0_matrix = load_from_pickle(os.path.join(analysis_directory, 'voxel_M0_matrix.pkl'))
    for (x, y) in roi_voxels:
        voxel_time_course = data[x, y, slice_index, :]
        T1 = T1_matrix[x, y, slice_index]
        M0 = M0_matrix[x, y, slice_index]
        C_t_0 = compute_CTC(voxel_time_course, T1, TD, r1=r1, m0=M0, slice=slice_index, prints=False)
        baseline_point = find_baseline_point_advanced(C_t_0)
        C_t = custom_shifter(C_t_0, baseline_point)
        all_C_t.append(C_t)  
        all_unnormalized_C_t.append(C_t_0 )
    # Averaging all the C_t curves
    avg_C_t_0 = np.mean(all_C_t, axis=0)
    avg_unnormalized_C_t_0 = np.mean(all_unnormalized_C_t, axis=0)
    baseline_point = find_baseline_point_advanced(avg_C_t_0)-1
    print('[!] Baseline point chosen: ', baseline_point)
    avg_C_t = custom_shifter(avg_C_t_0, baseline_point)
    fs = 15
    cutoff = 4.0
    order = 3

    fig, axs = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 1]})
    axs[0].plot(time_points_s, avg_C_t, color='k', label='Normalised')
    axs[0].set_xlabel('Time (sec)', fontsize=12)
    axs[0].set_ylabel('Concentration (mM)', fontsize=12)
    axs[0].set_title(f'Normalised {type} Concentration (Slice {slice_index + 1})', fontsize=14)
    axs[0].grid(which='minor', alpha=0.25)
    axs[0].minorticks_on()

    #np.save(os.path.join(analysis_directory, 'CTC Data', 'Tissue', type, f'CTC_slice_{slice_index+1}_unshifted.npy'), avg_C_t)
    # Equilibrium Magnetisation Map
    axs[1].plot(time_points_s, avg_unnormalized_C_t_0, color='k', label='Un-Normalised')
    #axs[1].scatter(time_points_s, avg_unnormalized_C_t_0, color='r', label='Normalised')
    axs[1].axvline(time_points_s[baseline_point], color='red', linestyle = '--')
    axs[1].set_xlabel('Time (sec)', fontsize=12)
    axs[1].set_ylabel('Concentration (mM)', fontsize=12)
    axs[1].set_title(f'Un-Normalised {type} Concentration (Slice {slice_index + 1})', fontsize=14)

    axs[1].grid(which='minor', alpha=0.25)
    axs[1].minorticks_on()
    #plt.savefig(os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', type, f'CTC+ROI_slice_{slice_index+1}_normalisation.png'), dpi=200)
    
    plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
    plt.show()
    plt.close()

    shift_manual = input('[!] Manually shift baseline point? (y/n): ')
    if shift_manual.lower() == 'y':
        indices = list(range(0, len(avg_unnormalized_C_t_0)))
        ax = plt.gca()  # Get current axes
        ax.plot(indices,avg_unnormalized_C_t_0, color='k', label='Un-Normalised')
        ax.scatter(indices,avg_unnormalized_C_t_0, color='r', label='Un-Normalised', s=5)
        ax.axvline(baseline_point, color='red', linestyle = '--')
        ax.set_xlabel('Time (sec)', fontsize=12)
        ax.set_ylabel('Concentration (mM)', fontsize=12)
        ax.set_title(f'Un-Normalised {type} Concentration (Slice {slice_index + 1})', fontsize=14)

        ax.grid(which='minor', alpha=0.25)
        ax.minorticks_on()
        plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
        plt.show()
        plt.close()
        baseline_point = int(input('[!] Pick baseline point: '))
        avg_C_t = custom_shifter(avg_C_t_0, baseline_point)

    # Create an empty 2D array of zeros
    rect_array = np.zeros((data.shape[0], data.shape[1]))

    # Fill in the original rectangle positions with ones
    for x, y in roi_voxels_upscaled:
        rect_array[x, y] = 1

    # Rotate the rectangle position array by 270 degrees
    rotated_rect_array = np.rot90(rect_array, 3)

    # Find the rotated coordinates
    rotated_roi_voxels = np.array(np.where(rotated_rect_array)).T

    fig, axs = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={'width_ratios': [2, 1, 1]})
    smoothed_values = butter_lowpass_filter(avg_C_t, cutoff, fs, order)
    
    # Concentration-Time Curve
    axs[0].plot(time_points_s, smoothed_values, color='black')
    axs[0].scatter(time_points_s, avg_C_t, color='r', s=5)
    axs[0].set_xlabel('Time (sec)', fontsize=12)
    axs[0].set_ylabel('Concentration (mM)', fontsize=12)
    axs[0].set_title(f'Average Concentration-Time Curve (Slice {slice_index + 1})', fontsize=14)
    axs[0].grid(which='minor', alpha=0.25)
    axs[0].minorticks_on()

    #np.save(os.path.join(analysis_directory, 'CTC Data', 'Tissue', type, f'CTC_slice_{slice_index+1}_unshifted.npy'), avg_C_t)
    # Equilibrium Magnetisation Map
    axs[1].imshow(np.rot90(data2[:, :, slice_index],3), cmap='magma', origin='lower')
    for x, y in rotated_roi_voxels:
        rect = Rectangle((y, x), 1, 1, linewidth=1, edgecolor='g', facecolor='none', alpha=0.5)
        axs[1].add_patch(rect)
    axs[1].set_title(f'T2-weighted Image (Slice {slice_index + 1})', fontsize=10)

    axs[2].imshow(np.rot90(data3[:, :, slice_index2],3), cmap='magma', origin='lower')
    for x, y in rotated_roi_voxels:
        rect = Rectangle((y, x), 1, 1, linewidth=1, edgecolor='g', facecolor='none', alpha=0.5)
        axs[2].add_patch(rect)
    axs[2].set_title(f'T1-weighted Image (Slice {slice_index2 + 1})', fontsize=10)
    plt.savefig(os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', type, f'CTC+ROI_slice_{slice_index+1}.png'), dpi=200)
    

    plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
    plt.show()
    plt.close()

    np.save(os.path.join(analysis_directory, 'CTC Data', 'Tissue', type, f'CTC_slice_{slice_index+1}.npy'), avg_C_t)
    return avg_C_t


def crop_and_downscale_3d(data_3d, crop_factor):
    """
    Crop and downscale a 3D array based on a given crop factor.

    Parameters:
        data_3d (numpy.ndarray): The 3D array to be cropped and downscaled.
        crop_factor (float): The factor by which to crop the data (0 to 1).

    Returns:
        numpy.ndarray: The cropped and downscaled 3D array.
    """
    # Calculate the margin for cropping
    crop_margin = int((1 - crop_factor) * data_3d.shape[0] // 2)
    
    # Crop the 3D data
    cropped_data_3d = data_3d[crop_margin:-crop_margin, crop_margin:-crop_margin, :]
    
    # Calculate the downscaling factor to fit 256x256xZ dimensions
    downscaling_factor = 256 / cropped_data_3d.shape[0]
    
    # Downscale the cropped 3D data
    downscaled_data_3d = zoom(cropped_data_3d, (downscaling_factor, downscaling_factor, 1), order=3)
    
    return downscaled_data_3d





def run(analysis_directory, nifti_directory, image_directory):
    possible_dce_filenames = ['WIPAxT2TSEmatrix.nii', 'WIPDelRec-AxT2TSEmatrix.nii']
    filename_dce = first_existing_dce_file(nifti_directory, possible_dce_filenames)
    nifti_img_dce = nib.load(filename_dce)
    possible_dce_filenames_real = ['WIPhperf120long.nii', 'WIPDelRec-hperf120long.nii']
    filename_dce_real = first_existing_dce_file(nifti_directory, possible_dce_filenames_real)
    TR = nib.load(filename_dce_real).header.get_zooms()[-1]
    num_volumes = nib.load(filename_dce_real).shape[-1]
    total_scan_duration = TR * num_volumes 
    time_points_s = np.linspace(0, total_scan_duration, num_volumes)
    np.save(os.path.join(analysis_directory, 'time_points_s.npy'), time_points_s)

    # Finding and loading T1 image
    pattern = r'ax([-_ ])?vwipcs_3D_Brain_VIEW_T2_32chSHC\.nii'
    filename_t1 = find_matching_file(nifti_directory, pattern)
    print(f"[!] Found: {filename_t1}" if filename_t1 else "[x] No matching file found.")

    # Finding and loading DCE image
    possible_dce_filenames = ['WIPAxT2TSEmatrix.nii', 'WIPDelRec-AxT2TSEmatrix.nii']
    filename_dce = first_existing_dce_file(nifti_directory, possible_dce_filenames)
    print(f"[!] Found: {filename_dce}" if filename_dce else "[x] No matching file found.")

    # Finding and loading "real" T1W image
    pattern_real = r'ax([-_ ])?vwipcs_t1w_3d_tfe_32channel\.nii'
    filename_t1_real = find_matching_file(nifti_directory, pattern_real)
    print(f"[!] Found: {filename_t1_real}" if filename_t1_real else "[x] No matching file found.")

    # Finding and loading "real" DCE image
    possible_dce_filenames_real = ['WIPhperf120long.nii', 'WIPDelRec-hperf120long.nii']
    filename_dce_real = first_existing_dce_file(nifti_directory, possible_dce_filenames_real)
    print(f"[!] Found: {filename_dce_real}" if filename_dce_real else "[x] No matching file found.")

    downscaled_boundary_voxels, best_dce_slice_idx, correlated_t1_slice = ROI_selector(filename_t1, filename_t1_real, filename_dce, filename_dce_real, analysis_directory, nifti_directory)


    data_4d = np.array(nib.load(filename_dce_real).get_fdata())
    data_3d = np.array(nib.load(filename_dce).get_fdata())
    data_3d = zoom(data_3d, (0.5, 0.5, 1), order=3) 
    data_3d_t1 = np.array(nib.load(filename_t1_real).get_fdata())
    data_3d_t1 = crop_and_downscale_3d(data_3d_t1, crop_factor = 0.89)
    curve = plot_time_intensity_curves_and_CTC_boundary(data_4d, data_3d, data_3d_t1, downscaled_boundary_voxels, downscaled_boundary_voxels, best_dce_slice_idx, correlated_t1_slice, type='Boundary', skipshift=False, time_points_s=time_points_s, analysis_directory=analysis_directory, image_directory=image_directory)

    correction_prompt = input('[!] Correct tissue concentration curve of anomalous behavior? (y/n): ')
    if correction_prompt == 'y':
        curve_path = os.path.join(analysis_directory, 'CTC Data', 'Tissue', 'Boundary', f'CTC_slice_{best_dce_slice_idx+1}.npy')
        while True:
            editor = ConcentrationCurveEditor(curve)
            corrected_curve_path = os.path.join(analysis_directory, 'CTC Data', 'Tissue', 'Boundary', f'CTC_slice_{best_dce_slice_idx+1}.npy')
            plot_corrected_tissue_curve(editor.data, data_3d, downscaled_boundary_voxels, best_dce_slice_idx, type='Boundary', time_points_s=time_points_s, image_directory = image_directory, rot90=True)
            break