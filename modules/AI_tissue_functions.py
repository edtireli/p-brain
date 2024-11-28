turbo_mode = False  # Set to True to suppress all plots

import nibabel as nib
import matplotlib.pyplot as plt
from skimage import color
import torch
import numpy as np
from torchvision import utils
import subprocess 
import os
from utils.fonts import *
from utils.loading import *
from utils.plotting import *
from scipy.ndimage import zoom
from skimage.transform import resize
import json
import os
from scipy.ndimage import binary_dilation
from matplotlib.gridspec import GridSpec
from skimage.exposure import rescale_intensity

def plot_predictions_with_masks(image, wm_mask, cortical_gm_mask, subcortical_gm_mask, image_directory):
    n_slices = image.shape[2]
    n_cols = 5
    n_rows = (n_slices + n_cols - 1) // n_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))

    for i in range(n_slices):
        row = i // n_cols
        col = i % n_cols

        image_slice = np.rot90(image[:, :, i])
        wm_slice = np.rot90(wm_mask[:, :, i])
        cortical_gm_slice = np.rot90(cortical_gm_mask[:, :, i])
        subcortical_gm_slice = np.rot90(subcortical_gm_mask[:, :, i])

        color_overlay = np.zeros((*image_slice.shape, 3))
        color_overlay[:, :, 2] = wm_slice  # Blue channel for white matter

        # Assign bright red to cortical gray matter
        color_overlay[:, :, 0][cortical_gm_slice == 1] = 1.0  # Bright red

        # Assign dark red to subcortical gray matter
        color_overlay[:, :, 0][subcortical_gm_slice == 1] = 0.5  # Darker red

        ax = axes[row, col]
        ax.imshow(image_slice, cmap='gray')
        ax.imshow(color_overlay, alpha=0.5)
        ax.set_title(f'Slice {i+1}')

        ax.grid(False)
        ax.axis("off")

    # Remove empty subplots
    for j in range(n_slices, n_rows * n_cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    os.makedirs(os.path.join(image_directory, 'AI', 'Segmentation'), exist_ok=True)
    plt.savefig(os.path.join(image_directory, 'AI', 'Segmentation', 'T2_WM_GM_masks.png'))
    if not turbo_mode:
        plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
        close_plot_after_delay(3, fig)
        plt.show()
    else:
        plt.close(fig)

def segmentation(fastsurfer_path, seg_mgz_path, t1_path, output_dir, sid, apple_metal=True):
    # Check if FastSurfer is installed
    if not os.path.exists(fastsurfer_path):
        raise Exception("FastSurfer not found, ensure correct installation and configuration of path.")

    # Run FastSurfer if the segmentation file doesn't exist
    if not os.path.exists(seg_mgz_path):
        print("Segmentation file not found, running FastSurfer...")
        if apple_metal:
            command = (
                f"export PYTORCH_ENABLE_MPS_FALLBACK=1 && "
                f"{fastsurfer_path} --seg_only --device mps "
                f"--t1 {t1_path} "
                f"--sid {sid} "
                f"--sd {output_dir}"
            )
        else:
            command = (
                f"{fastsurfer_path} --seg_only "
                f"--t1 {t1_path} "
                f"--sid {sid} "
                f"--sd {output_dir}"
            )
        subprocess.run(command, shell=True)
    else:
        print("Segmentation file already exists, skipping FastSurfer segmentation.")

    aseg_mgz_path = seg_mgz_path

    # Convert aseg.mgz to aseg.nii if needed
    aseg_nii_path = aseg_mgz_path.replace('.mgz', '.nii.gz')
    if not os.path.exists(aseg_nii_path):
        print(f"Converting {aseg_mgz_path} to {aseg_nii_path}...")
        subprocess.run(['mri_convert', aseg_mgz_path, aseg_nii_path])
    else:
        print(f"{aseg_nii_path} already exists, skipping conversion.")

    # Paths for the masks
    cortical_gm_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'cortical_gm.nii.gz')
    subcortical_gm_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'subcortical_gm.nii.gz')
    wm_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'wm.nii.gz')

    # Create masks using predefined flags
    # White Matter Mask
    if not os.path.exists(wm_mask_path):
        wm_command = f"mri_binarize --i {aseg_nii_path} --all-wm --o {wm_mask_path}"
        subprocess.run(wm_command, shell=True)
    else:
        print("WM mask already exists, skipping mri_binarize for WM.")

    # Subcortical Gray Matter Mask
    if not os.path.exists(subcortical_gm_mask_path):
        subcortical_gm_command = f"mri_binarize --i {aseg_nii_path} --subcort-gm --o {subcortical_gm_mask_path}"
        subprocess.run(subcortical_gm_command, shell=True)
    else:
        print("Subcortical GM mask already exists, skipping mri_binarize for subcortical GM.")

    # Cortical Gray Matter Mask
    # Create overall gray matter mask
    gm_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'gm.nii.gz')
    if not os.path.exists(gm_mask_path):
        gm_command = f"mri_binarize --i {aseg_nii_path} --gm --o {gm_mask_path}"
        subprocess.run(gm_command, shell=True)
    else:
        print("GM mask already exists, skipping mri_binarize for GM.")

    # Create cortical gray matter mask by subtracting subcortical gray matter from total gray matter
    if not os.path.exists(cortical_gm_mask_path):
        cortical_gm_command = f"fslmaths {gm_mask_path} -sub {subcortical_gm_mask_path} -thr 0.5 -bin {cortical_gm_mask_path}"
        subprocess.run(cortical_gm_command, shell=True)
    else:
        print("Cortical GM mask already exists, skipping creation.")

    # Optionally, remove the gm.nii.gz file if not needed
    if os.path.exists(gm_mask_path):
        os.remove(gm_mask_path)

def coregistration(seg_mgz_path, dce_path, t2_path):
    import subprocess
    import nibabel as nib
    import numpy as np
    import os

    # Step 1: Convert segmentation file from .mgz to .nii.gz format
    aseg_nii_path = seg_mgz_path.replace('.mgz', '.nii.gz')
    if not os.path.exists(aseg_nii_path):
        print(f"Converting {seg_mgz_path} to {aseg_nii_path}...")
        result = subprocess.run(['mri_convert', seg_mgz_path, aseg_nii_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_convert failed with error:\n{result.stderr}")
            raise RuntimeError("mri_convert command failed.")
    else:
        print(f"{aseg_nii_path} already exists, skipping conversion.")

    # Ensure the converted file exists
    if not os.path.exists(aseg_nii_path):
        raise FileNotFoundError(f"Converted segmentation file not found: {aseg_nii_path}")

    # Step 2: Align the segmentation image to the DCE space using -applyxfm -usesqform
    aseg_in_dce_path = aseg_nii_path.replace('.nii.gz', '_in_DCE.nii.gz')
    if not os.path.exists(aseg_in_dce_path):
        flirt_cmd_dce = [
            'flirt', '-in', aseg_nii_path, '-ref', dce_path,
            '-applyxfm', '-usesqform', '-interp', 'nearestneighbour',
            '-out', aseg_in_dce_path
        ]
        print(f"Running FLIRT command for DCE: {' '.join(flirt_cmd_dce)}")
        result = subprocess.run(flirt_cmd_dce, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FLIRT failed for DCE with error:\n{result.stderr}")
            raise RuntimeError("FLIRT command for DCE failed.")
    else:
        print(f"Aligned segmentation to DCE already exists at {aseg_in_dce_path}.")

    # Ensure the output file was created
    if not os.path.exists(aseg_in_dce_path):
        raise FileNotFoundError(f"Expected output not found: {aseg_in_dce_path}")

    # Step 3: Align the segmentation image to the T2 space using -applyxfm -usesqform
    aseg_in_t2_path = aseg_nii_path.replace('.nii.gz', '_in_T2.nii.gz')
    if not os.path.exists(aseg_in_t2_path):
        flirt_cmd_t2 = [
            'flirt', '-in', aseg_nii_path, '-ref', t2_path,
            '-applyxfm', '-usesqform', '-interp', 'nearestneighbour',
            '-out', aseg_in_t2_path
        ]
        print(f"Running FLIRT command for T2: {' '.join(flirt_cmd_t2)}")
        result = subprocess.run(flirt_cmd_t2, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FLIRT failed for T2 with error:\n{result.stderr}")
            raise RuntimeError("FLIRT command for T2 failed.")
    else:
        print(f"Aligned segmentation to T2 already exists at {aseg_in_t2_path}.")

    # Ensure the output file was created
    if not os.path.exists(aseg_in_t2_path):
        raise FileNotFoundError(f"Expected output not found: {aseg_in_t2_path}")

    # Step 4: Create masks from the aligned segmentation images
    # For DCE space
    wm_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_wm.nii.gz')
    if not os.path.exists(wm_mask_dce_path):
        wm_command = f"mri_binarize --i {aseg_in_dce_path} --all-wm --o {wm_mask_dce_path}"
        print(f"Running command: {wm_command}")
        result = subprocess.run(wm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for WM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for WM in DCE space failed.")
    else:
        print("WM mask in DCE space already exists, skipping mri_binarize for WM.")

    subcortical_gm_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_subcortical_gm.nii.gz')
    if not os.path.exists(subcortical_gm_mask_dce_path):
        subcortical_gm_command = f"mri_binarize --i {aseg_in_dce_path} --subcort-gm --o {subcortical_gm_mask_dce_path}"
        print(f"Running command: {subcortical_gm_command}")
        result = subprocess.run(subcortical_gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for subcortical GM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for subcortical GM in DCE space failed.")
    else:
        print("Subcortical GM mask in DCE space already exists, skipping mri_binarize.")

    gm_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_gm.nii.gz')
    if not os.path.exists(gm_mask_dce_path):
        gm_command = f"mri_binarize --i {aseg_in_dce_path} --gm --o {gm_mask_dce_path}"
        print(f"Running command: {gm_command}")
        result = subprocess.run(gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for GM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for GM in DCE space failed.")
    else:
        print("GM mask in DCE space already exists, skipping mri_binarize.")

    cortical_gm_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_cortical_gm.nii.gz')
    if not os.path.exists(cortical_gm_mask_dce_path):
        cortical_gm_command = f"fslmaths {gm_mask_dce_path} -sub {subcortical_gm_mask_dce_path} -thr 0.5 -bin {cortical_gm_mask_dce_path}"
        print(f"Running command: {cortical_gm_command}")
        result = subprocess.run(cortical_gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"fslmaths failed for cortical GM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("fslmaths command for cortical GM in DCE space failed.")
    else:
        print("Cortical GM mask in DCE space already exists, skipping creation.")

    # Similarly for T2 space
    wm_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_wm.nii.gz')
    if not os.path.exists(wm_mask_t2_path):
        wm_command = f"mri_binarize --i {aseg_in_t2_path} --all-wm --o {wm_mask_t2_path}"
        print(f"Running command: {wm_command}")
        result = subprocess.run(wm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for WM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for WM in T2 space failed.")
    else:
        print("WM mask in T2 space already exists, skipping mri_binarize for WM.")

    subcortical_gm_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_subcortical_gm.nii.gz')
    if not os.path.exists(subcortical_gm_mask_t2_path):
        subcortical_gm_command = f"mri_binarize --i {aseg_in_t2_path} --subcort-gm --o {subcortical_gm_mask_t2_path}"
        print(f"Running command: {subcortical_gm_command}")
        result = subprocess.run(subcortical_gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for subcortical GM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for subcortical GM in T2 space failed.")
    else:
        print("Subcortical GM mask in T2 space already exists, skipping mri_binarize.")

    gm_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_gm.nii.gz')
    if not os.path.exists(gm_mask_t2_path):
        gm_command = f"mri_binarize --i {aseg_in_t2_path} --gm --o {gm_mask_t2_path}"
        print(f"Running command: {gm_command}")
        result = subprocess.run(gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for GM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for GM in T2 space failed.")
    else:
        print("GM mask in T2 space already exists, skipping mri_binarize.")

    cortical_gm_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_cortical_gm.nii.gz')
    if not os.path.exists(cortical_gm_mask_t2_path):
        cortical_gm_command = f"fslmaths {gm_mask_t2_path} -sub {subcortical_gm_mask_t2_path} -thr 0.5 -bin {cortical_gm_mask_t2_path}"
        print(f"Running command: {cortical_gm_command}")
        result = subprocess.run(cortical_gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"fslmaths failed for cortical GM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("fslmaths command for cortical GM in T2 space failed.")
    else:
        print("Cortical GM mask in T2 space already exists, skipping creation.")

    # Ensure all mask files exist before loading
    required_files = [
        wm_mask_dce_path, cortical_gm_mask_dce_path, subcortical_gm_mask_dce_path,
        wm_mask_t2_path, cortical_gm_mask_t2_path, subcortical_gm_mask_t2_path
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required mask file not found: {file_path}")

    # Load the masks
    wm_mask_dce = nib.load(wm_mask_dce_path).get_fdata().astype(bool)
    cortical_gm_mask_dce = nib.load(cortical_gm_mask_dce_path).get_fdata().astype(bool)
    subcortical_gm_mask_dce = nib.load(subcortical_gm_mask_dce_path).get_fdata().astype(bool)

    wm_mask_t2 = nib.load(wm_mask_t2_path).get_fdata().astype(bool)
    cortical_gm_mask_t2 = nib.load(cortical_gm_mask_t2_path).get_fdata().astype(bool)
    subcortical_gm_mask_t2 = nib.load(subcortical_gm_mask_t2_path).get_fdata().astype(bool)

    return wm_mask_t2, wm_mask_dce, cortical_gm_mask_t2, cortical_gm_mask_dce, subcortical_gm_mask_t2, subcortical_gm_mask_dce

def plot_dce_grid(dce_image, wm_mask_downsampled, cortical_gm_mask_downsampled, subcortical_gm_mask_downsampled, image_directory):
    n_slices = dce_image.shape[2]
    n_cols = 5
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6))

    for i in range(n_slices):
        row = i // n_cols
        col = i % n_cols

        dce_slice = np.rot90(dce_image[:, :, i])
        wm_slice_dce = np.rot90(wm_mask_downsampled[:, :, i])
        cortical_gm_slice_dce = np.rot90(cortical_gm_mask_downsampled[:, :, i])
        subcortical_gm_slice_dce = np.rot90(subcortical_gm_mask_downsampled[:, :, i])

        color_overlay_dce = np.zeros((*dce_slice.shape, 3))
        color_overlay_dce[:, :, 2] = wm_slice_dce  # Blue channel for white matter

        # Assign bright red to cortical gray matter
        color_overlay_dce[:, :, 0][cortical_gm_slice_dce == 1] = 1.0  # Bright red

        # Assign dark red to subcortical gray matter
        color_overlay_dce[:, :, 0][subcortical_gm_slice_dce == 1] = 0.5  # Darker red

        ax_dce = axes[row, col]
        ax_dce.imshow(dce_slice, cmap='gray', alpha=1)
        ax_dce.imshow(color_overlay_dce, alpha=0.5)
        ax_dce.set_title(f'Slice {i+1}')
        ax_dce.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(image_directory, 'AI', 'Tissue functions', f'AI_Tissue_slice_{i+1}.png'))
    if not turbo_mode:
        plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
        close_plot_after_delay(3, fig)
        plt.show()
    else:
        plt.close(fig)

def patlak_analysis_plotting(c_tissue, c_input, time):
    frame_no = len(time)
    delta_t = np.diff(time)
    y_patlak = np.zeros(frame_no)
    x_patlak = np.zeros(frame_no)
    
    for i in range(frame_no - 1):
        if c_tissue.size == 0 or c_input.size == 0:
            return np.nan, np.nan, np.nan, np.array([]), np.array([]), np.array([], dtype=bool)   
        if c_input[i] != 0:
            y_patlak[i] = c_tissue[i] / c_input[i]
            x_patlak[i] = np.sum(c_input[:i+1] * delta_t[:i+1]) / c_input[i]         
    
    # Removing zero elements
    non_zero_indices = np.nonzero(y_patlak)
    y_patlak = y_patlak[non_zero_indices]
    x_patlak = x_patlak[non_zero_indices]
    
    if x_patlak.size == 0:
        return np.nan, np.nan, np.nan, np.array([]), np.array([]), np.array([], dtype=bool)
    
    calc_max = np.max(x_patlak)
    calc_min = calc_max / 3
    idx = np.where((x_patlak >= calc_min) & (x_patlak <= calc_max))
    x, y = x_patlak[idx], y_patlak[idx]
    
    if x.size < 2:
        return np.nan, np.nan, np.nan, x_patlak, y_patlak, np.array([], dtype=bool)
    
    Ki = np.dot(x - np.mean(x), y - np.mean(y)) / np.dot(x - np.mean(x), x - np.mean(x))
    lambda_ = np.mean(y) - Ki * np.mean(x)
    
    SD_Ki = np.sqrt(np.dot(y - (lambda_ + Ki * x), y - (lambda_ + Ki * x)) / (np.dot(x - np.mean(x), x - np.mean(x)) * (len(x) - 2)))
    
    # Apply the necessary multipliers
    Ki = Ki * 6000  # Convert Ki to ml/100g/min
    SD_Ki = SD_Ki * 6000  # Convert SD_Ki to ml/100g/min
    lambda_ = lambda_ * 100  # Convert lambda to ml/100g

    included_indices = np.isin(x_patlak, x)  # Identify which points were included in the analysis
    
    return Ki, lambda_, SD_Ki, x_patlak, y_patlak, included_indices

def find_baseline_point_advanced(y_data, fs=15, cutoff=4.0, order=3, radius=10):
    """
    Finds the baseline point in the given 1D array of y-values based on advanced filtering and gradient analysis.
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

def plot_ctcs_and_patlak(t2_img_slice, dce_img_slice, wm_mask_t2, cortical_gm_mask_t2, subcortical_gm_mask_t2, 
                         wm_mask_dce, cortical_gm_mask_dce, subcortical_gm_mask_dce, 
                         avg_wm_ctc, avg_cortical_gm_ctc, avg_subcortical_gm_ctc,
                         x_patlak_wm, y_patlak_wm, Ki_wm, lambda_wm, 
                         x_patlak_cortical_gm, y_patlak_cortical_gm, Ki_cortical_gm, lambda_cortical_gm,
                         x_patlak_subcortical_gm, y_patlak_subcortical_gm, Ki_subcortical_gm, lambda_subcortical_gm, 
                         slice_idx, save_path=None, boundary_mask=None, boundary_ctc=None, 
                         x_patlak_boundary=None, y_patlak_boundary=None, Ki_boundary=None, lambda_boundary=None, 
                         included_wm=None, included_cortical_gm=None, included_subcortical_gm=None, included_boundary=None):
    
    # Adjust image intensity to prevent overexposure
    # Clip intensities to the 100th percentile to avoid extreme bright pixels
    t2_vmin, t2_vmax = np.percentile(t2_img_slice, (1, 99))
    dce_vmin, dce_vmax = np.percentile(dce_img_slice, (1, 99))
    
    # Normalize images within the specified intensity range
    t2_img_slice_norm = np.clip(t2_img_slice, t2_vmin, t2_vmax)
    t2_img_slice_norm = (t2_img_slice_norm - t2_vmin) / (t2_vmax - t2_vmin)
    dce_img_slice_norm = np.clip(dce_img_slice, dce_vmin, dce_vmax)
    dce_img_slice_norm = (dce_img_slice_norm - dce_vmin) / (dce_vmax - dce_vmin)

    # Resize the masks to match the T2 and DCE image slice sizes
    wm_mask_resized_t2 = resize(wm_mask_t2, t2_img_slice.shape, order=0, preserve_range=True, anti_aliasing=False)
    cortical_gm_mask_resized_t2 = resize(cortical_gm_mask_t2, t2_img_slice.shape, order=0, preserve_range=True, anti_aliasing=False)
    subcortical_gm_mask_resized_t2 = resize(subcortical_gm_mask_t2, t2_img_slice.shape, order=0, preserve_range=True, anti_aliasing=False)
    wm_mask_resized_dce = resize(wm_mask_dce, dce_img_slice.shape, order=0, preserve_range=True, anti_aliasing=False)
    cortical_gm_mask_resized_dce = resize(cortical_gm_mask_dce, dce_img_slice.shape, order=0, preserve_range=True, anti_aliasing=False)
    subcortical_gm_mask_resized_dce = resize(subcortical_gm_mask_dce, dce_img_slice.shape, order=0, preserve_range=True, anti_aliasing=False)

    # Create color overlays
    color_overlay_t2 = np.zeros((*t2_img_slice.shape, 4))  # RGBA channels
    color_overlay_dce = np.zeros((*dce_img_slice.shape, 4))  # RGBA channels

    # Set blue channel for white matter
    color_overlay_t2[..., 2] = wm_mask_resized_t2  # Blue channel
    color_overlay_dce[..., 2] = wm_mask_resized_dce  # Blue channel

    # Assign bright red to cortical gray matter
    color_overlay_t2[..., 0][cortical_gm_mask_resized_t2 == 1] = 1.0  # Bright red
    color_overlay_dce[..., 0][cortical_gm_mask_resized_dce == 1] = 1.0  # Bright red

    # Assign dark red to subcortical gray matter
    color_overlay_t2[..., 0][subcortical_gm_mask_resized_t2 == 1] = 0.5  # Darker red
    color_overlay_dce[..., 0][subcortical_gm_mask_resized_dce == 1] = 0.5  # Darker red

    # Set alpha channel
    color_overlay_t2[..., 3] = (cortical_gm_mask_resized_t2 + subcortical_gm_mask_resized_t2 + wm_mask_resized_t2) * 0.5
    color_overlay_dce[..., 3] = (cortical_gm_mask_resized_dce + subcortical_gm_mask_resized_dce + wm_mask_resized_dce) * 0.5

    if boundary_mask is not None:
        boundary_mask_resized_t2 = resize(boundary_mask, t2_img_slice.shape, order=0, preserve_range=True, anti_aliasing=False)
        boundary_mask_resized_dce = resize(boundary_mask, dce_img_slice.shape, order=0, preserve_range=True, anti_aliasing=False)
        # Enhance green channel and alpha for boundary in T2
        color_overlay_t2[..., 1] += boundary_mask_resized_t2  # Enhance green channel
        color_overlay_t2[..., 3] += boundary_mask_resized_t2 * 0.5  # Adjust alpha
        # Enhance green channel and alpha for boundary in DCE
        color_overlay_dce[..., 1] += boundary_mask_resized_dce  # Enhance green channel
        color_overlay_dce[..., 3] += boundary_mask_resized_dce * 0.5  # Adjust alpha

    fig = plt.figure(figsize=(8, 18))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], width_ratios=[1, 1])

    # Top row with two side-by-side plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # T2 image with masks
    ax1.imshow(np.rot90(t2_img_slice_norm), cmap='gray', vmin=0, vmax=2)
    ax1.imshow(np.rot90(color_overlay_t2), interpolation='none')
    ax1.set_title(f'T2 Slice {slice_idx} with Masks')
    ax1.axis('off')

    # DCE image with masks
    ax2.imshow(np.rot90(dce_img_slice_norm), cmap='gray', vmin=0, vmax=2)
    ax2.imshow(np.rot90(color_overlay_dce), interpolation='none')
    ax2.set_title(f'DCE Slice {slice_idx} with Masks')
    ax2.axis('off')

    # Remove the axes and bounding box around the top plots
    for ax in [ax1, ax2]:
        ax.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_frame_on(False)

    # CTC plot in the middle row
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(avg_wm_ctc, label='White Matter CTC', color='blue')
    ax3.plot(avg_cortical_gm_ctc, label='Cortical Gray Matter CTC', color='red')
    ax3.plot(avg_subcortical_gm_ctc, label='Subcortical Gray Matter CTC', color='darkred')
    if boundary_ctc is not None and boundary_ctc.size > 0:
        ax3.plot(boundary_ctc, label='Boundary CTC', color='green')
    ax3.set_title('Concentration-Time Curves')
    ax3.legend(loc='upper right')
    ax3.grid(True)

    # Patlak plot on the bottom row
    ax4 = fig.add_subplot(gs[2, :])

    # Plot White Matter Patlak data if available
    if not np.isnan(Ki_wm):
        ax4.scatter(x_patlak_wm[included_wm], y_patlak_wm[included_wm], label='White Matter', color='blue', marker='o')
        ax4.scatter(x_patlak_wm[~included_wm], y_patlak_wm[~included_wm], facecolors='none', edgecolors='blue')
        ax4.plot(x_patlak_wm, lambda_wm / 100 + (Ki_wm / 6000) * x_patlak_wm, color='blue', linestyle='--')

    # Plot Cortical Gray Matter Patlak data if available
    if not np.isnan(Ki_cortical_gm):
        ax4.scatter(x_patlak_cortical_gm[included_cortical_gm], y_patlak_cortical_gm[included_cortical_gm], label='Cortical GM', color='red', marker='o')
        ax4.scatter(x_patlak_cortical_gm[~included_cortical_gm], y_patlak_cortical_gm[~included_cortical_gm], facecolors='none', edgecolors='red')
        ax4.plot(x_patlak_cortical_gm, lambda_cortical_gm / 100 + (Ki_cortical_gm / 6000) * x_patlak_cortical_gm, color='red', linestyle='--')

    # Plot Subcortical Gray Matter Patlak data if available
    if not np.isnan(Ki_subcortical_gm):
        ax4.scatter(x_patlak_subcortical_gm[included_subcortical_gm], y_patlak_subcortical_gm[included_subcortical_gm], label='Subcortical GM', color='darkred', marker='o')
        ax4.scatter(x_patlak_subcortical_gm[~included_subcortical_gm], y_patlak_subcortical_gm[~included_subcortical_gm], facecolors='none', edgecolors='darkred')
        ax4.plot(x_patlak_subcortical_gm, lambda_subcortical_gm / 100 + (Ki_subcortical_gm / 6000) * x_patlak_subcortical_gm, color='darkred', linestyle='--')

    # Plot Boundary Patlak data if available
    if boundary_ctc is not None and not np.isnan(Ki_boundary):
        ax4.scatter(x_patlak_boundary[included_boundary], y_patlak_boundary[included_boundary], label='Boundary', color='green', marker='o')
        ax4.scatter(x_patlak_boundary[~included_boundary], y_patlak_boundary[~included_boundary], facecolors='none', edgecolors='green')
        ax4.plot(x_patlak_boundary, lambda_boundary / 100 + (Ki_boundary / 6000) * x_patlak_boundary, color='green', linestyle='--')

    ax4.set_title('Patlak Plot')
    ax4.legend(loc='lower right')
    ax4.grid(True)

    # Compile fit text for display
    fit_text = ""
    if not np.isnan(Ki_wm):
        fit_text += f"White Matter: Ki = {Ki_wm:.5f} ml/100g/min, λ = {lambda_wm:.5f} ml/100g\n"
    if not np.isnan(Ki_cortical_gm):
        fit_text += f"Cortical GM: Ki = {Ki_cortical_gm:.5f} ml/100g/min, λ = {lambda_cortical_gm:.5f} ml/100g\n"
    if not np.isnan(Ki_subcortical_gm):
        fit_text += f"Subcortical GM: Ki = {Ki_subcortical_gm:.5f} ml/100g/min, λ = {lambda_subcortical_gm:.5f} ml/100g\n"
    if boundary_ctc is not None and not np.isnan(Ki_boundary):
        fit_text += f"Boundary: Ki = {Ki_boundary:.5f} ml/100g/min, λ = {lambda_boundary:.5f} ml/100g"

    # Display fit text
    ax4.text(0.5, -0.4, fit_text.strip(), transform=ax4.transAxes, fontsize=10, color='black', ha='center', bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)

    if not turbo_mode:
        plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
        close_plot_after_delay(1, fig)
        plt.show()
    else:
        plt.close(fig)

def compute_and_plot_ctcs_median(data_4d, t2_img, wm_mask_t2, cortical_gm_mask_t2, subcortical_gm_mask_t2,
                                 wm_mask_dce, cortical_gm_mask_dce, subcortical_gm_mask_dce,
                                 T1_matrix, M0_matrix, analysis_directory, time_points_s, image_directory,
                                 dce_path, boundary=False, compute_per_voxel_Ki=False):
    """
    Computes median CTCs for different tissue types across slices, performs Patlak analysis,
    saves the results, and generates plots. Also computes the total median for the entire tissue volume.
    Optionally computes K_i per voxel and generates overlay images and a NIfTI file.

    Parameters:
    - data_4d: 4D numpy array of DCE data.
    - t2_img: 3D numpy array of T2-weighted image.
    - wm_mask_t2, cortical_gm_mask_t2, subcortical_gm_mask_t2: 3D boolean arrays of tissue masks in T2 space.
    - wm_mask_dce, cortical_gm_mask_dce, subcortical_gm_mask_dce: 3D boolean arrays of tissue masks in DCE space.
    - T1_matrix, M0_matrix: 3D numpy arrays of T1 and M0 values.
    - analysis_directory: Path to save analysis results.
    - time_points_s: 1D numpy array of time points in seconds.
    - image_directory: Path to save images.
    - dce_path: Path to the DCE NIfTI file.
    - boundary: Boolean indicating whether to compute boundary region.
    - compute_per_voxel_Ki: Boolean indicating whether to compute K_i per voxel.

    Returns:
    - None
    """

    n_slices = t2_img.shape[2]

    # Load C_a once
    max_folder = os.path.join(analysis_directory, 'TSCC Data', 'Max')
    npy_files = [f for f in os.listdir(max_folder) if f.endswith('.npy')]

    if len(npy_files) != 1:
        raise ValueError(f"Expected exactly one .npy file in {max_folder}, but found {len(npy_files)}.")

    ca_file = npy_files[0]
    C_a_full = np.load(os.path.join(max_folder, ca_file))

    all_patlak_data = []
    Ki_wm_list = []
    Ki_cortical_gm_list = []
    Ki_subcortical_gm_list = []
    Ki_boundary_list = []

    # Initialize lists to collect all valid CTCs across slices
    wm_ctcs_total = []
    cortical_gm_ctcs_total = []
    subcortical_gm_ctcs_total = []
    boundary_ctcs_total = []

    # Initialize an empty 3D array to store K_i values per voxel
    Ki_per_voxel = np.full(data_4d.shape[:3], np.nan)

    for i in range(n_slices):
        # Extract relevant masks for the current slice
        wm_slice_t2 = wm_mask_t2[:, :, i]
        cortical_gm_slice_t2 = cortical_gm_mask_t2[:, :, i]
        subcortical_gm_slice_t2 = subcortical_gm_mask_t2[:, :, i]
        wm_slice_dce = wm_mask_dce[:, :, i]
        cortical_gm_slice_dce = cortical_gm_mask_dce[:, :, i]
        subcortical_gm_slice_dce = subcortical_gm_mask_dce[:, :, i]

        # Combine cortical and subcortical GM masks for boundary calculation
        gm_slice_dce = np.logical_or(cortical_gm_slice_dce, subcortical_gm_slice_dce)

        # Compute the boundary mask if required
        if boundary:
            wm_dilated = binary_dilation(wm_slice_dce, iterations=1)
            gm_dilated = binary_dilation(gm_slice_dce, iterations=1)
            boundary_mask = np.logical_and(wm_dilated, gm_dilated)
            boundary_indices = np.argwhere(boundary_mask)
        else:
            boundary_mask = None
            boundary_indices = []

        # Find voxel indices for white and gray matter in the slice
        wm_indices = np.argwhere(wm_slice_dce)
        cortical_gm_indices = np.argwhere(cortical_gm_slice_dce)
        subcortical_gm_indices = np.argwhere(subcortical_gm_slice_dce)

        # Initialize lists to store valid CTCs
        wm_ctcs = []
        cortical_gm_ctcs = []
        subcortical_gm_ctcs = []
        boundary_ctcs = []

        # Function to process CTCs for a given set of indices
        def process_ctcs(indices, label):
            ctcs = []
            for (x, y) in indices:
                voxel_time_course = data_4d[x, y, i, :]
                T1 = T1_matrix[x, y, i]
                M0 = M0_matrix[x, y, i]
                C_t_0 = compute_CTC(voxel_time_course, T1, m0=M0)
                baseline_point = find_baseline_point_advanced(C_t_0)
                C_t = custom_shifter(C_t_0, baseline_point)

                # Exclude CTCs with NaNs or zeros
                if np.isnan(C_t).any() or np.all(C_t == 0):
                    continue
                ctcs.append(C_t)
            return ctcs

        # Process CTCs for each tissue type
        wm_ctcs = process_ctcs(wm_indices, 'White Matter')
        cortical_gm_ctcs = process_ctcs(cortical_gm_indices, 'Cortical Gray Matter')
        subcortical_gm_ctcs = process_ctcs(subcortical_gm_indices, 'Subcortical Gray Matter')

        # Process CTCs for boundary if required
        if boundary and boundary_indices.size > 0:
            boundary_ctcs = process_ctcs(boundary_indices, 'Boundary')

        # Add the valid CTCs from this slice to the total lists
        wm_ctcs_total.extend(wm_ctcs)
        cortical_gm_ctcs_total.extend(cortical_gm_ctcs)
        subcortical_gm_ctcs_total.extend(subcortical_gm_ctcs)
        if boundary and boundary_ctcs:
            boundary_ctcs_total.extend(boundary_ctcs)

        # Compute median CTCs if valid CTCs are available
        avg_wm_ctc = np.median(wm_ctcs, axis=0) if wm_ctcs else np.array([])
        avg_cortical_gm_ctc = np.median(cortical_gm_ctcs, axis=0) if cortical_gm_ctcs else np.array([])
        avg_subcortical_gm_ctc = np.median(subcortical_gm_ctcs, axis=0) if subcortical_gm_ctcs else np.array([])
        avg_boundary_ctc = np.median(boundary_ctcs, axis=0) if boundary_ctcs else np.array([])

        # Save the tissue concentration curves as .npy files
        save_dir_ctc = os.path.join(analysis_directory, 'CTC Data', 'Tissue', 'AI')
        os.makedirs(save_dir_ctc, exist_ok=True)

        np.save(os.path.join(save_dir_ctc, f'wm_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_wm_ctc)
        np.save(os.path.join(save_dir_ctc, f'cortical_gm_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_cortical_gm_ctc)
        np.save(os.path.join(save_dir_ctc, f'subcortical_gm_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_subcortical_gm_ctc)
        if boundary and avg_boundary_ctc.size > 0:
            np.save(os.path.join(save_dir_ctc, f'bo_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_boundary_ctc)

        # Ensure the CTCs and C_a have the same length
        min_length = len(C_a_full)
        if avg_wm_ctc.size > 0:
            min_length = min(min_length, avg_wm_ctc.size)
        if avg_cortical_gm_ctc.size > 0:
            min_length = min(min_length, avg_cortical_gm_ctc.size)
        if avg_subcortical_gm_ctc.size > 0:
            min_length = min(min_length, avg_subcortical_gm_ctc.size)
        if boundary and avg_boundary_ctc.size > 0:
            min_length = min(min_length, avg_boundary_ctc.size)

        C_a_slice = C_a_full[:min_length]
        time_points = time_points_s[:min_length]

        # Truncate CTCs to match length
        C_t_wm = avg_wm_ctc[:min_length] if avg_wm_ctc.size > 0 else np.array([])
        C_t_cortical_gm = avg_cortical_gm_ctc[:min_length] if avg_cortical_gm_ctc.size > 0 else np.array([])
        C_t_subcortical_gm = avg_subcortical_gm_ctc[:min_length] if avg_subcortical_gm_ctc.size > 0 else np.array([])
        if boundary and avg_boundary_ctc.size > 0:
            C_t_boundary = avg_boundary_ctc[:min_length]
        else:
            C_t_boundary = np.array([])

        # Handle white matter
        if C_t_wm.size > 0:
            Ki_wm, lambda_wm, SD_Ki_wm, x_patlak_wm, y_patlak_wm, included_wm = patlak_analysis_plotting(C_t_wm, C_a_slice, time_points)
        else:
            Ki_wm = np.nan
            lambda_wm = np.nan
            SD_Ki_wm = np.nan
            x_patlak_wm = np.array([])
            y_patlak_wm = np.array([])
            included_wm = np.array([], dtype=bool)

        # Handle cortical gray matter
        if C_t_cortical_gm.size > 0:
            Ki_cortical_gm, lambda_cortical_gm, SD_Ki_cortical_gm, x_patlak_cortical_gm, y_patlak_cortical_gm, included_cortical_gm = patlak_analysis_plotting(C_t_cortical_gm, C_a_slice, time_points)
        else:
            Ki_cortical_gm = np.nan
            lambda_cortical_gm = np.nan
            SD_Ki_cortical_gm = np.nan
            x_patlak_cortical_gm = np.array([])
            y_patlak_cortical_gm = np.array([])
            included_cortical_gm = np.array([], dtype=bool)

        # Handle subcortical gray matter
        if C_t_subcortical_gm.size > 0:
            Ki_subcortical_gm, lambda_subcortical_gm, SD_Ki_subcortical_gm, x_patlak_subcortical_gm, y_patlak_subcortical_gm, included_subcortical_gm = patlak_analysis_plotting(C_t_subcortical_gm, C_a_slice, time_points)
        else:
            Ki_subcortical_gm = np.nan
            lambda_subcortical_gm = np.nan
            SD_Ki_subcortical_gm = np.nan
            x_patlak_subcortical_gm = np.array([])
            y_patlak_subcortical_gm = np.array([])
            included_subcortical_gm = np.array([], dtype=bool)

        # Handle boundary if required
        if boundary and C_t_boundary.size > 0:
            Ki_boundary, lambda_boundary, SD_Ki_boundary, x_patlak_boundary, y_patlak_boundary, included_boundary = patlak_analysis_plotting(C_t_boundary, C_a_slice, time_points)
        else:
            Ki_boundary = np.nan
            lambda_boundary = np.nan
            SD_Ki_boundary = np.nan
            x_patlak_boundary = np.array([])
            y_patlak_boundary = np.array([])
            included_boundary = np.array([], dtype=bool)

        # Compute voxel counts
        voxel_count_wm = np.sum(wm_slice_dce)
        voxel_count_cortical_gm = np.sum(cortical_gm_slice_dce)
        voxel_count_subcortical_gm = np.sum(subcortical_gm_slice_dce)
        voxel_count_boundary = np.sum(boundary_mask) if boundary and boundary_mask is not None else 0

        # Collect Ki values for plotting
        Ki_wm_list.append(Ki_wm)
        Ki_cortical_gm_list.append(Ki_cortical_gm)
        Ki_subcortical_gm_list.append(Ki_subcortical_gm)
        if boundary:
            Ki_boundary_list.append(Ki_boundary)

        # Plot the results for the current slice
        plot_ctcs_and_patlak(
            t2_img[:, :, i], data_4d[:, :, i, 20],
            wm_slice_t2, cortical_gm_slice_t2, subcortical_gm_slice_t2,  # Pass T2 masks
            wm_slice_dce, cortical_gm_slice_dce, subcortical_gm_slice_dce,  # Pass DCE masks
            avg_wm_ctc, avg_cortical_gm_ctc, avg_subcortical_gm_ctc,
            x_patlak_wm, y_patlak_wm, Ki_wm, lambda_wm,
            x_patlak_cortical_gm, y_patlak_cortical_gm, Ki_cortical_gm, lambda_cortical_gm,
            x_patlak_subcortical_gm, y_patlak_subcortical_gm, Ki_subcortical_gm, lambda_subcortical_gm,
            slice_idx=i+1,
            save_path=os.path.join(image_directory, 'AI', 'Tissue functions', f"AI_Tissue_slice_{i+1}_segmented_median.png"),
            boundary_mask=boundary_mask,  # Pass the boundary mask for visualization
            boundary_ctc=avg_boundary_ctc,
            x_patlak_boundary=x_patlak_boundary, y_patlak_boundary=y_patlak_boundary,
            Ki_boundary=Ki_boundary, lambda_boundary=lambda_boundary, included_wm=included_wm,
            included_cortical_gm=included_cortical_gm, included_subcortical_gm=included_subcortical_gm, included_boundary=included_boundary
        )

        # Collect data for JSON output
        patlak_data = {
            'slice': i + 1,
            'white_matter_median': {
                'Ki': Ki_wm,
                'SD_Ki': SD_Ki_wm,
                'lambda': lambda_wm,
                'voxel_count': int(voxel_count_wm)
            },
            'cortical_gray_matter_median': {
                'Ki': Ki_cortical_gm,
                'SD_Ki': SD_Ki_cortical_gm,
                'lambda': lambda_cortical_gm,
                'voxel_count': int(voxel_count_cortical_gm)
            },
            'subcortical_gray_matter_median': {
                'Ki': Ki_subcortical_gm,
                'SD_Ki': SD_Ki_subcortical_gm,
                'lambda': lambda_subcortical_gm,
                'voxel_count': int(voxel_count_subcortical_gm)
            }
        }

        if boundary and avg_boundary_ctc.size > 0:
            patlak_data['boundary_median'] = {
                'Ki': Ki_boundary,
                'SD_Ki': SD_Ki_boundary,
                'lambda': lambda_boundary,
                'voxel_count': int(voxel_count_boundary)
            }

        all_patlak_data.append(patlak_data)

        # Compute K_i per voxel if enabled
        if compute_per_voxel_Ki:
            # Combine WM and GM masks for the current slice
            brain_mask_slice = np.logical_or(wm_slice_dce, gm_slice_dce)
            brain_indices = np.argwhere(brain_mask_slice)

            # Initialize K_i slice array
            Ki_slice = np.full(brain_mask_slice.shape, np.nan)

            # For each voxel in the brain mask, compute K_i
            for (x, y) in brain_indices:
                voxel_time_course = data_4d[x, y, i, :]
                T1 = T1_matrix[x, y, i]
                M0 = M0_matrix[x, y, i]
                C_t_0 = compute_CTC(voxel_time_course, T1, m0=M0)
                baseline_point = find_baseline_point_advanced(C_t_0)
                C_t = custom_shifter(C_t_0, baseline_point)

                # Exclude CTCs with NaNs or zeros
                if np.isnan(C_t).any() or np.all(C_t == 0):
                    continue

                # Ensure C_t and C_a have the same length
                min_length_voxel = min(len(C_t), len(C_a_full))
                C_t_voxel = C_t[:min_length_voxel]
                C_a_voxel = C_a_full[:min_length_voxel]
                time_points_voxel = time_points_s[:min_length_voxel]

                # Perform Patlak analysis
                Ki_voxel, _, _, _, _, _ = patlak_analysis_plotting(C_t_voxel, C_a_voxel, time_points_voxel)

                Ki_slice[x, y] = Ki_voxel

            # Store the K_i slice in the 3D K_i array
            Ki_per_voxel[:, :, i] = Ki_slice

            # Generate and save the overlay image for the current slice
            save_dir_overlay = os.path.join(image_directory, 'AI', 'Ki Overlays')
            os.makedirs(save_dir_overlay, exist_ok=True)
            save_path_overlay = os.path.join(save_dir_overlay, f"Ki_overlay_slice_{i+1}.png")
            plot_Ki_overlay(data_4d[:, :, i, 20], Ki_slice, slice_idx=i+1, save_path=save_path_overlay)

    # Save all Patlak data to JSON file after processing all slices
    json_file_path = os.path.join(analysis_directory, "AI_values_median.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(all_patlak_data, json_file, indent=4)

    # Plot Ki values as a function of slice number
    if Ki_wm_list:
        num_processed_slices = len(Ki_wm_list)
        slice_numbers = range(1, num_processed_slices + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(slice_numbers, Ki_wm_list, label='White Matter Ki', marker='o')
        plt.plot(slice_numbers, Ki_cortical_gm_list, label='Cortical Gray Matter Ki', marker='o')
        plt.plot(slice_numbers, Ki_subcortical_gm_list, label='Subcortical Gray Matter Ki', marker='o')
        if boundary and Ki_boundary_list:
            plt.plot(slice_numbers, Ki_boundary_list, label='Boundary Ki', marker='o')
        plt.xlabel('Slice Number')
        plt.ylabel('K_i')
        plt.title('K_i values across Slices')
        plt.legend()
        plt.grid(True)

        # Ensure the directory exists
        save_dir = os.path.join(image_directory, 'AI', 'Tissue functions')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'Ki_vs_slice_median.png'))

        if not turbo_mode:
            plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
            close_plot_after_delay_plt(3)
            plt.show()
        else:
            plt.close()
    else:
        print("No Ki values were computed; skipping Ki plot.")

    # Compute the overall median CTC per tissue type
    avg_wm_ctc_total = np.median(wm_ctcs_total, axis=0) if wm_ctcs_total else np.array([])
    avg_cortical_gm_ctc_total = np.median(cortical_gm_ctcs_total, axis=0) if cortical_gm_ctcs_total else np.array([])
    avg_subcortical_gm_ctc_total = np.median(subcortical_gm_ctcs_total, axis=0) if subcortical_gm_ctcs_total else np.array([])
    avg_boundary_ctc_total = np.median(boundary_ctcs_total, axis=0) if boundary_ctcs_total else np.array([])

    # Ensure the CTCs and C_a have the same length
    min_length = len(C_a_full)
    if avg_wm_ctc_total.size > 0:
        min_length = min(min_length, avg_wm_ctc_total.size)
    if avg_cortical_gm_ctc_total.size > 0:
        min_length = min(min_length, avg_cortical_gm_ctc_total.size)
    if avg_subcortical_gm_ctc_total.size > 0:
        min_length = min(min_length, avg_subcortical_gm_ctc_total.size)
    if boundary and avg_boundary_ctc_total.size > 0:
        min_length = min(min_length, avg_boundary_ctc_total.size)

    C_a_total = C_a_full[:min_length]
    time_points_total = time_points_s[:min_length]

    # Truncate CTCs to match length
    C_t_wm_total = avg_wm_ctc_total[:min_length] if avg_wm_ctc_total.size > 0 else np.array([])
    C_t_cortical_gm_total = avg_cortical_gm_ctc_total[:min_length] if avg_cortical_gm_ctc_total.size > 0 else np.array([])
    C_t_subcortical_gm_total = avg_subcortical_gm_ctc_total[:min_length] if avg_subcortical_gm_ctc_total.size > 0 else np.array([])
    if boundary and avg_boundary_ctc_total.size > 0:
        C_t_boundary_total = avg_boundary_ctc_total[:min_length]
    else:
        C_t_boundary_total = np.array([])

    # Perform Patlak analysis on aggregated CTCs
    # White Matter
    if C_t_wm_total.size > 0:
        Ki_wm_total, lambda_wm_total, SD_Ki_wm_total, _, _, _ = patlak_analysis_plotting(C_t_wm_total, C_a_total, time_points_total)
    else:
        Ki_wm_total = np.nan
        lambda_wm_total = np.nan
        SD_Ki_wm_total = np.nan

    # Cortical Gray Matter
    if C_t_cortical_gm_total.size > 0:
        Ki_cortical_gm_total, lambda_cortical_gm_total, SD_Ki_cortical_gm_total, _, _, _ = patlak_analysis_plotting(C_t_cortical_gm_total, C_a_total, time_points_total)
    else:
        Ki_cortical_gm_total = np.nan
        lambda_cortical_gm_total = np.nan
        SD_Ki_cortical_gm_total = np.nan

    # Subcortical Gray Matter
    if C_t_subcortical_gm_total.size > 0:
        Ki_subcortical_gm_total, lambda_subcortical_gm_total, SD_Ki_subcortical_gm_total, _, _, _ = patlak_analysis_plotting(C_t_subcortical_gm_total, C_a_total, time_points_total)
    else:
        Ki_subcortical_gm_total = np.nan
        lambda_subcortical_gm_total = np.nan
        SD_Ki_subcortical_gm_total = np.nan

    # Boundary (if available)
    if boundary and C_t_boundary_total.size > 0:
        Ki_boundary_total, lambda_boundary_total, SD_Ki_boundary_total, _, _, _ = patlak_analysis_plotting(C_t_boundary_total, C_a_total, time_points_total)
    else:
        Ki_boundary_total = np.nan
        lambda_boundary_total = np.nan
        SD_Ki_boundary_total = np.nan

    # Create a dictionary to hold the total Patlak data
    patlak_data_total = {
        'white_matter_median_total': {
            'Ki': Ki_wm_total,
            'SD_Ki': SD_Ki_wm_total,
            'lambda': lambda_wm_total,
            'voxel_count': len(wm_ctcs_total)
        },
        'cortical_gray_matter_median_total': {
            'Ki': Ki_cortical_gm_total,
            'SD_Ki': SD_Ki_cortical_gm_total,
            'lambda': lambda_cortical_gm_total,
            'voxel_count': len(cortical_gm_ctcs_total)
        },
        'subcortical_gray_matter_median_total': {
            'Ki': Ki_subcortical_gm_total,
            'SD_Ki': SD_Ki_subcortical_gm_total,
            'lambda': lambda_subcortical_gm_total,
            'voxel_count': len(subcortical_gm_ctcs_total)
        }
    }

    if boundary and boundary_ctcs_total:
        patlak_data_total['boundary_median_total'] = {
            'Ki': Ki_boundary_total,
            'SD_Ki': SD_Ki_boundary_total,
            'lambda': lambda_boundary_total,
            'voxel_count': len(boundary_ctcs_total)
        }

    # Save the total Patlak data to JSON file
    json_file_path_total = os.path.join(analysis_directory, "AI_values_median_total.json")
    with open(json_file_path_total, 'w') as json_file:
        json.dump(patlak_data_total, json_file, indent=4)

    # Save Ki_per_voxel as a .nii file if computed
    if compute_per_voxel_Ki:
        Ki_per_voxel_nii = nib.Nifti1Image(Ki_per_voxel, affine=nib.load(dce_path).affine)
        Ki_per_voxel_path = os.path.join(analysis_directory, 'Ki_per_voxel.nii.gz')
        nib.save(Ki_per_voxel_nii, Ki_per_voxel_path)
        print(f"K_i per voxel saved to {Ki_per_voxel_path}")
        
        
def plot_Ki_overlay(dce_slice, Ki_slice, slice_idx, save_path):
    """
    Plots the DCE image slice with an overlay of K_i values.

    Parameters:
    - dce_slice: 2D numpy array of the DCE image at a specific time point.
    - Ki_slice: 2D numpy array of K_i values for the slice.
    - slice_idx: Integer indicating the slice number.
    - save_path: Path to save the overlay image.

    Returns:
    - None
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy.ma as ma

    # Mask out NaN values in K_i
    Ki_masked = ma.masked_invalid(Ki_slice)

    # Set up the figure and axis
    plt.figure(figsize=(8, 8))
    plt.imshow(np.rot90(dce_slice), cmap='gray', interpolation='nearest')

    # Overlay K_i values using a colormap
    plt.imshow(np.rot90(Ki_masked), cmap='jet', interpolation='nearest', alpha=0.6,
               norm=Normalize(vmin=np.nanmin(Ki_slice), vmax=np.nanmax(Ki_slice)))

    plt.colorbar(label='K_i (ml/100g/min)')
    plt.title(f'Slice {slice_idx} K_i Overlay')
    plt.axis('off')
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=300)
    plt.close()


def tissue_function_AI(analysis_directory, nifti_directory, image_directory, filenames, parameters):
    t1_3D_filename, axial_t1_3D_filename, t2_3D_filename, axial_t2_3D_filename, \
        flair_3D_filename, axial_flair_3D_filename, axial_t2_2D_filename, dce_filename = filenames

    IsVFA, IsIR, apple_metal, boundary = parameters

    fastsurfer_path = '/Users/edt/FastSurfer/run_fastsurfer.sh'
    t1_path = os.path.join(nifti_directory, t1_3D_filename)
    seg_dir = os.path.join(nifti_directory, 'segmentation')
    sid = 'segmentation'  # Define the subject ID
    seg_mgz_path = os.path.join(seg_dir, sid, 'mri', 'aparc.DKTatlas+aseg.deep.mgz')
    t2_path = os.path.join(nifti_directory, axial_t2_2D_filename)
    dce_path = os.path.join(nifti_directory, dce_filename)

    # Ensure segmentation directory exists
    os.makedirs(seg_dir, exist_ok=True)

    # Run segmentation and create masks
    segmentation(fastsurfer_path, seg_mgz_path, t1_path, seg_dir, sid, apple_metal)

    # Paths to masks in the same directory as aparc.DKTatlas+aseg.deep.mgz
    mask_dir = os.path.dirname(seg_mgz_path)
    cortical_gm_mask_path = os.path.join(mask_dir, 'cortical_gm.nii.gz')
    subcortical_gm_mask_path = os.path.join(mask_dir, 'subcortical_gm.nii.gz')
    wm_mask_path = os.path.join(mask_dir, 'wm.nii.gz')

    print('[!] Coregistering GM/WM masks onto T2 and DCE space')
    wm_mask_t2, wm_mask_dce, cortical_gm_mask_t2, cortical_gm_mask_dce, \
        subcortical_gm_mask_t2, subcortical_gm_mask_dce = coregistration(
            seg_mgz_path=seg_mgz_path,
            dce_path=dce_path,
            t2_path=t2_path
        )

    # Load the T2 image for visualization
    t2_img = nib.load(t2_path).get_fdata()

    # Plot the predictions with gray and white matter masks on T2
    plot_predictions_with_masks(t2_img, wm_mask_t2, cortical_gm_mask_t2, subcortical_gm_mask_t2, image_directory)

    # Load the DCE 4D data
    data_4d = np.array(nib.load(dce_path).get_fdata())

    # Load T1 and M0 matrices
    T1_matrix = load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_T1_matrix.pkl'))
    M0_matrix = load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_M0_matrix.pkl'))

    # Compute time_points_s
    TR = nib.load(dce_path).header.get_zooms()[-1]
    num_volumes = data_4d.shape[-1]
    total_scan_duration = TR * num_volumes
    time_points_s = np.linspace(0, total_scan_duration, num_volumes)

    # Compute and plot CTCs and Patlak fits, and save data
    compute_and_plot_ctcs_median(
        data_4d, t2_img, wm_mask_t2, cortical_gm_mask_t2, subcortical_gm_mask_t2,
        wm_mask_dce, cortical_gm_mask_dce, subcortical_gm_mask_dce,
        T1_matrix, M0_matrix, analysis_directory, time_points_s, image_directory,
        dce_path=dce_path, boundary=boundary, compute_per_voxel_Ki=True
    )
