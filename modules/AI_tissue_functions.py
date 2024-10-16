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


# Labels lookup dictionary
labels_lookup = {
    'Left-Lateral-Ventricle': 4,
    'Left-Inf-Lat-Vent': 5,
    'Left-Cerebellum-White-Matter': 7,
    'Left-Cerebellum-Cortex': 8,
    'Left-Thalamus-Proper': 10,
    'Left-Caudate': 11,
    'Left-Putamen': 12,
    'Left-Pallidum': 13,
    'Left-3rd-Ventricle': 14,
    'Left-4th-Ventricle': 15,
    'Left-Brain-Stem': 16,
    'Left-Hippocampus': 17,
    'Left-Amygdala': 18,
    'Left-CSF': 24,
    'Left-Accumbens-area': 26,
    'Left-VentralDC': 28,
    'Left-choroid-plexus': 31,
    'Right-Lateral-Ventricle': 43,
    'Right-Inf-Lat-Vent': 44,
    'Right-Cerebellum-White-Matter': 46,
    'Right-Cerebellum-Cortex': 47,
    'Right-Thalamus-Proper': 49,
    'Right-Caudate': 50,
    'Right-Putamen': 51,
    'Right-Pallidum': 52,
    'Right-Hippocampus': 53,
    'Right-Amygdala': 54,
    'Right-Accumbens-area': 58,
    'Right-VentralDC': 60,
    'Right-choroid-plexus': 63,
    'Right-3rd-Ventricle': 14,
    'Right-4th-Ventricle': 15,
    'Right-Brain-Stem': 16,
    'Right-CSF': 24,
    'ctx-lh-caudalanteriorcingulate': 1002,
    'ctx-lh-caudalmiddlefrontal': 1003,
    'ctx-lh-cuneus': 1005,
    'ctx-lh-entorhinal': 1006,
    'ctx-lh-fusiform': 1007,
    'ctx-lh-inferiorparietal': 1008,
    'ctx-lh-inferiortemporal': 1009,
    'ctx-lh-isthmuscingulate': 1010,
    'ctx-lh-lateraloccipital': 1011,
    'ctx-lh-lateralorbitofrontal': 1012,
    'ctx-lh-lingual': 1013,
    'ctx-lh-medialorbitofrontal': 1014,
    'ctx-lh-middletemporal': 1015,
    'ctx-lh-parahippocampal': 1016,
    'ctx-lh-paracentral': 1017,
    'ctx-lh-parsopercularis': 1018,
    'ctx-lh-parsorbitalis': 1019,
    'ctx-lh-parstriangularis': 1020,
    'ctx-lh-pericalcarine': 1021,
    'ctx-lh-postcentral': 1022,
    'ctx-lh-posteriorcingulate': 1023,
    'ctx-lh-precentral': 1024,
    'ctx-lh-precuneus': 1025,
    'ctx-lh-rostralanteriorcingulate': 1026,
    'ctx-lh-rostralmiddlefrontal': 1027,
    'ctx-lh-superiorfrontal': 1028,
    'ctx-lh-superiorparietal': 1029,
    'ctx-lh-superiortemporal': 1030,
    'ctx-lh-supramarginal': 1031,
    'ctx-lh-transversetemporal': 1034,
    'ctx-lh-insula': 1035,
    'ctx-rh-caudalanteriorcingulate': 2002,
    'ctx-rh-caudalmiddlefrontal': 2003,
    'ctx-rh-cuneus': 2005,
    'ctx-rh-entorhinal': 2006,
    'ctx-rh-fusiform': 2007,
    'ctx-rh-inferiorparietal': 2008,
    'ctx-rh-inferiortemporal': 2009,
    'ctx-rh-isthmuscingulate': 2010,
    'ctx-rh-lateraloccipital': 2011,
    'ctx-rh-lateralorbitofrontal': 2012,
    'ctx-rh-lingual': 2013,
    'ctx-rh-medialorbitofrontal': 2014,
    'ctx-rh-middletemporal': 2015,
    'ctx-rh-parahippocampal': 2016,
    'ctx-rh-paracentral': 2017,
    'ctx-rh-parsopercularis': 2018,
    'ctx-rh-parsorbitalis': 2019,
    'ctx-rh-parstriangularis': 2020,
    'ctx-rh-pericalcarine': 2021,
    'ctx-rh-postcentral': 2022,
    'ctx-rh-posteriorcingulate': 2023,
    'ctx-rh-precentral': 2024,
    'ctx-rh-precuneus': 2025,
    'ctx-rh-rostralanteriorcingulate': 2026,
    'ctx-rh-rostralmiddlefrontal': 2027,
    'ctx-rh-superiorfrontal': 2028,
    'ctx-rh-superiorparietal': 2029,
    'ctx-rh-superiortemporal': 2030,
    'ctx-rh-supramarginal': 2031,
    'ctx-rh-transversetemporal': 2034,
    'ctx-rh-insula': 2035
}

# Define gray and white matter labels
white_matter_labels = [
    2,  # Left-Cerebral-White-Matter
    41, # Right-Cerebral-White-Matter
    7,  # Left-Cerebellum-White-Matter
    46, # Right-Cerebellum-White-Matter
    251, # CC_Posterior
    252, # CC_Mid_Posterior
    253, # CC_Central
    254, # CC_Mid_Anterior
    255, # CC_Anterior
    28,  # Left-VentralDC
    60,  # Right-VentralDC
    11,  # Left-Caudate
    50,  # Right-Caudate
    12,  # Left-Putamen
    51,  # Right-Putamen
    13,  # Left-Pallidum
    52,  # Right-Pallidum
]

gray_matter_labels = [
    labels_lookup['Left-Cerebellum-Cortex'],
    labels_lookup['Right-Cerebellum-Cortex'],
    labels_lookup['Left-Hippocampus'],
    labels_lookup['Right-Hippocampus'],
    labels_lookup['Left-Amygdala'],
    labels_lookup['Right-Amygdala'],
    labels_lookup['Left-Accumbens-area'],
    labels_lookup['Right-Accumbens-area'],
    labels_lookup['ctx-lh-caudalanteriorcingulate'],
    labels_lookup['ctx-lh-caudalmiddlefrontal'],
    labels_lookup['ctx-lh-cuneus'],
    labels_lookup['ctx-lh-entorhinal'],
    labels_lookup['ctx-lh-fusiform'],
    labels_lookup['ctx-lh-inferiorparietal'],
    labels_lookup['ctx-lh-inferiortemporal'],
    labels_lookup['ctx-lh-isthmuscingulate'],
    labels_lookup['ctx-lh-lateraloccipital'],
    labels_lookup['ctx-lh-lateralorbitofrontal'],
    labels_lookup['ctx-lh-lingual'],
    labels_lookup['ctx-lh-medialorbitofrontal'],
    labels_lookup['ctx-lh-middletemporal'],
    labels_lookup['ctx-lh-parahippocampal'],
    labels_lookup['ctx-lh-paracentral'],
    labels_lookup['ctx-lh-parsopercularis'],
    labels_lookup['ctx-lh-parsorbitalis'],
    labels_lookup['ctx-lh-parstriangularis'],
    labels_lookup['ctx-lh-pericalcarine'],
    labels_lookup['ctx-lh-postcentral'],
    labels_lookup['ctx-lh-posteriorcingulate'],
    labels_lookup['ctx-lh-precentral'],
    labels_lookup['ctx-lh-precuneus'],
    labels_lookup['ctx-lh-rostralanteriorcingulate'],
    labels_lookup['ctx-lh-rostralmiddlefrontal'],
    labels_lookup['ctx-lh-superiorfrontal'],
    labels_lookup['ctx-lh-superiorparietal'],
    labels_lookup['ctx-lh-superiortemporal'],
    labels_lookup['ctx-lh-supramarginal'],
    labels_lookup['ctx-lh-transversetemporal'],
    labels_lookup['ctx-lh-insula'],
    labels_lookup['ctx-rh-caudalanteriorcingulate'],
    labels_lookup['ctx-rh-caudalmiddlefrontal'],
    labels_lookup['ctx-rh-cuneus'],
    labels_lookup['ctx-rh-entorhinal'],
    labels_lookup['ctx-rh-fusiform'],
    labels_lookup['ctx-rh-inferiorparietal'],
    labels_lookup['ctx-rh-inferiortemporal'],
    labels_lookup['ctx-rh-isthmuscingulate'],
    labels_lookup['ctx-rh-lateraloccipital'],
    labels_lookup['ctx-rh-lateralorbitofrontal'],
    labels_lookup['ctx-rh-lingual'],
    labels_lookup['ctx-rh-medialorbitofrontal'],
    labels_lookup['ctx-rh-middletemporal'],
    labels_lookup['ctx-rh-parahippocampal'],
    labels_lookup['ctx-rh-paracentral'],
    labels_lookup['ctx-rh-parsopercularis'],
    labels_lookup['ctx-rh-parsorbitalis'],
    labels_lookup['ctx-rh-parstriangularis'],
    labels_lookup['ctx-rh-pericalcarine'],
    labels_lookup['ctx-rh-postcentral'],
    labels_lookup['ctx-rh-posteriorcingulate'],
    labels_lookup['ctx-rh-precentral'],
    labels_lookup['ctx-rh-precuneus'],
    labels_lookup['ctx-rh-rostralanteriorcingulate'],
    labels_lookup['ctx-rh-rostralmiddlefrontal'],
    labels_lookup['ctx-rh-superiorfrontal'],
    labels_lookup['ctx-rh-superiorparietal'],
    labels_lookup['ctx-rh-superiortemporal'],
    labels_lookup['ctx-rh-supramarginal'],
    labels_lookup['ctx-rh-transversetemporal'],
    labels_lookup['ctx-rh-insula'],
    labels_lookup['Left-Thalamus-Proper'],
    labels_lookup['Right-Thalamus-Proper']
]

# Cortical Gray Matter Labels
cortical_gray_matter_labels = [
    labels_lookup['ctx-lh-caudalanteriorcingulate'],
    labels_lookup['ctx-lh-caudalmiddlefrontal'],
    labels_lookup['ctx-lh-cuneus'],
    labels_lookup['ctx-lh-entorhinal'],
    labels_lookup['ctx-lh-fusiform'],
    labels_lookup['ctx-lh-inferiorparietal'],
    labels_lookup['ctx-lh-inferiortemporal'],
    labels_lookup['ctx-lh-isthmuscingulate'],
    labels_lookup['ctx-lh-lateraloccipital'],
    labels_lookup['ctx-lh-lateralorbitofrontal'],
    labels_lookup['ctx-lh-lingual'],
    labels_lookup['ctx-lh-medialorbitofrontal'],
    labels_lookup['ctx-lh-middletemporal'],
    labels_lookup['ctx-lh-parahippocampal'],
    labels_lookup['ctx-lh-paracentral'],
    labels_lookup['ctx-lh-parsopercularis'],
    labels_lookup['ctx-lh-parsorbitalis'],
    labels_lookup['ctx-lh-parstriangularis'],
    labels_lookup['ctx-lh-pericalcarine'],
    labels_lookup['ctx-lh-postcentral'],
    labels_lookup['ctx-lh-posteriorcingulate'],
    labels_lookup['ctx-lh-precentral'],
    labels_lookup['ctx-lh-precuneus'],
    labels_lookup['ctx-lh-rostralanteriorcingulate'],
    labels_lookup['ctx-lh-rostralmiddlefrontal'],
    labels_lookup['ctx-lh-superiorfrontal'],
    labels_lookup['ctx-lh-superiorparietal'],
    labels_lookup['ctx-lh-superiortemporal'],
    labels_lookup['ctx-lh-supramarginal'],
    labels_lookup['ctx-lh-transversetemporal'],
    labels_lookup['ctx-lh-insula'],
    labels_lookup['ctx-rh-caudalanteriorcingulate'],
    labels_lookup['ctx-rh-caudalmiddlefrontal'],
    labels_lookup['ctx-rh-cuneus'],
    labels_lookup['ctx-rh-entorhinal'],
    labels_lookup['ctx-rh-fusiform'],
    labels_lookup['ctx-rh-inferiorparietal'],
    labels_lookup['ctx-rh-inferiortemporal'],
    labels_lookup['ctx-rh-isthmuscingulate'],
    labels_lookup['ctx-rh-lateraloccipital'],
    labels_lookup['ctx-rh-lateralorbitofrontal'],
    labels_lookup['ctx-rh-lingual'],
    labels_lookup['ctx-rh-medialorbitofrontal'],
    labels_lookup['ctx-rh-middletemporal'],
    labels_lookup['ctx-rh-parahippocampal'],
    labels_lookup['ctx-rh-paracentral'],
    labels_lookup['ctx-rh-parsopercularis'],
    labels_lookup['ctx-rh-parsorbitalis'],
    labels_lookup['ctx-rh-parstriangularis'],
    labels_lookup['ctx-rh-pericalcarine'],
    labels_lookup['ctx-rh-postcentral'],
    labels_lookup['ctx-rh-posteriorcingulate'],
    labels_lookup['ctx-rh-precentral'],
    labels_lookup['ctx-rh-precuneus'],
    labels_lookup['ctx-rh-rostralanteriorcingulate'],
    labels_lookup['ctx-rh-rostralmiddlefrontal'],
    labels_lookup['ctx-rh-superiorfrontal'],
    labels_lookup['ctx-rh-superiorparietal'],
    labels_lookup['ctx-rh-superiortemporal'],
    labels_lookup['ctx-rh-supramarginal'],
    labels_lookup['ctx-rh-transversetemporal'],
    labels_lookup['ctx-rh-insula']
]

# Subcortical Gray Matter Labels
subcortical_gray_matter_labels = [
    labels_lookup['Left-Cerebellum-Cortex'],
    labels_lookup['Right-Cerebellum-Cortex'],
    labels_lookup['Left-Hippocampus'],
    labels_lookup['Right-Hippocampus'],
    labels_lookup['Left-Amygdala'],
    labels_lookup['Right-Amygdala'],
    labels_lookup['Left-Accumbens-area'],
    labels_lookup['Right-Accumbens-area'],
    labels_lookup['Left-Thalamus-Proper'],
    labels_lookup['Right-Thalamus-Proper']
]


def plot_predictions_with_masks(image, wm_mask, cortical_gm_mask, subcortical_gm_mask, image_directory):
    n_slices = image.shape[2]
    n_cols = 5
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6))

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

    plt.tight_layout()
    os.makedirs(os.path.join(image_directory, 'AI', 'Segmentation'), exist_ok=True)
    plt.savefig(os.path.join(image_directory, 'AI', 'Segmentation', 'T2_WM_GM_masks.png'))
    plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
    close_plot_after_delay(3, fig)
    plt.show()

def segmentation(fastsurfer_path, seg_mgz_path, t1_path, output_dir, apple_metal=True):
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
                f"--sid segmentation "
                f"--sd {output_dir} --no_cereb"
            )
        else:
            command = (
                f"{fastsurfer_path} --seg_only "
                f"--t1 {t1_path} "
                f"--sid segmentation "
                f"--sd {output_dir} --no_cereb"
            )
        subprocess.run(command, shell=True)
    else:
        print("Segmentation file already exists, skipping FastSurfer segmentation.")

    aseg_mgz_path = seg_mgz_path

    # Convert aseg.mgz to aseg.nii if needed
    aseg_nii_path = aseg_mgz_path.replace('.mgz', '.nii')
    if not os.path.exists(aseg_nii_path):
        print(f"Converting {aseg_mgz_path} to {aseg_nii_path}...")
        subprocess.run(['mri_convert', aseg_mgz_path, aseg_nii_path])
    else:
        print(f"{aseg_nii_path} already exists, skipping conversion.")

    # Generate masks using mri_binarize with specific labels
    cortical_gm_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'cortical_gm.nii')
    subcortical_gm_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'subcortical_gm.nii')
    wm_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'wm.nii')

    # Create masks using the label lists
    labels_str = lambda labels: ' '.join(map(str, labels))

    # Cortical GM Mask
    if not os.path.exists(cortical_gm_mask_path):
        cortical_labels = labels_str(cortical_gray_matter_labels)
        cortical_gm_command = f"mri_binarize --i {aseg_nii_path} --match {cortical_labels} --o {cortical_gm_mask_path}"
        subprocess.run(cortical_gm_command, shell=True)
    else:
        print("Cortical GM mask already exists, skipping mri_binarize for cortical GM.")

    # Subcortical GM Mask
    if not os.path.exists(subcortical_gm_mask_path):
        subcortical_labels = labels_str(subcortical_gray_matter_labels)
        subcortical_gm_command = f"mri_binarize --i {aseg_nii_path} --match {subcortical_labels} --o {subcortical_gm_mask_path}"
        subprocess.run(subcortical_gm_command, shell=True)
    else:
        print("Subcortical GM mask already exists, skipping mri_binarize for subcortical GM.")

    # White Matter Mask
    if not os.path.exists(wm_mask_path):
        wm_labels = labels_str(white_matter_labels)
        wm_command = f"mri_binarize --i {aseg_nii_path} --match {wm_labels} --o {wm_mask_path}"
        subprocess.run(wm_command, shell=True)
    else:
        print("WM mask already exists, skipping mri_binarize for WM.")

def coregistration(seg_mgz_path, dce_path, t2_path, white_matter_labels, cortical_gm_labels, subcortical_gm_labels):
    import os
    import subprocess
    import nibabel as nib
    import numpy as np

    # Step 1: Convert segmentation file from .mgz to .nii format
    seg_nii_path = seg_mgz_path.replace('.mgz', '.nii')
    if not os.path.exists(seg_nii_path):
        print(f"Converting {seg_mgz_path} to {seg_nii_path}...")
        subprocess.run(['mri_convert', seg_mgz_path, seg_nii_path])
    else:
        print(f"{seg_nii_path} already exists, skipping conversion.")

    # Step 2: Align the segmentation image to the DCE space
    seg_in_dce_path = seg_nii_path.replace('.nii', '_in_DCE.nii.gz')
    if not os.path.exists(seg_in_dce_path):
        flirt_cmd_dce = [
            'flirt', '-in', seg_nii_path, '-ref', dce_path,
            '-out', seg_in_dce_path,
            '-interp', 'nearestneighbour',
            '-omat', seg_nii_path.replace('.nii', '_to_DCE.mat'),
            '-dof', '6'
        ]
        print(f"Running FLIRT command for DCE: {' '.join(flirt_cmd_dce)}")
        subprocess.run(flirt_cmd_dce)
    else:
        print(f"Aligned segmentation to DCE already exists at {seg_in_dce_path}.")

    # Step 3: Align the segmentation image to the T2 space
    seg_in_t2_path = seg_nii_path.replace('.nii', '_in_T2.nii.gz')
    if not os.path.exists(seg_in_t2_path):
        flirt_cmd_t2 = [
            'flirt', '-in', seg_nii_path, '-ref', t2_path,
            '-out', seg_in_t2_path,
            '-interp', 'nearestneighbour',
            '-omat', seg_nii_path.replace('.nii', '_to_T2.mat'),
            '-dof', '6'
        ]
        print(f"Running FLIRT command for T2: {' '.join(flirt_cmd_t2)}")
        subprocess.run(flirt_cmd_t2)
    else:
        print(f"Aligned segmentation to T2 already exists at {seg_in_t2_path}.")

    # Step 4: Load the aligned segmentation images
    print(f"Loading aligned segmentation images from {seg_in_dce_path} and {seg_in_t2_path}")
    seg_in_dce_img = nib.load(seg_in_dce_path).get_fdata()
    seg_in_t2_img = nib.load(seg_in_t2_path).get_fdata()

    # Step 5: Create masks from the aligned segmentation images
    wm_mask_dce = np.isin(seg_in_dce_img, white_matter_labels)
    cortical_gm_mask_dce = np.isin(seg_in_dce_img, cortical_gm_labels)
    subcortical_gm_mask_dce = np.isin(seg_in_dce_img, subcortical_gm_labels)

    wm_mask_t2 = np.isin(seg_in_t2_img, white_matter_labels)
    cortical_gm_mask_t2 = np.isin(seg_in_t2_img, cortical_gm_labels)
    subcortical_gm_mask_t2 = np.isin(seg_in_t2_img, subcortical_gm_labels)

    return wm_mask_t2, wm_mask_dce, cortical_gm_mask_t2, cortical_gm_mask_dce, subcortical_gm_mask_t2, subcortical_gm_mask_dce


def plot_dce_grid(dce_image, wm_mask_downsampled, cortical_gm_mask_downsampled, subcortical_gm_mask_downsampled):
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
    plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
    close_plot_after_delay(3, fig)
    plt.show()






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
    
    calc_max = np.max(x_patlak)
    calc_min = calc_max / 3
    idx = np.where((x_patlak >= calc_min) & (x_patlak <= calc_max))
    x, y = x_patlak[idx], y_patlak[idx]
    
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


from matplotlib.gridspec import GridSpec
from skimage.transform import resize

from matplotlib.gridspec import GridSpec
from skimage.transform import resize
from skimage.exposure import rescale_intensity

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

    # Initialize color overlays
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


    # Set alpha channel
    color_overlay_dce[..., 3] = (cortical_gm_mask_resized_dce + subcortical_gm_mask_resized_dce + wm_mask_resized_dce) * 0.5  # Adjust alpha

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
    if boundary_ctc is not None:
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

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.tight_layout()
    plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
    close_plot_after_delay(1, fig)
    plt.show()
























from scipy.ndimage import binary_dilation
def compute_and_plot_ctcs_median(data_4d, t2_img, wm_mask_t2, cortical_gm_mask_t2, subcortical_gm_mask_t2, 
                                 wm_mask_dce, cortical_gm_mask_dce, subcortical_gm_mask_dce, 
                                 T1_matrix, M0_matrix, analysis_directory, time_points_s, image_directory, boundary=False):
    
    n_slices = t2_img.shape[2]
    
    all_patlak_data = []
    Ki_wm_list = []
    Ki_cortical_gm_list = []
    Ki_subcortical_gm_list = []
    Ki_boundary_list = []

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

        # Find voxel indices for white and grey matter in the slice
        wm_indices = np.argwhere(wm_slice_dce)
        cortical_gm_indices = np.argwhere(cortical_gm_slice_dce)
        subcortical_gm_indices = np.argwhere(subcortical_gm_slice_dce)

        # Compute CTCs for white matter
        wm_ctcs = []
        for (x, y) in wm_indices:
            voxel_time_course = data_4d[x, y, i, :]
            T1 = T1_matrix[x, y, i]
            M0 = M0_matrix[x, y, i]
            C_t_0 = compute_CTC(voxel_time_course, T1, m0=M0)
            baseline_point = find_baseline_point_advanced(C_t_0)
            C_t = custom_shifter(C_t_0, baseline_point)
            wm_ctcs.append(C_t)
        
        avg_wm_ctc = np.median(wm_ctcs, axis=0) if wm_ctcs else np.array([])

        # Compute CTCs for cortical gray matter
        cortical_gm_ctcs = []
        for (x, y) in cortical_gm_indices:
            voxel_time_course = data_4d[x, y, i, :]
            T1 = T1_matrix[x, y, i]
            M0 = M0_matrix[x, y, i]
            C_t_0 = compute_CTC(voxel_time_course, T1, m0=M0)
            baseline_point = find_baseline_point_advanced(C_t_0)
            C_t = custom_shifter(C_t_0, baseline_point)
            cortical_gm_ctcs.append(C_t)
        
        avg_cortical_gm_ctc = np.median(cortical_gm_ctcs, axis=0) if cortical_gm_ctcs else np.array([])

        # Compute CTCs for subcortical gray matter
        subcortical_gm_ctcs = []
        for (x, y) in subcortical_gm_indices:
            voxel_time_course = data_4d[x, y, i, :]
            T1 = T1_matrix[x, y, i]
            M0 = M0_matrix[x, y, i]
            C_t_0 = compute_CTC(voxel_time_course, T1, m0=M0)
            baseline_point = find_baseline_point_advanced(C_t_0)
            C_t = custom_shifter(C_t_0, baseline_point)
            subcortical_gm_ctcs.append(C_t)
        
        avg_subcortical_gm_ctc = np.median(subcortical_gm_ctcs, axis=0) if subcortical_gm_ctcs else np.array([])

        # Compute CTCs for boundary if required
        if boundary:
            boundary_ctcs = []
            for (x, y) in boundary_indices:
                voxel_time_course = data_4d[x, y, i, :]
                T1 = T1_matrix[x, y, i]
                M0 = M0_matrix[x, y, i]
                C_t_0 = compute_CTC(voxel_time_course, T1, m0=M0)
                baseline_point = find_baseline_point_advanced(C_t_0)
                C_t = custom_shifter(C_t_0, baseline_point)
                boundary_ctcs.append(C_t)
            
            avg_boundary_ctc = np.median(boundary_ctcs, axis=0) if boundary_ctcs else np.array([])
            np.save(os.path.join(analysis_directory, 'CTC Data', 'Tissue', 'AI', f'bo_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_boundary_ctc)
        else:
            avg_boundary_ctc = np.array([])

        # Save the tissue concentration curves as .npy files
        save_dir_ctc = os.path.join(analysis_directory, 'CTC Data', 'Tissue', 'AI')
        os.makedirs(save_dir_ctc, exist_ok=True)

        np.save(os.path.join(save_dir_ctc, f'wm_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_wm_ctc)
        np.save(os.path.join(save_dir_ctc, f'cortical_gm_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_cortical_gm_ctc)
        np.save(os.path.join(save_dir_ctc, f'subcortical_gm_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_subcortical_gm_ctc)
        
        # Locate the only .npy file in the Max folder
        max_folder = os.path.join(analysis_directory, 'TSCC Data', 'Max')
        npy_files = [f for f in os.listdir(max_folder) if f.endswith('.npy')]

        if len(npy_files) != 1:
            raise ValueError(f"Expected exactly one .npy file in {max_folder}, but found {len(npy_files)}.")

        # Load the .npy file as C_a
        ca_file = npy_files[0]
        C_a = np.load(os.path.join(max_folder, ca_file))

        # Handle cases where CTCs might be empty
        C_t_wm = avg_wm_ctc[:len(C_a)] if avg_wm_ctc.size > 0 else np.array([])
        C_t_cortical_gm = avg_cortical_gm_ctc[:len(C_a)] if avg_cortical_gm_ctc.size > 0 else np.array([])
        C_t_subcortical_gm = avg_subcortical_gm_ctc[:len(C_a)] if avg_subcortical_gm_ctc.size > 0 else np.array([])
        
        time_points = time_points_s[:len(C_a)]

        # Handle white matter
        if avg_wm_ctc.size > 0:
            C_t_wm = avg_wm_ctc[:len(C_a)]
            Ki_wm, lambda_wm, SD_Ki_wm, x_patlak_wm, y_patlak_wm, included_wm = patlak_analysis_plotting(C_t_wm, C_a, time_points)
        else:
            Ki_wm = np.nan
            lambda_wm = np.nan
            SD_Ki_wm = np.nan
            x_patlak_wm = np.array([])
            y_patlak_wm = np.array([])
            included_wm = np.array([], dtype=bool)

        # Handle cortical gray matter
        if avg_cortical_gm_ctc.size > 0:
            C_t_cortical_gm = avg_cortical_gm_ctc[:len(C_a)]
            Ki_cortical_gm, lambda_cortical_gm, SD_Ki_cortical_gm, x_patlak_cortical_gm, y_patlak_cortical_gm, included_cortical_gm = patlak_analysis_plotting(C_t_cortical_gm, C_a, time_points)
        else:
            Ki_cortical_gm = np.nan
            lambda_cortical_gm = np.nan
            SD_Ki_cortical_gm = np.nan
            x_patlak_cortical_gm = np.array([])
            y_patlak_cortical_gm = np.array([])
            included_cortical_gm = np.array([], dtype=bool)

        # Handle subcortical gray matter
        if avg_subcortical_gm_ctc.size > 0:
            C_t_subcortical_gm = avg_subcortical_gm_ctc[:len(C_a)]
            Ki_subcortical_gm, lambda_subcortical_gm, SD_Ki_subcortical_gm, x_patlak_subcortical_gm, y_patlak_subcortical_gm, included_subcortical_gm = patlak_analysis_plotting(C_t_subcortical_gm, C_a, time_points)
        else:
            Ki_subcortical_gm = np.nan
            lambda_subcortical_gm = np.nan
            SD_Ki_subcortical_gm = np.nan
            x_patlak_subcortical_gm = np.array([])
            y_patlak_subcortical_gm = np.array([])
            included_subcortical_gm = np.array([], dtype=bool)

        # Handle boundary if required
        if boundary and avg_boundary_ctc.size > 0:
            C_t_boundary = avg_boundary_ctc[:len(C_a)]
            Ki_boundary, lambda_boundary, SD_Ki_boundary, x_patlak_boundary, y_patlak_boundary, included_boundary = patlak_analysis_plotting(C_t_boundary, C_a, time_points)
        else:
            Ki_boundary = np.nan
            lambda_boundary = np.nan
            SD_Ki_boundary = np.nan
            x_patlak_boundary = np.array([])
            y_patlak_boundary = np.array([])
            included_boundary = np.array([], dtype=bool)

        
        Ki_wm, lambda_wm, SD_Ki_wm, x_patlak_wm, y_patlak_wm, included_wm = patlak_analysis_plotting(C_t_wm, C_a, time_points)
        Ki_cortical_gm, lambda_cortical_gm, SD_Ki_cortical_gm, x_patlak_cortical_gm, y_patlak_cortical_gm, included_cortical_gm = patlak_analysis_plotting(C_t_cortical_gm, C_a, time_points)
        Ki_subcortical_gm, lambda_subcortical_gm, SD_Ki_subcortical_gm, x_patlak_subcortical_gm, y_patlak_subcortical_gm, included_subcortical_gm = patlak_analysis_plotting(C_t_subcortical_gm, C_a, time_points)

        # If boundary is being calculated, perform Patlak analysis for it
        if boundary and avg_boundary_ctc.size > 0:
            C_t_boundary = avg_boundary_ctc[:len(C_a)]
            Ki_boundary, lambda_boundary, SD_Ki_boundary, x_patlak_boundary, y_patlak_boundary, included_boundary = patlak_analysis_plotting(C_t_boundary, C_a, time_points)
            Ki_boundary_list.append(Ki_boundary)
        else:
            Ki_boundary, x_patlak_boundary, y_patlak_boundary, included_boundary = None, None, None, None

        Ki_wm_list.append(Ki_wm)
        Ki_cortical_gm_list.append(Ki_cortical_gm)
        Ki_subcortical_gm_list.append(Ki_subcortical_gm)
        
        plot_ctcs_and_patlak(
            t2_img[:, :, i], data_4d[:,:,i,20],
            wm_slice_t2, cortical_gm_slice_t2, subcortical_gm_slice_t2,  # Pass T2 masks
            wm_slice_dce, cortical_gm_slice_dce, subcortical_gm_slice_dce,  # Pass DCE masks
            avg_wm_ctc, avg_cortical_gm_ctc, avg_subcortical_gm_ctc,
            x_patlak_wm, y_patlak_wm, 
            Ki_wm, lambda_wm, x_patlak_cortical_gm, y_patlak_cortical_gm, 
            Ki_cortical_gm, lambda_cortical_gm, x_patlak_subcortical_gm, y_patlak_subcortical_gm, 
            Ki_subcortical_gm, lambda_subcortical_gm, slice_idx=i+1, 
            save_path=os.path.join(image_directory, 'AI', 'Tissue functions',f"AI_Tissue_slice_{i+1}_segmented_median.png"),
            boundary_mask=boundary_mask,  # Pass the boundary mask for visualization
            boundary_ctc=avg_boundary_ctc, 
            x_patlak_boundary=x_patlak_boundary, y_patlak_boundary=y_patlak_boundary,
            Ki_boundary=Ki_boundary, lambda_boundary=lambda_boundary, included_wm=included_wm,
            included_cortical_gm=included_cortical_gm, included_subcortical_gm=included_subcortical_gm, included_boundary=included_boundary
        )

        patlak_data = {
            'slice': i + 1,
            'white_matter_median': {
                'Ki': Ki_wm,
                'SD_Ki': SD_Ki_wm,
                'lambda': lambda_wm
            },
            'cortical_gray_matter_median': {
                'Ki': Ki_cortical_gm,
                'SD_Ki': SD_Ki_cortical_gm,
                'lambda': lambda_cortical_gm
            },
            'subcortical_gray_matter_median': {
                'Ki': Ki_subcortical_gm,
                'SD_Ki': SD_Ki_subcortical_gm,
                'lambda': lambda_subcortical_gm
            }
        }

        if boundary and avg_boundary_ctc.size > 0:
            patlak_data['boundary_median'] = {
                'Ki': Ki_boundary,
                'SD_Ki': SD_Ki_boundary,
                'lambda': lambda_boundary
            }

        all_patlak_data.append(patlak_data)

        json_file_path = os.path.join(analysis_directory, "AI_values_median.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(all_patlak_data, json_file, indent=4)

    # Plot Ki values as a function of slice number
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_slices + 1), Ki_wm_list, label='White Matter Ki', marker='o')
    plt.plot(range(1, n_slices + 1), Ki_cortical_gm_list, label='Cortical Gray Matter Ki', marker='o')
    plt.plot(range(1, n_slices + 1), Ki_subcortical_gm_list, label='Subcortical Gray Matter Ki', marker='o')
    if boundary:
        plt.plot(range(1, n_slices + 1), Ki_boundary_list, label='Boundary Ki', marker='o')
    plt.xlabel('Slice Number')
    plt.ylabel('K_i')
    plt.title('K_i values for White Matter, Cortical Gray Matter, Subcortical Gray Matter, and Boundary across Slices')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(image_directory, 'AI', 'Tissue functions', 'Ki_vs_slice_median.png'))
    close_plot_after_delay_plt(3)
    plt.show()


def tissue_function_AI(analysis_directory, nifti_directory, image_directory, filenames, parameters):
    t1_3D_filename, axial_t1_3D_filename, t2_3D_filename, axial_t2_3D_filename, \
        flair_3D_filename, axial_flair_3D_filename, axial_t2_2D_filename, dce_filename = filenames

    IsVFA, IsIR, apple_metal, boundary = parameters

    fastsurfer_path = '/Users/edt/FastSurfer/run_fastsurfer.sh'
    t1_path = os.path.join(nifti_directory, t1_3D_filename)
    seg_dir = os.path.join(nifti_directory, 'segmentation')
    seg_mgz_path = os.path.join(seg_dir, 'mri', 'aparc.DKTatlas+aseg.deep.mgz')
    t2_path = os.path.join(nifti_directory, axial_t2_2D_filename)
    dce_path = os.path.join(nifti_directory, dce_filename)

    # Ensure segmentation directory exists
    os.makedirs(seg_dir, exist_ok=True)

    # Run segmentation and create masks
    segmentation(fastsurfer_path, seg_mgz_path, t1_path, seg_dir, apple_metal)

    # Paths to masks in the same directory as aparc.DKTatlas+aseg.deep.mgz
    mask_dir = os.path.dirname(seg_mgz_path)
    cortical_gm_mask_path = os.path.join(mask_dir, 'cortical_gm.nii')
    subcortical_gm_mask_path = os.path.join(mask_dir, 'subcortical_gm.nii')
    wm_mask_path = os.path.join(mask_dir, 'wm.nii')

    print('[!] Coregistering GM/WM masks onto T2 and DCE space')
    wm_mask_t2, wm_mask_dce, cortical_gm_mask_t2, cortical_gm_mask_dce, subcortical_gm_mask_t2, subcortical_gm_mask_dce = coregistration(
        seg_mgz_path=seg_mgz_path,
        dce_path=dce_path,
        t2_path=t2_path,
        white_matter_labels=white_matter_labels,
        cortical_gm_labels=cortical_gray_matter_labels,
        subcortical_gm_labels=subcortical_gray_matter_labels
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
        T1_matrix, M0_matrix, analysis_directory, time_points_s, image_directory, boundary=boundary
    )