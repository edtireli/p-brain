import numpy as np
import os
import sys
import re
import time 

def _get_first_existing_file(filenames, nifti_directory):
    """Return the first filename from ``filenames`` that exists."""

    for filename in filenames:
        if not filename:
            continue
        if os.path.exists(os.path.join(nifti_directory, filename)):
            return filename
    return None


def get_dce_filename(primary, fallback, nifti_directory):
    return _get_first_existing_file((fallback, primary), nifti_directory)


def get_diffusion_filename(candidates, nifti_directory):
    """Return the configured diffusion volume if present in ``nifti_directory``."""

    if isinstance(candidates, str):
        candidates = (candidates,)
    return _get_first_existing_file(candidates, nifti_directory)

# Global parameters: 

def global_parameters():
    IsVFA = False #Variable flip angle for the T1/M0 fit
    IsIR = True #Inversion recovery method
    apple_metal = True # Enable if running on apple M1/M2/M3...
    boundary = True #compute boundary mask from GM/WM masks and plot/compute patlak values alongside wm/gm
    RERUN_SEGMENTATION = False  # Force rerun of FastSurfer segmentation
    SEGMENTATION_METHOD = "fastsurfer"  # Choose segmentation tool (default FastSurfer)
    COMPUTE_FA = False  # Compute fractional anisotropy from DWI
    return (
        IsVFA,
        IsIR,
        apple_metal,
        boundary,
        RERUN_SEGMENTATION,
        SEGMENTATION_METHOD,
        COMPUTE_FA,
    )

def refresh_nifti_directory(nifti_directory):
    return os.listdir(nifti_directory)

# Global filenames:
def global_filenames(nifti_directory):
    refresh_nifti_directory(nifti_directory)
    t1_3D_filename = 'WIPcs_T1W_3D_TFE_32channel.nii'
    axial_t1_3D_filename = r'ax([-_ ])?vwipcs_t1w_3d_tfe_32channel\.nii'

    t2_3D_filename = r'WIPcs_3D_Brain_VIEW_T2_32chSHC.nii'
    axial_t2_3D_filename = r'ax([-_ ])?vwipcs_3D_Brain_VIEW_T2_32chSHC\.nii'

    flair_3D_filename =  'WIPcs_3D_Brain_VIEW_FLAIR_SHC.nii'
    axial_flair_3D_filename = r'ax([-_ ])?VWIPcs_3D_Brain_VIEW_FLAIR_SHC\.nii'

    axial_t2_2D_filename = 'WIPAxT2TSEmatrix.nii'

    dce_filename_primary = 'WIPhperf120long.nii'
    dce_filename_fallback = 'WIPDelRec-hperf120long.nii'
    diffusion_filenames = (
        'WIPDTI_RSI_P.nii',
        'WIPDTI_RSI_P.nii.gz',
        'WIPDTI_RSI_A.nii',
        'WIPDTI_RSI_A.nii.gz',
        'WIPDWI_RSI_P.nii',
        'WIPDWI_RSI_P.nii.gz',
    )

    diffusion_filename = get_diffusion_filename(diffusion_filenames, nifti_directory)
    dce_filename = get_dce_filename(dce_filename_primary, dce_filename_fallback, nifti_directory)

    return (
        t1_3D_filename,
        axial_t1_3D_filename,
        t2_3D_filename,
        axial_t2_3D_filename,
        flair_3D_filename,
        axial_flair_3D_filename,
        axial_t2_2D_filename,
        diffusion_filename,
        dce_filename,
    )

# Separate filenames for control datasets used by the AI pipeline
def control_filenames(nifti_directory):
    refresh_nifti_directory(nifti_directory)
    t1_3D_filename = 'WIPT1W_3D_TFE.nii'
    axial_t1_3D_filename = r'ax([-_ ])?vwipcs_t1w_3d_tfe_32channel\.nii'

    t2_3D_filename = r'WIPcs_3D_Brain_VIEW_T2_32chSHC.nii'
    axial_t2_3D_filename = r'ax([-_ ])?vwipcs_3D_Brain_VIEW_T2_32chSHC\.nii'

    flair_3D_filename = 'WIPcs_3D_Brain_VIEW_FLAIR_SHC.nii'
    axial_flair_3D_filename = 'Ax_VWIPcs_3D_Brain_VIEW_FLAIR_SHC.nii'

    axial_t2_2D_filename = 'WIPAxT2TSEmatrix.nii'

    dce_filename_primary = 'WIPhperf120long.nii'
    dce_filename_fallback = 'WIPDelRec-hperf120long.nii'
    diffusion_filenames = (
        'WIPDTI_RSI_P.nii',
        'WIPDTI_RSI_P.nii.gz',
        'WIPDTI_RSI_A.nii',
        'WIPDTI_RSI_A.nii.gz',
        'WIPDWI_RSI_P.nii',
        'WIPDWI_RSI_P.nii.gz',
    )

    diffusion_filename = get_diffusion_filename(diffusion_filenames, nifti_directory)
    dce_filename = get_dce_filename(dce_filename_primary, dce_filename_fallback, nifti_directory)

    return (
        t1_3D_filename,
        axial_t1_3D_filename,
        t2_3D_filename,
        axial_t2_3D_filename,
        flair_3D_filename,
        axial_flair_3D_filename,
        axial_t2_2D_filename,
        diffusion_filename,
        dce_filename,
    )

