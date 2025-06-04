import numpy as np
import os
import sys
import re
import time 

def get_dce_filename(primary, fallback, nifti_directory):
    if os.path.exists(os.path.join(nifti_directory, fallback)):
        return fallback
    if os.path.exists(os.path.join(nifti_directory, primary)):
        return primary
    else:
        return None 

# Global parameters: 

def global_parameters():
    IsVFA = False #Variable flip angle for the T1/M0 fit
    IsIR = True #Inversion recovery method
    apple_metal = True # Enable if running on apple M1/M2/M3...
    boundary = True #compute boundary mask from GM/WM masks and plot/compute patlak values alongside wm/gm
    RERUN_SEGMENTATION = False  # Force rerun of FastSurfer segmentation
    SEGMENTATION_METHOD = "fastsurfer"  # Choose segmentation tool (default FastSurfer)
    return (
        IsVFA,
        IsIR,
        apple_metal,
        boundary,
        RERUN_SEGMENTATION,
        SEGMENTATION_METHOD,
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
    dce_filename = get_dce_filename(dce_filename_primary, dce_filename_fallback, nifti_directory)
    print(dce_filename)
    return t1_3D_filename, axial_t1_3D_filename, t2_3D_filename, axial_t2_3D_filename, flair_3D_filename, axial_flair_3D_filename, axial_t2_2D_filename, dce_filename

# Separate filenames for control datasets used by the AI pipeline
def control_filenames(nifti_directory):
    refresh_nifti_directory(nifti_directory)
    t1_3D_filename = 'Ax_VWIPT1W_3D_TFE.nii'
    axial_t1_3D_filename = r'control_ax([-_ ])?vwipcs_t1w_3d_tfe_32channel\.nii'

    t2_3D_filename = r'WIPcs_3D_Brain_VIEW_T2_32chSHC.nii'
    axial_t2_3D_filename = r'control_ax([-_ ])?vwipcs_3D_Brain_VIEW_T2_32chSHC\.nii'

    flair_3D_filename = 'WIPcs_3D_Brain_VIEW_FLAIR_SHC.nii'
    axial_flair_3D_filename = 'WIPcs_3D_Brain_VIEW_FLAIR_SHC.nii'

    axial_t2_2D_filename = 'WIPAxT2TSEmatrix.nii'

    dce_filename_primary = 'WIPhperf120long.nii'
    dce_filename_fallback = 'WIPDelRec-hperf120long.nii'
    dce_filename = get_dce_filename(dce_filename_primary, dce_filename_fallback, nifti_directory)
    return (
        t1_3D_filename,
        axial_t1_3D_filename,
        t2_3D_filename,
        axial_t2_3D_filename,
        flair_3D_filename,
        axial_flair_3D_filename,
        axial_t2_2D_filename,
        dce_filename,
    )

