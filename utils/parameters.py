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


# Diffusion acquisition configuration -------------------------------------------------

_LEGACY_DIFFUSION_FILENAMES = (
    "WIPDTI_RSI_P.nii",
    "WIPDTI_RSI_P.nii.gz",
    "WIPDTI_RSI_A.nii",
    "WIPDTI_RSI_A.nii.gz",
    "WIPDWI_RSI_P.nii",
    "WIPDWI_RSI_P.nii.gz",
)


DIFFUSION_FILE_GROUPS: dict[str, tuple[str, ...]] = {
    "dti": _LEGACY_DIFFUSION_FILENAMES,
    # Drift-corrected registered diffusion volume (preferred when available)
    "dwi_reg": (
        "Reg-DWInySENSE.nii",
        "Reg-DWInySENSE.nii.gz",
        "Reg-DWInySENSE_ADC.nii",
        "Reg-DWInySENSE_ADC.nii.gz",
    ),
    # Legacy isoDWI volumes retained for compatibility
    "dwi_iso": (
        "isoDWIb-1000.nii",
        "isoDWIb-1000.nii.gz",
    ),
}


_DEFAULT_DIFFUSION_PRIORITY = (
    "dti",
    "dwi_reg",
    "dwi_iso",
    "dwi",
)

_SUPPORTED_DIFFUSION_MODELS = {"DTI", "CSD"}

_DIFFUSION_MODEL_BY_GROUP = {
    "dti": "DTI",
    "dwi": "CSD",
    "dwi_reg": "CSD",
    "dwi_iso": "CSD",
}


def _parse_priority_list(raw: str) -> tuple[str, ...]:
    entries = []
    for token in raw.split(","):
        cleaned = token.strip().lower()
        if cleaned:
            entries.append(cleaned)
    return tuple(entries)


def diffusion_file_priority() -> tuple[str, ...]:
    env_value = os.environ.get("P_BRAIN_DIFFUSION_PRIORITY", "")
    if env_value:
        parsed = _parse_priority_list(env_value)
        if parsed:
            return parsed
    return _DEFAULT_DIFFUSION_PRIORITY


def diffusion_file_groups() -> dict[str, tuple[str, ...]]:
    return DIFFUSION_FILE_GROUPS


def diffusion_model_map() -> dict[str, str]:
    model_map: dict[str, str] = {}
    for group, default_model in _DIFFUSION_MODEL_BY_GROUP.items():
        env_key = f"P_BRAIN_DIFFUSION_MODEL_{group.upper()}"
        override = os.environ.get(env_key)
        if override:
            override_value = override.strip().upper()
            if override_value in _SUPPORTED_DIFFUSION_MODELS:
                model_map[group] = override_value
                continue
        model_map[group] = default_model
    return model_map


def ordered_diffusion_filenames() -> tuple[str, ...]:
    seen = set()
    ordered: list[str] = []
    groups = diffusion_file_priority()
    for group in groups:
        for pattern in DIFFUSION_FILE_GROUPS.get(group, ()):  # type: ignore[index]
            normalized = pattern.strip()
            if not normalized:
                continue
            if normalized in seen:
                continue
            ordered.append(normalized)
            seen.add(normalized)
    if not ordered:
        ordered.extend(_LEGACY_DIFFUSION_FILENAMES)
    return tuple(ordered)

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
    diffusion_candidates = ordered_diffusion_filenames()

    diffusion_filename = get_diffusion_filename(diffusion_candidates, nifti_directory)
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
    diffusion_candidates = ordered_diffusion_filenames()

    diffusion_filename = get_diffusion_filename(diffusion_candidates, nifti_directory)
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

