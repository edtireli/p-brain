import os
import sys
import json
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np
import matplotlib

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - fallback for very old Python versions
    import importlib_metadata  # type: ignore
_MPL_ENV_BACKEND = os.environ.get("P_BRAIN_MPL_BACKEND") or os.environ.get("MPLBACKEND")
if _MPL_ENV_BACKEND:
    matplotlib.use(_MPL_ENV_BACKEND)
else:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")

# Global toggle for analysing control data. The flag can also be
# overridden by setting the environment variable ``PBRAIN_CONTROLS`` to
# ``1``/``true``.
CONTROLS = 0#os.environ.get("PBRAIN_CONTROLS", "False").lower() in ("1", "true", "yes")
CONTROL_FLAG_FILENAME = "control.json"

# Toggle multiprocessing for compute-heavy routines
MULTIPROCESSING = True
# Default number of CPU cores used when multiprocessing is enabled
NUMBER_OF_CORES = int(os.environ.get("P_BRAIN_CORES", 4))

# Select the recovery equation to use when fitting T1/M0 from inversion or
# saturation recovery acquisitions. Accepts ``inversion`` (default) or
# ``saturation``.
T1_RECOVERY_MODEL = os.environ.get("P_BRAIN_T1_RECOVERY_MODEL", "inversion").strip().lower()
if T1_RECOVERY_MODEL not in {"inversion", "saturation"}:
    T1_RECOVERY_MODEL = "inversion"

# Select kinetic modelling strategy. Valid options are ``patlak``,
# ``two_compartment`` (regularised two-compartment fit) or ``both`` to execute the two
# approaches sequentially.  When the environment variable
# ``P_BRAIN_MODEL`` is not provided the default is ``both``.
KINETIC_MODEL = os.environ.get("P_BRAIN_MODEL", "both")

# Optionally use the Patlak permeability (Ki) as the initial guess for
# Ktrans in the two-compartment optimisation.
TWO_COMPARTMENT_INIT_FROM_PATLAK = False

# Control writing of additional voxelwise parametric maps derived from the
# deconvolution residue.
WRITE_MTT = True
WRITE_CTH = True
CTH_MTT_METHOD = os.environ.get("P_BRAIN_CTH_MTT_METHOD", "tikhonov")
CTH_MTT_GAMMA_VOXELWISE = os.environ.get(
    "P_BRAIN_CTH_MTT_GAMMA_VOXELWISE", "0"
).lower() in {"1", "true", "yes"}

# Regularisation strength for the two-compartment model
TIKHONOV_LAMBDA = float(os.environ.get("P_BRAIN_LAMBDA", 0.5))

# Physiological constants for residue-based perfusion metrics
TISSUE_DENSITY = float(os.environ.get("P_BRAIN_TISSUE_DENSITY", 1.04))
HEMATOCRIT = float(os.environ.get("P_BRAIN_HEMATOCRIT", 0.42))
PLASMA_DERIVED_AIF = os.environ.get("P_BRAIN_PLASMA_AIF", "0").lower() in {"1", "true", "yes"}

# Optional voxelwise delay alignment prior to deconvolution
ALIGN_AIF_BY_XCORR = os.environ.get("P_BRAIN_ALIGN_AIF", "0").lower() in {"1", "true", "yes"}
ALIGN_AIF_MAX_SHIFT_S = float(os.environ.get("P_BRAIN_ALIGN_AIF_MAX_SHIFT", 4.0))

# Select which input function to use when performing kinetic modelling. The
# default ``SSS`` corresponds to the legacy behaviour where the superior
# sagittal sinus curve is time shifted and rescaled to match the arterial
# signal.  Setting ``P_BRAIN_INPUT_FUNCTION`` to ``RICA`` uses the pure
# arterial concentration curve from the right internal carotid artery instead.
INPUT_FUNCTION_SOURCE = os.environ.get("P_BRAIN_INPUT_FUNCTION", "SSS").strip().upper()
if INPUT_FUNCTION_SOURCE not in {"SSS", "RICA"}:
    INPUT_FUNCTION_SOURCE = "SSS"

# Flip angle (degrees) used by signal->concentration conversion.
# Set `P_BRAIN_FLIP_ANGLE=auto` (default) to rely on metadata.
# Set `P_BRAIN_FLIP_ANGLE=<number>` to override (e.g. 30).
FLIP_ANGLE_SETTING = (os.environ.get("P_BRAIN_FLIP_ANGLE", "auto") or "auto").strip()
_flip_angle_lower = FLIP_ANGLE_SETTING.lower()
FLIP_ANGLE_DEG = None
if _flip_angle_lower and _flip_angle_lower != "auto":
    try:
        FLIP_ANGLE_DEG = float(FLIP_ANGLE_SETTING)
    except ValueError:
        FLIP_ANGLE_SETTING = "auto"
        FLIP_ANGLE_DEG = None

# Signal-to-concentration conversion model.
# - saturation: closed-form saturation-recovery inversion (legacy p-Brain)
# - turboflash: TurboFLASH readout-train forward model (MATLAB-style), numerically inverted
CTC_MODEL = (os.environ.get("P_BRAIN_CTC_MODEL", "saturation") or "saturation").strip().lower()
if CTC_MODEL not in {"saturation", "turboflash"}:
    CTC_MODEL = "saturation"

# TurboFLASH model parameters (only used when CTC_MODEL=turboflash).
# nph is the phase-encode line index (1-based) at ky=0 (k-space center).
TURBOFLASH_NPH = int(os.environ.get("P_BRAIN_TURBO_NPH", 1))

# T1/M0 fitting input source.
# - auto: prefer inversion-recovery if present, otherwise try VFA.
# - ir: require inversion-recovery series.
# - vfa: require VFA spoiled GRE series.
# - none: skip T1/M0 fitting (outputs NaNs).
T1_FIT_MODE = (os.environ.get("P_BRAIN_T1_FIT", "auto") or "auto").strip().lower()
if T1_FIT_MODE not in {"auto", "ir", "vfa", "none"}:
    T1_FIT_MODE = "auto"

# VFA discovery glob(s) used when T1_FIT_MODE is vfa/auto.
# Comma-separated patterns relative to the NIfTI directory.
VFA_FILE_GLOB = (os.environ.get("P_BRAIN_VFA_GLOB", "*VFA*.nii*") or "*VFA*.nii*").strip()

# When enabled, pick the regularisation weight automatically using an L-curve
# search across ``AUTO_LAMBDA_CANDIDATES``.
AUTO_LAMBDA = False
AUTO_LAMBDA_CANDIDATES = np.logspace(-2, 2, 30)
# Holds the most recently chosen value when ``AUTO_LAMBDA`` is True
AUTO_LAMBDA_VALUE = None

# Number of bolus peaks expected in the acquisition.  Defaults to two but can
# be overridden via the ``P_BRAIN_NUMBER_OF_PEAKS`` environment variable.
NUMBER_OF_PEAKS = int(os.environ.get("P_BRAIN_NUMBER_OF_PEAKS", 2))

# Lower bound for the Patlak inclusion window expressed as a fraction of the
# maximum Patlak x-coordinate.  Higher values restrict the fit to the tail.
PATLAK_WINDOW_START_FRACTION = float(
    os.environ.get("P_BRAIN_PATLAK_WINDOW_START_FRACTION", 1 / 3)
)

# Minimum coefficient of determination required when choosing the Patlak
# fitting segment.  The solver progressively discards early points until this
# target is reached (or the best achievable R² is used).
PATLAK_MIN_R2 = float(os.environ.get("P_BRAIN_PATLAK_MIN_R2", 0.985))

# Number of slices to omit from the inferior (bottom) and superior (top)
# ends when computing global Ki for white matter, cortical grey matter
# and boundary tissues.  These can be overridden via the environment
# variables ``P_BRAIN_GLOBAL_KI_SKIP_BOTTOM`` and
# ``P_BRAIN_GLOBAL_KI_SKIP_TOP``.
GLOBAL_KI_SKIP_BOTTOM = int(os.environ.get("P_BRAIN_GLOBAL_KI_SKIP_BOTTOM", 2))
GLOBAL_KI_SKIP_TOP = int(os.environ.get("P_BRAIN_GLOBAL_KI_SKIP_TOP", 2))

# Paths to the neural network models used for artery and vein ROI extraction.
# These can be overridden by environment variables to use custom models.
AI_MODEL_PATHS = {
    'slice_classifier_rica': os.environ.get(
        'SLICE_CLASSIFIER_RICA_MODEL', 'AI/slice_classifier_model_rica.keras'
    ),
    'rica_roi': os.environ.get(
        'RICA_ROI_MODEL', 'AI/rica_roi_model.keras'
    ),
    'slice_classifier_ss': os.environ.get(
        'SLICE_CLASSIFIER_SS_MODEL', 'AI/ss_slice_classifier.keras'
    ),
    'ss_roi': os.environ.get(
        'SS_ROI_MODEL', 'AI/ss_roi_model.keras'
    ),
}


def _default_data_root():
    """Return the default root directory for datasets.

    Prefers the ``P_BRAIN_DATA_DIR`` environment variable and falls back to a
    ``Data`` folder next to the executable.
    """

    env_root = os.environ.get("P_BRAIN_DATA_DIR")
    if env_root:
        return os.path.abspath(env_root)
    return os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "Data")

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def setup_directories(log_number, data_root=None):
    if data_root is None:
        data_root = _default_data_root()
    else:
        data_root = os.path.abspath(data_root)

    data_directory = os.path.join(data_root, log_number)

    # Look for control data if enabled
    if CONTROLS:
        control_directory = os.path.join(data_root, 'controls', log_number)
        if os.path.isdir(control_directory):
            data_directory = control_directory
            flag_path = os.path.join(control_directory, CONTROL_FLAG_FILENAME)
            if not os.path.exists(flag_path):
                with open(flag_path, 'w') as f:
                    json.dump({"control": True, "id": log_number}, f, indent=4)
    analysis_directory = os.path.join(data_directory, 'Analysis')
    nifti_directory = os.path.join(data_directory, 'NIfTI')
    image_directory = os.path.join(data_directory, 'Images')
    
    # Directories to create
    dirs_to_create = [
        analysis_directory,
        os.path.join(analysis_directory, 'TSCC Data'),
        os.path.join(analysis_directory, 'CTC Data'),
        os.path.join(analysis_directory, 'CTC Data', 'Artery'),
        os.path.join(analysis_directory, 'CTC Data', 'Vein'),
        os.path.join(analysis_directory, 'CTC Data', 'Vein', 'Sinus Sagittalis'),
        os.path.join(analysis_directory, 'CTC Data', 'Tissue', 'Grey Matter'),
        os.path.join(analysis_directory, 'CTC Data', 'Tissue', 'White Matter'),
        os.path.join(analysis_directory, 'CTC Data', 'Tissue', 'Boundary'),
        os.path.join(analysis_directory, 'CTC Data', 'Tissue', 'AI'),
        os.path.join(analysis_directory, 'ITC Data'),
        os.path.join(analysis_directory, 'TSCC Data', 'Max'),
        os.path.join(analysis_directory, 'ITC Data', 'Artery'),
        os.path.join(analysis_directory, 'ITC Data', 'Vein'),
        os.path.join(analysis_directory, 'ITC Data', 'Vein', 'Sinus Sagittalis'),
        os.path.join(analysis_directory, 'ROI Data'),
        os.path.join(analysis_directory, 'ROI Data', 'Vein', 'Sinus Sagittalis'),
        os.path.join(analysis_directory, 'Frame Data'),
        os.path.join(analysis_directory, 'Frame Data', 'Vein', 'Sinus Sagittalis'),
        os.path.join(analysis_directory, 'Fitting'),
        image_directory,
        os.path.join(image_directory, 'Intensity Time Curves'),
        os.path.join(image_directory, 'AI'),
        os.path.join(image_directory, 'AI', 'Input functions'),
        os.path.join(image_directory, 'AI', 'Tissue functions'),
        os.path.join(image_directory, 'AI', 'Segmentation'),
        os.path.join(image_directory, 'Fit'),
        os.path.join(image_directory, 'Intensity Time Curves', 'Artery'),
        os.path.join(image_directory, 'Intensity Time Curves', 'Vein'),
        os.path.join(image_directory, 'Intensity Time Curves', 'Vein', 'Sinus Sagittalis'),
        os.path.join(image_directory, 'Concentration Time Curves'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Artery'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Vein'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Vein', 'Sinus Sagittalis'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Tissue'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', 'White Matter'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', 'Grey Matter'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', 'Boundary'),
        os.path.join(image_directory, 'Time Shifted Concentration Curves'),
        os.path.join(image_directory, 'Time Shifted Concentration Curves', 'Max'),
        nifti_directory
    ]
    
    artery_names = ["Left Interior Carotid", "Right Interior Carotid", "Basilar", "Left Middle Cerebral", "Right Middle Cerebral"]
    
    # Append directories involving artery names
    for artery in artery_names:
        dirs_to_create.extend([
            os.path.join(analysis_directory, 'CTC Data', 'Artery', artery),
            os.path.join(analysis_directory, 'ITC Data', 'Artery', artery),
            os.path.join(analysis_directory, 'TSCC Data', artery),
            os.path.join(analysis_directory, 'ROI Data', 'Artery', artery),
            os.path.join(analysis_directory, 'Frame Data', 'Artery', artery),
            os.path.join(image_directory, 'Concentration Time Curves', 'Artery', artery),
            os.path.join(image_directory, 'Intensity Time Curves', 'Artery', artery),
            os.path.join(image_directory, 'Time Shifted Concentration Curves', artery)
        ])
    
    # Create all directories
    for dir_path in dirs_to_create:
        create_directory(dir_path)

    return data_directory, analysis_directory, nifti_directory, image_directory


def save_run_settings(analysis_directory, parameters):
    """Save the settings used for an analysis run.

    Parameters are provided as the tuple returned by ``global_parameters``.
    The summary is written to ``run_settings.json`` in ``analysis_directory``.
    """
    names = [
        "IsVFA",
        "IsIR",
        "apple_metal",
        "boundary",
        "RERUN_SEGMENTATION",
        "SEGMENTATION_METHOD",
        "COMPUTE_FA",
    ]
    settings = dict(zip(names, parameters))
    settings.update({
        "MULTIPROCESSING": MULTIPROCESSING,
        "NUMBER_OF_CORES": NUMBER_OF_CORES,
        "KINETIC_MODEL": KINETIC_MODEL,
        "AI_MODEL_PATHS": AI_MODEL_PATHS,
        "CONTROLS": CONTROLS,
        "TIKHONOV_LAMBDA": TIKHONOV_LAMBDA,
        "GLOBAL_KI_SKIP_BOTTOM": GLOBAL_KI_SKIP_BOTTOM,
        "GLOBAL_KI_SKIP_TOP": GLOBAL_KI_SKIP_TOP,
        "TWO_COMPARTMENT_INIT_FROM_PATLAK": TWO_COMPARTMENT_INIT_FROM_PATLAK,
        "WRITE_MTT": WRITE_MTT,
        "WRITE_CTH": WRITE_CTH,
        "CTH_MTT_METHOD": CTH_MTT_METHOD,
        "CTH_MTT_GAMMA_VOXELWISE": CTH_MTT_GAMMA_VOXELWISE,
        "ALIGN_AIF_BY_XCORR": ALIGN_AIF_BY_XCORR,
        "ALIGN_AIF_MAX_SHIFT_S": ALIGN_AIF_MAX_SHIFT_S,
    })
    if AUTO_LAMBDA:
        settings["AUTO_LAMBDA_VALUE"] = AUTO_LAMBDA_VALUE
    settings_path = os.path.join(analysis_directory, "run_settings.json")
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)


def _git_metadata():
    """Collect basic Git metadata if the repository is available."""

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None

    metadata = {"commit": commit}

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        metadata["branch"] = branch
    except Exception:
        pass

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        metadata["is_dirty"] = bool(status.strip())
    except Exception:
        pass

    return metadata


def save_runtime_metadata(analysis_directory, *, extra_environment_keys=None):
    """Record runtime metadata for the executed analysis pipeline.

    The metadata is written to ``runtime_metadata.json`` within the analysis
    directory.  The recorded information focuses on reproducibility and avoids
    dumping the entire environment to reduce the risk of leaking secrets.
    """

    env_keys = [
        "P_BRAIN_DATA_DIR",
        "P_BRAIN_MODEL",
        "P_BRAIN_CORES",
        "P_BRAIN_LAMBDA",
        "P_BRAIN_GLOBAL_KI_SKIP_BOTTOM",
        "P_BRAIN_GLOBAL_KI_SKIP_TOP",
        "PBRAIN_CONTROLS",
        "PBRAIN_TURBO",
        "P_BRAIN_CTH_MTT_METHOD",
        "P_BRAIN_CTH_MTT_GAMMA_VOXELWISE",
        "P_BRAIN_ALIGN_AIF",
        "P_BRAIN_ALIGN_AIF_MAX_SHIFT",
    ]
    if extra_environment_keys:
        env_keys.extend(k for k in extra_environment_keys if k not in env_keys)

    environment = {
        key: os.environ.get(key)
        for key in env_keys
        if os.environ.get(key) is not None
    }

    packages = {}
    for package in ("numpy", "scipy", "matplotlib", "nibabel", "torch"):
        try:
            packages[package] = importlib_metadata.version(package)
        except Exception:
            continue

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "argv": sys.argv,
    }

    if environment:
        metadata["environment"] = environment
    if packages:
        metadata["packages"] = packages

    git_info = _git_metadata()
    if git_info:
        metadata["git"] = git_info

    os.makedirs(analysis_directory, exist_ok=True)
    metadata_path = os.path.join(analysis_directory, "runtime_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return metadata_path
