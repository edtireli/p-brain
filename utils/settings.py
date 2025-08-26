import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

# Global toggle for analysing control data. The flag can also be
# overridden by setting the environment variable ``PBRAIN_CONTROLS`` to
# ``1``/``true``.
CONTROLS = 0#os.environ.get("PBRAIN_CONTROLS", "False").lower() in ("1", "true", "yes")
CONTROL_FLAG_FILENAME = "control.json"

# Toggle multiprocessing for compute-heavy routines
MULTIPROCESSING = True
# Default number of CPU cores used when multiprocessing is enabled
NUMBER_OF_CORES = int(os.environ.get("P_BRAIN_CORES", 4))

# Select kinetic modelling strategy. Valid options are ``patlak``,
# ``two_compartment`` (regularised two-compartment fit) or ``both`` to execute the two
# approaches sequentially.  When the environment variable
# ``P_BRAIN_MODEL`` is not provided the default is ``both``.
KINETIC_MODEL = os.environ.get("P_BRAIN_MODEL", "both")

# Regularisation strength for the two-compartment model
TIKHONOV_LAMBDA = float(os.environ.get("P_BRAIN_LAMBDA", 0.5))

# When enabled, pick the regularisation weight automatically using an L-curve
# search across ``AUTO_LAMBDA_CANDIDATES``.
AUTO_LAMBDA = False
AUTO_LAMBDA_CANDIDATES = np.logspace(-2, 2, 30)
# Holds the most recently chosen value when ``AUTO_LAMBDA`` is True
AUTO_LAMBDA_VALUE = None

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
    })
    if AUTO_LAMBDA:
        settings["AUTO_LAMBDA_VALUE"] = AUTO_LAMBDA_VALUE
    settings_path = os.path.join(analysis_directory, "run_settings.json")
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)
