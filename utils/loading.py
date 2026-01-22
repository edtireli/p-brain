import pickle
import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np
import json
import importlib
import time
from typing import Optional

import utils.settings as settings


def _nifti_sidecar_json_path(nifti_path: str) -> str:
    """Return the expected dcm2niix JSON sidecar path for a NIfTI file."""

    if nifti_path.endswith('.nii.gz'):
        return nifti_path[:-7] + '.json'
    if nifti_path.endswith('.nii'):
        return nifti_path[:-4] + '.json'
    base, _ext = os.path.splitext(nifti_path)
    return base + '.json'


def read_nifti_sidecar_json(nifti_path: str):
    """Load the JSON sidecar produced by dcm2niix for a NIfTI file.

    Returns the parsed dict, or None when the sidecar is missing/unreadable.
    """

    if not nifti_path:
        return None
    json_path = _nifti_sidecar_json_path(nifti_path)
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def read_flip_angle_deg_from_sidecar(nifti_path: str, default=None):
    """Return excitation flip angle (degrees) from the NIfTI JSON sidecar."""

    data = read_nifti_sidecar_json(nifti_path)
    if not isinstance(data, dict):
        return default
    for key in (
        'FlipAngle',
        'FlipAngleDeg',
        'FlipAngleDegrees',
        'FlipAngle_deg',
        'FlipAngle(deg)',
    ):
        if key not in data:
            continue
        try:
            return float(data[key])
        except (TypeError, ValueError):
            return default
    return default


def resolve_flip_angle_deg(nifti_path: str, default=None):
    """Resolve flip angle (degrees) from config override or metadata.

    Resolution order:
    1) `P_BRAIN_FLIP_ANGLE=<number>` override.
    2) NIfTI JSON sidecar `FlipAngle` (or equivalent keys).
    """

    override = getattr(settings, "FLIP_ANGLE_DEG", None)
    if override is not None:
        try:
            return float(override)
        except (TypeError, ValueError):
            pass
    return read_flip_angle_deg_from_sidecar(nifti_path, default=default)


def read_repetition_time_s_from_sidecar(nifti_path: str, default=None):
    """Return repetition time (seconds) from the NIfTI JSON sidecar."""

    data = read_nifti_sidecar_json(nifti_path)
    if not isinstance(data, dict):
        return default
    for key in (
        "RepetitionTimeExcitation",
        "RepetitionTime",
        "RepetitionTime_s",
        "RepetitionTimeSeconds",
    ):
        if key not in data:
            continue
        try:
            value = float(data[key])
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return default


def discover_vfa_series(
    nifti_directory: str,
    patterns: Optional[str] = None,
    *,
    min_series: int = 2,
):
    """Discover VFA spoiled GRE series within ``nifti_directory``.

    Returns a list of dicts: {"nifti": path, "flip_angle_deg": float, "tr_s": float}
    sorted by flip angle.

    Patterns are glob expressions (comma-separated). When omitted, uses
    ``settings.VFA_FILE_GLOB``.
    """

    if not nifti_directory or not os.path.isdir(nifti_directory):
        return []

    raw_patterns = patterns if patterns is not None else getattr(settings, "VFA_FILE_GLOB", "*VFA*.nii*")
    pat_list = [p.strip() for p in str(raw_patterns).split(",") if p.strip()]
    if not pat_list:
        pat_list = ["*VFA*.nii*"]

    candidates: set[str] = set()
    for pat in pat_list:
        for match in glob.glob(os.path.join(nifti_directory, pat)):
            if os.path.isfile(match):
                candidates.add(match)

    series = []
    for nifti_path in sorted(candidates):
        fa = read_flip_angle_deg_from_sidecar(nifti_path, default=None)
        tr = read_repetition_time_s_from_sidecar(nifti_path, default=None)
        if fa is None or tr is None:
            continue
        series.append({"nifti": nifti_path, "flip_angle_deg": float(fa), "tr_s": float(tr)})

    series.sort(key=lambda item: item.get("flip_angle_deg", 0.0))
    if len(series) < int(min_series):
        return []
    return series


def discover_ir_series(nifti_directory: str):
    """Discover inversion-recovery NIfTIs expected by the released pipeline.

    Returns a list of 7 NIfTI paths (one per TI) when complete; otherwise []
    so callers can fall back to VFA or "none".
    """

    if not nifti_directory or not os.path.isdir(nifti_directory):
        return []

    TI = ["00120", "00300", "00600", "01000", "02000", "04000", "10000"]
    patterns = ["WIPTI_", "WIPDelRec-TI_"]
    paths = []
    for ti in TI:
        hit = None
        for suf in (".nii", ".nii.gz"):
            hit = first_existing_file(nifti_directory, patterns, ti, suf)
            if hit:
                break
        if not hit:
            return []
        paths.append(hit)
    return paths


def _sanitize_dcm2niix_basename(name: str) -> str:
    """Approximate dcm2niix filename sanitisation for matching outputs."""

    if name is None:
        return ""
    text = str(name).strip()
    # Replace whitespace with underscores, drop obvious path separators.
    text = re.sub(r"\s+", "_", text)
    text = text.replace(os.sep, "_")
    # Keep alphanumerics, underscore, dash.
    text = re.sub(r"[^0-9A-Za-z_\-]+", "", text)
    return text


sanitize_dcm2niix_basename = _sanitize_dcm2niix_basename


def read_protocol_name_from_par(par_path: str, default=None):
    """Extract protocol name from a Philips PAR header (used by dcm2niix %p)."""

    if not par_path or not os.path.exists(par_path):
        return default
    try:
        with open(par_path, "r", errors="ignore") as f:
            for line in f:
                if "protocol name" in line.lower():
                    # Typical: "Protocol name                         : WIPhperf120long"
                    if ":" in line:
                        value = line.split(":", 1)[1].strip()
                        return value or default
        return default
    except OSError:
        return default


def read_flip_angle_deg_from_par(par_path: str, default=None):
    """Extract excitation flip angle (degrees) from a Philips PAR file."""

    if not par_path or not os.path.exists(par_path):
        return default

    try:
        with open(par_path, "r", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return default

    col_index_1based = None
    for line in lines:
        if not line.lstrip().startswith("#"):
            continue
        lower = line.lower()
        if "image_flip_angle" not in lower:
            continue
        # Example: "#  10. image_flip_angle (in degrees)        (float)"
        m = re.search(r"#\s*(\d+)\s*\.\s*image_flip_angle", lower)
        if m:
            try:
                col_index_1based = int(m.group(1))
                break
            except ValueError:
                col_index_1based = None

    if not col_index_1based:
        return default

    # Find IMAGE INFORMATION section (not DEFINITION) and parse data lines.
    in_image_info = False
    values = []
    for line in lines:
        lower = line.lower()
        if line.lstrip().startswith("#") and "=== image information" in lower and "definition" not in lower:
            in_image_info = True
            continue
        if not in_image_info:
            continue
        if line.lstrip().startswith("#"):
            continue
        tokens = line.strip().split()
        if len(tokens) < col_index_1based:
            continue
        try:
            values.append(float(tokens[col_index_1based - 1]))
        except ValueError:
            continue

    if not values:
        return default
    try:
        return float(np.median(np.asarray(values, dtype=float)))
    except Exception:
        return default


def inject_flip_angle_into_sidecar_json(nifti_path: str, flip_angle_deg: float) -> bool:
    """Ensure NIfTI JSON sidecar includes `FlipAngle`.

    Returns True when the file was written/updated.
    """

    if not nifti_path:
        return False
    json_path = _nifti_sidecar_json_path(nifti_path)
    if not os.path.exists(json_path):
        return False
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    if "FlipAngle" in data:
        return False
    try:
        data["FlipAngle"] = float(flip_angle_deg)
    except (TypeError, ValueError):
        return False
    try:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        return True
    except OSError:
        return False



def list_addons():
    addon_folders = [f for f in os.listdir("addons") if os.path.isdir(os.path.join("addons", f))]
    print('--------------------------------')
    for i, addon in enumerate(addon_folders):
        print(f"{i+1}. {addon}")
    print('--------------------------------')    
    choice = int(input("[!] Choose addon: ")) - 1
    return addon_folders[choice]

import sys
def load_addon(addon_folder_name, *args):
    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
        base_dir = os.path.dirname(current_script_dir)
        addon_path = os.path.join(base_dir, 'addons')
        sys.path.append(addon_path)  

        addon = importlib.import_module(f'{addon_folder_name}.{addon_folder_name}')
        addon.run(*args)
    except ModuleNotFoundError as e:
        print(f"Addon {addon_folder_name} not available! Error: {str(e)}")
        print("Make sure the addon is correctly placed in the 'addons' directory and named correctly.")
        print(f"Attempted to import from {addon_path}")
    except Exception as e:
        print(f"An error occurred while loading the addon: {str(e)}")

def replace_max_with_artery_type_and_delete(values_json_path, max_info_json_path):
    # Read and parse max_info.json
    with open(max_info_json_path, 'r') as f:
        max_info_json = json.load(f)
    
    # Check if 'Max artery type' exists in max_info.json
    artery_type_list = [entry.split(": ")[1] for entry in max_info_json if "Max artery type" in entry]
    
    if not artery_type_list:
        raise ValueError("No 'Max artery type' found in max_info.json")
    
    artery_type = artery_type_list[0]
    
    # Read and parse values.json
    with open(values_json_path, 'r') as f:
        values_json = json.load(f)
    
    # Replace "Max" with the extracted "Artery type" in values.json
    updated_values_json = []
    
    for entry in values_json:
        new_entry = {k: v.replace("Max", artery_type) if "Max" in v else v for k, v in entry.items()}
        updated_values_json.append(new_entry)
    
    # Update values.json with the modified data
    with open(values_json_path, 'w') as f:
        json.dump(updated_values_json, f)
    
    # Uncomment the following line to remove max_info.json after processing
    # os.remove(max_info_json_path)

    return updated_values_json


def find_matching_file(directory, pattern):
    regex = re.compile(pattern, re.IGNORECASE)
    for filename in os.listdir(directory):
        if regex.match(filename):
            return os.path.join(directory, filename)
    return None

def save_values(Ki, SD_Ki, lambda_, P, P_std, subtype_tissue, slice_tissue,
                subtype_artery, venous_slice, arterial_slice,
                analysis_directory, suffix=""):
    """Save permeability results to ``values{suffix}.json``."""
    values_file_path = os.path.join(analysis_directory, f'values{suffix}.json')
    
    if os.path.exists(values_file_path):
        with open(values_file_path, 'r') as f:
            existing_values = json.load(f)
    else:
        existing_values = []
    
    new_entry = {
        "Ki": f"{Ki*6000} (+- {SD_Ki*6000})",
        "Lambda": f"{lambda_*100}",
        "Tissue": f"{subtype_tissue} (Slice {slice_tissue})",
        "Artery": f"{subtype_artery} (Venous Slice {venous_slice}, Arterial Slice {arterial_slice})",
        "Ki_f": f"{P} (+- {P_std})"
    }
    
    updated_values = []
    found = False
    
    for entry in existing_values:
        if entry.get("Tissue") == f"{subtype_tissue} (Slice {slice_tissue})":
            updated_values.append(new_entry)
            found = True
        else:
            updated_values.append(entry)
    
    if not found:
        updated_values.append(new_entry)
    
    with open(values_file_path, 'w') as f:
        json.dump(updated_values, f)




def leaver():
    leave = input('[!] Return to main menu? (y/n): ')    
    if leave == 'y' or leave=='':
        return
    if leave == 'n':
        leaver()
    else:
        print('This was not an option...')
        time.sleep(2)
        leaver()      

def find_matching_file(directory, pattern):
    regex = re.compile(pattern, re.IGNORECASE)
    for filename in os.listdir(directory):
        if regex.match(filename):
            return os.path.join(directory, filename)
    return None


def load_curves(venous_slice, arterial_slice, artery_choice, analysis_directory):
    shifted_vein_file = os.path.join(analysis_directory, 'CTC Data', 'Vein', 'Sinus Sagittalis', f'CTC_shifted_slice_{venous_slice}.npy')
    vein_file = os.path.join(analysis_directory, 'CTC Data', 'Vein', 'Sinus Sagittalis', f'CTC_slice_{venous_slice}.npy')
    
    shifted_artery_file = os.path.join(analysis_directory, 'CTC Data', 'Artery', artery_choice, f'CTC_shifted_slice_{arterial_slice}.npy')
    artery_file = os.path.join(analysis_directory, 'CTC Data', 'Artery', artery_choice, f'CTC_slice_{arterial_slice}.npy')

    if os.path.exists(shifted_vein_file):
        vein_curve = np.load(shifted_vein_file)
    else:
        vein_curve = np.load(vein_file)
    
    if os.path.exists(shifted_artery_file):
        artery_curve = np.load(shifted_artery_file)
    else:
        artery_curve = np.load(artery_file)
    
    return vein_curve, artery_curve



def save_as_pickle(matrix, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(matrix, file)


def load_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        matrix = pickle.load(file)
    return matrix


def first_existing_file(directory, patterns, time, suffix):
    for pattern in patterns:
        file_path = os.path.join(directory, f"{pattern}{time}{suffix}")
        if os.path.exists(file_path):
            return file_path
    return None


def _read_max_artery_type(analysis_directory):
    info_path = os.path.join(analysis_directory, 'max_info.json')
    if not os.path.exists(info_path):
        return None

    try:
        with open(info_path, 'r') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    for entry in data:
        if isinstance(entry, str) and entry.startswith('Max artery type:'):
            return entry.split(':', 1)[1].strip()
    return None


def _parse_tscc_filename(filename):
    match = re.match(r'TSCC_slice_(\d+)_(\d+)\.npy$', filename)
    if not match:
        raise ValueError(f"Unrecognised TSCC filename format: {filename}")
    venous_slice, arterial_slice = map(int, match.groups())
    return venous_slice, arterial_slice


def _load_shifted_input_function(analysis_directory, subtype, venous_slice, arterial_slice):
    tscc_root = os.path.join(analysis_directory, 'TSCC Data')

    if subtype is None or subtype == 'Max':
        max_dir = os.path.join(tscc_root, 'Max')
        npy_files = [
            f for f in os.listdir(max_dir)
            if f.endswith('.npy') and not f.startswith('.')
        ]
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {max_dir}.")
        npy_files.sort()
        filename = npy_files[0]
        venous_slice, arterial_slice = _parse_tscc_filename(filename)
        artery_type = _read_max_artery_type(analysis_directory) or 'Max'
        path = os.path.join(max_dir, filename)
    else:
        if venous_slice is None or arterial_slice is None:
            raise ValueError(
                "Both venous and arterial slice indices are required for TSCC input functions."
            )
        filename = f'TSCC_slice_{venous_slice}_{arterial_slice}.npy'
        path = os.path.join(tscc_root, subtype, filename)
        artery_type = subtype
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input function file not found: {path}")

    curve = np.load(path)
    metadata = {
        'source': 'SSS',
        'path': path,
        'artery_subtype': artery_type,
        'venous_slice': venous_slice,
        'arterial_slice': arterial_slice,
    }
    return curve, metadata


def _load_pure_input_function(analysis_directory, subtype, venous_slice, arterial_slice):
    artery_root = os.path.join(analysis_directory, 'CTC Data', 'Artery')
    if not os.path.isdir(artery_root):
        raise FileNotFoundError(f"Artery directory not found: {artery_root}")

    def available_slices(artery_dir):
        slices = []
        for fname in os.listdir(artery_dir):
            match = re.match(r'CTC_slice_(\d+)\.npy$', fname)
            if match:
                slices.append((int(match.group(1)), fname))
        return sorted(slices)

    best_curve = None
    best_metadata = None

    if subtype is None or subtype == 'Max':
        for current_subtype in os.listdir(artery_root):
            artery_dir = os.path.join(artery_root, current_subtype)
            if not os.path.isdir(artery_dir):
                continue
            for slice_idx, fname in available_slices(artery_dir):
                path = os.path.join(artery_dir, fname)
                curve = np.load(path)
                peak = float(np.max(curve))
                if best_curve is None or peak > best_metadata['peak']:
                    best_curve = curve
                    best_metadata = {
                        'source': 'RICA',
                        'path': path,
                        'artery_subtype': current_subtype,
                        'venous_slice': None,
                        'arterial_slice': slice_idx,
                        'peak': peak,
                    }
        if best_curve is None:
            raise FileNotFoundError(
                f"No arterial concentration curves found in {artery_root}."
            )
        metadata = best_metadata.copy()
        metadata.pop('peak', None)
        return best_curve, metadata

    artery_dir = os.path.join(artery_root, subtype)
    if not os.path.isdir(artery_dir):
        raise FileNotFoundError(f"Artery subtype directory not found: {artery_dir}")

    slices = available_slices(artery_dir)
    if not slices:
        raise FileNotFoundError(
            f"No pure arterial curves available for subtype '{subtype}'."
        )

    if arterial_slice is None or str(arterial_slice).lower() == 'auto':
        best_slice = None
        best_curve = None
        best_peak = None
        for slice_idx, fname in slices:
            path = os.path.join(artery_dir, fname)
            curve = np.load(path)
            peak = float(np.max(curve))
            if best_peak is None or peak > best_peak:
                best_slice = slice_idx
                best_curve = curve
                best_peak = peak
                best_path = path
        metadata = {
            'source': 'RICA',
            'path': best_path,
            'artery_subtype': subtype,
            'venous_slice': None,
            'arterial_slice': best_slice,
        }
        return best_curve, metadata

    try:
        slice_idx = int(arterial_slice)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid arterial slice index '{arterial_slice}' for pure input function."
        )

    filename = f'CTC_slice_{slice_idx}.npy'
    path = os.path.join(artery_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input function file not found: {path}")

    curve = np.load(path)
    metadata = {
        'source': 'RICA',
        'path': path,
        'artery_subtype': subtype,
        'venous_slice': None,
        'arterial_slice': slice_idx,
    }
    return curve, metadata


def get_input_function_curve(analysis_directory, subtype='Max',
                             venous_slice=None, arterial_slice=None):
    """Load the selected arterial input function as configured.

    Parameters
    ----------
    analysis_directory : str
        Root analysis directory for the current dataset.
    subtype : str, optional
        Artery subtype requested by the user.  ``'Max'`` selects the default
        time-shifted curve in SSS mode or the highest peak pure curve when the
        pure arterial option is enabled.
    venous_slice : str or int, optional
        Venous slice index (used for time-shifted SSS curves).
    arterial_slice : str or int, optional
        Arterial slice index.  When ``None`` or ``'auto'`` in pure mode, the
        slice with the highest peak is selected automatically.

    Returns
    -------
    tuple
        ``(curve, metadata)`` where *curve* is a NumPy array containing the
        input function and *metadata* a dictionary describing the selection.
    """

    source = settings.INPUT_FUNCTION_SOURCE
    if source == 'RICA':
        return _load_pure_input_function(analysis_directory, subtype, venous_slice, arterial_slice)
    return _load_shifted_input_function(analysis_directory, subtype, venous_slice, arterial_slice)

