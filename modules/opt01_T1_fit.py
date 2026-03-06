import os
from scipy.optimize import least_squares, curve_fit, fmin
import nibabel as nib
from tqdm import tqdm
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import multiprocessing
import functools
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, binary_dilation, ball
from skimage.morphology import remove_small_objects
import utils.settings as settings
from utils.fonts import *
from utils.loading import *
from utils.plotting import *
from utils.cli_logging import auto_logging_suppressed
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import threading
import sys

turbo_mode = True #doesnt show plots


def _resolve_t1_recovery_model(explicit: str | None = None) -> str:
    """Resolve the T1 recovery model at runtime.

    Important: do not import a snapshot of settings.T1_RECOVERY_MODEL because
    main() may update it after module import (e.g. when --ctc-model=turboflash).
    """

    if explicit:
        model = str(explicit).strip().lower()
        if model:
            return model

    env = (os.environ.get('P_BRAIN_T1_RECOVERY_MODEL') or '').strip().lower()
    if env:
        return env

    model = getattr(settings, 'T1_RECOVERY_MODEL', '')
    return (str(model).strip().lower() or 'inversion')


def _reference_img_for_t1fit(nifti_directory, dce_filename, *, prefer_ir=True):
    """Resolve a reference NIfTI for affine/header when exporting T1/M0 maps.

    Prefer an inversion-recovery/VFA source volume if present so that the
    exported maps match the fitted voxel grid.
    """

    def _try_load(path):
        if not path:
            return None
        if not os.path.exists(path):
            return None
        try:
            return nib.load(path)
        except Exception:
            return None

    if prefer_ir:
        # Inversion recovery series naming conventions used in this module.
        for ti in ("00120", "00300", "00600", "01000", "02000", "04000", "10000"):
            for prefix in ("WIPTI_", "WIPDelRec-TI_"):
                candidate = os.path.join(nifti_directory, f"{prefix}{ti}.nii")
                img = _try_load(candidate)
                if img is not None:
                    return img

    # Fall back to the DCE reference if available.
    if dce_filename:
        img = _try_load(os.path.join(nifti_directory, dce_filename))
        if img is not None:
            return img

    return None


def _shape_from_parrec(data_directory):
    """Try to determine the 3-D volume shape from a PAR/REC file in *data_directory*.

    Scans one level deep for .PAR files and loads the first one with nibabel
    to extract the spatial shape.  Returns a 3-tuple (x, y, z) or ``None``.
    """
    if not data_directory or not os.path.isdir(data_directory):
        return None

    candidates = [data_directory]
    try:
        for entry in os.scandir(data_directory):
            if entry.is_dir() and not entry.name.startswith("."):
                candidates.append(entry.path)
    except Exception:
        pass

    for base_dir in candidates:
        try:
            entries = list(os.scandir(base_dir))
        except Exception:
            continue
        pars = {}
        recs = set()
        for e in entries:
            if not e.is_file():
                continue
            stem, ext = os.path.splitext(e.name.lower())
            if ext == ".par":
                pars[stem] = e.path
            elif ext == ".rec":
                recs.add(stem)
        for stem, par_path in pars.items():
            if stem not in recs:
                continue
            try:
                img = nib.parrec.load(par_path, permit_truncated=True, strict_sort=False)
                s = img.shape
                if len(s) >= 3:
                    return s[:3]
            except Exception:
                continue

    return None


def _export_map_nifti(map_data, reference_img, out_path):
    if reference_img is None:
        return False
    data = np.asarray(map_data)
    if data.ndim != 3:
        data = np.squeeze(data)
    if data.ndim != 3:
        return False

    header = None
    try:
        header = reference_img.header.copy() if reference_img.header is not None else None
        if header is not None:
            header.set_data_dtype(np.float32)
    except Exception:
        header = None

    img = nib.Nifti1Image(data.astype(np.float32, copy=False), reference_img.affine, header)
    try:
        nib.save(img, out_path)
        return True
    except Exception:
        return False

def close_plot_after_delay_plt(delay):
    """Close the plot after ``delay`` seconds when running in turbo mode."""
    if turbo_mode:
        def close():
            plt.close(plt.gcf())  # Close the current figure

        timer = threading.Timer(delay, close)
        timer.start()

        # If there is user interaction, cancel the timer
        plt.gcf().canvas.mpl_connect('key_press_event', lambda event: timer.cancel())


def extract_vfa_params(vfa_filenames, nifti_directory):
    repetition_times = []
    flip_angles = []

    for filename in vfa_filenames:
        # Remove the extension and construct the JSON filename
        base_name = os.path.splitext(filename)[0]  # Removes only the last extension (e.g., .nii)
        if base_name.endswith("_DCE"):  # Specific to your case, adjust as necessary
            base_name = base_name.rsplit("_", 1)[0]  # Remove the last part after '_'

        json_filename = base_name + ".json"
        json_path = os.path.join(nifti_directory, json_filename)

        try:
            # Read and extract data from the JSON file
            with open(json_path, 'r') as json_file:
                json_data = json.load(json_file)
                repetition_times.append(json_data.get("RepetitionTime"))
                flip_angles.append(json_data.get("FlipAngle"))
        except FileNotFoundError:
            print(f"Warning: JSON file not found: {json_path}")
            repetition_times.append(None)  # Append None to keep data aligned
            flip_angles.append(None)

    return repetition_times, flip_angles


def build_voxel_matrix(dce_data, *, volume_index: int = 0):
    if not dce_data:
        raise ValueError("No data files provided.")

    missing = [idx for idx, path in enumerate(dce_data) if not path]
    if missing:
        raise FileNotFoundError(
            "One or more input NIfTI paths are missing (None/empty) at indices "
            f"{missing}. This usually means the expected series was not found in the NIfTI directory."
        )

    # Load the first file to determine the shape
    first_data_shape = nib.load(dce_data[0]).get_fdata().shape
    # Adjust the shape to accommodate all data files
    matrix_shape = (len(dce_data), *first_data_shape[:3])

    matrix = np.zeros(matrix_shape)

    vol_idx = int(volume_index) if volume_index is not None else 0

    with auto_logging_suppressed():
        for idx, file in tqdm(enumerate(dce_data), desc="Building Voxel Matrix", total=len(dce_data)):
            data_4d = nib.load(file).get_fdata()
            # If data is 4D (e.g. Philips PAR/REC dynamics), select the requested
            # volume; MATLAB case-3 T1 fitting uses dynamic=2 (1-based), i.e.
            # volume_index=1 (0-based).
            if data_4d.ndim == 4:
                chosen = vol_idx if 0 <= vol_idx < data_4d.shape[3] else 0
                matrix[idx, :, :, :] = data_4d[:, :, :, chosen]
            else:
                matrix[idx, :, :, :] = data_4d

    return matrix

# -----------------------------------------------------------------------------
# Brain masking utilities
# -----------------------------------------------------------------------------


def _normalise_image(data):
    """Normalise image intensities to [0, 1] while avoiding NaNs."""
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    finite_vals = data[np.isfinite(data)]
    if finite_vals.size == 0:
        return data
    high_percentile = np.percentile(finite_vals, 99)
    if high_percentile == 0:
        return data
    data = data / high_percentile
    data[data > 1] = 1
    data[data < 0] = 0
    return data


def compute_brain_mask(t1_path):
    """Generate a crude brain mask from the provided T1 image."""
    t1_img = nib.load(t1_path)
    t1_data = t1_img.get_fdata()
    # Many clinical datasets store structural scans with a trailing singleton
    # dimension (e.g. ``(x, y, z, 1)``).  The downstream fitting code expects
    # a 3-D mask, so squeeze away any singleton axes before we start
    # processing.  This keeps the brain mask compatible with the voxel matrix
    # that will be generated from the IR/VFA series.
    t1_data = np.squeeze(t1_data)
    t1_data = _normalise_image(t1_data)

    non_zero = t1_data[t1_data > 0]
    if non_zero.size == 0:
        raise ValueError("T1 image appears to be empty; cannot compute brain mask.")

    threshold = threshold_otsu(non_zero)
    mask = t1_data > threshold

    # Morphological operations to clean the mask
    mask = binary_closing(mask, ball(2))
    mask = binary_opening(mask, ball(1))
    mask = remove_small_objects(mask, 500)
    mask = binary_dilation(mask, ball(1))

    mask = np.squeeze(mask)
    return mask.astype(bool)


def _compute_mask_from_voxel_matrix(voxel_matrix):
    """Derive a fallback brain mask directly from the acquisition series."""
    if voxel_matrix is None:
        return None

    if voxel_matrix.ndim < 4:
        return None

    # Use the maximum intensity projection over time to capture all anatomy
    summary_image = np.nanmax(voxel_matrix, axis=0)
    summary_image = _normalise_image(summary_image)

    non_zero = summary_image[summary_image > 0]
    if non_zero.size == 0:
        return None

    threshold = threshold_otsu(non_zero)
    mask = summary_image > threshold

    mask = binary_closing(mask, ball(1))
    mask = binary_opening(mask, ball(1))
    mask = remove_small_objects(mask, 100)
    mask = binary_dilation(mask, ball(1))

    return mask.astype(bool)


def load_brain_mask(analysis_directory, t1_path):
    """Load a cached brain mask or compute a new one if missing."""
    mask_path = os.path.join(analysis_directory, 'Fitting', 'brain_mask.npy')
    brain_mask = None

    if os.path.exists(mask_path):
        try:
            brain_mask = np.load(mask_path)
        except Exception:
            brain_mask = None

    if brain_mask is None:
        if not os.path.exists(t1_path):
            print(f"Warning: T1 image missing at {t1_path}; cannot compute brain mask.")
            return None
        try:
            brain_mask = compute_brain_mask(t1_path)
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            np.save(mask_path, brain_mask)
        except Exception as exc:
            print(f"Warning: Unable to compute brain mask ({exc}). Proceeding without masking.")
            brain_mask = None

    return brain_mask

# Variable flip angle not inversion recovery
def model_function_VFA(alfas, TR, M0, R1):
    TR = np.array(TR)*1e3 #conversion to ms from s
    R1 = np.array(R1)*1e-3 #conversion to 1/ms from 1/s
    alfas_rad = np.radians(alfas)  # Convert degrees to radians
    a = np.cos(alfas_rad) * np.exp(-np.array(TR) * R1)
    b = 1 - np.exp(-TR * R1)
    return M0 * np.sin(alfas_rad) * b / (1 - a)


def model_residuals_VFA(params, voxel_values, alfas, TRs):
    M0, T1 = params
    R1 = 1 / T1
    return model_function_VFA(alfas, TRs, M0, R1) - voxel_values

# Inversion recovery model following A - B * exp(-TI / T1)
def model_function_ir(TIs, A, B, T1):
    TIs_array = np.asarray(TIs, dtype=float)
    return A - B * np.exp(-TIs_array / T1)


def model_residuals_ir(params, TIs, voxel_values):
    A, B, T1 = params
    return model_function_ir(TIs, A, B, T1) - voxel_values


def model_function_sr(TIs, M0, T1):
    """Saturation recovery model following M0 * (1 - exp(-TI / T1))."""
    TIs_array = np.asarray(TIs, dtype=float)
    return M0 * (1 - np.exp(-TIs_array / T1))


def model_residuals_sr(params, TIs, voxel_values):
    M0, T1 = params
    return model_function_sr(TIs, M0, T1) - voxel_values


def model_function_turboflash_ti(TIs_ms, M0, T1_ms, *, alpha_rad: float, tr_ms: float, nph: int):
    """TurboFLASH TI-series model matching legacy MATLAB `sigdif_turbof_r1_90_2`.

    Units:
    - TIs_ms, T1_ms, tr_ms are in milliseconds
    - alpha_rad is in radians

    Signal model:
      a = cos(alpha) * exp(-TR*R1)
      b = 1 - exp(-TR*R1)
      s = M0*sin(alpha) * ( (1-exp(-R1*TI))*a^(nph-1) + b*(1-a^(nph-1))/(1-a) )

    where R1 = 1/T1.
    """

    TIs = np.asarray(TIs_ms, dtype=float)
    T1_ms = float(T1_ms)
    if not np.isfinite(T1_ms) or T1_ms <= 0:
        return np.full_like(TIs, np.nan, dtype=float)

    r1_ms = 1.0 / T1_ms
    tr_ms = float(tr_ms)
    nph = int(nph) if int(nph) > 0 else 1

    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        a = np.cos(alpha_rad) * np.exp(-tr_ms * r1_ms)
        b = 1.0 - np.exp(-tr_ms * r1_ms)
        # Guard a≈1 to avoid division blow-ups.
        denom = (1.0 - a)
        if not np.isfinite(denom) or abs(denom) < 1e-10:
            denom = 1e-10 if denom >= 0 else -1e-10
        a_pow = a ** (nph - 1)
        term1 = (1.0 - np.exp(-r1_ms * TIs)) * a_pow
        term2 = b * (1.0 - a_pow) / denom
        s = float(M0) * np.sin(alpha_rad) * (term1 + term2)
    return s


def model_residuals_turboflash_ti(params, TIs_ms, voxel_values, *, alpha_rad: float, tr_ms: float, nph: int):
    M0, T1_ms = params
    return model_function_turboflash_ti(TIs_ms, M0, T1_ms, alpha_rad=alpha_rad, tr_ms=tr_ms, nph=nph) - voxel_values


def _fit_single(voxel_values, IsVFA, TI_values, alfas, TRs, *, tf_alpha_rad=None, tf_tr_ms=None, tf_nph=None, recovery_model=None):
    if max(voxel_values) == 0:
        return (0.0, 0.0)
    if IsVFA:
        alfas_rad = np.radians(alfas)
        initial_M0 = max(voxel_values) / np.sin(alfas_rad[0])
        initial_T1 = 750
        bounds = ([0, 500], [2 * initial_M0, 5000])
        result = least_squares(
            model_residuals_VFA,
            [initial_M0, initial_T1],
            args=(voxel_values,),
            kwargs={"alfas": alfas, "TRs": TRs},
            bounds=bounds,
            method="trf",
        )
    else:
        max_signal = float(np.max(voxel_values))
        if max_signal <= 0:
            return (0.0, 0.0)
        model = _resolve_t1_recovery_model(recovery_model)
        if model == "turboflash":
            # TurboFLASH TI-series fit (matches MATLAB turbof_r1_m0_90_fit_2).
            # Requires alpha/TR/nph; fall back to simple saturation model if unavailable.
            if tf_alpha_rad is None or tf_tr_ms is None:
                model = "saturation"
            else:
                # MATLAB create_t1map_philips6_NewParRec2015:
                # - TI is in seconds (default [0.12 0.3 0.6 1 2 4 10])
                # - TR is in seconds (PAR "Repetition time"; for these NIfTIs: RepetitionTimeExcitation)
                # - nph defaults to 1
                alpha_rad = float(tf_alpha_rad)
                tr_ms = float(tf_tr_ms)
                nph = int(tf_nph) if tf_nph is not None else 1

                opt = (os.environ.get("P_BRAIN_T1FIT_TURBO_OPT") or "matlab").strip().lower()
                if opt in {"matlab", "fminsearch", "nelder-mead", "nelder_mead"}:
                    # Match MATLAB exactly (slow per-voxel):
                    #   m0_start = max(s)/sin(alfa)
                    #   r1_start = 1/0.75
                    #   x = fminsearch(@sigdif_turbof_r1_90_2, [r1_start,m0_start], ...)
                    # where sigdif uses: s - abs(s_calc)
                    tr_s = tr_ms * 1e-3
                    # In this module, `TI_values` for IR is passed in *seconds*.
                    ti_s = np.asarray(TI_values, dtype=float)
                    s = np.asarray(voxel_values, dtype=float).reshape(-1)

                    sin_a = float(np.sin(alpha_rad))
                    if not np.isfinite(sin_a) or abs(sin_a) < 1e-12:
                        sin_a = 1e-12

                    m0_start = float(np.nanmax(s)) / sin_a
                    r1_start = 1.0 / 0.75

                    def _sigdif_turbof_r1_90_2(x):
                        r1 = float(x[0])
                        m0 = float(x[1])
                        with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
                            a = np.cos(alpha_rad) * np.exp(-tr_s * r1)
                            b = 1.0 - np.exp(-tr_s * r1)
                            a_pow = a ** (nph - 1)
                            denom = (1.0 - a)
                            if not np.isfinite(denom) or abs(denom) < 1e-12:
                                denom = 1e-12 if denom >= 0 else -1e-12
                            s_calc = m0 * np.sin(alpha_rad) * (
                                (1.0 - np.exp(-r1 * ti_s)) * a_pow + b * (1.0 - a_pow) / denom
                            )
                            s_diff = s - np.abs(s_calc)
                            ss = float(np.dot(s_diff, s_diff))
                            n = int(s.size)
                            return ss / float(n - 1) if n > 1 else ss

                    x0 = np.array([r1_start, m0_start], dtype=float)
                    x = fmin(_sigdif_turbof_r1_90_2, x0, ftol=1e-3, maxiter=1000, disp=False)

                    r1_hat = float(x[0])
                    m0_hat = float(x[1])
                    if not np.isfinite(r1_hat) or r1_hat <= 0:
                        return (np.nan, np.nan)
                    t1_ms = 1000.0 / r1_hat
                    return (m0_hat, t1_ms)

                # Fast default: bounded least-squares on (M0, T1_ms).
                # `model_function_turboflash_ti` expects TI/TR in milliseconds.
                initial_M0 = max_signal / max(np.sin(alpha_rad), 1e-6)
                initial_T1 = 750.0
                # MATLAB case-3 outputs can exceed 6000 ms (e.g. up to ~8.2 s),
                # so keep the upper bound comfortably above the longest TI.
                bounds = ([1e-6, 100.0], [np.inf, 12000.0])
                result = least_squares(
                    model_residuals_turboflash_ti,
                    [initial_M0, initial_T1],
                    args=(np.asarray(TI_values, dtype=float) * 1e3, voxel_values),
                    kwargs={"alpha_rad": alpha_rad, "tr_ms": tr_ms, "nph": nph},
                    bounds=bounds,
                    method="trf",
                )
                return float(result.x[0]), float(result.x[1])

        if model == "saturation":
            initial_M0 = max_signal
            # TI_values are provided in seconds; keep the fit in seconds to
            # match the MATLAB TurboFLASH case-12 workflow.
            initial_T1 = 0.75
            bounds = ([1e-6, 0.05], [np.inf, 12.0])
            result = least_squares(
                model_residuals_sr,
                [initial_M0, initial_T1],
                args=(TI_values, voxel_values),
                bounds=bounds,
                method="trf",
            )
            return result.x[0], result.x[1]
        else:
            initial_A = max_signal
            initial_B = 2 * max_signal
            # Fit in seconds (TI_values already in seconds).
            initial_T1 = 0.75
            bounds = ([1e-6, 1e-6, 0.05], [np.inf, np.inf, 12.0])
            result = least_squares(
                model_residuals_ir,
                [initial_A, initial_B, initial_T1],
                args=(TI_values, voxel_values),
                bounds=bounds,
                method="trf",
            )
            return result.x[0], result.x[2]
    return result.x[0], result.x[1]

def _ensure_mask_matches_shape(brain_mask, expected_shape):
    """Return a boolean mask if the shape matches, otherwise ``None``."""
    if brain_mask is None:
        return None

    mask_array = np.asarray(brain_mask)
    expected_shape = tuple(expected_shape)

    # Allow masks saved with trailing singleton dimensions (e.g. ``(x, y, z, 1)``)
    # by squeezing them before performing the comparison.  This situation is
    # common when loading NIfTI volumes that were stored with an extra axis.
    if mask_array.ndim > len(expected_shape):
        mask_array = np.squeeze(mask_array)

    if mask_array.shape != expected_shape:
        print(
            "Warning: Brain mask shape"
            f" {mask_array.shape} does not match data shape {expected_shape}; ignoring mask."
        )
        return None

    return mask_array.astype(bool)


def _resolve_brain_mask(brain_mask, voxel_matrix, mask_path, existing_source=None):
    """Ensure the brain mask matches the voxel matrix, falling back if needed."""
    mask_source = existing_source
    if brain_mask is not None and mask_source is None:
        mask_source = "t1"

    if voxel_matrix is None:
        return brain_mask, mask_source

    ensured_mask = _ensure_mask_matches_shape(brain_mask, voxel_matrix.shape[1:])
    if ensured_mask is not None:
        return ensured_mask, mask_source or "t1"

    fallback_mask = _compute_mask_from_voxel_matrix(voxel_matrix)
    if fallback_mask is not None:
        try:
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            np.save(mask_path, fallback_mask)
            print("Info: Brain mask recomputed from voxel data to match acquisition shape.")
        except Exception:
            pass
        return fallback_mask, "voxel_matrix"

    return None, None


def fit_all_voxels(voxel_matrix, TI_values, IsVFA, brain_mask=None, signal_mask=None, **kwargs):
    shape_x, shape_y, shape_z = voxel_matrix.shape[1:]
    total_voxels = shape_x * shape_y * shape_z

    voxels = voxel_matrix.reshape(voxel_matrix.shape[0], -1).T
    alfas = kwargs.get('alfas')
    TRs = kwargs.get('TRs')

    combined_mask = None
    if brain_mask is not None:
        mask_flat = np.asarray(brain_mask, dtype=bool).reshape(-1)
        if mask_flat.size != total_voxels:
            print(
                "Warning: Brain mask contains"
                f" {mask_flat.size} voxels but data contains {total_voxels}; ignoring mask."
            )
        else:
            combined_mask = mask_flat

    if signal_mask is not None:
        sig_flat = np.asarray(signal_mask, dtype=bool).reshape(-1)
        if sig_flat.size != total_voxels:
            print(
                "Warning: Signal mask contains"
                f" {sig_flat.size} voxels but data contains {total_voxels}; ignoring signal mask."
            )
        else:
            combined_mask = sig_flat if combined_mask is None else (combined_mask & sig_flat)

    if combined_mask is not None:
        indices = np.where(combined_mask)[0]
        voxels_to_fit = voxels[indices]
    else:
        indices = np.arange(total_voxels)
        voxels_to_fit = voxels

    partial_fit = functools.partial(
        _fit_single,
        IsVFA=IsVFA,
        TI_values=TI_values,
        alfas=alfas,
        TRs=TRs,
        tf_alpha_rad=kwargs.get("tf_alpha_rad"),
        tf_tr_ms=kwargs.get("tf_tr_ms"),
        tf_nph=kwargs.get("tf_nph"),
        recovery_model=kwargs.get("recovery_model"),
    )

    if getattr(settings, "MULTIPROCESSING", False):
        n_cores = int(getattr(settings, "NUMBER_OF_CORES", 1) or 1)

        # On macOS, the default start method is often 'spawn', which can be
        # significantly slower and more fragile for large numeric workloads.
        # Prefer 'fork' unless the user overrides it.
        start_method = (os.environ.get("P_BRAIN_MP_START_METHOD") or "").strip().lower()
        if not start_method:
            start_method = "fork" if sys.platform == "darwin" else multiprocessing.get_start_method(allow_none=True) or "spawn"

        try:
            ctx = multiprocessing.get_context(start_method)
        except Exception:
            ctx = multiprocessing.get_context("spawn")

        pool = ctx.Pool(processes=max(1, n_cores))
        try:
            with auto_logging_suppressed():
                iterator = tqdm(
                    pool.imap(partial_fit, voxels_to_fit, chunksize=256),
                    total=len(indices),
                    desc=" Fitting Voxel Matrix",
                )
                results = list(iterator)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        except Exception:
            pool.terminate()
            pool.join()
            raise
        else:
            pool.close()
            pool.join()
    else:
        with auto_logging_suppressed():
            iterator = tqdm(voxels_to_fit, total=len(indices), desc=" Fitting Voxel Matrix")
            results = [partial_fit(v) for v in iterator]

    M0_flat = np.full(total_voxels, np.nan)
    T1_flat = np.full(total_voxels, np.nan)

    fitted_M0 = np.array([r[0] for r in results])
    fitted_T1 = np.array([r[1] for r in results])

    # ------------------------------------------------------------------
    # Normalise T1 to milliseconds.  The turboflash recovery model
    # returns T1 in ms, but the saturation/inversion models fit in
    # seconds.  Downstream code (turboflash CTC conversion) always
    # divides by 1000, so we must ensure the stored matrix is in ms.
    # Heuristic: if the median positive T1 is < 50 it is in seconds.
    # ------------------------------------------------------------------
    valid_t1 = fitted_T1[np.isfinite(fitted_T1) & (fitted_T1 > 0)]
    if valid_t1.size > 0:
        t1_median = float(np.median(valid_t1))
        if np.isfinite(t1_median) and t1_median < 50.0:
            print(f"Info: T1 median = {t1_median:.4f} (appears to be in seconds); converting to milliseconds.")
            fitted_T1 = fitted_T1 * 1000.0

    M0_flat[indices] = fitted_M0
    T1_flat[indices] = fitted_T1

    M0_values = M0_flat.reshape(shape_x, shape_y, shape_z)
    T1_values = T1_flat.reshape(shape_x, shape_y, shape_z)
    return M0_values, T1_values

def plot_histograms(M0_matrix, T1_matrix, image_directory):
    def on_esc(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)
    M0_values = M0_matrix.flatten()
    T1_values = T1_matrix.flatten()

    M0_values = M0_values[np.isfinite(M0_values)]
    T1_values = T1_values[np.isfinite(T1_values)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot histogram of M0 values
    axes[0].hist(M0_values, bins=50, color=blaa1, histtype='step')
    axes[0].set_title('Histogram of M0 Values', fontproperties=prop, fontsize=14)
    axes[0].set_xlabel('Equilibrium Magnetisation [M0]', fontproperties=prop, fontsize=12)
    axes[0].set_ylabel('Frequency', fontproperties=prop, fontsize=12)
    axes[0].grid(which='minor', alpha=0.25)
    axes[0].minorticks_on()
    axes[0].set_yscale('log')

    # Plot histogram of T1 values
    axes[1].hist(T1_values, bins=50, color=roed, histtype='step')
    axes[1].set_title('Histogram of T1 Values', fontproperties=prop, fontsize=12)
    axes[1].set_xlabel('Longitudinal (T1) Relaxation Time [ms]', fontproperties=prop, fontsize=12)
    axes[1].set_ylabel('Frequency', fontproperties=prop, fontsize=12)
    axes[1].grid(which='minor', alpha=0.25)
    axes[1].minorticks_on()
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(image_directory, 'Fit', 'M0+T1_Histogram.png'), dpi=200)
    plt.gcf().canvas.mpl_connect('key_press_event', on_esc) 
    if not turbo_mode:
        close_plot_after_delay(3, fig)
        plt.show()


def plot_brain_slices_grid(M0_matrix, T1_matrix, image_directory, mask=None, output_name='M0+T1_Maps.png'):
    def on_esc(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)

    ensured_mask = None
    if mask is not None:
        ensured_mask = _ensure_mask_matches_shape(mask, M0_matrix.shape)
        if ensured_mask is None:
            print("Warning: Provided segmentation mask does not match map dimensions; skipping mask.")

    slices = M0_matrix.shape[2]
    grid_size = slices

    fig, axes = plt.subplots(2, grid_size, figsize=(20, 6))
    for i in range(slices):
        ax = axes[0, i]
        slice_data = np.ma.masked_invalid(T1_matrix[:, :, i])
        if ensured_mask is not None:
            slice_mask = ensured_mask[:, :, i]
            slice_data = np.ma.masked_where(~slice_mask, slice_data)
        im_t1 = ax.imshow(slice_data.T, cmap='viridis', origin='lower')
        ax.axis('off')
        ax.set_title(f'Slice {i+1}')
    cbar_ax_t1 = fig.add_axes([0.92, 0.58, 0.01, 0.3])
    cbar_t1 = fig.colorbar(im_t1, cax=cbar_ax_t1)
    cbar_t1.set_label('Longitudinal (T1) Relaxation Time [ms]', fontproperties=prop, fontsize=9) 
    for i in range(slices):
        ax = axes[1, i]
        slice_data = np.ma.masked_invalid(M0_matrix[:, :, i])
        if ensured_mask is not None:
            slice_mask = ensured_mask[:, :, i]
            slice_data = np.ma.masked_where(~slice_mask, slice_data)
        im_m0 = ax.imshow(slice_data.T, cmap='plasma', origin='lower')
        ax.axis('off')
    cbar_ax_m0 = fig.add_axes([0.92, 0.15, 0.01, 0.3])
    cbar_m0 = fig.colorbar(im_m0, cax=cbar_ax_m0)
    cbar_m0.set_label('Equilibrium Magnetization [M0]', fontproperties=prop, fontsize=9)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(os.path.join(image_directory, 'Fit', output_name), dpi=200)
    plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
    if not turbo_mode:
        close_plot_after_delay(3, fig)
        plt.show()


def _maybe_write_matlab_t1m0_compare(data_directory: str, analysis_directory: str, image_directory: str) -> None:
    """Best-effort QA: compare MATLAB T1/M0 .mat exports against p-brain maps.

    Looks for a MATLAB file matching '*T1*M0*plusError*maps*.mat' in the dataset
    root (or an explicit `P_BRAIN_T1M0_COMPARE_MAT_PATH`). When present, writes:
    - Images/Fit/T1_matlab_pbrain_diff.png
    - Images/Fit/M0_matlab_pbrain_diff.png

    Each output is a 10-row grid with 3 columns: MATLAB | p-brain | diff.
    """

    # Comparisons are opt-in. Enable via `--compare-matlab` (sets env), or by
    # setting `P_BRAIN_COMPARE_MATLAB=1` directly. Keep legacy envs for back-compat.
    compare_env = (os.environ.get('P_BRAIN_COMPARE_MATLAB') or '').strip().lower()
    legacy_env = (os.environ.get('P_BRAIN_T1M0_COMPARE') or '').strip().lower()
    if compare_env not in {'1', 'true', 'yes', 'on'} and legacy_env not in {'1', 'true', 'yes', 'on'}:
        return

    mat_path = (os.environ.get('P_BRAIN_T1M0_COMPARE_MAT_PATH') or '').strip()
    if not mat_path:
        try:
            import glob

            matches = glob.glob(os.path.join(data_directory, '*T1*M0*plusError*maps*.mat'))
            matches += glob.glob(os.path.join(data_directory, '*t1*m0*plusError*maps*.mat'))
            mat_path = matches[0] if matches else ''
        except Exception:
            mat_path = ''

    if not mat_path or not os.path.isfile(mat_path):
        return

    fitting_dir = os.path.join(analysis_directory, 'Fitting')
    t1_nii = os.path.join(fitting_dir, 't1_map.nii.gz')
    m0_nii = os.path.join(fitting_dir, 'm0_map.nii.gz')
    if not (os.path.isfile(t1_nii) and os.path.isfile(m0_nii)):
        return

    try:
        from scipy.io import loadmat
    except Exception:
        return

    try:
        md = loadmat(mat_path)
        mat_m0 = np.asarray(md.get('m0_map'), dtype=float)
        mat_r1 = np.asarray(md.get('r1_map'), dtype=float) if md.get('r1_map') is not None else None
        mat_t1 = np.asarray(md.get('t1_map'), dtype=float) if md.get('t1_map') is not None else None
        if mat_m0.ndim != 3:
            return
        if mat_r1 is not None and mat_r1.ndim != 3:
            mat_r1 = None
        if mat_t1 is not None and mat_t1.ndim != 3:
            mat_t1 = None
        # Prefer comparing T1 directly when available.
        if mat_t1 is None and mat_r1 is None:
            return
    except Exception:
        return

    try:
        pb_t1 = np.asarray(nib.load(t1_nii).get_fdata(), dtype=float)
        pb_m0 = np.asarray(nib.load(m0_nii).get_fdata(), dtype=float)
        pb_t1 = np.squeeze(pb_t1)
        pb_m0 = np.squeeze(pb_m0)
        if pb_t1.ndim != 3 or pb_m0.ndim != 3:
            return
    except Exception:
        return

    ref_z = mat_t1.shape[2] if mat_t1 is not None else mat_r1.shape[2]
    z = int(min(ref_z, mat_m0.shape[2], pb_t1.shape[2], pb_m0.shape[2], 10))
    if z <= 0:
        return

    if mat_r1 is not None:
        mat_r1 = mat_r1[:, :, :z]
    if mat_t1 is not None:
        mat_t1 = mat_t1[:, :, :z]
    mat_m0 = mat_m0[:, :, :z]
    pb_t1 = pb_t1[:, :, :z]
    pb_m0 = pb_m0[:, :, :z]

    # Convert p-brain T1 to seconds for direct comparison to MATLAB t1_map.
    # Heuristic: median > 50 => ms.
    try:
        t1_vals = pb_t1[np.isfinite(pb_t1) & (pb_t1 > 0)]
        t1_med = float(np.median(t1_vals)) if t1_vals.size else float('nan')
    except Exception:
        t1_med = float('nan')
    pb_t1_s = pb_t1 * (1e-3 if (np.isfinite(t1_med) and t1_med > 50.0) else 1.0)

    # Best-effort orientation alignment search (rot90 + flips) for numeric comparison.
    def _apply_xform(vol3d: np.ndarray, rot_k: int, flip_ud: bool, flip_lr: bool) -> np.ndarray:
        out = np.stack([np.rot90(vol3d[:, :, k], int(rot_k) % 4) for k in range(vol3d.shape[2])], axis=2)
        if flip_ud:
            out = np.flipud(out)
        if flip_lr:
            out = np.fliplr(out)
        return out

    def _best_xform(mat_vol: np.ndarray, pb_vol: np.ndarray):
        best = None
        best_score = None
        for rot_k in range(4):
            for flip_ud in (False, True):
                for flip_lr in (False, True):
                    cand = _apply_xform(pb_vol, rot_k, flip_ud, flip_lr)
                    if cand.shape != mat_vol.shape:
                        continue
                    m = np.isfinite(mat_vol) & np.isfinite(cand) & (mat_vol != 0)
                    if np.count_nonzero(m) < 100:
                        continue
                    d = cand[m] - mat_vol[m]
                    score = float(np.nanmean(d * d))
                    if best_score is None or score < best_score:
                        best_score = score
                        best = (rot_k, flip_ud, flip_lr)
        return best, best_score

    # Choose reference volume for alignment: MATLAB t1_map if available, else r1_map.
    mat_ref = mat_t1 if mat_t1 is not None else mat_r1
    pb_ref = pb_t1_s if mat_t1 is not None else (1.0 / pb_t1_s)
    with np.errstate(divide='ignore', invalid='ignore'):
        if mat_t1 is None:
            pb_ref = 1.0 / pb_t1_s

    xf, xf_score = _best_xform(mat_ref, pb_ref)
    if xf is None:
        xf = (1, False, False)  # legacy display default
    rot_k, flip_ud, flip_lr = xf

    pb_t1_s = _apply_xform(pb_t1_s, rot_k, flip_ud, flip_lr)
    pb_m0 = _apply_xform(pb_m0, rot_k, flip_ud, flip_lr)
    if mat_r1 is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            pb_r1 = 1.0 / pb_t1_s
    else:
        pb_r1 = None

    # Also align MATLAB volumes for consistent rendering (keep MATLAB as-is; pb is transformed).
    if xf_score is not None:
        print(f"Info: MATLAB compare alignment rot90={rot_k}, flip_ud={flip_ud}, flip_lr={flip_lr}, mse={xf_score:.6g}")

    def _robust_range(a, b):
        x = np.concatenate([
            np.ravel(np.asarray(a, dtype=float)),
            np.ravel(np.asarray(b, dtype=float)),
        ])
        x = x[np.isfinite(x)]
        x = x[x != 0]
        if x.size == 0:
            return 0.0, 1.0
        lo = float(np.percentile(x, 1))
        hi = float(np.percentile(x, 99))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(x))
            hi = float(np.max(x))
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def _diff_range(d):
        x = np.ravel(np.asarray(d, dtype=float))
        x = x[np.isfinite(x)]
        if x.size == 0:
            return -1.0, 1.0
        m = float(np.percentile(np.abs(x), 99))
        if not np.isfinite(m) or m <= 0:
            m = float(np.max(np.abs(x)))
        if not np.isfinite(m) or m <= 0:
            m = 1.0
        return -m, m

    def _copy_cmap(name: str, n: int | None = None):
        # Matplotlib returns shared instances; copy so we can set bad/under/over safely.
        try:
            base = plt.get_cmap(name, n) if n is not None else plt.get_cmap(name)
            return base.copy()
        except Exception:
            base = plt.get_cmap(name, n) if n is not None else plt.get_cmap(name)
            return base

    def _mask_for_display(vol: np.ndarray, *, zero_is_bg: bool) -> np.ndarray:
        v = np.asarray(vol, dtype=float)
        m = ~np.isfinite(v)
        if zero_is_bg:
            m = m | (v == 0)
        return np.ma.array(v, mask=m)

    def _write_coverage(out_path: str, label: str, mat_vol: np.ndarray, pb_vol: np.ndarray) -> None:
        os.makedirs(os.path.join(image_directory, 'Fit'), exist_ok=True)

        mat_vol = np.asarray(mat_vol, dtype=float)
        pb_vol = np.asarray(pb_vol, dtype=float)

        mat_mask = np.isfinite(mat_vol) & (mat_vol != 0)
        pb_mask = np.isfinite(pb_vol) & (pb_vol != 0)

        overlap = mat_mask & pb_mask
        missing_in_pb = mat_mask & ~pb_mask
        extra_in_pb = pb_mask & ~mat_mask

        mat_n = int(np.count_nonzero(mat_mask))
        pb_n = int(np.count_nonzero(pb_mask))
        ov_n = int(np.count_nonzero(overlap))
        miss_n = int(np.count_nonzero(missing_in_pb))
        extra_n = int(np.count_nonzero(extra_in_pb))

        cov_mat = (ov_n / mat_n * 100.0) if mat_n else 0.0
        cov_pb = (ov_n / pb_n * 100.0) if pb_n else 0.0

        fig, axs = plt.subplots(nrows=z, ncols=4, figsize=(10, 2.5 * z), dpi=160)
        if z == 1:
            axs = np.asarray([axs])

        cm_mask = _copy_cmap('gray')
        cm_mask.set_bad('black')
        cm_miss = _copy_cmap('Reds')
        cm_miss.set_bad('black')
        cm_extra = _copy_cmap('Blues')
        cm_extra.set_bad('black')

        for k in range(z):
            mm = mat_mask[:, :, k]
            pm = pb_mask[:, :, k]
            mi = missing_in_pb[:, :, k]
            ex = extra_in_pb[:, :, k]

            axs[k, 0].imshow(np.ma.array(mm.astype(float), mask=~np.isfinite(mm)), cmap=cm_mask, vmin=0, vmax=1, origin='lower')
            axs[k, 1].imshow(np.ma.array(pm.astype(float), mask=~np.isfinite(pm)), cmap=cm_mask, vmin=0, vmax=1, origin='lower')
            axs[k, 2].imshow(np.ma.array(mi.astype(float), mask=~np.isfinite(mi)), cmap=cm_miss, vmin=0, vmax=1, origin='lower')
            axs[k, 3].imshow(np.ma.array(ex.astype(float), mask=~np.isfinite(ex)), cmap=cm_extra, vmin=0, vmax=1, origin='lower')

            axs[k, 0].set_ylabel(f'slice {k+1}', fontsize=8)
            for j in range(4):
                h, w = mm.shape
                axs[k, j].set_xticks([0, int(w // 2), int(w - 1)])
                axs[k, j].set_yticks([0, int(h // 2), int(h - 1)])
                axs[k, j].tick_params(labelsize=6)

        axs[0, 0].set_title('MATLAB mask (!=0)', fontsize=10)
        axs[0, 1].set_title('p-brain mask (!=0)', fontsize=10)
        axs[0, 2].set_title('missing in p-brain', fontsize=10)
        axs[0, 3].set_title('extra in p-brain', fontsize=10)

        fig.suptitle(
            f'{label} coverage ({os.path.basename(mat_path)})\n'
            f'mat={mat_n} pb={pb_n} overlap={ov_n}  missing={miss_n} extra={extra_n}  '
            f'overlap/mat={cov_mat:.2f}%  overlap/pb={cov_pb:.2f}%',
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(out_path)
        plt.close(fig)

    def _render(out_path, label, mat_vol, pb_vol):
        os.makedirs(os.path.join(image_directory, 'Fit'), exist_ok=True)
        vmin, vmax = _robust_range(mat_vol, pb_vol)
        diff = pb_vol - mat_vol
        dvmin, dvmax = _diff_range(diff)

        # Ensure NaNs/invalid render as black so MATLAB(zeros) and p-brain(NaNs) backgrounds look comparable.
        gray_cmap = _copy_cmap('gray')
        gray_cmap.set_bad('black')
        diff_cmap = _copy_cmap('coolwarm', 13)
        diff_cmap.set_bad('black')

        fig, axs = plt.subplots(nrows=z, ncols=3, figsize=(9, 3 * z), dpi=150)
        if z == 1:
            axs = np.asarray([axs])

        for k in range(z):
            a = _mask_for_display(mat_vol[:, :, k], zero_is_bg=True)
            b = _mask_for_display(pb_vol[:, :, k], zero_is_bg=True)
            d = _mask_for_display(diff[:, :, k], zero_is_bg=False)
            axs[k, 0].imshow(a, cmap=gray_cmap, vmin=vmin, vmax=vmax, origin='lower')
            axs[k, 1].imshow(b, cmap=gray_cmap, vmin=vmin, vmax=vmax, origin='lower')
            axs[k, 2].imshow(d, cmap=diff_cmap, vmin=dvmin, vmax=dvmax, origin='lower')

            axs[k, 0].set_ylabel(f'slice {k+1}', fontsize=8)
            for j in range(3):
                # Keep axes on for quantitative inspection.
                h, w = a.shape
                axs[k, j].set_xticks([0, int(w // 2), int(w - 1)])
                axs[k, j].set_yticks([0, int(h // 2), int(h - 1)])
                axs[k, j].tick_params(labelsize=6)

        axs[0, 0].set_title('MATLAB', fontsize=10)
        axs[0, 1].set_title('p-brain', fontsize=10)
        axs[0, 2].set_title('diff (p-brain - MATLAB)', fontsize=10)
        fig.suptitle(f'{label} comparison ({os.path.basename(mat_path)})', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(out_path)
        plt.close(fig)

    def _render_quant(out_path: str, label: str, mat_vol: np.ndarray, pb_vol: np.ndarray, *, unit: str, diff_bins, rel_bins_pct) -> None:
        """Quantitative comparison plot with discrete colorbars.

        Produces a grid per-slice showing:
        - signed diff (pb - matlab) with discrete bins
        - abs diff with discrete bins
        - relative abs diff (%) with discrete bins
        Each panel includes axis ticks; each row includes slice-level summary.
        """

        from matplotlib.colors import BoundaryNorm

        os.makedirs(os.path.join(image_directory, 'Fit'), exist_ok=True)

        mat_vol = np.asarray(mat_vol, dtype=float)
        pb_vol = np.asarray(pb_vol, dtype=float)

        # Compare only where MATLAB has signal and both are finite.
        base_mask = np.isfinite(mat_vol) & np.isfinite(pb_vol) & (mat_vol != 0)
        if not np.any(base_mask):
            return

        diff = np.where(base_mask, (pb_vol - mat_vol), np.nan)
        abs_diff = np.abs(diff)
        rel_pct = np.where(base_mask, (abs_diff / (np.abs(mat_vol) + 1e-12)) * 100.0, np.nan)

        # Global stats
        ad = abs_diff[base_mask]
        rp = rel_pct[base_mask]
        g = {
            "n": int(np.count_nonzero(base_mask)),
            "abs_median": float(np.nanmedian(ad)),
            "abs_mean": float(np.nanmean(ad)),
            "abs_p95": float(np.nanpercentile(ad, 95)),
            "abs_max": float(np.nanmax(ad)),
            "rel_median_pct": float(np.nanmedian(rp)),
            "rel_p95_pct": float(np.nanpercentile(rp, 95)),
            "rel_max_pct": float(np.nanmax(rp)),
        }

        # Discrete bins/norms
        diff_bins = [float(x) for x in diff_bins]
        rel_bins_pct = [float(x) for x in rel_bins_pct]
        abs_bins = [0.0] + [abs(x) for x in diff_bins if x > 0]
        abs_bins = sorted(set(abs_bins))

        diff_cmap = plt.get_cmap('coolwarm', len(diff_bins) - 1)
        abs_cmap = plt.get_cmap('viridis', len(abs_bins) - 1)
        rel_cmap = plt.get_cmap('magma', len(rel_bins_pct) - 1)

        diff_norm = BoundaryNorm(diff_bins, diff_cmap.N, clip=True)
        abs_norm = BoundaryNorm(abs_bins, abs_cmap.N, clip=True)
        rel_norm = BoundaryNorm(rel_bins_pct, rel_cmap.N, clip=True)

        fig, axs = plt.subplots(nrows=z, ncols=3, figsize=(11.5, 2.5 * z), dpi=160)
        if z == 1:
            axs = np.asarray([axs])

        # Track last images for colorbars
        im_diff = None
        im_abs = None
        im_rel = None

        for k in range(z):
            m2 = base_mask[:, :, k]
            if not np.any(m2):
                # Still render empty panels with axes.
                h, w = mat_vol[:, :, k].shape
                for j in range(3):
                    axs[k, j].set_xticks([0, int(w // 2), int(w - 1)])
                    axs[k, j].set_yticks([0, int(h // 2), int(h - 1)])
                    axs[k, j].tick_params(labelsize=6)
                axs[k, 0].set_ylabel(f'slice {k+1}', fontsize=8)
                continue

            d2 = diff[:, :, k]
            ad2 = abs_diff[:, :, k]
            rp2 = rel_pct[:, :, k]

            # Slice stats
            s_ad = ad2[m2]
            s_rp = rp2[m2]
            s = {
                "abs_med": float(np.nanmedian(s_ad)),
                "abs_max": float(np.nanmax(s_ad)),
                "rel_med": float(np.nanmedian(s_rp)),
                "rel_max": float(np.nanmax(s_rp)),
            }

            im_diff = axs[k, 0].imshow(d2, cmap=diff_cmap, norm=diff_norm, origin='lower')
            im_abs = axs[k, 1].imshow(ad2, cmap=abs_cmap, norm=abs_norm, origin='lower')
            im_rel = axs[k, 2].imshow(rp2, cmap=rel_cmap, norm=rel_norm, origin='lower')

            axs[k, 0].set_ylabel(f'slice {k+1}\n|Δ| med={s["abs_med"]:.3g}{unit}\n|Δ| max={s["abs_max"]:.3g}{unit}', fontsize=7)
            axs[k, 2].set_title(f'|Δ|% med={s["rel_med"]:.3g}%  max={s["rel_max"]:.3g}%', fontsize=7)

            for j in range(3):
                h, w = d2.shape
                axs[k, j].set_xticks([0, int(w // 2), int(w - 1)])
                axs[k, j].set_yticks([0, int(h // 2), int(h - 1)])
                axs[k, j].tick_params(labelsize=6)

        axs[0, 0].set_title('Δ = p-brain − MATLAB', fontsize=10)
        axs[0, 1].set_title('|Δ|', fontsize=10)
        axs[0, 2].set_title('|Δ| (%)', fontsize=10)

        fig.suptitle(
            f'{label} quantitative diff ({os.path.basename(mat_path)})\n'
            f"N={g['n']}  |Δ| med={g['abs_median']:.3g}{unit}  mean={g['abs_mean']:.3g}{unit}  p95={g['abs_p95']:.3g}{unit}  max={g['abs_max']:.3g}{unit}   "
            f"|Δ|% med={g['rel_median_pct']:.3g}%  p95={g['rel_p95_pct']:.3g}%  max={g['rel_max_pct']:.3g}%",
            fontsize=10,
        )

        # Colorbars with discrete ticks and visible axes.
        # Place one colorbar per column.
        fig.subplots_adjust(top=0.94, right=0.88, wspace=0.15, hspace=0.25)
        cax0 = fig.add_axes([0.90, 0.68, 0.02, 0.22])
        cax1 = fig.add_axes([0.90, 0.40, 0.02, 0.22])
        cax2 = fig.add_axes([0.90, 0.12, 0.02, 0.22])

        unit_suffix = f' ({unit})' if unit else ''

        cb0 = fig.colorbar(im_diff, cax=cax0, ticks=diff_bins)
        cb0.set_label(f'Δ{unit_suffix}', fontsize=8)
        cb0.ax.tick_params(labelsize=7)

        cb1 = fig.colorbar(im_abs, cax=cax1, ticks=abs_bins)
        cb1.set_label(f'|Δ|{unit_suffix}', fontsize=8)
        cb1.ax.tick_params(labelsize=7)

        cb2 = fig.colorbar(im_rel, cax=cax2, ticks=rel_bins_pct)
        cb2.set_label('|Δ| (%)', fontsize=8)
        cb2.ax.tick_params(labelsize=7)

        plt.savefig(out_path)
        plt.close(fig)

    try:
        if mat_t1 is not None:
            _render(os.path.join(image_directory, 'Fit', 'T1_matlab_pbrain_diff.png'), 'T1 (s)', mat_t1, pb_t1_s)
            _write_coverage(os.path.join(image_directory, 'Fit', 'T1_matlab_pbrain_coverage.png'), 'T1 (s)', mat_t1, pb_t1_s)
            # Quantitative plot: choose bins appropriate for typical matching errors.
            _render_quant(
                os.path.join(image_directory, 'Fit', 'T1_matlab_pbrain_quant.png'),
                'T1 (s)',
                mat_t1,
                pb_t1_s,
                unit='s',
                diff_bins=[-2e-3, -1e-3, -5e-4, -2e-4, -1e-4, -5e-5, 0.0, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3],
                rel_bins_pct=[0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            )
        if mat_r1 is not None and pb_r1 is not None:
            _render(os.path.join(image_directory, 'Fit', 'R1_matlab_pbrain_diff.png'), 'R1 (1/s)', mat_r1, pb_r1)
        _render(os.path.join(image_directory, 'Fit', 'M0_matlab_pbrain_diff.png'), 'M0', mat_m0, pb_m0)
        _write_coverage(os.path.join(image_directory, 'Fit', 'M0_matlab_pbrain_coverage.png'), 'M0', mat_m0, pb_m0)
        _render_quant(
            os.path.join(image_directory, 'Fit', 'M0_matlab_pbrain_quant.png'),
            'M0',
            mat_m0,
            pb_m0,
            unit='',
            diff_bins=[-2.0, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            rel_bins_pct=[0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
        )
    except Exception:
        return


def _maybe_write_orientation_debug(*, analysis_directory: str, nifti_directory: str, image_directory: str, dce_filename: str) -> None:
    """Best-effort QA: write a quick orientation sanity-check image.

    Shows the raw NIfTI slice and the display-rotated slice used by the MATLAB
    comparison (90° CCW) for DCE/T1/M0.

    Output:
      Images/Fit/orientation_debug.png
    """

    compare_env = (os.environ.get('P_BRAIN_COMPARE_MATLAB') or '').strip().lower()
    legacy_env = (os.environ.get('P_BRAIN_T1M0_COMPARE') or '').strip().lower()
    if compare_env not in {'1', 'true', 'yes', 'on'} and legacy_env not in {'1', 'true', 'yes', 'on'}:
        return

    fitting_dir = os.path.join(analysis_directory, 'Fitting')
    t1_nii = os.path.join(fitting_dir, 't1_map.nii.gz')
    m0_nii = os.path.join(fitting_dir, 'm0_map.nii.gz')
    if not (os.path.isfile(t1_nii) and os.path.isfile(m0_nii)):
        return

    dce_path = os.path.join(nifti_directory, dce_filename) if dce_filename else ''
    if not dce_path or not os.path.isfile(dce_path):
        # DCE may be absent for some runs; still write T1/M0 debug.
        dce_path = ''

    def _load_3d(path: str) -> np.ndarray:
        a = np.asarray(nib.load(path).get_fdata(), dtype=float)
        a = np.squeeze(a)
        if a.ndim == 4:
            a = a[:, :, :, 0]
        return np.squeeze(a)

    def _rot_ccw2(a2: np.ndarray) -> np.ndarray:
        return np.rot90(a2, 1)

    try:
        t1 = _load_3d(t1_nii)
        m0 = _load_3d(m0_nii)
        dce = _load_3d(dce_path) if dce_path else None
        if t1.ndim != 3 or m0.ndim != 3:
            return
        z = int(min(t1.shape[2], m0.shape[2]))
        if z <= 0:
            return
        z0 = z // 2

        os.makedirs(os.path.join(image_directory, 'Fit'), exist_ok=True)

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 10), dpi=150)

        def _imshow_pair(row: int, vol: np.ndarray, title: str, *, cmap: str = 'gray'):
            a = np.asarray(vol[:, :, z0], dtype=float)
            # Robust window to avoid outliers.
            vals = a[np.isfinite(a)]
            if vals.size:
                vmin = float(np.percentile(vals, 2))
                vmax = float(np.percentile(vals, 98))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    vmin, vmax = None, None
            else:
                vmin, vmax = None, None
            axs[row, 0].imshow(a, cmap=cmap, vmin=vmin, vmax=vmax)
            axs[row, 1].imshow(_rot_ccw2(a), cmap=cmap, vmin=vmin, vmax=vmax)
            axs[row, 0].set_title(f'{title} (raw)', fontsize=10)
            axs[row, 1].set_title(f'{title} (rot90 CCW)', fontsize=10)
            h, w = a.shape
            for j in (0, 1):
                axs[row, j].set_xticks([0, int(w // 2), int(w - 1)])
                axs[row, j].set_yticks([0, int(h // 2), int(h - 1)])
                axs[row, j].tick_params(labelsize=7)

        if dce is not None and isinstance(dce, np.ndarray) and dce.ndim == 3:
            _imshow_pair(0, dce, 'DCE t=0', cmap='gray')
        else:
            axs[0, 0].axis('off')
            axs[0, 1].axis('off')
            axs[0, 0].set_title('DCE (missing)', fontsize=10)
            axs[0, 1].set_title('DCE (missing)', fontsize=10)

        _imshow_pair(1, t1, 'T1 map', cmap='gray')
        _imshow_pair(2, m0, 'M0 map', cmap='gray')

        fig.suptitle('Orientation debug (what p-brain-web will load)', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = os.path.join(image_directory, 'Fit', 'orientation_debug.png')
        plt.savefig(out_path)
        plt.close(fig)
    except Exception:
        return


# --------------------------------------------------------------------------- #
#   M0 / DCE intensity-scale recalibration                                    #
# --------------------------------------------------------------------------- #

def _recalibrate_m0_to_dce(
    M0_matrix: np.ndarray,
    T1_matrix: np.ndarray,
    nifti_directory: str,
    dce_filename: str | None,
    *,
    baseline_frames: tuple[int, int] = (2, 10),
    tolerance: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale M0 so that the TurboFLASH signal prediction matches DCE baseline.

    When T1/M0 are fitted from an IR (or VFA) series whose NIfTI proxy slope
    differs from the DCE NIfTI proxy slope, the fitted M0 lives on a different
    intensity scale than the DCE signal.  This manifests as a wrong
    ``S / (M0 * sin α)`` ratio in the turboflash CTC conversion.

    The function computes a per-slice correction factor by comparing the
    actual mean DCE baseline signal with the signal predicted from M0/T1 via
    the TurboFLASH saturation-recovery equation:

        S_predicted = M0 · sin α · (1 − exp(−TD / T1))

    where TD is the TurboFLASH inversion/recovery time (typically 120 ms).

    This uses the **same signal model** as the CTC conversion formula, so the
    calibrated M0 guarantees that ``S / (M0 * sin α) = 1 − exp(−TD/T1)``
    which is always in (0, 1) for any positive T1 — the log argument in the
    CTC formula is therefore always valid regardless of enhancement level.

    If the median ratio ``S_actual / S_predicted`` deviates from 1.0 by more
    than *tolerance* for any slice, M0 for that slice is multiplied by the
    ratio so that the CTC conversion receives internally consistent data.

    When the ratios are within tolerance, M0 and T1 are returned unchanged.

    Returns (M0_matrix, T1_matrix) — T1 is never modified.
    """

    # Guard: need DCE NIfTI + metadata ----------------------------------------
    if not dce_filename or not nifti_directory:
        return M0_matrix, T1_matrix

    dce_path = os.path.join(nifti_directory, dce_filename)
    if not os.path.isfile(dce_path):
        return M0_matrix, T1_matrix

    # Flip angle from JSON sidecar / environment -------------------------------
    try:
        flip_deg = resolve_flip_angle_deg(dce_path, default=None)
    except Exception:
        flip_deg = None
    if flip_deg is None:
        return M0_matrix, T1_matrix

    flip_deg = float(flip_deg)
    if not np.isfinite(flip_deg) or flip_deg <= 0:
        return M0_matrix, T1_matrix

    # TD (TurboFLASH inversion/recovery time) ----------------------------------
    try:
        td_s = resolve_turboflash_ti_s(dce_path, default=0.12)
    except Exception:
        td_s = 0.12
    if td_s is None or not np.isfinite(td_s) or td_s <= 0:
        td_s = 0.12

    # Load DCE baseline --------------------------------------------------------
    try:
        dce_img = nib.load(dce_path)
        dce_data = np.asarray(dce_img.dataobj, dtype=np.float32)
    except Exception:
        return M0_matrix, T1_matrix

    if dce_data.ndim < 4:
        return M0_matrix, T1_matrix

    # Spatial shape check
    spatial_dce = dce_data.shape[:3]
    if M0_matrix.shape != spatial_dce:
        # Shape mismatch between T1 fit grid and DCE grid — skip recalibration.
        return M0_matrix, T1_matrix

    bl_start, bl_end = baseline_frames
    bl_end = min(bl_end, dce_data.shape[3])
    if bl_start >= bl_end:
        return M0_matrix, T1_matrix

    dce_baseline = np.mean(dce_data[:, :, :, bl_start:bl_end], axis=-1)

    # TurboFLASH saturation-recovery prediction per voxel ----------------------
    alpha_rad = np.radians(flip_deg)
    sin_a = float(np.sin(alpha_rad))
    if abs(sin_a) < 1e-12:
        return M0_matrix, T1_matrix

    # T1_matrix is in milliseconds (post-normalisation); TD is in seconds
    t1_s = T1_matrix / 1000.0
    td_ms = td_s * 1000.0
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        s_predicted = M0_matrix * sin_a * (1.0 - np.exp(-td_ms / T1_matrix))

    # Per-slice scale factor ---------------------------------------------------
    n_slices = M0_matrix.shape[2] if M0_matrix.ndim >= 3 else 1
    any_corrected = False
    factors = []

    for k in range(n_slices):
        if M0_matrix.ndim >= 3:
            mask_k = (
                np.isfinite(T1_matrix[:, :, k]) &
                (T1_matrix[:, :, k] > 100) &
                np.isfinite(s_predicted[:, :, k]) &
                (s_predicted[:, :, k] > 0) &
                np.isfinite(dce_baseline[:, :, k]) &
                (dce_baseline[:, :, k] > 0)
            )
            s_act = dce_baseline[:, :, k][mask_k]
            s_pred = s_predicted[:, :, k][mask_k]
        else:
            mask_k = (
                np.isfinite(T1_matrix) & (T1_matrix > 100) &
                np.isfinite(s_predicted) & (s_predicted > 0) &
                np.isfinite(dce_baseline) & (dce_baseline > 0)
            )
            s_act = dce_baseline[mask_k]
            s_pred = s_predicted[mask_k]

        if s_act.size < 50:
            factors.append(1.0)
            continue

        ratio = s_act / s_pred
        factor = float(np.median(ratio))

        if not np.isfinite(factor) or factor <= 0:
            factors.append(1.0)
            continue

        factors.append(factor)
        if abs(factor - 1.0) > tolerance:
            any_corrected = True

    if not any_corrected:
        return M0_matrix, T1_matrix

    # Apply correction ---------------------------------------------------------
    M0_out = M0_matrix.copy()
    for k in range(n_slices):
        f = factors[k]
        if abs(f - 1.0) <= tolerance:
            continue
        print(
            f"Info: M0 recalibration slice {k+1}: "
            f"scale factor = {f:.4f} (M0 was {1/f:.1f}x too large for DCE scale)."
        )
        if M0_out.ndim >= 3:
            M0_out[:, :, k] *= f
        else:
            M0_out *= f

    return M0_out, T1_matrix


def T1_fit(data_directory, analysis_directory, nifti_directory, image_directory, filenames, parameters):
    _, IsVFA, IsIR, _, _, _, _ = parameters
    (
        t1_3D_filename,
        axial_t1_3D_filename,
        t2_3D_filename,
        axial_t2_3D_filename,
        flair_3D_filename,
        axial_flair_3D_filename,
        axial_t2_2D_filename,
        diffusion_filename,
        dce_filename,
    ) = filenames

    voxel_matrix_path = os.path.join(analysis_directory, 'Fitting', 'voxel_matrix.pkl')
    M0_matrix_path = os.path.join(analysis_directory, 'Fitting', 'voxel_M0_matrix.pkl')
    T1_matrix_path = os.path.join(analysis_directory, 'Fitting', 'voxel_T1_matrix.pkl')
    # Initialize alfas and TRs to default values (empty lists or None)
    alfas = []
    TRs = []

    voxel_matrix_exists = os.path.exists(voxel_matrix_path)
    M0_matrix_exists = os.path.exists(M0_matrix_path)
    T1_matrix_exists = os.path.exists(T1_matrix_path)

    voxel_matrix = None
    if voxel_matrix_exists:
        voxel_matrix = load_from_pickle(voxel_matrix_path)

    use_cached = voxel_matrix_exists and M0_matrix_exists and T1_matrix_exists

    if use_cached:
        M0_matrix = load_from_pickle(M0_matrix_path)
        T1_matrix = load_from_pickle(T1_matrix_path)
    else:
        # Decide which source to use for T1/M0 fitting.
        t1_mode = getattr(settings, "T1_FIT_MODE", "auto")
        if t1_mode not in {"auto", "ir", "vfa", "none"}:
            t1_mode = "auto"

        use_vfa = False
        use_ir = False

        if t1_mode == "vfa":
            use_vfa = True
        elif t1_mode == "ir":
            use_ir = True
        elif t1_mode == "none":
            use_vfa = False
            use_ir = False
        else:
            # auto: prefer complete IR series, otherwise fall back to VFA.
            ir_paths = discover_ir_series(nifti_directory)
            if ir_paths:
                use_ir = True
            else:
                vfa_series = discover_vfa_series(nifti_directory)
                if vfa_series:
                    use_vfa = True

        # Allow explicit legacy parameters to override auto when they are set.
        if t1_mode == "auto":
            if IsVFA:
                use_vfa = True
                use_ir = False
            elif IsIR:
                # Only keep IR if it is complete; otherwise let VFA/none handle it.
                if discover_ir_series(nifti_directory):
                    use_ir = True
                    use_vfa = False

        if use_vfa:
            vfa_series = discover_vfa_series(nifti_directory)
            if not vfa_series:
                raise FileNotFoundError(
                    "T1_FIT_MODE=vfa requested but no VFA series was discovered in "
                    f"{nifti_directory}. Provide VFA NIfTIs+JSON sidecars (with FlipAngle and RepetitionTime) "
                    "or set P_BRAIN_VFA_GLOB to match your filenames."
                )
            vfa_data = [entry["nifti"] for entry in vfa_series]
            alfas = [entry["flip_angle_deg"] for entry in vfa_series]
            TRs = [entry["tr_s"] for entry in vfa_series]
            if voxel_matrix is None:
                voxel_matrix = build_voxel_matrix(vfa_data)
            M0_matrix, T1_matrix = fit_all_voxels(voxel_matrix, None, True, alfas=alfas, TRs=TRs)
        elif use_ir:
            # MATLAB create_t1map_philips6_NewParRec2015 uses TI in seconds
            # (default [0.12 0.3 0.6 1 2 4 10]). Our IR filenames encode ms.
            TI_ms = [120, 300, 600, 1000, 2000, 4000, 10000]
            TI_values = [v * 1e-3 for v in TI_ms]  # seconds
            ir_paths = discover_ir_series(nifti_directory)
            if not ir_paths:
                patterns = ['WIPTI_', 'WIPDelRec-TI_']
                TI_codes = [f"{v:05d}" for v in TI_ms]
                raise FileNotFoundError(
                    "Missing inversion-recovery NIfTI files for TI "
                    f"{TI_codes} in {nifti_directory}. Expected filenames like "
                    f"{patterns[0]}<TI>.nii(.gz) or {patterns[1]}<TI>.nii(.gz)."
                )
            if voxel_matrix is None:
                # MATLAB case-3 uses dynamic=2 (1-based) from the PAR/REC.
                voxel_matrix = build_voxel_matrix(ir_paths, volume_index=1)

            # Optional MATLAB-style background suppression: only fit voxels
            # whose longest-TI signal exceeds a fraction of the per-slice max.
            signal_mask = None
            try:
                frac = float(os.environ.get("P_BRAIN_T1FIT_SIGNAL_FRACTION", "0.1"))
            except Exception:
                frac = 0.1
            if np.isfinite(frac) and frac > 0:
                try:
                    ref = np.asarray(voxel_matrix[-1, :, :, :], dtype=float)
                    signal_mask = np.zeros(ref.shape, dtype=bool)
                    for k in range(ref.shape[2]):
                        sl = ref[:, :, k]
                        m = float(np.nanmax(sl)) if np.isfinite(sl).any() else 0.0
                        thr = float(frac) * m
                        if m > 0:
                            signal_mask[:, :, k] = np.isfinite(sl) & (sl > thr)
                except Exception:
                    signal_mask = None

            # MATLAB case-3 uses only a per-slice signal threshold (on the
            # longest-TI image). Do not additionally apply our Otsu-derived
            # brain mask by default, because it can shrink coverage and create
            # apparent "missing" regions vs MATLAB.
            brain_mask = None
            use_brain_mask_env = (os.environ.get('P_BRAIN_T1FIT_USE_BRAIN_MASK') or '').strip().lower()
            if use_brain_mask_env in {'1', 'true', 'yes', 'on'}:
                try:
                    brain_mask = _compute_mask_from_voxel_matrix(voxel_matrix)
                except Exception:
                    brain_mask = None

            # TurboFLASH TI-series fit parameters (used when P_BRAIN_T1_RECOVERY_MODEL=turboflash).
            tf_alpha_rad = None
            tf_tr_ms = None
            tf_nph = None
            try:
                flip_deg = resolve_flip_angle_deg(ir_paths[0], default=None)
                if flip_deg is not None and np.isfinite(flip_deg):
                    tf_alpha_rad = float(np.deg2rad(float(flip_deg)))
            except Exception:
                tf_alpha_rad = None
            try:
                # Match MATLAB extract_t1_philips6_NewParRec2015: TR is the sequence
                # repetition time (seconds). For dcm2niix NIfTIs this is typically
                # stored as RepetitionTimeExcitation.
                tr_s = resolve_turboflash_tr_s(ir_paths[0], default=None, mode="excitation")
                if tr_s is not None and float(tr_s) > 0:
                    tf_tr_ms = float(tr_s) * 1e3
            except Exception:
                tf_tr_ms = None
            try:
                # MATLAB default is nph=1 (low-high sampling).
                tf_nph = int(resolve_turboflash_nph(ir_paths[0], default=1))
            except Exception:
                tf_nph = 1

            if _resolve_t1_recovery_model(None) == "turboflash" and (tf_alpha_rad is None or tf_tr_ms is None):
                print(
                    "Warning: P_BRAIN_T1_RECOVERY_MODEL=turboflash requested but flip angle/TR metadata is missing for the TI series; "
                    "falling back to the simpler saturation model for T1/M0 fitting."
                )

            M0_matrix, T1_matrix = fit_all_voxels(
                voxel_matrix,
                TI_values,
                False,
                brain_mask=brain_mask,
                signal_mask=signal_mask,
                tf_alpha_rad=tf_alpha_rad,
                tf_tr_ms=tf_tr_ms,
                tf_nph=tf_nph,
                recovery_model=_resolve_t1_recovery_model(None),
            )
        else:
            # No fitting performed without IR or VFA data
            if voxel_matrix is not None:
                shape = voxel_matrix.shape[1:]
            else:
                ref_img = _reference_img_for_t1fit(nifti_directory, dce_filename, prefer_ir=True)
                if ref_img is not None:
                    shape = ref_img.shape[:3]
                else:
                    # NIfTI conversion may have failed (e.g. dcm2niix not
                    # installed on Windows).  Try to determine the 3-D shape
                    # directly from a PAR/REC file in the data directory.
                    shape = _shape_from_parrec(data_directory)
                    if shape is None:
                        print(
                            "[T1_fit] WARNING: No NIfTI files and no PAR/REC "
                            "files found – cannot determine volume shape.  "
                            "T1/M0 maps will be skipped."
                        )
                        # Return empty arrays so downstream never gets None.
                        M0_matrix = np.full((1,), np.nan)
                        T1_matrix = np.full((1,), np.nan)
                        # Jump past the NaN-fill below.
                        shape = None
                if shape is not None:
                    M0_matrix = np.full(shape, np.nan)
                    T1_matrix = np.full(shape, np.nan)

        # ------------------------------------------------------------------
        # M0 / DCE scale recalibration.
        #
        # When T1/M0 are fitted from an IR (or VFA) series whose NIfTI proxy
        # slope differs from the DCE NIfTI slope the fitted M0 lives on a
        # different intensity scale than the DCE signal.  The turboflash CTC
        # conversion computes S/(M0*sin α) and expects both numerator and
        # denominator to be on the same scale.
        #
        # Fix: use the TurboFLASH saturation-recovery equation
        #     S = M0 · sinα · (1 − exp(−TD/T1))
        # to predict the pre-contrast DCE signal from the fitted M0/T1,
        # compare with the actual DCE baseline, and scale M0 per-slice so
        # the two agree.  This ensures S/(M0·sinα) = 1−exp(−TD/T1) < 1
        # always, keeping the log argument in the CTC formula valid.
        # When the ratio is already close to 1.0 (±20 %) the step is a no-op.
        # ------------------------------------------------------------------
        M0_matrix, T1_matrix = _recalibrate_m0_to_dce(
            M0_matrix, T1_matrix,
            nifti_directory, dce_filename,
        )

        save_as_pickle(M0_matrix, os.path.join(analysis_directory, 'Fitting', 'voxel_M0_matrix.pkl'))
        save_as_pickle(T1_matrix, os.path.join(analysis_directory, 'Fitting', 'voxel_T1_matrix.pkl'))
        if not voxel_matrix_exists:
            save_as_pickle(voxel_matrix, os.path.join(analysis_directory, 'Fitting', 'voxel_matrix.pkl'))

    # Export fitted volumes as NIfTI so the montage pipeline can render them.
    # These live alongside the cached pickles under Analysis/Fitting.
    try:
        prefer_ir = False
        try:
            prefer_ir = bool(discover_ir_series(nifti_directory))
        except Exception:
            prefer_ir = False
        ref_img = _reference_img_for_t1fit(nifti_directory, dce_filename, prefer_ir=prefer_ir)
        fitting_dir = os.path.join(analysis_directory, 'Fitting')
        os.makedirs(fitting_dir, exist_ok=True)
        _export_map_nifti(T1_matrix, ref_img, os.path.join(fitting_dir, 't1_map.nii.gz'))
        _export_map_nifti(M0_matrix, ref_img, os.path.join(fitting_dir, 'm0_map.nii.gz'))
    except Exception:
        pass

    try:
        _maybe_write_orientation_debug(
            analysis_directory=analysis_directory,
            nifti_directory=nifti_directory,
            image_directory=image_directory,
            dce_filename=dce_filename,
        )
    except Exception:
        pass

    try:
        _maybe_write_matlab_t1m0_compare(data_directory, analysis_directory, image_directory)
    except Exception:
        pass



    plot_histograms(M0_matrix, T1_matrix, image_directory)
    plot_brain_slices_grid(M0_matrix, T1_matrix, image_directory)


def _segmentation_mask_path(nifti_directory):
    return os.path.join(
        nifti_directory,
        'segmentation',
        'segmentation',
        'mri',
        'aparc.DKTatlas+aseg.deep_in_DCE.nii.gz'
    )


def _load_segmentation_mask(nifti_directory):
    mask_path = _segmentation_mask_path(nifti_directory)
    if not os.path.exists(mask_path):
        return None

    try:
        mask_img = nib.load(mask_path)
    except Exception as exc:
        print(f"Warning: Unable to load segmentation mask ({exc}).")
        return None

    mask_data = mask_img.get_fdata()
    mask_data = np.squeeze(mask_data)
    if mask_data.ndim != 3:
        print("Warning: Segmentation mask is not 3-D; skipping segmented map generation.")
        return None

    return mask_data > 0


def generate_segmented_m0_t1_maps(analysis_directory, image_directory, nifti_directory):
    M0_matrix_path = os.path.join(analysis_directory, 'Fitting', 'voxel_M0_matrix.pkl')
    T1_matrix_path = os.path.join(analysis_directory, 'Fitting', 'voxel_T1_matrix.pkl')

    if not (os.path.exists(M0_matrix_path) and os.path.exists(T1_matrix_path)):
        print("Warning: Unable to locate cached M0/T1 matrices; skipping segmented map rendering.")
        return False

    M0_matrix = load_from_pickle(M0_matrix_path)
    T1_matrix = load_from_pickle(T1_matrix_path)

    segmentation_mask = _load_segmentation_mask(nifti_directory)
    if segmentation_mask is None:
        print("Info: Segmentation mask not available; skipping segmented M0/T1 map rendering.")
        return False

    ensured_mask = _ensure_mask_matches_shape(segmentation_mask, M0_matrix.shape)
    if ensured_mask is None:
        print("Warning: Segmentation mask shape mismatch; skipping segmented map rendering.")
        return False

    os.makedirs(os.path.join(image_directory, 'Fit'), exist_ok=True)
    plot_brain_slices_grid(
        M0_matrix,
        T1_matrix,
        image_directory,
        mask=ensured_mask,
        output_name='M0+T1_Maps_segmented.png'
    )

    return True
