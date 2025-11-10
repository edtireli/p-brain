import os
from scipy.optimize import least_squares, curve_fit
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
from utils.settings import MULTIPROCESSING, NUMBER_OF_CORES, T1_RECOVERY_MODEL
from utils.fonts import *
from utils.loading import *
from utils.plotting import *
from utils.cli_logging import auto_logging_suppressed
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import threading

turbo_mode = True #doesnt show plots

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


def build_voxel_matrix(dce_data):
    if not dce_data:
        raise ValueError("No data files provided.")

    # Load the first file to determine the shape
    first_data_shape = nib.load(dce_data[0]).get_fdata().shape
    # Adjust the shape to accommodate all data files
    matrix_shape = (len(dce_data), *first_data_shape[:3])

    matrix = np.zeros(matrix_shape)

    with auto_logging_suppressed():
        for idx, file in tqdm(enumerate(dce_data), desc="Building Voxel Matrix", total=len(dce_data)):
            data_4d = nib.load(file).get_fdata()
            # Ensure that the 4th dimension is handled correctly
            if data_4d.ndim == 4:
                matrix[idx, :, :, :] = data_4d[:, :, :, 0]
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


def _fit_single(voxel_values, IsVFA, TI_values, alfas, TRs):
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
        if T1_RECOVERY_MODEL == "saturation":
            initial_M0 = max_signal
            initial_T1 = 750
            bounds = ([1e-6, 100], [np.inf, 6000])
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
            initial_T1 = 750
            bounds = ([1e-6, 1e-6, 100], [np.inf, np.inf, 6000])
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


def fit_all_voxels(voxel_matrix, TI_values, IsVFA, brain_mask=None, **kwargs):
    shape_x, shape_y, shape_z = voxel_matrix.shape[1:]
    total_voxels = shape_x * shape_y * shape_z

    voxels = voxel_matrix.reshape(voxel_matrix.shape[0], -1).T
    alfas = kwargs.get('alfas')
    TRs = kwargs.get('TRs')

    if brain_mask is not None:
        mask_flat = np.asarray(brain_mask, dtype=bool).reshape(-1)
        if mask_flat.size != total_voxels:
            print(
                "Warning: Brain mask contains"
                f" {mask_flat.size} voxels but data contains {total_voxels}; ignoring mask."
            )
            indices = np.arange(total_voxels)
            voxels_to_fit = voxels
        else:
            indices = np.where(mask_flat)[0]
            voxels_to_fit = voxels[indices]
    else:
        indices = np.arange(total_voxels)
        voxels_to_fit = voxels

    partial_fit = functools.partial(_fit_single, IsVFA=IsVFA, TI_values=TI_values, alfas=alfas, TRs=TRs)

    if MULTIPROCESSING:
        with multiprocessing.Pool(NUMBER_OF_CORES) as pool:
            with auto_logging_suppressed():
                iterator = tqdm(
                    pool.imap(partial_fit, voxels_to_fit),
                    total=len(indices),
                    desc=" Fitting Voxel Matrix",
                )
                results = list(iterator)
    else:
        with auto_logging_suppressed():
            iterator = tqdm(voxels_to_fit, total=len(indices), desc=" Fitting Voxel Matrix")
            results = [partial_fit(v) for v in iterator]

    M0_flat = np.full(total_voxels, np.nan)
    T1_flat = np.full(total_voxels, np.nan)

    fitted_M0 = np.array([r[0] for r in results])
    fitted_T1 = np.array([r[1] for r in results])

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
def T1_fit(data_directory, analysis_directory, nifti_directory, image_directory, filenames, parameters):
    _, IsVFA, IsIR, _, _, _, _ = parameters
    t1_3D_filename, axial_t1_3D_filename, t2_3D_filename, axial_t2_3D_filename, \
    (
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
        # Handling for VFA
        IsVFA = False
        if IsVFA:
            vfa_flip_angles = range(1, 6)  # Assuming 5 flip angles
            # Extract base name without '_FLAIR_DCE' or similar
            dce_filename_base = os.path.splitext(dce_filename)[0]
            dce_filename_base = dce_filename_base.replace('_FLAIR_DCE', '').replace('_DCE', '')

            # Generate VFA filenames
            vfa_filenames = [f"{dce_filename_base}_flip-{str(flip).zfill(2)}_VFA.json" for flip in vfa_flip_angles]
            vfa_data = [os.path.join(nifti_directory, f"{dce_filename_base}_flip-{str(flip).zfill(2)}_VFA.nii") for flip in vfa_flip_angles]

            # Extract TRs and flip angles
            TRs, alfas = extract_vfa_params(vfa_filenames, nifti_directory)

            # Check for missing files
            for file_path in vfa_data:
                if not os.path.exists(file_path):
                    print(f"Warning: File not found: {file_path}")
            
            # Build voxel matrix and fit VFA model
            if voxel_matrix is None:
                voxel_matrix = build_voxel_matrix(vfa_data)
            M0_matrix, T1_matrix = fit_all_voxels(voxel_matrix, None, True, alfas=alfas, TRs=TRs)

        # Handling for IR
        if not IsVFA:
            if IsIR:
                TI = ['00120', '00300', '00600', '01000', '02000', '04000', '10000']
                TI_values = [int(times) for times in TI]
                patterns = ['WIPTI_', 'WIPDelRec-TI_']
                dce_data = [first_existing_file(nifti_directory, patterns, time, '.nii') for time in TI]
                if voxel_matrix is None:
                    voxel_matrix = build_voxel_matrix(dce_data)
                M0_matrix, T1_matrix = fit_all_voxels(voxel_matrix, TI_values, False)
            else:
                # No fitting performed without IR or VFA data
                if voxel_matrix is not None:
                    shape = voxel_matrix.shape[1:]
                else:
                    raise RuntimeError("Unable to determine volume shape for T1/M0 outputs.")
                M0_matrix = np.full(shape, np.nan)
                T1_matrix = np.full(shape, np.nan)

        save_as_pickle(M0_matrix, os.path.join(analysis_directory, 'Fitting', 'voxel_M0_matrix.pkl'))
        save_as_pickle(T1_matrix, os.path.join(analysis_directory, 'Fitting', 'voxel_T1_matrix.pkl'))
        if not voxel_matrix_exists:
            save_as_pickle(voxel_matrix, os.path.join(analysis_directory, 'Fitting', 'voxel_matrix.pkl'))



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
