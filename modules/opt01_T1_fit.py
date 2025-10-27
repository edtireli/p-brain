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
from utils.settings import MULTIPROCESSING, NUMBER_OF_CORES
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

# 90 deg flip angle + inversion recovery
def model_function(TIs, M0, T1):
    TIs_array = np.array(TIs)
    return M0 * np.sin(np.pi / 2) * (1 - np.exp(-TIs_array / T1))

def model_residuals(params, TIs, voxel_values):
    M0, T1 = params
    return model_function(TIs, M0, T1) - voxel_values


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
        initial_M0 = max(voxel_values) / np.sin(np.pi / 2)
        max_signal = max(voxel_values)
        initial_T1 = 750
        bounds = ([1e-3, 500], [max_signal, 5000])
        result = least_squares(
            model_residuals,
            [initial_M0, initial_T1],
            args=(TI_values, voxel_values),
            bounds=bounds,
            method="trf",
        )
    return result.x[0], result.x[1]

def fit_all_voxels(voxel_matrix, TI_values, IsVFA, **kwargs):
    shape_x, shape_y, shape_z = voxel_matrix.shape[1:]
    total_voxels = shape_x * shape_y * shape_z

    voxels = voxel_matrix.reshape(voxel_matrix.shape[0], -1).T
    alfas = kwargs.get('alfas')
    TRs = kwargs.get('TRs')

    partial_fit = functools.partial(_fit_single, IsVFA=IsVFA, TI_values=TI_values, alfas=alfas, TRs=TRs)

    if MULTIPROCESSING:
        with multiprocessing.Pool(NUMBER_OF_CORES) as pool:
            with auto_logging_suppressed():
                iterator = tqdm(
                    pool.imap(partial_fit, voxels),
                    total=total_voxels,
                    desc=" Fitting Voxel Matrix",
                )
                results = list(iterator)
    else:
        with auto_logging_suppressed():
            iterator = tqdm(voxels, total=total_voxels, desc=" Fitting Voxel Matrix")
            results = [partial_fit(v) for v in iterator]

    M0_values = np.array([r[0] for r in results]).reshape(shape_x, shape_y, shape_z)
    T1_values = np.array([r[1] for r in results]).reshape(shape_x, shape_y, shape_z)
    return M0_values, T1_values



def plot_histograms(M0_matrix, T1_matrix, image_directory):
    def on_esc(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)
    M0_values = M0_matrix.flatten()
    T1_values = T1_matrix.flatten()

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


def plot_brain_slices_grid(M0_matrix, T1_matrix, image_directory):
    def on_esc(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)
             
    slices = M0_matrix.shape[2]
    grid_size = slices

    fig, axes = plt.subplots(2, grid_size, figsize=(20, 6))
    for i in range(slices):
        ax = axes[0, i]
        slice_data = T1_matrix[:, :, i]
        im_t1 = ax.imshow(slice_data.T, cmap='viridis', origin='lower')
        ax.axis('off')
        ax.set_title(f'Slice {i+1}')
    cbar_ax_t1 = fig.add_axes([0.92, 0.58, 0.01, 0.3])
    cbar_t1 = fig.colorbar(im_t1, cax=cbar_ax_t1)
    cbar_t1.set_label('Longitudinal (T1) Relaxation Time [ms]', fontproperties=prop, fontsize=9) 
    for i in range(slices):
        ax = axes[1, i]
        slice_data = M0_matrix[:, :, i]
        im_m0 = ax.imshow(slice_data.T, cmap='plasma', origin='lower')
        ax.axis('off')
    cbar_ax_m0 = fig.add_axes([0.92, 0.15, 0.01, 0.3])
    cbar_m0 = fig.colorbar(im_m0, cax=cbar_ax_m0)
    cbar_m0.set_label('Equilibrium Magnetization [M0]', fontproperties=prop, fontsize=9)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(os.path.join(image_directory, 'Fit', 'M0+T1_Maps.png'), dpi=200)
    plt.gcf().canvas.mpl_connect('key_press_event', on_esc) 
    if not turbo_mode:
        close_plot_after_delay(3, fig)
        plt.show()
def T1_fit(data_directory, analysis_directory, nifti_directory, image_directory, filenames, parameters):
    _, IsVFA, IsIR, _, _, _, _ = parameters
    t1_3D_filename, axial_t1_3D_filename, t2_3D_filename, axial_t2_3D_filename, \
    flair_3D_filename, axial_flair_3D_filename, axial_t2_2D_filename, dce_filename = filenames
    
    voxel_matrix_path = os.path.join(analysis_directory, 'Fitting', 'voxel_matrix.pkl')
    M0_matrix_path = os.path.join(analysis_directory, 'Fitting', 'voxel_M0_matrix.pkl')
    T1_matrix_path = os.path.join(analysis_directory, 'Fitting', 'voxel_T1_matrix.pkl')
    
    # Initialize alfas and TRs to default values (empty lists or None)
    alfas = []
    TRs = []

    if os.path.exists(voxel_matrix_path):
        voxel_matrix = load_from_pickle(voxel_matrix_path)
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
            voxel_matrix = build_voxel_matrix(vfa_data)
            M0_matrix, T1_matrix = fit_all_voxels(voxel_matrix, None, True, alfas=alfas, TRs=TRs)
        
        # Handling for IR
        if not IsVFA:
            if IsIR: 
                TI = ['00120', '00300', '00600', '01000', '02000', '04000', '10000']
                TI_values = [int(times) for times in TI]
                patterns = ['WIPTI_', 'WIPDelRec-TI_']
                dce_data = [first_existing_file(nifti_directory, patterns, time, '.nii') for time in TI]
                voxel_matrix = build_voxel_matrix(dce_data)
                M0_matrix, T1_matrix = fit_all_voxels(voxel_matrix, TI_values, False)

        save_as_pickle(M0_matrix, os.path.join(analysis_directory, 'Fitting', 'voxel_M0_matrix.pkl'))
        save_as_pickle(T1_matrix, os.path.join(analysis_directory, 'Fitting', 'voxel_T1_matrix.pkl'))
        save_as_pickle(voxel_matrix, os.path.join(analysis_directory, 'Fitting', 'voxel_matrix.pkl'))


    
    plot_histograms(M0_matrix, T1_matrix, image_directory) 
    plot_brain_slices_grid(M0_matrix, T1_matrix, image_directory)
