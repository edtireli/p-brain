import os
from scipy.optimize import least_squares, curve_fit, minimize
import nibabel as nib
from tqdm import tqdm
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from utils.fonts import *
from utils.loading import *
<<<<<<< HEAD
from utils.plotting import *
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import threading

turbo_mode = True #doesnt show plots

def close_plot_after_delay_plt(delay):
    """
    Close the plot automatically after a delay if no interaction occurs.
    :param delay: Time in seconds to wait before closing the plot.
    """
    def close():
        plt.close(plt.gcf())  # Close the current figure

    timer = threading.Timer(delay, close)
    timer.start()

    # If there is user interaction, cancel the timer
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: timer.cancel())


=======
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

>>>>>>> a0de673fc033368a127dc6bae55e4b3363958e21
def extract_vfa_params(vfa_filenames, nifti_directory):
    repetition_times = []
    flip_angles = []

    for filename in vfa_filenames:
        # Construct the path to the corresponding JSON file
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(nifti_directory, json_filename)

        # Read and extract data from the JSON file
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
            repetition_times.append(json_data.get("RepetitionTime"))
            flip_angles.append(json_data.get("FlipAngle"))

    return repetition_times, flip_angles

def build_voxel_matrix(dce_data):
    if not dce_data:
        raise ValueError("No data files provided.")

    # Load the first file to determine the shape
    first_data_shape = nib.load(dce_data[0]).get_fdata().shape
    # Adjust the shape to accommodate all data files
    matrix_shape = (len(dce_data), *first_data_shape[:3])

    matrix = np.zeros(matrix_shape)

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

def fit_all_voxels(voxel_matrix, TI_values, IsVFA, **kwargs):
    shape_x, shape_y, shape_z = voxel_matrix.shape[1:]
    M0_matrix = np.zeros((shape_x, shape_y, shape_z))
    T1_matrix = np.zeros((shape_x, shape_y, shape_z))

    total_voxels = shape_x * shape_y * shape_z
    with tqdm(total=total_voxels, desc=" Fitting Voxel Matrix") as pbar:
        for i in range(shape_x):
            for j in range(shape_y):
                for k in range(shape_z):
                    voxel_values = voxel_matrix[:, i, j, k]
                    if max(voxel_values) == 0:
                        pbar.update(1)
                        continue

                    #if IsVFA:
                        # Skip voxels with low signal for VFA method
                    #    if max(voxel_values) < 0.1 * np.max(voxel_matrix[:, i, j, :]):
                    #        pbar.update(1)
                    #        continue

                    if IsVFA:
                        # Extract alfas and TRs from kwargs
                        alfas = kwargs.get('alfas')
                        alfas_rad = np.radians(alfas)
                        TRs = kwargs.get('TRs')

                        initial_M0 = max(voxel_values)/np.sin(alfas_rad[0])

                        initial_T1 = 750  # Adjust initial guess as needed
                        bounds = ([0, 500], [2*initial_M0, 5000])  # Adjust bounds as needed

                        result = least_squares(model_residuals_VFA, [initial_M0, initial_T1], args=(voxel_values,), kwargs={'alfas': alfas, 'TRs': TRs}, bounds=bounds, method='trf')
                    else:
                        initial_M0 = max(voxel_values)/np.sin(np.pi/2)
                        max_signal = max(voxel_values)
                        initial_T1 = 750  # Adjust initial guess as needed
                        bounds = ([1e-3, 500], [max_signal, 5000])  # Adjust bounds as needed

                        result = least_squares(model_residuals, [initial_M0, initial_T1], args=(TI_values, voxel_values), bounds=bounds, method='trf')

                    M0_matrix[i, j, k], T1_matrix[i, j, k] = result.x
                    pbar.update(1)

    return M0_matrix, T1_matrix


def plot_voxel_fit(X, Y, Z, M0_matrix, T1_matrix, voxel_matrix, values, isVFA=False):
    def on_esc(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)

    voxel_values = voxel_matrix[:, X, Y, Z]
    M0_value = M0_matrix[X, Y, Z]
    T1_value = T1_matrix[X, Y, Z]

    # Choose the model function based on the method
    if isVFA:
        fine_values = np.linspace(min(values), max(values), 1000)
        fitted_values = model_function_VFA(fine_values, TRs, M0_value, 1/T1_value)  # Assuming TR is defined globally or passed as an argument
        xlabel = 'Flip Angle [degrees]'
    else:
        fine_values = np.linspace(min(values), max(values), 1000)
        fitted_values = model_function(fine_values, M0_value, T1_value)
        xlabel = 'TI Values [ms]'

    # Plotting
    plt.plot(values, voxel_values, 'o', color='red', markersize=3, label='Voxel Signal Values')
    plt.plot(fine_values, fitted_values, '--', color='black', alpha=0.75, label='Fitted Curve')
    plt.xlabel(xlabel)
    plt.ylabel('Signal Intensity')
    plt.legend()
    plt.title(f'Voxel Signal and Fitted Curve at X={X}, Y={Y}, Z={Z}')
    plt.grid(which='minor', alpha=0.25)
    plt.minorticks_on()
    plt.gcf().canvas.mpl_connect('key_press_event', on_esc) 
    if not turbo_mode:
        close_plot_after_delay_plt(3)
        plt.show()

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
    
<<<<<<< HEAD
    IsVFA, IsIR,_,_ = parameters
=======
    IsVFA, IsIR = parameters
>>>>>>> a0de673fc033368a127dc6bae55e4b3363958e21
    
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
        if IsVFA:
            vfa_flip_angles = range(1, 6)  # Assuming 5 flip angles
            # renaming logic from dce_filename to VFA files - change if needed
            dce_filename_base = dce_filename.replace('_dce.nii', '')
            vfa_filenames = [f"{dce_filename_base}_flip-{str(flip).zfill(2)}_VFA.nii" for flip in vfa_flip_angles]
            vfa_data = [os.path.join(nifti_directory, fname) for fname in vfa_filenames]

            TRs, alfas = extract_vfa_params(vfa_filenames, nifti_directory)
            voxel_matrix = build_voxel_matrix(vfa_data)
            M0_matrix, T1_matrix = fit_all_voxels(voxel_matrix, None, True, alfas=alfas, TRs=TRs)
        
        # Handling for IR
        if not IsVFA:
            if IsIR: 
                TI=['00120', '00300', '00600', '01000', '02000', '04000', '10000']
                TI_values = [int(times) for times in TI]
                patterns = ['WIPTI_', 'WIPDelRec-TI_']
                dce_data = [first_existing_file(nifti_directory, patterns, time, '.nii') for time in TI]
                voxel_matrix = build_voxel_matrix(dce_data)
                M0_matrix, T1_matrix = fit_all_voxels(voxel_matrix, TI_values, False)

        save_as_pickle(M0_matrix, os.path.join(analysis_directory, 'Fitting', 'voxel_M0_matrix.pkl'))
        save_as_pickle(T1_matrix, os.path.join(analysis_directory, 'Fitting', 'voxel_T1_matrix.pkl'))
        save_as_pickle(voxel_matrix, os.path.join(analysis_directory, 'Fitting', 'voxel_matrix.pkl'))

    X_center = voxel_matrix.shape[1] // 2
    Y_center = voxel_matrix.shape[2] // 2
    Z_center = voxel_matrix.shape[3] // 2
    
    #if not IsVFA:
    #    plot_voxel_fit(X_center, Y_center, Z_center, M0_matrix, T1_matrix, voxel_matrix, TI_values, isVFA=IsVFA)
    #elif IsVFA:
    #    plot_voxel_fit(X_center, Y_center, Z_center, M0_matrix, T1_matrix, voxel_matrix, alfas, isVFA=IsVFA)

    plot_histograms(M0_matrix, T1_matrix, image_directory) 
    plot_brain_slices_grid(M0_matrix, T1_matrix, image_directory)
