import os
from scipy.optimize import least_squares, curve_fit, minimize
import nibabel as nib
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils.fonts import *
from utils.loading import *


def build_voxel_matrix(dce_data):
    matrix_shape = (len(dce_data), 256, 256, 10)
    matrix = np.zeros(matrix_shape)

    for idx, file in tqdm(enumerate(dce_data), desc="Building Voxel Matrix", total=len(dce_data)):
        data_4d = nib.load(file).get_fdata()
        matrix[idx, :, :, :] = data_4d[:, :, :, 0]

    return matrix


def model_function(TIs, M0, T1):
    TIs_array = np.array(TIs)
    return M0 * np.sin(np.pi / 2) * (1 - np.exp(-TIs_array / T1))


def model_residuals(params, TIs, voxel_values):
    M0, T1 = params
    return model_function(TIs, M0, T1) - voxel_values



def fit_all_voxels(voxel_matrix, TI_values):
    shape_x, shape_y, shape_z = voxel_matrix.shape[1:]
    M0_matrix = np.zeros((shape_x, shape_y, shape_z))
    T1_matrix = np.zeros((shape_x, shape_y, shape_z))

    total_voxels = shape_x * shape_y * shape_z
    with tqdm(total=total_voxels, desc="Fitting Voxel Matrix ") as pbar:
        for i in range(shape_x):
            for j in range(shape_y):
                for k in range(shape_z):
                    voxel_values = voxel_matrix[:, i, j, k]
                    if max(voxel_values) == 0:
                        pbar.update(1)
                        continue
                    initial_M0 = max(voxel_values)
                    initial_T1 = 750
                    bounds = ([0, 0], [2*initial_M0, 5000])
                    result = least_squares(model_residuals, [initial_M0, initial_T1], args=(TI_values, voxel_values), bounds=bounds, method='trf')
                    M0_matrix[i, j, k], T1_matrix[i, j, k] = result.x
                    pbar.update(1)

    return M0_matrix, T1_matrix



def plot_voxel_fit(X, Y, Z, M0_matrix, T1_matrix, voxel_matrix, TI_values): #Plot a central voxel fit to visually check if the fit is okay, as an extra check
    def on_esc(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)
    voxel_values = voxel_matrix[:, X, Y, Z]
    M0_value = M0_matrix[X, Y, Z]
    T1_value = T1_matrix[X, Y, Z]
    fine_TI_values = np.linspace(min(TI_values), max(TI_values), 1000)
    fitted_values = model_function(fine_TI_values, M0_value, T1_value)

    # Plotting
    plt.plot(TI_values, voxel_values, 'o', color='red', markersize=3, label='Voxel Signal Values')
    plt.plot(fine_TI_values, fitted_values, '--', color='black', alpha=0.75, label='Fitted Curve')
    plt.xlabel('TI Values [ms]')
    plt.ylabel('Signal Intensity')
    plt.legend()
    plt.title(f'Voxel Signal and Fitted Curve at X={X}, Y={Y}, Z={Z}')
    plt.grid(which='minor', alpha=0.25)
    plt.minorticks_on()
    plt.gcf().canvas.mpl_connect('key_press_event', on_esc) 
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
    plt.show()




def T1_fit(data_directory, analysis_directory, nifti_directory, image_directory):
    TI=['00120', '00300', '00600', '01000', '02000', '04000', '10000']
    TI_values = [int(times) for times in TI]
    patterns = ['WIPTI_', 'WIPDelRec-TI_']
    dce_data = [first_existing_file(nifti_directory, patterns, time, '.nii') for time in TI]


    if os.path.exists(os.path.join(analysis_directory, 'voxel_matrix.pkl')):
        voxel_matrix = load_from_pickle(os.path.join(analysis_directory, 'voxel_matrix.pkl'))
        print("Loading previously built voxel matrix!")
    else:    
        voxel_matrix = build_voxel_matrix(dce_data)
        save_as_pickle(voxel_matrix, os.path.join(analysis_directory, 'voxel_matrix.pkl'))
    if os.path.exists(os.path.join(analysis_directory, 'voxel_M0_matrix.pkl')):
        M0_matrix = load_from_pickle(os.path.join(analysis_directory, 'voxel_M0_matrix.pkl'))
        print("Loading previously fitted M0 fit matrix!")
    if os.path.exists(os.path.join(analysis_directory, 'voxel_T1_matrix.pkl')):
        T1_matrix = load_from_pickle(os.path.join(analysis_directory, 'voxel_T1_matrix.pkl'))
        print("Loading previously fitted T1 fit matrix!")
    else:        
        M0_matrix, T1_matrix = fit_all_voxels(voxel_matrix, TI_values)
        save_as_pickle(M0_matrix, os.path.join(analysis_directory, 'voxel_M0_matrix.pkl'))
        save_as_pickle(T1_matrix, os.path.join(analysis_directory, 'voxel_T1_matrix.pkl'))
    
    plot_voxel_fit(X=120, Y=140, Z=0, M0_matrix=M0_matrix, T1_matrix=T1_matrix, voxel_matrix=voxel_matrix, TI_values=TI_values)
    plot_histograms(M0_matrix, T1_matrix, image_directory) 
    plot_brain_slices_grid(M0_matrix, T1_matrix, image_directory)