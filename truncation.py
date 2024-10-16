import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Function to detect truncation in DCE data
def dce_truncation(input_file):
    """
    Detects if there is truncation in DCE data (clipping of peak).
    
    Args:
        input_file (str): Path to the 4D DCE .PAR file
    """
    if not os.path.exists(input_file):
        print(f"File does not exist: {input_file}")
        return

    print(f"dce_truncation('{input_file}')")

    # Load PAR/REC file using nibabel
    parrec_data = nib.load(input_file)
    data = parrec_data.get_fdata()
    hdr = parrec_data.header

    # Extract nframes, nslices, and time, handle missing time
    try:
        nframes = int(hdr.general_info['max_dynamics'])
        nslices = int(hdr.general_info['max_slices'])
        # Placeholder for time if actual time data is missing
        time = np.linspace(0, nframes - 1, nframes)
    except KeyError as e:
        print(f"KeyError: {e}")
        return

    ntypes = 3  # Assuming types {'Magnitude', 'Real', 'Imag'}

    # Reshape data into [rows, cols, slices, frames, types]
    deck = data.reshape(data.shape[0], data.shape[1], nslices, nframes, ntypes)

    types = ['Magnitude', 'Real', 'Imag']

    # Detect truncation
    number_of_truncated_voxels = np.zeros(ntypes)
    mask = np.zeros_like(deck, dtype=bool)
    
    for i in range(ntypes):
        print(f"Data stats for {types[i]}: min={np.min(deck[..., i])}, max={np.max(deck[..., i])}, mean={np.mean(deck[..., i])}")
        mask[..., i], number_of_truncated_voxels[i] = dce_detect_truncation(deck[..., i], [np.nan, 4095])
        print(f"Number of truncated voxels ({types[i]}): {int(number_of_truncated_voxels[i])}")

    # Selected time points
    start_frame = 0
    end_frame = 250
    time_slice = slice(start_frame, end_frame)
    time_selected = time[time_slice]

    # Plotting the data
    f = plt.figure(figsize=(10,9))
    if ntypes == 3:
        t = plt.GridSpec(ntypes + 1, 1)
    else:
        t = plt.GridSpec(ntypes, 1)

    # Plot an average signal over all slices, rows, and columns for the selected frames
    for i in range(ntypes):
        ax = f.add_subplot(t[i, 0])
        deck_avg = np.mean(deck[..., i], axis=(0, 1, 2))  # Average over rows, columns, and slices
        
        # Plot only the selected frames
        ax.plot(time_selected, deck_avg[time_slice], 'r', linewidth=1)
        ax.grid(which='both')
        
        if ntypes == 3 and i == 0:
            # Calculate the complex signal (Real and Imaginary)
            deck_cpx = deck[..., 1] + 1j * deck[..., 2]
            deck_mod = np.abs(deck_cpx)
            deck_mod_avg = np.mean(deck_mod, axis=(0, 1, 2))  # Average over rows, columns, and slices
            
            # Plot only the selected frames
            ax.plot(time_selected, deck_mod_avg[time_slice], 'b', linewidth=2)
            ax.legend(['Modulus', 'Magnitude'])
            ax.grid(which='both')
        
        ax.set_ylabel(types[i])
        if i == ntypes - 1:
            ax.set_xlabel('Time [s]')

    if ntypes == 3:
        ax = f.add_subplot(t[ntypes, 0])
        deck_pha = np.angle(deck[..., 1] + 1j * deck[..., 2])
        deck_pha_avg = np.mean(deck_pha, axis=(0, 1, 2))  # Average over rows, columns, and slices
        
        # Plot only the selected frames
        ax.plot(time_selected, deck_pha_avg[time_slice], 'r', linewidth=1)
        ax.set_ylabel('Phase')
        ax.set_xlabel('Time [s]')
        ax.grid(which='both')
    plt.tight_layout()
    plt.show()

def dce_detect_truncation(data, limits):
    """
    Detects truncation in the data.
    
    Args:
        data (np.ndarray): 4D array containing the DCE data
        limits (list): [minval, maxval] to check for clipping
    
    Returns:
        mask (np.ndarray): A mask indicating truncated voxels
        nv (int): Number of truncated voxels
    """
    # Reshape data to 2D for easier processing
    data_rs = data.reshape(-1, data.shape[-1])

    # Allocate mask
    mask = np.zeros_like(data_rs, dtype=bool)

    # Set limits for clipping detection
    minval = limits[0] if not np.isnan(limits[0]) else np.min(data_rs)
    maxval = limits[1] if not np.isnan(limits[1]) else np.max(data_rs)

    for i in range(data_rs.shape[0]):
        # Adjust clipping logic for more accuracy
        idx_max = (data_rs[i, :] == maxval)
        idx_min = (data_rs[i, :] == minval)

        if np.sum(idx_max) > 1:
            mask[i, :] = idx_max
        elif np.sum(idx_min) > 1:
            mask[i, :] = idx_min

    # Reshape mask back to the original data shape
    mask = mask.reshape(data.shape)

    # Calculate total number of truncated voxels
    mask_3D = np.sum(mask, axis=-1)
    nv = np.sum(mask_3D > 0)

    return mask, nv

if __name__ == "__main__":
    # Ensure the user provides a PAR file as input
    if len(sys.argv) < 2:
        print("Usage: python3 truncation.py path/to/.PAR file")
        sys.exit(1)

    input_file = sys.argv[1]
    dce_truncation(input_file)
