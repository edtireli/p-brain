import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image

def load_data(logfolder_path):
    wip_perf_path = os.path.join(logfolder_path, 'NIfTI', 'WIPhperf120long.nii')
    wip_delrec_path = os.path.join(logfolder_path, 'NIfTI', 'WIPDelRec-hperf120long.nii')
    
    if os.path.exists(wip_perf_path):
        data_path = wip_perf_path
    elif os.path.exists(wip_delrec_path):
        data_path = wip_delrec_path
    else:
        raise FileNotFoundError("Neither WIPhperf120long.nii nor WIPDelRec-hperf120long.nii found.")

    roi_path = os.path.join(logfolder_path, 'Analysis', 'ROI Data', 'Artery', 'Right Interior Carotid')
    mri_data = np.rot90(nib.load(data_path).get_fdata(), k=-1, axes=(0, 1))

    roi_files = [f for f in os.listdir(roi_path) if f.startswith('ROI_voxels_slice')]
    roi_data = {}
    for roi_file in roi_files:
        slice_index = int(roi_file.split('_')[-1].split('.')[0]) - 1  # Adjusting for 0-based indexing
        roi_voxels = np.load(os.path.join(roi_path, roi_file))
        roi_data[slice_index] = roi_voxels
    
    return mri_data, roi_data

def save_slices_as_png(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logfolders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    for logfolder in logfolders:
        logfolder_path = os.path.join(data_dir, logfolder)
        mri_data, roi_data = load_data(logfolder_path)
        
        # Compute time-averaged intensity across all time frames
        time_averaged_data = np.mean(mri_data, axis=-1)
        
        for slice_index, rois in roi_data.items():
            # Extract the specific slice
            slice_data = time_averaged_data[:, :, slice_index]

            # Normalize the slice data
            slice_data_normalized = ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)

            # Save the slice as a PNG image
            slice_image = Image.fromarray(slice_data_normalized)
            slice_filename = os.path.join(output_dir, f"{logfolder}_slice_{slice_index + 1}.png")
            slice_image.save(slice_filename, format='PNG')

            # Create and save the ROI mask as a PNG image
            mask = np.zeros((256, 256), dtype=np.uint8)
            for roi in rois:
                x, y = roi
                if 0 <= x < 256 and 0 <= y < 256:
                    mask[x, y] = 255  # Set the mask to white (255) for the ROI region

            mask_image = Image.fromarray(mask)
            mask_filename = os.path.join(output_dir, f"{logfolder}_mask_{slice_index + 1}.png")
            mask_image.save(mask_filename, format='PNG')

            print(f"Saved slice and mask for {logfolder}, slice {slice_index + 1}.")

# Example usage
data_dir = '/Users/edt/Desktop/p-brain/data'
output_dir = '/Users/edt/Desktop/p-brain/AI/output_pngs_rica'
save_slices_as_png(data_dir, output_dir)
