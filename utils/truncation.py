#!/usr/bin/env python3

import os
import json
import numpy as np
import nibabel as nib

def dce_truncation(INPFILE):
    # Check that file exists
    if not os.path.exists(INPFILE):
        print(f'File does not exist: {INPFILE}')
        return None

    print(f"Processing file: {INPFILE}")

    try:
        # Load PAR file using nibabel with unscaled data (raw pixel values)
        data_pv, hdr = load_parrec(INPFILE, scale=None)

        # Get image info
        image_defs = hdr.image_defs
        slice_numbers = np.unique(image_defs['slice number'])
        dynamic_numbers = np.unique(image_defs['dynamic scan number'])
        image_types_codes = np.unique(image_defs['image_type_mr'])

        # Map image_type_mr codes to types
        image_type_mr_codes = {
            0: 'Magnitude',
            1: 'Real',
            2: 'Imaginary',
            3: 'Phase'
        }
        types = [image_type_mr_codes.get(code, f'Type{code}') for code in image_types_codes]

        nslices = slice_numbers.size
        nframes = dynamic_numbers.size
        ntypes = image_types_codes.size

        # Create mappings from parameter values to indices
        slice_number_to_index = {num: idx for idx, num in enumerate(slice_numbers)}
        dynamic_number_to_index = {num: idx for idx, num in enumerate(dynamic_numbers)}
        image_type_to_index = {code: idx for idx, code in enumerate(image_types_codes)}

        # Get data dimensions
        data_shape = data_pv.shape
        x_dim, y_dim = data_shape[:2]
        n_slices = data_shape[2]
        n_volumes = data_shape[3] if len(data_shape) > 3 else 1
        n_images = n_slices * n_volumes

        # Reshape data
        data_pv = data_pv.reshape((x_dim, y_dim, n_images))

        # Create deck array
        deck = np.zeros((x_dim, y_dim, nslices, nframes, ntypes), dtype=data_pv.dtype)

        # Assign data to deck
        for i in range(n_images):
            img_data = data_pv[..., i]
            img_def = image_defs[i]
            slice_num = img_def['slice number']
            dyn_num = img_def['dynamic scan number']
            img_type_code = img_def['image_type_mr']

            # Map to indices
            slice_idx = slice_number_to_index[slice_num]
            dyn_idx = dynamic_number_to_index[dyn_num]
            type_idx = image_type_to_index[img_type_code]

            # Assign data
            deck[..., slice_idx, dyn_idx, type_idx] = img_data

        # Detect truncation in magnitude signal
        NumberOfTruncatedVoxels = np.zeros(ntypes, dtype=int)
        mask = np.zeros(deck.shape, dtype=bool)
        for i in range(ntypes):
            mask[..., i], NumberOfTruncatedVoxels[i] = dce_detect_truncation(deck[..., i], [None, 4095])
            print(f'Number of truncated voxels ({types[i]}): {NumberOfTruncatedVoxels[i]}')

        total_truncated_voxels = int(np.sum(NumberOfTruncatedVoxels))
        if total_truncated_voxels == 0:
            print("No truncation detected.")
            return total_truncated_voxels
        else:
            print(f"Total number of truncated voxels: {total_truncated_voxels}")
            return total_truncated_voxels
    except Exception as e:
        print(f"Error processing file {INPFILE}: {e}")
        return None

def dce_detect_truncation(DATA, LIMITS):
    # Reshape DATA to 2D
    data_rs = DATA.reshape((-1, DATA.shape[-1]))

    # Allocate mask
    mask = np.zeros(data_rs.shape, dtype=bool)

    if LIMITS is None or len(LIMITS) == 0:
        minval = np.nanmin(data_rs)
        maxval = np.nanmax(data_rs)
    else:
        minval = LIMITS[0]
        maxval = LIMITS[1]

    for i in range(data_rs.shape[0]):
        idx_max = data_rs[i, :] == maxval if maxval is not None else np.zeros(data_rs.shape[1], dtype=bool)
        idx_min = data_rs[i, :] == minval if minval is not None else np.zeros(data_rs.shape[1], dtype=bool)
        if np.sum(idx_max) > 1:
            mask[i, :] = idx_max
        elif np.sum(idx_min) > 1:
            mask[i, :] = idx_min

    # Reshape mask to be size of DATA
    mask = mask.reshape(DATA.shape)

    # Calculate total number of truncated voxels
    mask_3D = np.sum(mask, axis=-1)
    nv = np.sum(mask_3D > 0)
    return mask, nv

def load_parrec(filename, scale='fp'):
    # Load the PAR/REC file using nibabel
    img = nib.parrec.load(filename)
    hdr = img.header
    if scale == 'fp':
        data = img.get_fdata()
    elif scale == 'dv':
        data = img.get_fdata(scaling='dv')
    elif scale is None:
        data = img.dataobj.get_unscaled()
    else:
        raise ValueError(f"Unknown scaling method '{scale}'.")
    return data, hdr

def find_par_files_with_protocol(root_dir, protocol_name):
    par_files = []
    # Get sorted list of all subdirectories
    subdirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Sort dirnames and filenames alphanumerically
        dirnames.sort()
        filenames.sort()
        # Prevent os.walk from going into subdirectories of subdirectories
        subdirs.extend([os.path.join(dirpath, d) for d in dirnames])
        # Break to prevent recursive walk
        break

    # Sort subdirectories alphanumerically
    subdirs.sort()

    for subdir in subdirs:
        subfolder_name = os.path.basename(subdir)
        metadata_file = os.path.join(subdir, 'truncation_metadata.json')
        if os.path.exists(metadata_file):
            print(f"Subfolder '{subfolder_name}' already processed. Skipping.")
            continue  # Skip processed subfolders

        # Walk through the subdir
        for dirpath, dirnames, filenames in os.walk(subdir):
            # Sort dirnames and filenames alphanumerically
            dirnames.sort()
            filenames.sort()
            for filename in filenames:
                if filename.endswith('.PAR') or filename.endswith('.par'):
                    par_file = os.path.join(dirpath, filename)
                    # Check protocol name
                    if check_protocol_name(par_file, protocol_name):
                        par_files.append((par_file, subdir))
            # We assume that all .PAR files are in the immediate subdirectory
            # If you have nested subdirectories, remove the following break
            break

    return par_files

def check_protocol_name(par_file, protocol_name):
    try:
        with open(par_file, 'r') as f:
            for line in f:
                if line.startswith('.    Protocol name'):
                    # Extract the protocol name
                    line_parts = line.strip().split(':', 1)
                    if len(line_parts) > 1:
                        current_protocol_name = line_parts[1].strip()
                        if current_protocol_name == protocol_name:
                            return True
                    break
    except Exception as e:
        print(f"Error reading file {par_file}: {e}")
    return False

def main():
    root_dir = '/Users/edt/Desktop/p-brain/data'
    protocol_name = 'WIP hperf120 long'

    # Find PAR files with the specified protocol, excluding processed subfolders
    par_files_info = find_par_files_with_protocol(root_dir, protocol_name)
    if not par_files_info:
        print(f"No new .PAR files with protocol name '{protocol_name}' found in {root_dir}.")
        return

    print(f"Found {len(par_files_info)} new .PAR files with protocol name '{protocol_name}':")
    for par_file, subfolder_path in par_files_info:
        print(f"- {par_file}")

    # Process files and write metadata in each subfolder
    for par_file, subfolder_path in par_files_info:
        total_truncated_voxels = dce_truncation(par_file)
        metadata = {
            'file': par_file,
            'truncated_voxels': total_truncated_voxels
        }

        # Write metadata to a file in the subfolder
        metadata_file = os.path.join(subfolder_path, 'truncation_metadata.json')
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"Metadata written to {metadata_file}")
        except Exception as e:
            print(f"Error writing metadata to {metadata_file}: {e}")

    # Print summary report
    print("\nSummary Report:")
    for par_file, subfolder_path in par_files_info:
        subfolder_name = os.path.basename(subfolder_path)
        metadata_file = os.path.join(subfolder_path, 'truncation_metadata.json')
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            truncated_voxels = metadata.get('truncated_voxels', 0)
            if truncated_voxels > 0:
                print(f"Subfolder: {subfolder_name}")
                print(f"  File: {par_file}")
                print(f"  Total truncated voxels: {truncated_voxels}")
        except Exception as e:
            print(f"Error reading metadata from {metadata_file}: {e}")

if __name__ == "__main__":
    main()
