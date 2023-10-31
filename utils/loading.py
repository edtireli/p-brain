import pickle
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import json



def plot_rois_and_curves(selected_voxels, data_4d, data_3d, T1_matrix, M0_matrix, time_points_, choice = 1):
    num_rois = sum(len(roi_list) for roi_list in selected_voxels.values())
    gs = gridspec.GridSpec(3, num_rois, height_ratios=[1, 1.5, 1])
    fig = plt.figure(figsize=(20, 12))
    idx = 0
    for slice_index, roi_voxels_list in selected_voxels.items():
        for roi_num, roi_voxels in enumerate(roi_voxels_list):
            all_C_t = []
            all_unnormalized_C_t = []
            roi_voxels_downsampled = np.floor_divide(roi_voxels, 2)
            for (x, y) in roi_voxels_downsampled:
                voxel_time_course = data_4d[x, y, slice_index, :]
                T1 = T1_matrix[x, y, slice_index]
                M0 = M0_matrix[x, y, slice_index]
                C_t_0 = compute_CTC(voxel_time_course, T1, r1=4000, TD=120, m0=M0, slice=slice_index, prints=False)
                baseline_point = find_baseline_point_advanced(C_t_0)
                C_t = custom_shifter(C_t_0, baseline_point)
                all_C_t.append(C_t)
                all_unnormalized_C_t.append(C_t_0)
            avg_C_t_0 = np.mean(all_C_t, axis=0)
            baseline_point = find_baseline_point_advanced(avg_C_t_0) - 1
            avg_C_t = custom_shifter(avg_C_t_0, baseline_point)
            
            # Get Patlak data
            max_file = os.listdir(os.path.join(analysis_directory, 'TSCC Data', 'Max'))[0]
            chosen_venous_slice, chosen_arterial_slice = max_file.split('_')[2:4]
            chosen_arterial_slice = chosen_arterial_slice.split('.')[0]
            C_a = np.load(os.path.join(analysis_directory, 'TSCC Data', 'Max', f'TSCC_slice_{chosen_venous_slice}_{chosen_arterial_slice}.npy'))
            C_t = avg_C_t[0:len(C_a)]
            time_points = time_points_[0:len(C_a)]
            Ki, lambda_, SD_Ki, x_patlak, y_patlak = patlak_analysis_plotting(C_t, C_a, time_points)
            baseline_point_f = find_shifted_baseline(C_t)+1
            P, P_std = compute_average_permeability(C_a, C_t, time_points_s, baseline_point=baseline_point_f)
            
            ax1 = plt.subplot(gs[0, idx])
            ax1.plot(avg_C_t)
            ax1.set_title(f'Concentration (Slice {slice_index+1} - ROI {roi_num+1})', fontsize=8)
            ax1.grid(True)
            
            ax2 = plt.subplot(gs[1, idx])
            ax2.imshow(data_3d[:, :, slice_index], cmap='magma', origin='lower')
            for x, y in roi_voxels:
                rect = Rectangle((y, x), 1, 1, linewidth=1, edgecolor='g', facecolor='none', alpha=0.5)
                ax2.add_patch(rect)
            ax2.set_title(f'T2 Image (Slice {slice_index+1} - ROI {roi_num+1})', fontsize=8)
            
            ax3 = plt.subplot(gs[2, idx])
            ax3.scatter(x_patlak, y_patlak, c='black', s=2)
            ax3.plot(x_patlak, lambda_ + Ki * x_patlak, c='red', linestyle='--')
            ax3.set_ylim(min(y_patlak), max(y_patlak))
            ax3.set_title(f'$K_i = {round(Ki*6000, 5)}$, $\\lambda = {round(lambda_*100, 5)}$', fontsize=8)
            ax3.set_xlabel(f'P = {round(P, 5)}')
            ax3.grid(True)
            
            idx += 1

    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    if choice == 2:
        plt.savefig(os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', f'Grey_Matter.png'), dpi=200) 
    elif choice == 1:
        plt.savefig(os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', f'White_Matter.png'), dpi=200)   
    elif choice == 3: 
        plt.savefig(os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', f'Mixed_Matter.png'), dpi=200)    
    plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
    plt.show()
    plt.tight_layout()
    plt.close()


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

def save_values(Ki, SD_Ki, lambda_, subtype_tissue, slice_tissue, subtype_artery, venous_slice, arterial_slice, analysis_directory):
    values_file_path = os.path.join(analysis_directory, 'values.json')
    
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

def quitter():
    leave = input('[!] Quit program? (y/n): ')    
    if leave == 'y':
        exit()
    elif leave == 'n':
        leaver()  


def nii2anat_extension(filename):
    import os

    # Extract the base name and directory from the filename
    base_name = os.path.basename(filename)
    directory = os.path.dirname(filename)

    # Remove the .nii extension and append .anat
    base_name_without_extension = os.path.splitext(base_name)[0]
    new_base_name = base_name_without_extension + ".anat"

    # Create the new directory path
    new_directory = os.path.join(directory, new_base_name)

    return new_directory


def find_matching_file(directory, pattern):
    regex = re.compile(pattern, re.IGNORECASE)
    for filename in os.listdir(directory):
        if regex.match(filename):
            return os.path.join(directory, filename)
    return None


def load_curves(venous_slice, arterial_slice, artery_choice, analysis_directory):
    vein_curve = np.load(os.path.join(analysis_directory, 'CTC Data', 'Vein', 'Sinus Sagittalis', f'CTC_slice_{venous_slice}.npy'))
    artery_curve = np.load(os.path.join(analysis_directory, 'CTC Data', 'Artery', artery_choice, f'CTC_slice_{arterial_slice}.npy'))
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

def first_existing_dce_file(directory, filenames, preferred_filename='WIPDelRec-hperf120long.nii'):
    # Check the preferred filename first
    preferred_file_path = os.path.join(directory, preferred_filename)
    if os.path.exists(preferred_file_path):
        return preferred_file_path
    
    for fname in filenames:
        if fname == preferred_filename:
            continue 
        file_path = os.path.join(directory, fname)
        if os.path.exists(file_path):
            return file_path
            
    return None
