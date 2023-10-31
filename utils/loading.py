import pickle
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import json
from utils.plotting import *
from utils.loading import *

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

def save_values(Ki, SD_Ki, lambda_, P, P_std, subtype_tissue, slice_tissue, subtype_artery, venous_slice, arterial_slice, analysis_directory):
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
