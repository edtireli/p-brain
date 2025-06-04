import pickle
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import json
import importlib
import time



def list_addons():
    addon_folders = [f for f in os.listdir("addons") if os.path.isdir(os.path.join("addons", f))]
    print('--------------------------------')
    for i, addon in enumerate(addon_folders):
        print(f"{i+1}. {addon}")
    print('--------------------------------')    
    choice = int(input("[!] Choose addon: ")) - 1
    return addon_folders[choice]

import sys
def load_addon(addon_folder_name, *args):
    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
        base_dir = os.path.dirname(current_script_dir)
        addon_path = os.path.join(base_dir, 'addons')
        sys.path.append(addon_path)  

        addon = importlib.import_module(f'{addon_folder_name}.{addon_folder_name}')
        addon.run(*args)
    except ModuleNotFoundError as e:
        print(f"Addon {addon_folder_name} not available! Error: {str(e)}")
        print("Make sure the addon is correctly placed in the 'addons' directory and named correctly.")
        print(f"Attempted to import from {addon_path}")
    except Exception as e:
        print(f"An error occurred while loading the addon: {str(e)}")

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

def find_matching_file(directory, pattern):
    regex = re.compile(pattern, re.IGNORECASE)
    for filename in os.listdir(directory):
        if regex.match(filename):
            return os.path.join(directory, filename)
    return None


def load_curves(venous_slice, arterial_slice, artery_choice, analysis_directory):
    shifted_vein_file = os.path.join(analysis_directory, 'CTC Data', 'Vein', 'Sinus Sagittalis', f'CTC_shifted_slice_{venous_slice}.npy')
    vein_file = os.path.join(analysis_directory, 'CTC Data', 'Vein', 'Sinus Sagittalis', f'CTC_slice_{venous_slice}.npy')
    
    shifted_artery_file = os.path.join(analysis_directory, 'CTC Data', 'Artery', artery_choice, f'CTC_shifted_slice_{arterial_slice}.npy')
    artery_file = os.path.join(analysis_directory, 'CTC Data', 'Artery', artery_choice, f'CTC_slice_{arterial_slice}.npy')

    if os.path.exists(shifted_vein_file):
        vein_curve = np.load(shifted_vein_file)
    else:
        vein_curve = np.load(vein_file)
    
    if os.path.exists(shifted_artery_file):
        artery_curve = np.load(shifted_artery_file)
    else:
        artery_curve = np.load(artery_file)
    
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

