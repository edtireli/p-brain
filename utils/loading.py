import pickle
import os
import matplotlib.pyplot as plt
import numpy as np


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
