"""Interactive start/menu helpers.

This module contains both GUI helpers (Tkinter) and headless helpers used by
batch runs (e.g. PAR/REC conversion). Tkinter is optional so the pipeline can
run on headless machines.
"""

# A GUI that enumerates over the data directory's subfolders and shows them to the user so that they may continue with analysis of a single subfolder.
# The idea is the user has MRI data within the subfolders, and analyses them sequentially or individually, and not in bulk.
try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover
    tk = None
    ttk = None
import sys
import os
import gzip
import glob
import json

import utils.settings as settings
from utils.loading import (
    inject_flip_angle_into_sidecar_json,
    read_flip_angle_deg_from_par,
    read_protocol_name_from_par,
    sanitize_dcm2niix_basename,
)

from termcolor import colored

def print_banner():
    """Display the p-brain banner."""
    os.system('clear')
    line = "=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-="
    print(colored("         Welcome to p-brain - a neuroimaging & analysis tool", "white"))
    print(colored(line, "white"))
    print("")
    print(colored("        / /                                                  / /", "cyan"))
    print(colored("       / /    eeeee      eeeee  eeeee  eeeee e  eeeee       / /", "cyan"))
    print(colored("      / /     8   8      8   8  8   8  8   8 8  8   8      / /", "cyan"))
    print(colored("eeee / /      8eee8 eeee 8eee8e 8eee8e 8eee8 8e 8e  8     / /    eeee", "cyan"))
    print(colored("    / /       88         88   8 88   8 88  8 88 88  8    / /", "cyan"))
    print(colored("   / /        88         88eee8 88   8 88  8 88 88  8   / /", "cyan"))
    print("")
    print(colored("                  Developed by Edis Devin Tireli", "white"))
    print(colored("                     University of Copenhagen", "white"))
    print(colored(line, "white"))


availability_toggled = False


def select_log_number(data_root=None):
    """GUI for selecting a dataset.

    ``data_root`` specifies the directory containing subject folders. When not
    provided, the ``P_BRAIN_DATA_DIR`` environment variable is honoured before
    falling back to a local ``Data`` directory.
    """

    global selected_log_number

    if data_root is None:
        data_root = os.environ.get("P_BRAIN_DATA_DIR")
        if data_root is None:
            data_root = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'Data')
    data_root = os.path.abspath(data_root)

    if tk is None or ttk is None:
        raise RuntimeError("tkinter is not available in this Python; run with --id/--mode or install a Python build with Tk support")

    root = tk.Tk()
    root.title('Select Log Number')
    root.geometry("200x450")

    frame = tk.Frame(root)
    frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

    current_path = data_root
    log_numbers = []

    log_numbers_listbox = tk.Listbox(frame, height=20, width=30)
    log_numbers_listbox.pack(side=tk.TOP, padx=20, pady=5)

    def ensure_controls_flag(path):
        flag_path = os.path.join(path, 'controls.json')
        if not os.path.exists(flag_path):
            with open(flag_path, 'w') as f:
                json.dump({"controls": True}, f, indent=4)

    def refresh_list(path):
        nonlocal current_path, log_numbers
        current_path = path
        if os.path.basename(current_path).lower() == 'controls':
            ensure_controls_flag(current_path)
        log_numbers = [f.name for f in os.scandir(current_path) if f.is_dir()]
        log_numbers.sort()
        log_numbers_listbox.delete(0, tk.END)
        for item in log_numbers:
            log_numbers_listbox.insert(tk.END, item)
        log_numbers_listbox.yview(tk.END)
        if current_path != data_root:
            back_button.pack(side=tk.TOP, pady=5, anchor=tk.CENTER)
        else:
            back_button.pack_forget()

    def on_select(event):
        global selected_log_number
        if not log_numbers_listbox.curselection():
            return
        selected_log_number = log_numbers_listbox.get(log_numbers_listbox.curselection())
        selected_log_number = selected_log_number.rstrip('*')

    def on_double_click(event):
        if not log_numbers_listbox.curselection():
            return
        item = log_numbers_listbox.get(log_numbers_listbox.curselection())
        next_path = os.path.join(current_path, item)
        if os.path.isdir(next_path) and current_path == data_root and item.lower() == 'controls':
            refresh_list(next_path)
        else:
            on_select(None)
            on_accept()

    def on_accept():
        global selected_log_number
        if selected_log_number:
            root.destroy()

    def toggle_availability():
        global availability_toggled
        availability_toggled = not availability_toggled
        current_scroll = log_numbers_listbox.yview()
        for i, log in enumerate(log_numbers):
            path_to_check = os.path.join(current_path, log, 'Analysis', 'values.json')
            nifti_file_path = os.path.join(current_path, log, 'NIfTI', 'WIPDelRec-hperf120long.nii')
            nifti_file_exists = os.path.exists(nifti_file_path)
            if availability_toggled:
                display_text = log
                if nifti_file_exists:
                    display_text += '*'
                if os.path.exists(path_to_check):
                    log_numbers_listbox.delete(i)
                    log_numbers_listbox.insert(i, display_text)
                    log_numbers_listbox.itemconfig(i, {'fg': 'green'})
                else:
                    log_numbers_listbox.delete(i)
                    log_numbers_listbox.insert(i, display_text)
                    log_numbers_listbox.itemconfig(i, {'fg': 'red'})
            else:
                log_numbers_listbox.delete(i)
                log_numbers_listbox.insert(i, log)
                log_numbers_listbox.itemconfig(i, {'fg': 'white'})

        log_numbers_listbox.yview_moveto(current_scroll[0])

    def go_back():
        refresh_list(data_root)

    accept_button = ttk.Button(root, text="Accept", command=on_accept)
    availability_button = ttk.Button(root, text="Availability", command=toggle_availability)
    back_button = ttk.Button(root, text="Back", command=go_back)

    accept_button.pack(side=tk.TOP, pady=5, anchor=tk.CENTER)
    availability_button.pack(side=tk.TOP, pady=5, anchor=tk.CENTER)

    refresh_list(data_root)

    log_numbers_listbox.bind('<<ListboxSelect>>', on_select)
    log_numbers_listbox.bind('<Double-1>', on_double_click)

    selected_log_number = None
    root.mainloop()
    root.update()
    return selected_log_number


#A simple implementation of a CLI interface
import time
import os


def welcome_screen():
    print_banner()
    #print('----------------------------------------------------------------------')
    print(colored('=-=-= Choose between the following options =-=-==-=-==-=-==-=-==-=-=-=', 'white'))
    print('| '+colored('0', 'cyan')+' | View MRI images')
    print('| '+colored('1', 'cyan')+' | Compute M0 and T1 map from MRI data')
    print('| '+colored('2', 'cyan')+' | Generate Concentration Time Curves (CTC) based on ROI')
    print('| '+colored('3', 'cyan')+' | Generate Time-Shifted Concentration Curves (TSCC) from CTC')
    print('| '+colored('4', 'cyan')+' | Create Tissue (Grey/White matter) CTCs')
    print('| '+colored('5', 'cyan')+' | Compute BBB permeability and perfusion parameters')
    print('| '+colored('6', 'cyan')+' | Add analysis notes')
    print('| '+colored('7', 'cyan')+' | Addons')
    print('| '+colored('9', 'red')+' | ' +colored('Exit program', 'red'))
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))

def welcome_screen_pseudo():
    """Menu shown after running the automatic pipeline in pseudo mode."""
    print_banner()
    print(colored('=-=-= Choose between the following options =-=-==-=-==-=-==-=-=-=-=', 'white'))
    print('| '+colored('0', 'cyan')+' | View MRI images')
    print('| '+colored('1', 'cyan')+' | Compute M0 and T1 map from MRI data')
    print('| '+colored('2', 'cyan')+' | Change Input Function')
    print('| '+colored('3', 'cyan')+' | Generate Time-Shifted Concentration Curves (TSCC) from CTC')
    print('| '+colored('4', 'cyan')+' | Change Tissue (Grey/White matter) CTCs')
    print('| '+colored('5', 'cyan')+' | Compute BBB permeability and perfusion parameters')
    print('| '+colored('6', 'cyan')+' | Add analysis notes')
    print('| '+colored('7', 'cyan')+' | Addons')
    print('| '+colored('9', 'red')+' | ' +colored('Exit program', 'red'))
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
    
def welcome_screen_choice():
    choice = input('['+colored('!', 'cyan')+'] Enter option ('+colored('1', 'cyan')+'-'+colored('7', 'cyan')+', or 9 to exit): ')
    
    if not choice.isdigit():
        print('[' +colored('!', 'red') +"] Only integer input!")
        time.sleep(2)
        print('[' +colored('!', 'cyan') +'] Try again!  ^-^')
        time.sleep(2)
        return welcome_screen_choice()  # Recursively call itself
    
    return int(choice) 


import subprocess
from collections import defaultdict
import os


def decompress_gz_in_directory(directory):
    """Decompress .gz files under a directory.

    WARNING: This can be extremely expensive on large cohorts.
    By default we skip decompressing `.nii.gz` because most tooling supports
    compressed NIfTI directly.
    """

    decompress_nii_gz = os.environ.get("PBRAIN_DECOMPRESS_NIFTI_GZ") == "1"
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith('.gz'):
                continue

            # Avoid expanding compressed NIfTI unless explicitly requested.
            if (not decompress_nii_gz) and file.endswith('.nii.gz'):
                continue

            gz_path = os.path.join(root, file)
            with gzip.open(gz_path, 'rb') as f_in:
                with open(gz_path[:-3], 'wb') as f_out:
                    f_out.write(f_in.read())
            print(f'Decompressed: {gz_path}')


def parrec2nifti(directory, nifti_directory):
    os.makedirs(nifti_directory, exist_ok=True)

    def _nifti_path_from_json(json_path: str) -> str:
        base = json_path[:-5] if json_path.endswith('.json') else os.path.splitext(json_path)[0]
        gz = base + '.nii.gz'
        nii = base + '.nii'
        if os.path.exists(gz):
            return gz
        return nii

    # Check for .PAR files in the directory
    try:
        par_files = [
            f
            for f in os.listdir(directory)
            if f.lower().endswith('.par')
        ]
    except Exception:
        par_files = []

    # If NIfTI already exists, we still try to enrich JSON metadata from PAR.
    nifti_present = False
    try:
        nifti_files = os.listdir(nifti_directory)
        nifti_present = any(f.endswith('.nii') or f.endswith('.nii.gz') for f in nifti_files)
    except Exception:
        nifti_present = False

    if nifti_present and par_files:
        for file in par_files:
            file_to_convert = os.path.join(directory, file)
            flip_angle_deg = settings.FLIP_ANGLE_DEG
            if flip_angle_deg is None:
                flip_angle_deg = read_flip_angle_deg_from_par(file_to_convert, default=None)
            if flip_angle_deg is None:
                continue

            protocol_name = read_protocol_name_from_par(file_to_convert, default=None)
            protocol_sanitized = sanitize_dcm2niix_basename(protocol_name) if protocol_name else None

            candidate_jsons = []
            try:
                for fname in os.listdir(nifti_directory):
                    if not fname.lower().endswith('.json'):
                        continue
                    if protocol_sanitized:
                        base = os.path.splitext(fname)[0]
                        base_sanitized = sanitize_dcm2niix_basename(base)
                        if base_sanitized.lower() == protocol_sanitized.lower():
                            candidate_jsons.append(os.path.join(nifti_directory, fname))
                if not candidate_jsons and protocol_sanitized:
                    # Fallback: substring match.
                    for fname in os.listdir(nifti_directory):
                        if not fname.lower().endswith('.json'):
                            continue
                        base = os.path.splitext(fname)[0]
                        if protocol_sanitized.lower() in sanitize_dcm2niix_basename(base).lower():
                            candidate_jsons.append(os.path.join(nifti_directory, fname))
            except Exception:
                candidate_jsons = []

            for json_path in candidate_jsons:
                inject_flip_angle_into_sidecar_json(
                    _nifti_path_from_json(json_path),
                    flip_angle_deg,
                )
        return

    if par_files:
        # Convert each .PAR file
        for file in par_files:
            file_to_convert = os.path.join(directory, file)
            before_json = set()
            try:
                before_json = {
                    f for f in os.listdir(nifti_directory) if f.lower().endswith('.json')
                }
            except Exception:
                before_json = set()

            command = f"dcm2niix -f %p -o {nifti_directory} -v n {file_to_convert}"
            try:
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    # If there was an error, output it
                    print(f"Error converting {file}: {stderr.decode('utf-8')}")
                else:
                    # If conversion was successful, print a confirmation message
                    print(f"Converted {file} successfully.")

                    flip_angle_deg = settings.FLIP_ANGLE_DEG
                    if flip_angle_deg is None:
                        flip_angle_deg = read_flip_angle_deg_from_par(file_to_convert, default=None)
                    if flip_angle_deg is not None:
                        after_json = set()
                        try:
                            after_json = {
                                f for f in os.listdir(nifti_directory) if f.lower().endswith('.json')
                            }
                        except Exception:
                            after_json = set()
                        new_json = sorted(after_json - before_json)
                        for fname in new_json:
                            json_path = os.path.join(nifti_directory, fname)
                            inject_flip_angle_into_sidecar_json(
                                _nifti_path_from_json(json_path),
                                flip_angle_deg,
                            )
            except Exception as e:
                # If there was an exception, output the exception details
                print(f"Exception during conversion of {file}: {e}")
    else:
        # No .PAR files. Avoid decompressing by default (can be hours on large cohorts).
        # Opt-in via: PBRAIN_DECOMPRESS_NIFTI_GZ=1
        if os.environ.get("PBRAIN_DECOMPRESS_NIFTI_GZ") == "1":
            decompress_gz_in_directory(nifti_directory)

