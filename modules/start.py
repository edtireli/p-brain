#A GUI that enumerates over the data directory's subfolders and shows them to the user so that they may continue with analysis of a single subfolder.
#The idea is the user has MRI data within the subfolders, and analyses them sequentially or individually, and not in bulk. 
import tkinter as tk
from tkinter import ttk
import sys
import os
import gzip
import glob
import json


def print_banner():
    """Display the p-brain banner."""
    os.system('clear')
    print('         Welcome to p-brain - a neuroimaging & analysis tool')
    print('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=')
    print('')
    print('        / /                                                  / /')
    print('       / /    eeeee      eeeee  eeeee  eeeee e  eeeee       / /')
    print('      / /     8   8      8   8  8   8  8   8 8  8   8      / /')
    print('eeee / /      8eee8 eeee 8eee8e 8eee8e 8eee8 8e 8e  8     / /    eeee')
    print('    / /       88         88   8 88   8 88  8 88 88  8    / /')
    print('   / /        88         88eee8 88   8 88  8 88 88  8   / /')
    print('')
    print('                  Developed by Edis Devin Tireli')
    print('                     University of Copenhagen')
    print('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=')


availability_toggled = False

def select_log_number():
    """GUI for selecting a dataset."""

    global selected_log_number

    root = tk.Tk()
    root.title('Select Log Number')
    root.geometry("200x450")

    frame = tk.Frame(root)
    frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

    data_root = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'Data')
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
from termcolor import colored
import time
import os


def welcome_screen():
    print_banner()
    #print('----------------------------------------------------------------------')
    print('=-=-= Choose between the following options =-=-==-=-==-=-==-=-==-=-=-= ')
    print('| '+colored('0', 'cyan')+' | View MRI images')
    print('| '+colored('1', 'cyan')+' | Compute M0 and T1 map from MRI data')
    print('| '+colored('2', 'cyan')+' | Generate Concentration Time Curves (CTC) based on ROI')
    print('| '+colored('3', 'cyan')+' | Generate Time-Shifted Concentration Curves (TSCC) from CTC')
    print('| '+colored('4', 'cyan')+' | Create Tissue (Grey/White matter) CTCs')
    print('| '+colored('5', 'cyan')+' | Compute BBB permeability and perfusion parameters')
    #print('| '+colored('6', 'cyan')+' | Cross-sequence axial reconstructions')
    print('| '+colored('6', 'cyan')+' | Full automatic AI enhanced analysis')
    print('| '+colored('7', 'cyan')+' | Add analysis notes')
    print('| '+colored('8', 'cyan')+' | Addons')
    print('| '+colored('9', 'red')+' | ' +colored('Exit program', 'red'))
    print('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=')
    

def welcome_screen_choice():
    choice = input('['+colored('!', 'cyan')+'] Enter option ('+colored('1', 'cyan')+'-'+colored('9', 'cyan')+'): ')
    
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
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.gz'):
                gz_path = os.path.join(root, file)
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(gz_path[:-3], 'wb') as f_out:
                        f_out.write(f_in.read())
                print(f'Decompressed: {gz_path}')


def parrec2nifti(directory, nifti_directory):
    # Check if any .nii files are already present; if so, return
    if any(file.endswith('.nii') for file in os.listdir(nifti_directory)):
        return

    # Check for .PAR files in the directory
    par_files = [f for f in os.listdir(directory) if f.endswith('.PAR')]

    if par_files:
        # Convert each .PAR file
        for file in par_files:
            file_to_convert = os.path.join(directory, file)
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
            except Exception as e:
                # If there was an exception, output the exception details
                print(f"Exception during conversion of {file}: {e}")
    else:
        # No .PAR files, so decompress gzipped files if present
        decompress_gz_in_directory(nifti_directory)

