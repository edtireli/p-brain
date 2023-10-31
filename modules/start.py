#A GUI that enumerates over the data directory's subfolders and shows them to the user so that they may continue with analysis of a single subfolder.
#The idea is the user has MRI data within the subfolders, and analyses them sequentially or individually, and not in bulk. 
import tkinter as tk
from tkinter import ttk
import sys
import os


availability_toggled = False
def select_log_number():
    global selected_log_number
    
    def on_select(event):
        global selected_log_number
        selected_log_number = log_numbers_listbox.get(log_numbers_listbox.curselection())
        # Remove the asterisk from the selected log number if it exists
        selected_log_number = selected_log_number.rstrip('*')

    
    def on_accept():
        global selected_log_number 
        if selected_log_number:
            root.destroy()
            
    def toggle_availability():
        global availability_toggled
        availability_toggled = not availability_toggled
        # Save the current scrollbar position
        current_scroll = log_numbers_listbox.yview()
        for i, log in enumerate(log_numbers):
            path_to_check = os.path.join(base_path, log, 'Analysis', 'values.json')
            nifti_file_path = os.path.join(base_path, log, 'NIfTI', 'WIPDelRec-hperf120long.nii')
            nifti_file_exists = os.path.exists(nifti_file_path)
            if availability_toggled:
                display_text = log
                if nifti_file_exists:
                    display_text += '*'
                    
                if os.path.exists(path_to_check):
                    log_numbers_listbox.delete(i)
                    log_numbers_listbox.insert(i, display_text)
                    log_numbers_listbox.itemconfig(i, {'fg':'green'})
                else:
                    log_numbers_listbox.delete(i)
                    log_numbers_listbox.insert(i, display_text)
                    log_numbers_listbox.itemconfig(i, {'fg':'red'})
            else:
                log_numbers_listbox.delete(i)
                log_numbers_listbox.insert(i, log)
                log_numbers_listbox.itemconfig(i, {'fg':'white'})

        log_numbers_listbox.yview_moveto(current_scroll[0])
    root = tk.Tk()
    root.title('Select Log Number')
    root.geometry("200x450")
    
    frame = tk.Frame(root)
    frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
    
    base_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'Data')
    log_numbers = [f.name for f in os.scandir(base_path) if f.is_dir()]
    log_numbers.sort()
    
    log_numbers_listbox = tk.Listbox(frame, height=20, width=30)
    log_numbers_listbox.pack(side=tk.TOP, padx=20, pady=5)  # Horizontally centered using padx
    
    for item in log_numbers:
        log_numbers_listbox.insert(tk.END, item)
        
    log_numbers_listbox.yview(tk.END)
    log_numbers_listbox.bind('<<ListboxSelect>>', on_select)
    
    availability_toggled = False
    
    accept_button = ttk.Button(root, text="Accept", command=on_accept)
    accept_button.pack(side=tk.TOP, pady=5, anchor=tk.CENTER)
    
    availability_button = ttk.Button(root, text="Availability", command=toggle_availability)
    availability_button.pack(side=tk.TOP, pady=5, anchor=tk.CENTER)
    
    selected_log_number = None
    
    root.mainloop()
    root.update()
    return selected_log_number


#A simple implementation of a CLI interface
from pyfiglet import Figlet
from termcolor import colored
import time
import os


def welcome_screen():
    os.system('clear')
    custom_fig = Figlet(font='computer')
    print(colored('         Welcome to p-brain - a neuroimaging & analysis tool', 'white'))
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'cyan'))
    print('')
    print(custom_fig.renderText('-// p-brain //-'))
    print('                  Developed by Edis Devin Tireli')
    print('                     University of Copenhagen')
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'cyan'))
    #print('----------------------------------------------------------------------')
    print('=-=-= Choose between the following options =-=-==-=-==-=-==-=-==-=-=-= ')
    print('| '+colored('0', 'cyan')+' | View MRI images')
    print('| '+colored('1', 'cyan')+' | Compute M0 and T1 map from MRI data')
    print('| '+colored('2', 'cyan')+' | Generate Concentration Time Curves (CTC) based on ROI')
    print('| '+colored('3', 'cyan')+' | Generate Time-Shifted Concentration Curves (TSCC) from CTC')
    print('| '+colored('4', 'cyan')+' | Create Tissue (Grey/White matter) CTCs')
    print('| '+colored('5', 'cyan')+' | Compute BBB permeability and perfusion parameters')
    print('| '+colored('6', 'cyan')+' | Add analysis notes')
    print('| '+colored('7', 'cyan')+' | Addons')
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

#A simple function for transforming the .PAR files if available to .nii/.json assuming the user has dcm2niix installed
def parrec2nifti(directory, nifti_directory):
    if any(file.endswith('.nii') for file in os.listdir(nifti_directory)):
        return

    par_files = [f for f in os.listdir(directory) if f.endswith('.PAR')]
    prefix_dict = defaultdict(int)

    for file in par_files:
        parts = file.rsplit('_', 1)
        prefix, num = parts[0], int(parts[1].split('.')[0])
        prefix_dict[prefix] = max(prefix_dict[prefix], num)

    for prefix, max_num in prefix_dict.items():
        file_to_convert = f"{prefix}_{max_num}.PAR"
        command = f"dcm2niix -f %p -v n -o {nifti_directory} {os.path.join(directory, file_to_convert)}"
        process = subprocess.Popen(command, shell=True)
        process.wait()