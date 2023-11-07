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

def parrec2nifti(directory, nifti_directory):
    # Check if any .nii files are already present; if so, return
    if any(file.endswith('.nii') for file in os.listdir(nifti_directory)):
        return

    # List all .PAR files in the given directory
    par_files = [f for f in os.listdir(directory) if f.endswith('.PAR')]

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
         