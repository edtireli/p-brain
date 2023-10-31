
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