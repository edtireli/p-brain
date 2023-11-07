import os
import nibabel as nib
import matplotlib.pyplot as plt
import re
from termcolor import colored
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

class MRIViewer:
    def __init__(self, nifti_directory):
        self.nifti_directory = nifti_directory
        self.slice_idx = 0
        self.frame_idx = 0

    def find_matching_file(self, patterns):
        for root, dirs, files in os.walk(self.nifti_directory):
            for file in files:
                for pattern in patterns:
                    if re.fullmatch(pattern, file, re.IGNORECASE):
                        return file
        return None



    def display(self):
        patterns_and_names = [
            ([r'WIPcs_T1W_3D_TFE_32channel\.nii'], 'Saggital T1'),
            ([r'ax([-_ ])?vwipcs_t1w_3d_tfe_32channel\.nii'], '3D Axial T1'),
            ([r'WIPcs_3D_Brain_VIEW_T2_32chSHC\.nii'], 'Saggital T2'),
            ([r'ax([-_ ])?vwipcs_3D_Brain_VIEW_T2_32chSHC\.nii'], '3D Axial T2'),
            ([r'WIPcs_3D_Brain_VIEW_FLAIR_SHC\.nii'], 'Saggital FLAIR'),
            ([r'ax([-_ ])?VWIPcs_3D_Brain_VIEW_FLAIR_SHC\.nii'], '3D Axial FLAIR'),
            ([r'WIPAxT2TSEmatrix\.nii'], 'Axial T2 (DCE)'),
            ([r'WIPhperf120long\.nii', r'WIPDelRec-hperf120long'], 'Axial DCE')
        ]

        while True:  # Outer loop to restart the choice
            print("------------------------------")
            available_files = {}
            for i, (patterns, name) in enumerate(patterns_and_names):
                filename = self.find_matching_file(patterns)
                if filename:
                    available_files[i + 1] = (filename, name)
                    print(f"| {colored(i + 1, 'cyan')} | {colored(name, 'green')}")
                else:
                    print(f"| {colored(i + 1, 'cyan')} | {colored(name, 'red')}")
            print(f"| {colored('q', 'red')} | {colored('Exit', 'red')}")
            print("------------------------------")

            choice = input("Select an option or 'q' to quit: ")
            if choice == 'q':
                return  # Exit the program

            try:
                choice = int(choice)
                if choice in available_files:
                    selected_file, selected_name = available_files[choice]

                    # Load the NIFTI file
                    img = nib.load(os.path.join(self.nifti_directory, selected_file))
                    self.img_data = img.get_fdata()
                    self.num_slices = self.img_data.shape[2]
                    self.num_frames = self.img_data.shape[3] if self.img_data.ndim == 4 else None

                    self.slice_idx = 0  # Reset slice index to start at slice 1

                    # Inner loop for the GUI
                    while True:
                        # Create a new figure and axis
                        self.fig, self.ax = plt.subplots()

                        # Connect to events and show the figure
                        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
                        self.redraw()
                        plt.show()

                        # Close the GUI and break the inner loop when 'Escape' is pressed
                        if plt.get_fignums():
                            continue  # Reopen the GUI if it is still open
                        else:
                            break  # Break the inner loop to return to the user choice

                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number or 'q' to quit.")




    def on_key(self, event):
        if event.key == 'escape':
            plt.close(self.fig)
            return
        elif event.key == 'up':
            self.slice_idx = (self.slice_idx + 1) % self.num_slices
        elif event.key == 'down':
            self.slice_idx = (self.slice_idx - 1) % self.num_slices
        elif self.num_frames and event.key == 'left':
            self.frame_idx = (self.frame_idx - 1) % self.num_frames
        elif self.num_frames and event.key == 'right':
            self.frame_idx = (self.frame_idx + 1) % self.num_frames
        self.redraw()

    def redraw(self):
        self.ax.clear()
        if self.num_frames:
            self.ax.imshow(np.rot90(self.img_data[:, :, self.slice_idx, self.frame_idx], k=1), cmap="viridis")
            self.ax.set_title(f"Slice: {self.slice_idx + 1}/{self.num_slices}, Frame: {self.frame_idx + 1}/{self.num_frames}")
        else:
            self.ax.imshow(np.rot90(self.img_data[:, :, self.slice_idx], k=1), cmap="viridis")
            self.ax.set_title(f"Slice: {self.slice_idx + 1}/{self.num_slices}")
        self.fig.canvas.draw()
