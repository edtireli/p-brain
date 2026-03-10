import os
import re
import numpy as np
import nibabel as nib
import matplotlib

_MPL_ENV_BACKEND = os.environ.get("P_BRAIN_MPL_BACKEND") or os.environ.get("MPLBACKEND")
if _MPL_ENV_BACKEND:
    matplotlib.use(_MPL_ENV_BACKEND)
else:
    try:
        import _tkinter  # noqa: F401 — test C extension before committing to TkAgg
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")

import matplotlib.pyplot as plt
from termcolor import colored

turbo_mode = True  # When True, suppress interactive plotting

class MRIViewer:
    def __init__(self, nifti_directory, filenames):
        self.nifti_directory = nifti_directory
        self.filenames = filenames
        self.slice_idx = 0
        self.frame_idx = 0

    def find_matching_file(self, patterns):
        for root, _, files in os.walk(self.nifti_directory):
            for file in files:
                for pattern in patterns:
                    if re.fullmatch(pattern, file, re.IGNORECASE):
                        return file
        return None



    def display(self):
        if turbo_mode:
            print("[!] Turbo mode enabled; skipping image viewer.")
            return
        (
            t1_3D_filename,
            axial_t1_3D_filename,
            t2_3D_filename,
            axial_t2_3D_filename,
            flair_3D_filename,
            axial_flair_3D_filename,
            axial_t2_2D_filename,
            diffusion_filename,
            dce_filename,
        ) = self.filenames
        patterns_and_names = [
            ([t1_3D_filename], 'Saggital T1'),
            ([axial_t1_3D_filename], '3D Axial T1'),
            ([t2_3D_filename], 'Saggital T2'),
            ([axial_t2_3D_filename], '3D Axial T2'),
            ([flair_3D_filename], 'Saggital FLAIR'),
            ([axial_flair_3D_filename], '3D Axial FLAIR'),
            ([axial_t2_2D_filename], 'Axial T2 (DCE)'),
            ([dce_filename], 'Axial DCE')
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
                    selected_file, _ = available_files[choice]

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
                        if not turbo_mode:
                            plt.show()
                        else:
                            plt.close(self.fig)
                            break

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
