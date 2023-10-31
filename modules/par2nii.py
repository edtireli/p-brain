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