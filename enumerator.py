import sys
import subprocess
import os

# Path to the directory containing the datasets
data_directory = "/Users/edt/Desktop/p-brain/data"

# Get arguments from the command line
args = sys.argv[1:]

# Check if '--all' is specified
if "--all" in args:
    # Collect subfolder names in alphanumeric order
    ids = []
    for name in sorted(os.listdir(data_directory)):
        full_path = os.path.join(data_directory, name)
        if not os.path.isdir(full_path):
            continue

        lower_name = name.lower()

        # If a control directory is found, enumerate over its subfolders
        if lower_name in {"control", "controls"}:
            for ctrl in sorted(os.listdir(full_path)):
                ctrl_path = os.path.join(full_path, ctrl)
                if os.path.isdir(ctrl_path):
                    ids.append(ctrl)
        else:
            ids.append(name)
else:
    # Otherwise, use the supplied IDs
    ids = args

# Template for the command to run
command_template = "python3 main.py --id {} --option 66"

# Iterate through IDs and execute the command
for id in ids:
    command = command_template.format(id)
    print(f"Running: {command}")
    subprocess.run(command, shell=True)
