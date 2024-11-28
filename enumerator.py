import sys
import subprocess
import os

# Path to the directory containing subfolders
data_directory = "/Users/edt/Desktop/p-brain/data"

# Get arguments from the command line
args = sys.argv[1:]

# Check if '--all' is specified
if "--all" in args:
    # Get all subfolder names in alphanumeric order
    ids = sorted(name for name in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, name)))
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
