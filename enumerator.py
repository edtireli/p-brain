import sys
import subprocess
import os

# Path to the directory containing the datasets
data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Get arguments from the command line
args = sys.argv[1:]

# Normalise arguments for easy checks
args_lower = [arg.lower() for arg in args]

# Flags
use_controls = "--controls" in args_lower
use_all = "--all" in args_lower

# Collect numeric dataset IDs in the order provided
ids = [arg for arg in args if not arg.startswith("--")]

# Determine which datasets to process and whether they are controls
datasets = []  # list of tuples (id, is_control)

if use_all:
    # Enumerate every dataset in the data directory
    for name in sorted(os.listdir(data_directory)):
        full_path = os.path.join(data_directory, name)
        if not os.path.isdir(full_path):
            continue

        lower_name = name.lower()

        # If a control directory is found, enumerate over its subfolders
        if lower_name in {"control", "controls"}:
            if use_controls:
                # Only process controls when --controls is supplied
                for ctrl in sorted(os.listdir(full_path)):
                    ctrl_path = os.path.join(full_path, ctrl)
                    if os.path.isdir(ctrl_path):
                        datasets.append((ctrl, True))
            else:
                for ctrl in sorted(os.listdir(full_path)):
                    ctrl_path = os.path.join(full_path, ctrl)
                    if os.path.isdir(ctrl_path):
                        datasets.append((ctrl, True))
        elif not use_controls:
            # Only add non-control folders when not limiting to controls
            datasets.append((name, False))
elif ids:
    # Use the supplied IDs
    for id_str in ids:
        is_ctrl = use_controls or os.path.isdir(os.path.join(data_directory, "controls", id_str))
        datasets.append((id_str, is_ctrl))
else:
    print("No datasets specified. Provide log numbers or use --all.")
    sys.exit(1)

# Template for the command to run
command_template = "python3 main.py --id {} --option 6"

# Iterate through datasets and execute the command
for id, is_control in datasets:
    command = command_template.format(id)
    env = os.environ.copy()
    if is_control:
        env["PBRAIN_CONTROLS"] = "true"
    else:
        env.pop("PBRAIN_CONTROLS", None)
    print(f"Running: {command} (control={is_control})")
    subprocess.run(command, shell=True, env=env)
