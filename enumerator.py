import sys
import subprocess
import os

# Path to the directory containing the datasets
data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Get arguments from the command line
args = sys.argv[1:]

# Normalize args for easy checks
args_lower = [arg.lower() for arg in args]

# Determine which datasets to process and whether they are controls
datasets = []  # list of tuples (id, is_control)

if "--all" in args_lower:
    # Collect subfolder names in alphanumeric order
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
                    datasets.append((ctrl, True))
        else:
            datasets.append((name, False))
elif "--controls" in args_lower:
    # Only enumerate over the control datasets
    for name in sorted(os.listdir(data_directory)):
        full_path = os.path.join(data_directory, name)
        if not os.path.isdir(full_path):
            continue
        if name.lower() in {"control", "controls"}:
            for ctrl in sorted(os.listdir(full_path)):
                ctrl_path = os.path.join(full_path, ctrl)
                if os.path.isdir(ctrl_path):
                    datasets.append((ctrl, True))
            break
else:
    # Otherwise, use the supplied IDs
    for arg in args:
        if arg.startswith("--"):
            continue
        ctrl_path = os.path.join(data_directory, "controls", arg)
        is_ctrl = os.path.isdir(ctrl_path)
        datasets.append((arg, is_ctrl))

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
