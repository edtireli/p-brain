import argparse
import subprocess
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run p-brain on multiple datasets")
    parser.add_argument('--data-dir', dest='data_dir', type=str,
                        default=os.environ.get(
                            'P_BRAIN_DATA_DIR',
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
                        ),
                        help='Directory containing dataset folders')
    parser.add_argument('--controls', action='store_true', help='Process control datasets')
    parser.add_argument('--all', action='store_true', help='Process all datasets in the data directory')
    parser.add_argument('ids', nargs='*', help='Specific dataset IDs to process')
    return parser.parse_args()


args = parse_args()
data_directory = os.path.abspath(args.data_dir)
use_controls = args.controls
use_all = args.all
ids = args.ids

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
# Use the dedicated automatic mode instead of the manual option.
command_template = "python3 main.py --id {} --mode auto"

# Iterate through datasets and execute the command
for id, is_control in datasets:
    command = command_template.format(id)
    env = os.environ.copy()
    env["P_BRAIN_DATA_DIR"] = data_directory
    if is_control:
        env["PBRAIN_CONTROLS"] = "true"
    else:
        env.pop("PBRAIN_CONTROLS", None)
    # Disable plotting and interactive windows when running in batch mode
    # by enabling turbo mode in ``main.py``.
    env["PBRAIN_TURBO"] = "1"
    print(f"Running: {command} (control={is_control})")
    subprocess.run(command, shell=True, env=env)
