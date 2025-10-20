import argparse
import subprocess
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run p-brain on multiple datasets")
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        type=str,
        default=os.environ.get(
            "P_BRAIN_DATA_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        ),
        help="Directory containing dataset folders",
    )
    parser.add_argument("--controls", action="store_true", help="Process control datasets")
    parser.add_argument("--all", action="store_true", help="Process all datasets in the data directory")
    parser.add_argument("ids", nargs="*", help="Specific dataset IDs to process")
    return parser.parse_args()


def collect_datasets(data_directory, use_all, ids, use_controls):
    """Return a list of ``(dataset_id, is_control)`` tuples to process.

    Parameters
    ----------
    data_directory:
        Root directory containing patient datasets and, optionally, a
        ``controls`` sub-directory.
    use_all:
        When ``True`` every dataset in ``data_directory`` will be returned.
    ids:
        Optional explicit list of dataset identifiers supplied on the command
        line.
    use_controls:
        When ``True`` only control datasets are returned for ``use_all``.  For
        explicit ``ids`` the flag forces the resulting entries to be marked as
        controls, mirroring the behaviour of the command line interface.
    """

    datasets = []

    if use_all:
        top = sorted(os.listdir(data_directory))
        for name in top:
            full_path = os.path.join(data_directory, name)
            if not os.path.isdir(full_path):
                continue

            lower_name = name.lower()
            if lower_name in {"control", "controls"}:
                if not use_controls:
                    continue

                for ctrl in sorted(os.listdir(full_path)):
                    ctrl_path = os.path.join(full_path, ctrl)
                    if os.path.isdir(ctrl_path):
                        datasets.append((ctrl, True))
            else:
                if use_controls:
                    # When ``--controls`` is supplied with ``--all`` the user
                    # expects only control datasets.  Skip patient folders in
                    # that case.
                    continue
                datasets.append((name, False))

    elif ids:
        for id_str in ids:
            control_dir = os.path.join(data_directory, "controls", id_str)
            is_ctrl = use_controls or os.path.isdir(control_dir)
            datasets.append((id_str, is_ctrl))

    else:
        raise ValueError("No datasets specified")

    return datasets


def main():
    args = parse_args()
    data_directory = os.path.abspath(args.data_dir)
    use_controls = args.controls
    use_all = args.all
    ids = args.ids

    # If user gave no ids and no --all, default to all
    if not ids and not use_all:
        use_all = True

    try:
        datasets = collect_datasets(data_directory, use_all, ids, use_controls)
    except FileNotFoundError:
        print(f"Data dir missing: {data_directory}")
        sys.exit(1)
    except ValueError as exc:
        print(f"{exc}. Provide log numbers or use --all.")
        sys.exit(1)

    if not datasets:
        print("No datasets found to process.")
        sys.exit(0)

    # Build command
    command_template = "python3 main.py --id {} --mode auto --data-dir {}"

    for dataset_id, is_control in datasets:
        command = command_template.format(dataset_id, data_directory)
        env = os.environ.copy()
        env["P_BRAIN_DATA_DIR"] = data_directory
        if is_control:
            env["PBRAIN_CONTROLS"] = "true"
        else:
            env.pop("PBRAIN_CONTROLS", None)
        env["PBRAIN_TURBO"] = "1"
        print(f"Running: {command} (control={is_control})")
        subprocess.run(command, shell=True, env=env)


if __name__ == "__main__":
    main()
