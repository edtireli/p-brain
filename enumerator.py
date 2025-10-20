import argparse
import subprocess
import os
import sys

import utils.settings as settings


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
    parser.add_argument("--all", action="store_true", help="Process all datasets in the data directory")
    parser.add_argument("ids", nargs="*", help="Specific dataset IDs to process")
    return parser.parse_args()


def _list_subdirectories(path):
    """Return sorted subdirectories under ``path`` (non-recursive)."""

    try:
        entries = sorted(os.listdir(path))
    except FileNotFoundError:
        return []

    result = []
    for name in entries:
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path):
            result.append(name)
    return result


def collect_datasets(data_directory, use_all, ids, *, use_controls=False):
    """Return a list of dataset identifiers to process.

    Parameters
    ----------
    data_directory:
        Root directory containing dataset folders to process.
    use_all:
        When ``True`` every dataset in ``data_directory`` will be returned.
    ids:
        Optional explicit list of dataset identifiers supplied on the command
        line.
    use_controls:
        When ``True`` operate on control datasets stored inside a ``controls``
        subdirectory.
    """

    datasets = []
    data_directory = os.fspath(data_directory)
    controls_directory = os.path.join(data_directory, "controls")

    if use_all:
        if use_controls:
            for name in _list_subdirectories(controls_directory):
                datasets.append((name, True))
        else:
            for name in _list_subdirectories(data_directory):
                if name == "controls":
                    continue
                datasets.append((name, False))

    elif ids:
        for dataset_id in ids:
            dataset_id = str(dataset_id)
            control_path = os.path.join(controls_directory, dataset_id)
            patient_path = os.path.join(data_directory, dataset_id)

            is_control = False
            if os.path.isdir(control_path):
                is_control = True
                dataset_path = control_path
            else:
                dataset_path = patient_path

            if not os.path.isdir(dataset_path):
                raise FileNotFoundError(f"Dataset {dataset_id} not found in {data_directory}.")

            # Honour explicit ``use_controls`` flag even if the directory lives
            # outside ``controls``.
            if use_controls:
                is_control = True

            datasets.append((dataset_id, is_control))

    else:
        raise ValueError("No datasets specified")

    return datasets


def main():
    args = parse_args()
    data_directory = os.path.abspath(args.data_dir)
    use_all = args.all
    ids = args.ids

    # If user gave no ids and no --all, default to all
    if not ids and not use_all:
        use_all = True

    try:
        datasets = collect_datasets(
            data_directory,
            use_all,
            ids,
            use_controls=settings.CONTROLS,
        )
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
        env["PBRAIN_TURBO"] = "1"
        if is_control:
            env["PBRAIN_CONTROLS"] = "1"
        else:
            env.pop("PBRAIN_CONTROLS", None)
        print(f"Running: {command}")
        subprocess.run(command, shell=True, env=env)


if __name__ == "__main__":
    main()
