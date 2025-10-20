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
    parser.add_argument("--all", action="store_true", help="Process all datasets in the data directory")
    parser.add_argument("ids", nargs="*", help="Specific dataset IDs to process")
    return parser.parse_args()


def collect_datasets(data_directory, use_all, ids):
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
    """

    datasets = []

    if use_all:
        top = sorted(os.listdir(data_directory))
        for name in top:
            full_path = os.path.join(data_directory, name)
            if not os.path.isdir(full_path):
                continue
            datasets.append(name)

    elif ids:
        datasets.extend(ids)

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
        datasets = collect_datasets(data_directory, use_all, ids)
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

    for dataset_id in datasets:
        command = command_template.format(dataset_id, data_directory)
        env = os.environ.copy()
        env["P_BRAIN_DATA_DIR"] = data_directory
        env["PBRAIN_TURBO"] = "1"
        print(f"Running: {command}")
        subprocess.run(command, shell=True, env=env)


if __name__ == "__main__":
    main()
