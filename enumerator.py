import argparse
import subprocess
import os
import sys

import utils.settings as settings
import utils.parameters as parameters


def _load_montage_dependencies():
    """Import heavy montage modules lazily."""

    from utils.montage import (
        generate_parametric_montages,
        generate_projection_montages,
        build_population_projection_stats,
    )
    from utils import parameters

    return (
        generate_parametric_montages,
        generate_projection_montages,
        parameters,
        build_population_projection_stats,
    )


def _prepare_projection_stats(data_root: str):
    """Return cached atlas statistics for projection montages."""

    try:
        deps = _load_montage_dependencies()
    except ImportError as exc:  # pragma: no cover - import side effect
        raise RuntimeError(f"Unable to import montage dependencies: {exc}") from exc

    build_population_projection_stats = None
    if len(deps) >= 4:
        build_population_projection_stats = deps[3]

    if build_population_projection_stats is None:
        return None

    return build_population_projection_stats(
        data_root,
        include_controls=bool(settings.CONTROLS),
    )


def _resolve_dataset_root(data_root, dataset_id, is_control):
    """Return the filesystem path for ``dataset_id``."""

    candidates = []
    if is_control:
        candidates.append(os.path.join(data_root, "controls", dataset_id))
    candidates.append(os.path.join(data_root, dataset_id))

    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def _run_montage_for_dataset(
    data_root,
    dataset_id,
    is_control,
    *,
    use_projection=False,
    projection_stats=None,
):
    """Render parametric montages for ``dataset_id`` if possible."""

    dataset_root = _resolve_dataset_root(data_root, dataset_id, is_control)
    if not os.path.isdir(dataset_root):
        print(f"[montage] Dataset directory missing – skipping: {dataset_root}")
        return False

    analysis_directory = os.path.join(dataset_root, "Analysis")
    image_directory = os.path.join(dataset_root, "Images")
    nifti_directory = os.path.join(dataset_root, "NIfTI")

    required_dirs = (
        (analysis_directory, "Analysis"),
        (image_directory, "Images"),
        (nifti_directory, "NIfTI"),
    )
    for path, label in required_dirs:
        if not os.path.isdir(path):
            print(f"[montage] {label} directory missing – skipping: {path}")
            return False

    try:
        deps = _load_montage_dependencies()
    except ImportError as exc:
        print(f"[montage] Unable to import montage dependencies: {exc}")
        return False

    if len(deps) == 2:
        generate_parametric_montages, parameters = deps
        generate_projection_montages = None
    elif len(deps) == 3:
        generate_parametric_montages, generate_projection_montages, parameters = deps
    else:
        (
            generate_parametric_montages,
            generate_projection_montages,
            parameters,
            *_
        ) = deps

    try:
        if bool(is_control or settings.CONTROLS):
            filenames = parameters.control_filenames(nifti_directory)
        else:
            filenames = parameters.global_filenames(nifti_directory)
    except Exception as exc:  # noqa: BLE001 - surface helpful context to CLI users
        print(f"[montage] Failed to discover DCE filename for {dataset_id}: {exc}")
        return False

    dce_filename = filenames[-1] if filenames else None
    if not dce_filename:
        print(f"[montage] No DCE filename available – skipping montage rendering for {dataset_id}.")
        return False

    dce_path = os.path.join(nifti_directory, dce_filename)
    if not os.path.isfile(dce_path):
        print(f"[montage] DCE file missing – skipping montage rendering: {dce_path}")
        return False

    print(f"[montage] Generating montages for {dataset_id}")
    overall_success = True
    try:
        generate_parametric_montages(analysis_directory, image_directory, dce_path)
    except Exception as exc:  # noqa: BLE001 - runtime errors should surface to the CLI
        print(f"[montage] Failed to generate montages for {dataset_id}: {exc}")
        overall_success = False

    if use_projection:
        if generate_projection_montages is None:
            print("[projection] Projection rendering unavailable – skipping.")
            return overall_success
        try:
            projection_ok = generate_projection_montages(
                analysis_directory,
                image_directory,
                nifti_directory,
                dce_path,
                population_stats=projection_stats,
            )
            overall_success &= projection_ok
        except Exception as exc:  # noqa: BLE001 - runtime errors should surface to the CLI
            print(
                f"[projection] Failed to generate projection montages for {dataset_id}: {exc}"
            )
            overall_success = False

    return overall_success


def _run_diffusion_for_dataset(data_root, dataset_id, is_control):
    """Execute the diffusion tensor processing workflow for ``dataset_id``."""

    dataset_root = _resolve_dataset_root(data_root, dataset_id, is_control)
    if not os.path.isdir(dataset_root):
        print(f"[diffusion] Dataset directory missing – skipping: {dataset_root}")
        return False

    analysis_directory = os.path.join(dataset_root, "Analysis")
    nifti_directory = os.path.join(dataset_root, "NIfTI")
    image_directory = os.path.join(dataset_root, "Images")

    for path, label in (
        (analysis_directory, "Analysis"),
        (nifti_directory, "NIfTI"),
    ):
        if not os.path.isdir(path):
            print(f"[diffusion] {label} directory missing – skipping: {path}")
            return False

    try:
        if bool(is_control or settings.CONTROLS):
            filenames = parameters.control_filenames(nifti_directory)
        else:
            filenames = parameters.global_filenames(nifti_directory)
    except Exception as exc:
        print(f"[diffusion] Failed to discover configured filenames for {dataset_id}: {exc}")
        return False

    diffusion_filename = filenames[-2] if filenames else None
    if not diffusion_filename:
        print(
            f"[diffusion] No configured diffusion volume for {dataset_id} – skipping diffusion processing."
        )
        return False

    try:
        from modules.opt08_fa import compute_fa
    except ImportError as exc:
        print(f"[diffusion] Unable to import diffusion workflow: {exc}")
        return False

    print(f"[diffusion] Computing diffusion metrics for {dataset_id}")
    try:
        compute_fa(
            nifti_directory,
            analysis_directory,
            image_directory,
            diffusion_filename=diffusion_filename,
        )
    except Exception as exc:  # noqa: BLE001 - expose runtime issues to CLI users
        print(f"[diffusion] Failed to compute diffusion metrics for {dataset_id}: {exc}")
        return False

    return True


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
    parser.add_argument(
        "--from",
        dest="start_id",
        type=str,
        help="Start processing from the specified dataset ID (inclusive)",
    )
    parser.add_argument("ids", nargs="*", help="Specific dataset IDs to process")
    parser.add_argument(
        "--montage",
        action="store_true",
        help="Only generate montage images for the selected datasets",
    )
    parser.add_argument(
        "--projection",
        action="store_true",
        help=(
            "When used with --montage, also render parcel projection montages for atlas metrics"
        ),
    )
    parser.add_argument(
        "--diffusion",
        action="store_true",
        help="Run the diffusion tensor workflow for each dataset",
    )
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
    start_id = args.start_id

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

    if start_id:
        start_id = str(start_id)
        try:
            start_index = next(
                index for index, (dataset_id, _) in enumerate(datasets) if dataset_id == start_id
            )
        except StopIteration:
            print(f"Start dataset {start_id} not found in selection.")
            sys.exit(1)
        datasets = datasets[start_index:]

    if not datasets:
        print("No datasets found to process.")
        sys.exit(0)

    projection_stats = None
    if args.projection:
        try:
            projection_stats = _prepare_projection_stats(data_directory)
        except RuntimeError as exc:
            print(f"[projection] {exc}")
            sys.exit(1)

    if args.montage:
        exit_code = 0
        for dataset_id, is_control in datasets:
            if args.diffusion:
                diffusion_ok = _run_diffusion_for_dataset(
                    data_directory,
                    dataset_id,
                    is_control,
                )
                if not diffusion_ok:
                    exit_code = 1
            success = _run_montage_for_dataset(
                data_directory,
                dataset_id,
                is_control,
                use_projection=args.projection,
                projection_stats=projection_stats,
            )
            if not success:
                exit_code = 1
        sys.exit(exit_code)

    # Build command
    command_template = "python3 main.py --id {} --mode auto --data-dir {}"
    if args.diffusion:
        command_template += " --diffusion"

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
        result = subprocess.run(command, shell=True, env=env)
        if result.returncode != 0:
            print(
                f"[montage] Skipping montage rendering for {dataset_id} "
                f"due to pipeline failure (exit code {result.returncode})."
            )
            continue

        _run_montage_for_dataset(
            data_directory,
            dataset_id,
            is_control,
            use_projection=args.projection,
            projection_stats=projection_stats,
        )


if __name__ == "__main__":
    main()
