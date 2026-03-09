import argparse
import glob
import inspect
import os
import shutil
import shlex
import subprocess
import sys

import utils.settings as settings
import utils.parameters as parameters


_SEGMENTATION_COMPONENTS = (
    "segmentation",
    "segmentation",
    "mri",
    "aparc.DKTatlas+aseg.deep_in_DCE.nii.gz",
)


def _load_montage_dependencies():
    """Import heavy montage modules lazily."""

    from utils.montage import (
        compute_fixed_colorbar_width,
        generate_parametric_montages,
        generate_projection_montages,
        build_population_projection_stats,
    )
    from utils import parameters

    return (
        compute_fixed_colorbar_width,
        generate_parametric_montages,
        generate_projection_montages,
        parameters,
        build_population_projection_stats,
    )


def _expected_segmentation_path(nifti_directory: str) -> str:
    return os.path.join(nifti_directory, *_SEGMENTATION_COMPONENTS)


def _prepare_projection_stats(data_root: str):
    """Return cached atlas statistics for projection montages."""

    try:
        deps = _load_montage_dependencies()
    except ImportError as exc:  # pragma: no cover - import side effect
        raise RuntimeError(f"Unable to import montage dependencies: {exc}") from exc

    build_population_projection_stats = None
    # New format: (compute_fixed_colorbar_width, generate_parametric_montages,
    #              generate_projection_montages, parameters, build_population_projection_stats)
    if len(deps) >= 5 and hasattr(deps[3], "global_filenames"):
        build_population_projection_stats = deps[4]
    # Old format: (generate_parametric_montages, generate_projection_montages, parameters, build_population_projection_stats)
    elif len(deps) >= 4 and hasattr(deps[2], "global_filenames"):
        build_population_projection_stats = deps[3]

    if build_population_projection_stats is None:
        return None

    return build_population_projection_stats(
        data_root,
        include_controls=bool(settings.CONTROLS),
    )


def _find_anatomical_overlay(nifti_directory: str) -> str | None:
    """Return a T1 volume aligned to DCE space when available."""

    preferred = (
        os.path.join(
            nifti_directory,
            "segmentation",
            "segmentation",
            "mri",
            "T1w_conformed_in_DCE.nii.gz",
        ),
        os.path.join(
            nifti_directory,
            "segmentation",
            "segmentation",
            "mri",
            "T1w_in_DCE.nii.gz",
        ),
        os.path.join(
            nifti_directory,
            "segmentation",
            "segmentation",
            "mri",
            "T1_in_DCE.nii.gz",
        ),
    )

    for candidate in preferred:
        if os.path.isfile(candidate):
            return candidate

    search_roots = (
        os.path.join(nifti_directory, "segmentation", "segmentation", "mri"),
        os.path.join(nifti_directory, "segmentation", "mri"),
        os.path.join(nifti_directory, "segmentation"),
        nifti_directory,
    )

    patterns = (
        "*T1*DCE*.nii.gz",
        "*T1*DCE*.nii",
        "*T1*_in_DCE*.nii.gz",
        "*T1*_in_DCE*.nii",
    )

    for root in search_roots:
        if not os.path.isdir(root):
            continue

        for pattern in patterns:
            search_pattern = os.path.join(root, "**", pattern)
            for path in sorted(glob.iglob(search_pattern, recursive=True)):
                if os.path.isfile(path):
                    return path

    return None


def _maybe_generate_anatomical_overlay(nifti_directory: str, dce_path: str) -> str | None:
    """Return an anatomical overlay, generating one when necessary."""

    overlay = _find_anatomical_overlay(nifti_directory)
    if overlay is not None:
        return overlay

    generated = _generate_anatomical_overlay(nifti_directory, dce_path)
    if generated is not None:
        return generated

    return None


def _generate_anatomical_overlay(nifti_directory: str, dce_path: str) -> str | None:
    """Generate a T1 overlay coregistered to the provided DCE reference."""

    t1_candidates = _discover_t1_candidates(nifti_directory)
    if not t1_candidates:
        return None

    print("[montage] Anatomical overlay missing – attempting to resample T1 to DCE space.")

    for candidate in t1_candidates:
        try:
            prepared = _ensure_nifti_volume(candidate)
        except Exception as exc:  # noqa: BLE001 - surface helpful context
            print(
                f"[montage] Failed to prepare anatomical candidate {candidate!r}: {exc}"
            )
            continue

        overlay_path = _derive_overlay_output_path(prepared)
        try:
            produced = _coregister_volume_to_reference(prepared, dce_path, overlay_path)
        except Exception as exc:  # noqa: BLE001 - surface helpful context
            print(
                "[montage] Failed to resample anatomical volume into DCE space: "
                f"{exc}"
            )
            continue

        if produced and os.path.isfile(produced):
            return produced

    return None


def _discover_t1_candidates(nifti_directory: str) -> list[str]:
    """Return candidate anatomical T1 volumes from segmentation outputs."""

    search_roots = (
        os.path.join(nifti_directory, "segmentation", "segmentation", "mri"),
        os.path.join(nifti_directory, "segmentation", "mri"),
        os.path.join(nifti_directory, "segmentation"),
    )

    patterns = (
        "T1w_conformed*",
        "T1w*",
        "T1_*",
        "T1.*",
        "orig.mgz",
    )

    candidates: list[str] = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for pattern in patterns:
            search_pattern = os.path.join(root, "**", pattern)
            for path in sorted(glob.iglob(search_pattern, recursive=True)):
                if not os.path.isfile(path):
                    continue
                if "_in_DCE" in os.path.basename(path):
                    continue
                candidates.append(path)

    return candidates


def _ensure_nifti_volume(path: str) -> str:
    """Convert ``path`` to a NIfTI file if necessary and return the usable path."""

    lower = path.lower()
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        return path

    if lower.endswith(".mgz"):
        converted = path[:-4] + ".nii.gz"
        if os.path.isfile(converted):
            return converted

        try:
            import nibabel as nib
            import numpy as np
        except Exception as exc:  # noqa: BLE001 - expose helpful context
            raise RuntimeError(
                "nibabel is required to convert MGZ anatomical volumes to NIfTI"
            ) from exc

        image = nib.load(path)
        data = np.asanyarray(image.dataobj)
        nifti = nib.Nifti1Image(data, image.affine)
        nib.save(nifti, converted)
        return converted

    return path


def _derive_overlay_output_path(source_path: str) -> str:
    """Return the expected overlay filename for ``source_path``."""

    directory, filename = os.path.split(source_path)
    base = filename
    if base.lower().endswith(".nii.gz"):
        base = base[:-7]
    elif base.lower().endswith(".nii"):
        base = base[:-4]
    elif base.lower().endswith(".mgz"):
        base = base[:-4]

    return os.path.join(directory, f"{base}_in_DCE.nii.gz")


def _coregister_volume_to_reference(
    source_path: str, reference_path: str, output_path: str
) -> str:
    """Resample ``source_path`` into ``reference_path`` space and return ``output_path``."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    flirt_path = shutil.which("flirt")
    if flirt_path:
        command = [
            flirt_path,
            "-in",
            source_path,
            "-ref",
            reference_path,
            "-applyxfm",
            "-usesqform",
            "-interp",
            "trilinear",
            "-out",
            output_path,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and os.path.isfile(output_path):
            return output_path

        stderr = (result.stderr or "").strip()
        print(
            "[montage] FLIRT command failed while generating anatomical overlay: "
            f"{stderr or 'unknown error'}"
        )

    try:
        import nibabel as nib
        from nibabel.processing import resample_from_to
    except Exception as exc:  # noqa: BLE001 - expose helpful context
        raise RuntimeError(
            "Unable to generate anatomical overlay without FSL flirt or nibabel"
        ) from exc

    source_img = nib.load(source_path)
    reference_img = nib.load(reference_path)
    resampled = resample_from_to(source_img, reference_img)
    nib.save(resampled, output_path)
    return output_path


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
    use_anatomical=False,
    transparent_background=False,
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

    compute_fixed_colorbar_width = None
    generate_parametric_montages = None
    generate_projection_montages = None
    params_mod = parameters

    # Backwards compatible unpacking for tests/older runners.
    if len(deps) == 2 and hasattr(deps[1], "global_filenames"):
        generate_parametric_montages, params_mod = deps
    elif len(deps) == 3 and hasattr(deps[2], "global_filenames"):
        generate_parametric_montages, generate_projection_montages, params_mod = deps
    elif len(deps) >= 5 and hasattr(deps[3], "global_filenames"):
        compute_fixed_colorbar_width = deps[0]
        generate_parametric_montages = deps[1]
        generate_projection_montages = deps[2]
        params_mod = deps[3]
    else:
        generate_parametric_montages = deps[0] if deps else None
        generate_projection_montages = deps[1] if len(deps) >= 2 else None

    if generate_parametric_montages is None:
        print("[montage] Montage rendering unavailable – skipping.")
        return False

    try:
        if bool(is_control or settings.CONTROLS):
            filenames = params_mod.control_filenames(nifti_directory)
        else:
            filenames = params_mod.global_filenames(nifti_directory)
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

    anatomical_overlay = None
    if use_anatomical:
        anatomical_overlay = _maybe_generate_anatomical_overlay(
            nifti_directory, dce_path
        )
        if anatomical_overlay is None:
            print(
                "[montage] Anatomical overlay requested but no T1 reference was found – "
                "continuing without overlay."
            )
        else:
            print(
                "[montage] Using anatomical overlay for montages: "
                f"{os.path.relpath(anatomical_overlay, start=data_root)}"
            )

    print(f"[montage] Generating montages for {dataset_id}")
    overall_success = True
    segmentation_path = _expected_segmentation_path(nifti_directory)
    if not os.path.isfile(segmentation_path):
        segmentation_path = None

    fixed_cb_w = None
    try:
        if callable(compute_fixed_colorbar_width):
            fixed_cb_w = compute_fixed_colorbar_width(
                analysis_directory,
                nifti_directory,
                dce_path,
                anatomical_overlay=anatomical_overlay,
                segmentation_path=segmentation_path,
                population_stats=projection_stats if use_projection else None,
                transparent_background=transparent_background,
                include_projections=bool(use_projection),
            )
    except TypeError:
        # Backwards compatibility when signature differs.
        fixed_cb_w = None
    except Exception as exc:  # noqa: BLE001
        print(f"[montage] Failed to precompute fixed width – continuing without: {exc}")
        fixed_cb_w = None

    try:
        montage_kwargs = {
            "anatomical_overlay": anatomical_overlay,
            "segmentation_path": segmentation_path,
            "transparent_background": transparent_background,
        }
        if fixed_cb_w is not None:
            try:
                if "fixed_colorbar_width" in inspect.signature(generate_parametric_montages).parameters:
                    montage_kwargs["fixed_colorbar_width"] = fixed_cb_w
            except Exception:
                pass

        generate_parametric_montages(
            analysis_directory,
            image_directory,
            dce_path,
            **montage_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - runtime errors should surface to the CLI
        print(f"[montage] Failed to generate montages for {dataset_id}: {exc}")
        overall_success = False

    if use_projection:
        if generate_projection_montages is None:
            print("[projection] Projection rendering unavailable – skipping.")
            return overall_success
        try:
            projection_kwargs = {
                "population_stats": projection_stats,
                "transparent_background": transparent_background,
            }
            if fixed_cb_w is not None:
                try:
                    if "fixed_colorbar_width" in inspect.signature(generate_projection_montages).parameters:
                        projection_kwargs["fixed_colorbar_width"] = fixed_cb_w
                except Exception:
                    pass

            projection_ok = generate_projection_montages(
                analysis_directory,
                image_directory,
                nifti_directory,
                dce_path,
                **projection_kwargs,
            )
            overall_success &= projection_ok
        except Exception as exc:  # noqa: BLE001 - runtime errors should surface to the CLI
            print(
                f"[projection] Failed to generate projection montages for {dataset_id}: {exc}"
            )
            overall_success = False

    return overall_success


def _run_diffusion_for_dataset(
    data_root,
    dataset_id,
    is_control,
    *,
    diffusion_filename_override: str | None = None,
):
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

    diffusion_filename = diffusion_filename_override or (filenames[-2] if filenames else None)
    dce_filename = filenames[-1] if filenames else None
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
            dce_path=(
                os.path.join(nifti_directory, dce_filename)
                if dce_filename
                else None
            ),
        )
    except Exception as exc:  # noqa: BLE001 - expose runtime issues to CLI users
        print(f"[diffusion] Failed to compute diffusion metrics for {dataset_id}: {exc}")
        return False

    return True


def _run_tracks_for_dataset(
    data_root,
    dataset_id,
    is_control,
    *,
    create_montage=False,
    use_anatomical=False,
    tracks_only=False,
    render_only=False,
    orientation_model: str | None = None,
    diffusion_filename_override: str | None = None,
    act_tracking: bool | None = None,
    pft_tracking: bool | None = None,
    force_regenerate: bool = False,
    compute_connectome: bool = False,
    connectome_min_streamlines: int = 1,
    connectome_normalize_volume: bool = False,
):
    """Compute tractography renderings for ``dataset_id`` when possible."""

    dataset_root = _resolve_dataset_root(data_root, dataset_id, is_control)
    if not os.path.isdir(dataset_root):
        print(f"[tracks] Dataset directory missing – skipping: {dataset_root}")
        return False

    analysis_directory = os.path.join(dataset_root, "Analysis")
    nifti_directory = os.path.join(dataset_root, "NIfTI")
    image_directory = os.path.join(dataset_root, "Images")

    for path, label in (
        (analysis_directory, "Analysis"),
        (nifti_directory, "NIfTI"),
    ):
        if not os.path.isdir(path):
            print(f"[tracks] {label} directory missing – skipping: {path}")
            return False

    try:
        if bool(is_control or settings.CONTROLS):
            filenames = parameters.control_filenames(nifti_directory)
        else:
            filenames = parameters.global_filenames(nifti_directory)
    except Exception as exc:
        print(f"[tracks] Failed to discover configured filenames for {dataset_id}: {exc}")
        return False

    diffusion_filename = diffusion_filename_override or (filenames[-2] if filenames else None)
    dce_filename = filenames[-1] if filenames else None

    if not diffusion_filename:
        print(
            f"[tracks] No configured diffusion volume for {dataset_id} – skipping tractography."
        )
        return False

    anatomical_overlay = None
    if use_anatomical:
        dce_path = os.path.join(nifti_directory, dce_filename) if dce_filename else None
        if dce_path and not os.path.isfile(dce_path):
            dce_path = None
        if dce_path:
            anatomical_overlay = _maybe_generate_anatomical_overlay(nifti_directory, dce_path)
        else:
            anatomical_overlay = _find_anatomical_overlay(nifti_directory)
        if anatomical_overlay is None:
            print(
                "[tracks] Anatomical overlay requested but unavailable – proceeding without it."
            )

    try:
        from modules.tractography import generate_tractography
    except ImportError as exc:
        print(f"[tracks] Unable to import tractography workflow: {exc}")
        return False

    try:
        outputs = generate_tractography(
            nifti_directory,
            analysis_directory,
            image_directory,
            diffusion_filename=diffusion_filename,
            diffusion_model=orientation_model.upper() if orientation_model else None,
            anatomical_overlay=anatomical_overlay,
            create_montage=create_montage,
            montage_title=(
                "Tractography over anatomical T1"
                if (create_montage and anatomical_overlay and use_anatomical)
                else "Tractography overlay"
            ),
            render_only=render_only,
            force_regenerate=force_regenerate,
            enable_act=act_tracking,
            enable_pft=pft_tracking,
        )
    except Exception as exc:  # noqa: BLE001 - surface helpful context to CLI users
        print(f"[tracks] Failed to compute tractography for {dataset_id}: {exc}")
        return False

    print(
        "[tracks] Generated tractography: "
        f"{os.path.relpath(outputs.tract_path, start=data_root)}"
    )
    if outputs.render_path:
        print(
            "[tracks] Render saved to "
            f"{os.path.relpath(outputs.render_path, start=data_root)}"
        )
    if outputs.montage_path:
        print(
            "[tracks] Montage saved to "
            f"{os.path.relpath(outputs.montage_path, start=data_root)}"
        )

    if compute_connectome:
        try:
            from modules.connectome import compute_connectome as _compute_connectome

            diffusion_filename_for_connectome = diffusion_filename
            _compute_connectome(
                nifti_directory,
                analysis_directory,
                diffusion_filename=diffusion_filename_for_connectome,
                tractography_path=outputs.tract_path,
                min_streamlines=max(1, int(connectome_min_streamlines)),
                normalize_by_nodepair_volume=bool(connectome_normalize_volume),
            )
            print(
                "[connectome] Saved connectome outputs under "
                f"{os.path.relpath(os.path.join(analysis_directory, 'diffusion'), start=data_root)}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[connectome] Failed to compute connectome for {dataset_id}: {exc}")

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
        "--montage_anatomical",
        action="store_true",
        help=(
            "Overlay montage values on the anatomical T1 volume resliced to DCE space "
            "(requires precomputed alignment, implies --montage)"
        ),
    )
    parser.add_argument(
        "--montage_transparent",
        action="store_true",
        help=(
            "Render montages with a transparent background and without anatomical underlays "
            "(implies --montage and disables --montage_anatomical)."
        ),
    )
    parser.add_argument(
        "--tracks_force",
        action="store_true",
        help=(
            "Force regeneration of tractography streamlines even if a prior tractography.trk exists."
        ),
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
    parser.add_argument(
        "--diffusion_only",
        action="store_true",
        help="Only run the diffusion tensor workflow for patient datasets",
    )
    parser.add_argument(
        "--tracks",
        action="store_true",
        help=(
            "Compute diffusion tractography streamlines and render visualisations. "
            "Implies --diffusion so the tensor fit is available."
        ),
    )
    parser.add_argument(
        "--tracks_only",
        action="store_true",
        help=(
            "Re-render existing tractography outputs without recomputing streamlines. "
            "Implies --tracks but skips diffusion/tract generation; requires prior run."
        ),
    )
    parser.add_argument(
        "--tracks_dont_recompute",
        action="store_true",
        help=(
            "When used with --tracks/--tracks_only, skip streamline regeneration and only refresh "
            "renders/montages (requires existing tractography.trk)."
        ),
    )
    parser.add_argument(
        "--tracks_act",
        action="store_true",
        default=None,
        help=(
            "Enable anatomically constrained tractography (ACT) when tissue probability maps "
            "are available. Without this flag the run falls back to legacy streamline tracking."
        ),
    )
    parser.add_argument(
        "--tracks_pft",
        action="store_true",
        default=None,
        help=(
            "Enable particle filtering tractography (CMC/PFT). Can be combined with --tracks_act; "
            "defaults to disabled."
        ),
    )
    parser.add_argument(
        "--tracks_advanced",
        action="store_true",
        help=(
            "Convenience flag that enables both ACT and PFT (equivalent to --tracks_act --tracks_pft)."
        ),
    )
    parser.add_argument(
        "--connectome",
        action="store_true",
        help=(
            "Compute a structural connectome + graph topology metrics from tractography and "
            "atlas segmentation (runs after --tracks)."
        ),
    )
    parser.add_argument(
        "--connectome_min_streamlines",
        type=int,
        default=1,
        help=(
            "Minimum streamline count for an edge to be considered present in binary topology metrics."
        ),
    )
    parser.add_argument(
        "--connectome_norm_volume",
        action="store_true",
        help=(
            "Normalize edge weights by node-pair volume (streamlines / (vol_i + vol_j))."
        ),
    )
    parser.add_argument(
        "--diffusion-file",
        dest="diffusion_filename",
        type=str,
        help=(
            "Explicit diffusion NIfTI filename (relative to each dataset's NIfTI folder or absolute) "
            "to use for diffusion metrics and tractography."
        ),
    )
    parser.add_argument(
        "--orientation",
        dest="orientation_model",
        type=str,
        choices=("tensor", "dti", "csd", "mt_csd", "msmt", "msmt_csd", "qball", "gqi"),
        help=(
            "Override the diffusion orientation model for tractography. "
            "Examples: --orientation csd (single-shell) or --orientation mt_csd (multi-tissue)."
        ),
    )
    parser.add_argument(
        "--flip-angle",
        dest="flip_angle",
        type=str,
        default=None,
        help='Flip angle in degrees (number) or "auto" (default: from metadata). Passed to main pipeline.',
    )
    parser.add_argument(
        "--t1-fit",
        dest="t1_fit",
        type=str,
        choices=("auto", "ir", "vfa", "none"),
        default=None,
        help='T1/M0 fitting source: auto|ir|vfa|none (default: auto). Passed to main pipeline.',
    )
    parser.add_argument(
        "--vfa-glob",
        dest="vfa_glob",
        type=str,
        default=None,
        help='Comma-separated glob(s) for VFA NIfTI discovery (default: *VFA*.nii*). Passed to main pipeline.',
    )
    parser.add_argument(
        "--ctc-model",
        dest="ctc_model",
        type=str,
        choices=("saturation", "turboflash", "advanced"),
        default=None,
        help=(
            "Signal-to-concentration conversion model: saturation (legacy), turboflash (reference), or advanced (validated TurboFLASH case12). "
            "When set, exported as P_BRAIN_CTC_MODEL for the main pipeline."
        ),
    )
    parser.add_argument(
        "--turbo-nph",
        dest="turbo_nph",
        type=int,
        default=None,
        help=(
            "TurboFLASH nph (1-based ky=0 line index within readout train). "
            "Used when --ctc-model=turboflash; exported as P_BRAIN_TURBO_NPH."
        ),
    )
    parser.add_argument(
        "--roi-method",
        dest="roi_method",
        type=str,
        choices=("ai", "deterministic", "geometry", "file"),
        default=None,
        help=(
            "ROI extraction method for input-function ROIs. "
            "When set, exported as P_BRAIN_ROI_METHOD."
        ),
    )
    parser.add_argument(
        "--roi-aif-mat",
        dest="roi_aif_mat",
        type=str,
        default=None,
        help="Path to .mat file containing arterial ROI mask (BW_input/BW_input_big).",
    )
    parser.add_argument(
        "--roi-aif-conc-mat",
        dest="roi_aif_conc_mat",
        type=str,
        default=None,
        help="Optional path to .mat file containing arterial concentration curve (c_input).",
    )
    parser.add_argument(
        "--roi-sss-mat",
        dest="roi_sss_mat",
        type=str,
        default=None,
        help="Path to .mat file containing SSS ROI mask (BW_input/BW_input_big).",
    )
    parser.add_argument(
        "--roi-sss-conc-mat",
        dest="roi_sss_conc_mat",
        type=str,
        default=None,
        help="Optional path to .mat file containing SSS concentration curve (c_input).",
    )
    parser.add_argument(
        '--roi-tscc-mat',
        dest='roi_tscc_mat',
        type=str,
        default=None,
        help='Optional path to .mat file containing TSCC input function (c_input) used for modelling. If provided, TSCC generation will import this curve directly.'
    )
    parser.add_argument(
        "--roi-only",
        dest="roi_only",
        action="store_true",
        help=(
            "Only run ROI extraction (input-function step) and exit. "
            "Skips the rest of the main pipeline and any montage/tracks work."
        ),
    )
    parser.add_argument(
        "--single-bolus",
        dest="single_bolus",
        action="store_true",
        help=(
            "Single contrast injection protocol. Sets NUMBER_OF_PEAKS=1 so "
            "time-shifting and tissue-function stages look for one peak only."
        ),
    )
    parser.add_argument(
        "--num-peaks",
        dest="num_peaks",
        type=int,
        default=None,
        help=(
            "Explicit number of bolus peaks expected in the DCE signal. "
            "Overrides --single-bolus. Default is 2 (dual injection)."
        ),
    )
    parser.add_argument(
        "--dce-skip-volumes",
        dest="dce_skip_volumes",
        type=int,
        default=None,
        help=(
            "Number of leading DCE volumes (time frames) to discard before "
            "analysis. Useful when the first few acquisitions are unstable."
        ),
    )
    parser.add_argument(
        "--pk-model", "--pk-models",
        dest="pk_model",
        type=str,
        default=None,
        help=(
            "PK model(s) to run.  Single: patlak | tikhonov | etofts | both | all.  "
            "Combine with '+': e.g. patlak+etofts, etofts+patlak+tikhonov.  "
            "Default: both (patlak + tikhonov)."
        ),
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
        # p-brain-web stores app metadata in a hidden folder inside the data root.
        # Enumerator should treat hidden directories as non-datasets.
        if str(name).startswith("."):
            continue
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
    if getattr(args, "tracks_advanced", False):
        args.tracks_act = True
        args.tracks_pft = True
    if args.tracks_dont_recompute and not (args.tracks or args.tracks_only):
        # Allow --tracks_dont_recompute to work on its own by falling back to rerender mode.
        args.tracks_only = True
    data_directory = os.path.abspath(args.data_dir)
    use_all = args.all
    ids = args.ids
    start_id = args.start_id

    # Common mistake: ``python enumerator.py --all /path/to/data`` puts the
    # path into ``ids`` instead of ``--data-dir``.  Detect and fix.
    if use_all and ids and len(ids) == 1 and os.path.isdir(ids[0]):
        print(f"Hint: treating positional argument as --data-dir {ids[0]}")
        data_directory = os.path.abspath(ids[0])
        ids = []

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
    except FileNotFoundError as exc:
        # collect_datasets may raise FileNotFoundError both when the data root is
        # missing and when one of the requested dataset IDs does not exist.
        if not os.path.isdir(data_directory):
            print(f"Data dir missing: {data_directory}")
        else:
            print(str(exc) or f"Dataset not found in {data_directory}.")
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

    if args.montage_transparent:
        args.montage = True
        if args.montage_anatomical:
            print("[montage] --montage_transparent disables anatomical overlays.")
            args.montage_anatomical = False

    if args.montage_anatomical:
        args.montage = True
        args.diffusion = True

    if args.diffusion_only:
        args.diffusion = True

    if args.tracks_only:
        args.tracks = True
        if not args.diffusion_only:
            args.diffusion = False
            args.diffusion_only = False
    elif args.tracks:
        args.diffusion = True

    args.tracks_render_only = bool(args.tracks_dont_recompute or args.tracks_only)
    args.tracks_force = bool(getattr(args, "tracks_force", False))
    force_tracks_regen = bool(args.tracks_force or (args.tracks_only and not args.tracks_render_only))

    projection_stats = None
    if args.projection:
        try:
            projection_stats = _prepare_projection_stats(data_directory)
        except RuntimeError as exc:
            print(f"[projection] {exc}")
            sys.exit(1)

    if args.tracks_only and not args.diffusion_only:
        exit_code = 0
        for dataset_id, is_control in datasets:
            tracks_ok = _run_tracks_for_dataset(
                data_directory,
                dataset_id,
                is_control,
                create_montage=args.montage,
                use_anatomical=args.montage_anatomical,
                tracks_only=True,
                render_only=args.tracks_render_only,
                force_regenerate=force_tracks_regen,
                orientation_model=args.orientation_model,
                diffusion_filename_override=args.diffusion_filename,
                act_tracking=args.tracks_act,
                pft_tracking=args.tracks_pft,
            )
            if not tracks_ok:
                exit_code = 1

            if args.montage:
                success = _run_montage_for_dataset(
                    data_directory,
                    dataset_id,
                    is_control,
                    use_projection=args.projection,
                    projection_stats=projection_stats,
                    use_anatomical=args.montage_anatomical,
                    transparent_background=args.montage_transparent,
                )
                if not success:
                    exit_code = 1

        sys.exit(exit_code)

    if args.diffusion_only:
        exit_code = 0
        for dataset_id, is_control in datasets:
            if is_control:
                print(
                    f"[diffusion] Skipping control dataset {dataset_id} in diffusion-only mode."
                )
                continue

            diffusion_ok = _run_diffusion_for_dataset(
                data_directory,
                dataset_id,
                is_control,
                diffusion_filename_override=args.diffusion_filename,
            )
            if not diffusion_ok:
                exit_code = 1

            if args.tracks and diffusion_ok:
                tracks_ok = _run_tracks_for_dataset(
                    data_directory,
                    dataset_id,
                    is_control,
                    create_montage=args.montage,
                    use_anatomical=args.montage_anatomical,
                    tracks_only=args.tracks_only,
                    render_only=args.tracks_render_only,
                    force_regenerate=force_tracks_regen,
                    orientation_model=args.orientation_model,
                    diffusion_filename_override=args.diffusion_filename,
                    act_tracking=args.tracks_act,
                    pft_tracking=args.tracks_pft,
                )
                if not tracks_ok:
                    exit_code = 1

            if args.montage:
                success = _run_montage_for_dataset(
                    data_directory,
                    dataset_id,
                    is_control,
                    use_projection=args.projection,
                    projection_stats=projection_stats,
                    use_anatomical=args.montage_anatomical,
                    transparent_background=args.montage_transparent,
                )
                if not success:
                    exit_code = 1

        sys.exit(exit_code)

    if args.montage:
        exit_code = 0
        for dataset_id, is_control in datasets:
            if args.diffusion:
                diffusion_ok = _run_diffusion_for_dataset(
                    data_directory,
                    dataset_id,
                    is_control,
                    diffusion_filename_override=args.diffusion_filename,
                )
                if not diffusion_ok:
                    exit_code = 1
            if args.tracks and (not args.diffusion or diffusion_ok):
                tracks_ok = _run_tracks_for_dataset(
                    data_directory,
                    dataset_id,
                    is_control,
                    create_montage=True,
                    use_anatomical=args.montage_anatomical,
                    tracks_only=args.tracks_only,
                    render_only=args.tracks_render_only,
                    force_regenerate=force_tracks_regen,
                    orientation_model=args.orientation_model,
                    diffusion_filename_override=args.diffusion_filename,
                    act_tracking=args.tracks_act,
                    pft_tracking=args.tracks_pft,
                    compute_connectome=args.connectome,
                    connectome_min_streamlines=args.connectome_min_streamlines,
                    connectome_normalize_volume=args.connectome_norm_volume,
                )
                if not tracks_ok:
                    exit_code = 1
            success = _run_montage_for_dataset(
                data_directory,
                dataset_id,
                is_control,
                use_projection=args.projection,
                projection_stats=projection_stats,
                use_anatomical=args.montage_anatomical,
                transparent_background=args.montage_transparent,
            )
            if not success:
                exit_code = 1
        sys.exit(exit_code)

    # Build command
    if getattr(args, "roi_only", False):
        command_template = "python3 roi_only.py --id {} --data-dir {}"
    else:
        command_template = "python3 main.py --id {} --mode auto --data-dir {}"
    if args.diffusion:
        command_template += " --diffusion"
    if args.flip_angle is not None:
        command_template += " --flip-angle {}"
    if getattr(args, "t1_fit", None) is not None:
        command_template += " --t1-fit {}"
    if getattr(args, "vfa_glob", None) is not None:
        command_template += " --vfa-glob {}"

    for dataset_id, is_control in datasets:
        fmt_args = [dataset_id, data_directory]
        if args.flip_angle is not None:
            fmt_args.append(str(args.flip_angle))
        if getattr(args, "t1_fit", None) is not None:
            fmt_args.append(str(args.t1_fit))
        if getattr(args, "vfa_glob", None) is not None:
            fmt_args.append(shlex.quote(str(args.vfa_glob)))
        command = command_template.format(*fmt_args)
        env = os.environ.copy()
        env["P_BRAIN_DATA_DIR"] = data_directory
        env["PBRAIN_TURBO"] = "1"
        if getattr(args, "ctc_model", None):
            env["P_BRAIN_CTC_MODEL"] = str(args.ctc_model)
        if getattr(args, "turbo_nph", None) is not None:
            env["P_BRAIN_TURBO_NPH"] = str(int(args.turbo_nph))
        if getattr(args, "roi_method", None):
            env["P_BRAIN_ROI_METHOD"] = str(args.roi_method)
        if getattr(args, "pk_model", None):
            env["P_BRAIN_MODEL"] = str(args.pk_model)

        if getattr(args, "num_peaks", None) is not None:
            env["P_BRAIN_NUMBER_OF_PEAKS"] = str(int(args.num_peaks))
        elif getattr(args, "single_bolus", False):
            env["P_BRAIN_NUMBER_OF_PEAKS"] = "1"
        if getattr(args, "dce_skip_volumes", None) is not None:
            env["P_BRAIN_DCE_SKIP_VOLUMES"] = str(int(args.dce_skip_volumes))
        if getattr(args, "roi_aif_mat", None):
            env["P_BRAIN_ROI_AIF_MAT"] = str(args.roi_aif_mat)
        if getattr(args, "roi_aif_conc_mat", None):
            env["P_BRAIN_ROI_AIF_CONC_MAT"] = str(args.roi_aif_conc_mat)
        if getattr(args, "roi_sss_mat", None):
            env["P_BRAIN_ROI_SSS_MAT"] = str(args.roi_sss_mat)
        if getattr(args, "roi_sss_conc_mat", None):
            env["P_BRAIN_ROI_SSS_CONC_MAT"] = str(args.roi_sss_conc_mat)
        if getattr(args, "roi_tscc_mat", None):
            env["P_BRAIN_ROI_TSCC_MAT"] = str(args.roi_tscc_mat)
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

        if getattr(args, "roi_only", False):
            # User requested ROI extraction only; do not run montage/tracks.
            continue

        _run_montage_for_dataset(
            data_directory,
            dataset_id,
            is_control,
            use_projection=args.projection,
            projection_stats=projection_stats,
            use_anatomical=args.montage_anatomical,
            transparent_background=args.montage_transparent,
        )

        if args.tracks:
            _run_tracks_for_dataset(
                data_directory,
                dataset_id,
                is_control,
                create_montage=False,
                use_anatomical=args.montage_anatomical,
                tracks_only=args.tracks_only,
                render_only=args.tracks_render_only,
                orientation_model=args.orientation_model,
                diffusion_filename_override=args.diffusion_filename,
                act_tracking=args.tracks_act,
                pft_tracking=args.tracks_pft,
            )


if __name__ == "__main__":
    main()
