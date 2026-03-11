"""
Windows pipeline orchestrator.

Mirrors the macOS ``main.py`` auto-mode flow but replaces all FreeSurfer /
FastSurfer / FSL subprocess calls with pure-Python equivalents from
``windows.neuroimaging`` and ``windows.segmentation``.

Existing p-brain modules (T1 fitting, input function, time-shifting,
kinetic modelling) are imported and called directly — they are pure Python
and work on any platform.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
#  Ensure p-brain root is on sys.path so we can import its modules.
# ---------------------------------------------------------------------------
_PBRAIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PBRAIN_ROOT not in sys.path:
    sys.path.insert(0, _PBRAIN_ROOT)

# Tell matplotlib to use a non-interactive backend (no display on Windows CI).
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["PBRAIN_TURBO"] = "1"  # Suppress interactive plots


def _apply_defaults_json(path: str, args) -> None:
    """Load a p-brain-web Defaults JSON and inject into env/settings."""
    try:
        from utils.defaults import apply_defaults_json
        apply_defaults_json(path, args=args)
    except Exception as exc:
        raise RuntimeError(f"Failed to apply --defaults-json={path}: {exc}")


def run_pipeline(
    subject_id: str,
    *,
    data_dir: str = "./Data",
    segmentation_method: str = "synthseg",
    synthseg_home: Optional[str] = None,
    pk_model: str = "both",
    t1_fit: str = "auto",
    roi_method: str = "deterministic",
    tissue_roi_method: str = "automatic",
    defaults_json: Optional[str] = None,
    force_recreate_masks: bool = False,
    cpu: bool = False,
    tikhonov_lambda: Optional[float] = None,
    flip_angle: Optional[str] = None,
) -> None:
    """Execute the full p-brain analysis pipeline (Windows variant).

    This is the Windows equivalent of ``main.py``'s ``mode == 'auto'`` path.
    """
    start = time.time()

    # ---- env setup ----
    os.environ["P_BRAIN_DATA_DIR"] = os.path.abspath(data_dir)
    if force_recreate_masks:
        os.environ["FORCE_RECREATE_MASKS"] = "1"

    # ---- late imports (after sys.path manipulation) ----
    from utils import settings
    import utils.plotting as plotting
    from utils.loading import discover_ir_series, discover_vfa_series

    import modules.opt01_T1_fit as opt01_T1_fit
    import modules.opt03_time_shifting as opt03_time_shifting
    from modules.input_function_dispatch import run_input_function

    # Apply defaults
    if defaults_json:
        import types
        _ns = types.SimpleNamespace()
        _apply_defaults_json(defaults_json, _ns)

    # ---- PK model ----
    from models import normalise_pk_model
    pk_model = normalise_pk_model(pk_model)
    settings.KINETIC_MODEL = pk_model

    if tikhonov_lambda is not None:
        settings.TIKHONOV_LAMBDA = tikhonov_lambda
        settings.AUTO_LAMBDA = False

    # ---- Flip angle ----
    if flip_angle is not None:
        raw = str(flip_angle).strip()
        if raw.lower() == "auto" or raw == "":
            settings.FLIP_ANGLE_SETTING = "auto"
            settings.FLIP_ANGLE_DEG = None
        else:
            settings.FLIP_ANGLE_SETTING = raw
            settings.FLIP_ANGLE_DEG = float(raw)

    # ---- ROI method ----
    settings.ROI_METHOD = roi_method
    os.environ["P_BRAIN_ROI_METHOD"] = roi_method

    # ---- Tissue ROI method ----
    settings.TISSUE_ROI_METHOD = tissue_roi_method
    os.environ["P_BRAIN_TISSUE_ROI_METHOD"] = tissue_roi_method

    # ---- Segmentation method (always synthseg on Windows) ----
    settings.SEGMENTATION_METHOD = segmentation_method

    # ---- T1 fit mode ----
    settings.T1_FIT_MODE = t1_fit

    # ---- Turbo mode for all modules ----
    for m in [plotting, opt01_T1_fit, opt03_time_shifting]:
        m.turbo_mode = True

    # ---- Directory setup ----
    from utils.settings import (
        setup_directories,
        save_run_settings,
        CONTROLS,
    )
    from utils.parameters import (
        global_filenames,
        control_filenames,
        global_parameters,
        refresh_nifti_directory,
    )
    from modules.opt07_axials import check_axial
    from modules.start import parrec2nifti

    data_directory, analysis_directory, nifti_directory, image_directory = (
        setup_directories(subject_id, os.path.abspath(data_dir))
    )

    if CONTROLS:
        filenames = control_filenames(nifti_directory)
    else:
        filenames = global_filenames(nifti_directory)
    parameters = global_parameters()

    parrec2nifti(data_directory, nifti_directory)
    if CONTROLS:
        filenames = control_filenames(nifti_directory)
    else:
        filenames = global_filenames(nifti_directory)
    parameters = global_parameters()
    refresh_nifti_directory(nifti_directory)
    check_axial(nifti_directory, filenames)

    # ---- Auto-detect T1 fit source ----
    if settings.T1_FIT_MODE == "auto":
        has_ir = bool(discover_ir_series(nifti_directory))
        has_vfa = bool(discover_vfa_series(
            nifti_directory,
            patterns=getattr(settings, "VFA_FILE_GLOB", None),
        ))
        if has_ir:
            settings.T1_FIT_MODE = "ir"
        elif has_vfa:
            settings.T1_FIT_MODE = "vfa"
        else:
            settings.T1_FIT_MODE = "none"
        parameters = global_parameters()

    # ---- STAGE 1: T1 fitting ----
    print("\n" + "=" * 60)
    print("  STAGE 1 / 4 :  T1 fitting")
    print("=" * 60)
    from utils import T1_fit
    T1_fit(data_directory, analysis_directory, nifti_directory,
           image_directory, filenames, parameters)

    # ---- STAGE 2: Input function ----
    print("\n" + "=" * 60)
    print("  STAGE 2 / 4 :  Input function extraction")
    print("=" * 60)
    run_input_function(analysis_directory, nifti_directory,
                       image_directory, filenames, parameters)

    # ---- STAGE 3: Time-shifted concentration curves ----
    print("\n" + "=" * 60)
    print("  STAGE 3 / 4 :  Time-shifted concentration curves")
    print("=" * 60)
    opt03_time_shifting.time_shifting(
        analysis_directory, nifti_directory, image_directory,
    )

    # ---- STAGE 4: Segmentation + tissue modelling ----
    print("\n" + "=" * 60)
    print("  STAGE 4 / 4 :  Segmentation & tissue kinetic modelling")
    print("=" * 60)
    _tissue_roi_method = getattr(settings, "TISSUE_ROI_METHOD", "automatic").strip().lower()
    if _tissue_roi_method == "manual":
        from modules.manual_tissue_roi import tissue_function_manual
        tissue_function_manual(
            analysis_directory,
            nifti_directory,
            image_directory,
            filenames,
            parameters,
        )
    else:
        _run_tissue_analysis_windows(
            analysis_directory=analysis_directory,
            nifti_directory=nifti_directory,
            image_directory=image_directory,
            filenames=filenames,
            parameters=parameters,
            synthseg_home=synthseg_home,
            cpu=cpu,
            force_masks=force_recreate_masks,
        )

    save_run_settings(analysis_directory, parameters)
    settings.save_runtime_metadata(analysis_directory)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline completed in {elapsed:.1f}s")
    print(f"{'=' * 60}")


def _run_tissue_analysis_windows(
    analysis_directory: str,
    nifti_directory: str,
    image_directory: str,
    filenames: tuple,
    parameters: tuple,
    *,
    synthseg_home: Optional[str] = None,
    cpu: bool = False,
    force_masks: bool = False,
) -> None:
    """Windows-specific tissue analysis: SynthSeg + pure-Python masks.

    This replaces the macOS ``_tissue_function_AI`` which shells out to
    FreeSurfer/FSL/FastSurfer.
    """
    import nibabel as nib
    import numpy as np

    from utils import settings
    from utils.loading import load_dce_4d, resolve_flip_angle_deg

    from windows.neuroimaging import (
        mgz_to_nifti,
        create_all_masks,
        create_coregistered_masks,
    )
    from windows.segmentation import run_synthseg

    (
        t1_3D_filename,
        axial_t1_3D_filename,
        t2_3D_filename,
        axial_t2_3D_filename,
        flair_3D_filename,
        axial_flair_3D_filename,
        axial_t2_2D_filename,
        diffusion_filename,
        dce_filename,
    ) = filenames

    IsVFA, IsIR, apple_metal, boundary, RERUN_SEG, SEG_METHOD, _ = parameters

    t1_path = os.path.join(nifti_directory, t1_3D_filename) if t1_3D_filename else None
    dce_path = os.path.join(nifti_directory, dce_filename) if dce_filename else None
    t2_path = os.path.join(nifti_directory, axial_t2_2D_filename)

    if not dce_path or not os.path.exists(dce_path):
        raise RuntimeError("Missing DCE NIfTI.")

    flip_angle_deg = resolve_flip_angle_deg(dce_path, default=None)

    # ---- Decide whether we have a structural T1 ----
    has_struct_t1 = bool(t1_path and os.path.exists(t1_path))
    voxelwise_only = not has_struct_t1

    seg_dir = os.path.join(nifti_directory, "segmentation")
    sid = "segmentation"
    mri_dir = os.path.join(seg_dir, sid, "mri")
    os.makedirs(mri_dir, exist_ok=True)

    # The macOS pipeline writes the seg as:
    #   <seg_dir>/segmentation/mri/aparc.DKTatlas+aseg.deep.mgz
    # SynthSeg will output NIfTI directly.  We convert if needed.
    seg_nii_path = os.path.join(mri_dir, "aparc.DKTatlas+aseg.deep.nii.gz")
    seg_mgz_path = os.path.join(mri_dir, "aparc.DKTatlas+aseg.deep.mgz")

    if not voxelwise_only:
        # ---- Run SynthSeg ----
        if force_masks or not os.path.exists(seg_nii_path):
            run_synthseg(
                input_path=t1_path,
                output_path=seg_nii_path,
                synthseg_home=synthseg_home,
                robust=True,
                cpu=cpu,
                vol_csv=os.path.join(mri_dir, "synthseg_volumes.csv"),
                qc_csv=os.path.join(mri_dir, "synthseg_qc.csv"),
            )
        else:
            print("[segmentation] Segmentation already exists, skipping.")

        # ---- Create T1-space masks ----
        print("[masks] Creating T1-space tissue masks ...")
        create_all_masks(seg_nii_path, mri_dir, force=force_masks)

        # ---- Coregister to DCE and T2 ----
        print("[masks] Coregistering masks to DCE space ...")
        dce_masks = create_coregistered_masks(
            seg_nii_path, dce_path, "DCE", force=force_masks,
        )

        print("[masks] Coregistering masks to T2 space ...")
        t2_masks = create_coregistered_masks(
            seg_nii_path, t2_path, "T2", force=force_masks,
        )

    # ---- Load DCE data ----
    ref_img, data_4d = load_dce_4d(dce_path, prefer_complex_mag=True, dtype=np.float32)
    data_4d = np.asarray(data_4d)
    ref_affine = ref_img.affine
    ref_header = ref_img.header.copy()

    # ---- Load T1/M0 ----
    import pickle

    def _load_pkl(name):
        p = os.path.join(analysis_directory, "Fitting", name)
        with open(p, "rb") as f:
            return pickle.load(f)

    T1_matrix = _load_pkl("voxel_T1_matrix.pkl")
    M0_matrix = _load_pkl("voxel_M0_matrix.pkl")

    # ---- Time points ----
    from modules.AI_tissue_functions import (
        resolve_dce_time_step_s,
        build_time_points_s,
        compute_and_plot_ctcs_median,
    )

    time_path = os.path.join(analysis_directory, "Fitting", "time_points_s.npy")
    time_points_s = None
    if os.path.isfile(time_path):
        try:
            time_points_s = np.load(time_path)
        except Exception:
            pass
    if time_points_s is None:
        dt_s = resolve_dce_time_step_s(dce_path, default=None)
        time_points_s = build_time_points_s(data_4d.shape[-1], dt_s)
        os.makedirs(os.path.dirname(time_path), exist_ok=True)
        np.save(time_path, time_points_s)

    # ---- Model selection ----
    model = (settings.KINETIC_MODEL or "both").strip().lower()
    _model_parts = set(model.split("+")) if "+" in model else {model}
    compute_ki = bool({"patlak", "both", "all"} & _model_parts)
    compute_cbf = bool({"tikhonov", "both", "all"} & _model_parts)
    compute_etofts = bool({"extended_tofts", "etofts", "all"} & _model_parts)

    if voxelwise_only:
        compute_and_plot_ctcs_median(
            data_4d, None, None, None, None, None, None, None,
            T1_matrix, M0_matrix, analysis_directory, time_points_s,
            image_directory,
            dce_path=dce_path,
            ref_affine=ref_affine,
            ref_header=ref_header,
            boundary=False,
            compute_per_voxel_Ki=compute_ki,
            compute_per_voxel_CBF=compute_cbf,
            compute_per_voxel_ETofts=compute_etofts,
            flip_angle_deg=flip_angle_deg,
            voxelwise_only=True,
        )
    else:
        # Load masks
        t2_img = nib.load(t2_path).get_fdata()
        _load = lambda p: nib.load(p).get_fdata().astype(bool)

        wm_mask_t2 = _load(t2_masks["wm"])
        cortical_gm_mask_t2 = _load(t2_masks["cortical_gm"])
        subcortical_gm_mask_t2 = _load(t2_masks["subcortical_gm"])
        wm_mask_dce = _load(dce_masks["wm"])
        cortical_gm_mask_dce = _load(dce_masks["cortical_gm"])
        subcortical_gm_mask_dce = _load(dce_masks["subcortical_gm"])

        gm_brainstem_mask_t2 = _load(t2_masks["gm_brainstem"])
        gm_brainstem_mask_dce = _load(dce_masks["gm_brainstem"])
        gm_cerebellum_mask_t2 = _load(t2_masks["gm_cerebellum"])
        gm_cerebellum_mask_dce = _load(dce_masks["gm_cerebellum"])
        wm_cerebellum_mask_t2 = _load(t2_masks["wm_cerebellum"])
        wm_cerebellum_mask_dce = _load(dce_masks["wm_cerebellum"])
        wm_cc_mask_t2 = _load(t2_masks["wm_cc"])
        wm_cc_mask_dce = _load(dce_masks["wm_cc"])

        # WMH masks (optional — may not be present or may be empty)
        wmh_mask_dce = None
        wmh_mask_t2 = None
        if "wmh" in dce_masks and os.path.isfile(dce_masks["wmh"]):
            _wmh = _load(dce_masks["wmh"])
            if np.any(_wmh):
                wmh_mask_dce = _wmh
                # Subtract WMH from WM in DCE space
                wm_mask_dce = wm_mask_dce & ~wmh_mask_dce
                n_wmh = int(np.sum(wmh_mask_dce))
                print(f"[masks] WM-hypointensities in DCE space: {n_wmh} voxels")
        if "wmh" in t2_masks and os.path.isfile(t2_masks["wmh"]):
            _wmh = _load(t2_masks["wmh"])
            if np.any(_wmh):
                wmh_mask_t2 = _wmh
                wm_mask_t2 = wm_mask_t2 & ~wmh_mask_t2

        compute_and_plot_ctcs_median(
            data_4d, t2_img,
            wm_mask_t2, cortical_gm_mask_t2, subcortical_gm_mask_t2,
            wm_mask_dce, cortical_gm_mask_dce, subcortical_gm_mask_dce,
            T1_matrix, M0_matrix, analysis_directory, time_points_s,
            image_directory,
            dce_path=dce_path,
            ref_affine=ref_affine,
            ref_header=ref_header,
            boundary=boundary,
            compute_per_voxel_Ki=compute_ki,
            compute_per_voxel_CBF=compute_cbf,
            compute_per_voxel_ETofts=compute_etofts,
            gm_brainstem_mask_t2=gm_brainstem_mask_t2,
            gm_brainstem_mask_dce=gm_brainstem_mask_dce,
            gm_cerebellum_mask_t2=gm_cerebellum_mask_t2,
            gm_cerebellum_mask_dce=gm_cerebellum_mask_dce,
            wm_cerebellum_mask_t2=wm_cerebellum_mask_t2,
            wm_cerebellum_mask_dce=wm_cerebellum_mask_dce,
            wm_cc_mask_t2=wm_cc_mask_t2,
            wm_cc_mask_dce=wm_cc_mask_dce,
            flip_angle_deg=flip_angle_deg,
        )

    print("[tissue] Tissue analysis completed.")
