import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np

import utils.settings as settings
from utils.loading import build_time_points_s, resolve_dce_time_step_s, load_dce_4d


def _load_roi_voxels_by_slice(roi_dir: Path) -> dict[int, np.ndarray]:
    voxels_by_slice: dict[int, np.ndarray] = {}
    if not roi_dir.exists() or not roi_dir.is_dir():
        return voxels_by_slice

    for p in sorted(roi_dir.glob("ROI_voxels_slice_*.npy")):
        if not p.is_file() or p.name.startswith("._"):
            continue
        m = re.search(r"ROI_voxels_slice_(\d+)\.npy$", p.name)
        if not m:
            continue
        z = int(m.group(1)) - 1
        if z < 0:
            continue
        try:
            arr = np.asarray(np.load(str(p)))
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        vox = arr[:, :2].astype(int, copy=False)
        if vox.size == 0:
            continue
        voxels_by_slice[int(z)] = vox

    return voxels_by_slice


def _centers_from_voxels_by_slice(voxels_by_slice: dict[int, np.ndarray]) -> list[tuple[int, int, int, float]]:
    centers: list[tuple[int, int, int, float]] = []
    for z, vox in sorted(voxels_by_slice.items()):
        if vox is None or np.asarray(vox).size == 0:
            continue
        v = np.asarray(vox)
        cx = int(round(float(np.mean(v[:, 0]))))
        cy = int(round(float(np.mean(v[:, 1]))))
        centers.append((cx, cy, int(z), 1.0))
    return centers


def input_function_file(analysis_directory, nifti_directory, image_directory, filenames, parameters):
    """Load user-provided ROI voxel lists and produce standard p-brain ROI/ITC/CTC artifacts.

    Expected inputs (written by p-brain-web or external tooling):
      Analysis/ROI Data/<type>/<subtype>/ROI_voxels_slice_<N>.npy

    This function regenerates the conventional artifacts under:
      Analysis/ITC Data, Analysis/CTC Data, Analysis/Frame Data, Analysis/ROI NIfTI
    so downstream stages (TSCC + modelling) can run unchanged.
    """

    (
        _t1_3D_filename,
        _axial_t1_3D_filename,
        _t2_3D_filename,
        _axial_t2_3D_filename,
        _flair_3D_filename,
        _axial_flair_3D_filename,
        _axial_t2_2D_filename,
        _diffusion_filename,
        dce_filename,
    ) = filenames

    is_vfa, _is_ir, _apple_metal, _boundary, *_rest = parameters

    dce_path = os.path.join(nifti_directory, dce_filename)
    ref_img, dce4d = load_dce_4d(dce_path, prefer_complex_mag=True, dtype=np.float32)
    if dce4d.ndim != 4:
        raise ValueError(f"Expected DCE to be 4D (x,y,z,t); got shape {dce4d.shape} from {dce_path}")

    # Resolve time axis (prefer existing time_points_s.npy if present).
    time_path = os.path.join(analysis_directory, "Fitting", "time_points_s.npy")
    time_points_s = None
    if os.path.isfile(time_path):
        try:
            time_points_s = np.asarray(np.load(time_path), dtype=float).reshape(-1)
        except Exception:
            time_points_s = None

    if time_points_s is None or time_points_s.size < 3 or not np.all(np.isfinite(time_points_s)):
        n_volumes = int(dce4d.shape[-1])
        dt_s = resolve_dce_time_step_s(dce_path, default=None)
        time_points_s = build_time_points_s(n_volumes, dt_s)
        os.makedirs(os.path.join(analysis_directory, "Fitting"), exist_ok=True)
        np.save(time_path, time_points_s)

    # Load ROI voxel lists.
    roi_root = Path(analysis_directory) / "ROI Data"
    if not roi_root.exists():
        raise FileNotFoundError(f"Missing ROI Data directory: {roi_root}")

    # Implementation uses the same writer as the deterministic geometry pipeline.
    from modules.geometry_input_functions import _save_roi_outputs  # noqa: PLC0415

    wrote_any = False

    for roi_type_dir in sorted([d for d in roi_root.iterdir() if d.is_dir()]):
        roi_type = roi_type_dir.name
        for subtype_dir in sorted([d for d in roi_type_dir.iterdir() if d.is_dir()]):
            roi_subtype = subtype_dir.name
            voxels_by_slice = _load_roi_voxels_by_slice(subtype_dir)
            if not voxels_by_slice:
                continue

            centers = _centers_from_voxels_by_slice(voxels_by_slice)
            if not centers:
                continue

            _save_roi_outputs(
                roi_type=str(roi_type),
                roi_subtype=str(roi_subtype),
                centers=centers,
                voxels_by_slice=voxels_by_slice,
                dce4d=dce4d,
                ref_img=ref_img,
                analysis_dir=str(analysis_directory),
                image_dir=str(image_directory),
                nifti_dir=str(nifti_directory),
                time_points_s=np.asarray(time_points_s, dtype=float),
                filenames=filenames,
                is_vfa=bool(is_vfa),
            )
            wrote_any = True

    if not wrote_any:
        raise FileNotFoundError(
            f"No ROI_voxels_slice_*.npy found under {roi_root}. "
            "Expected Analysis/ROI Data/<type>/<subtype>/ROI_voxels_slice_<N>.npy"
        )

    # Keep return value compatible with other input-function modules.
    return {
        "roi_method": "file",
        "roi_root": str(roi_root),
        "vascular_curve_method": getattr(settings, "VASCULAR_ROI_CURVE_METHOD", "max"),
    }
