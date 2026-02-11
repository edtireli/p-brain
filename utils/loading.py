import pickle
import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np
import json
import importlib
import time
from typing import Optional

import utils.settings as settings
from utils.qc import detect_truncated_bolus

try:  # Optional dependency (required for PAR/REC metadata enrichment)
    import nibabel as nib  # type: ignore
except Exception:  # pragma: no cover
    nib = None


def _nifti_peer_path(nifti_path: str, peer_suffix: str) -> str:
    """Return a peer NIfTI path like '<base><peer_suffix>.nii[.gz]'."""

    if not nifti_path:
        return ""
    if nifti_path.endswith(".nii.gz"):
        return nifti_path[:-7] + f"{peer_suffix}.nii.gz"
    if nifti_path.endswith(".nii"):
        return nifti_path[:-4] + f"{peer_suffix}.nii"
    base, ext = os.path.splitext(nifti_path)
    return base + peer_suffix + ext


def load_nifti_scaled_data(path: str, *, dtype=np.float32) -> np.ndarray:
    """Load a NIfTI volume with nibabel's proxy scaling applied.

    Notes:
    - Many Philips NIfTIs (dcm2niix) store scaling in the nibabel ArrayProxy
      (slope/inter) even when header scl_slope/scl_inter are NaN.
    - Using `np.asarray(img.dataobj, ...)` ensures those proxy slope/inter are applied.
    """

    if nib is None:
        raise ImportError("Loading NIfTI requires 'nibabel'.")
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=dtype)


def load_dce_4d(
    dce_path: str,
    *,
    prefer_complex_mag: bool = True,
    dtype=np.float32,
) -> tuple["nib.Nifti1Image", np.ndarray]:
    """Load a DCE 4D series in consistent (scaled) signal units.

    Strategy:
    - If `<base>_real.nii*` and `<base>_imaginary.nii*` exist and `prefer_complex_mag`
      is True, return `sqrt(real^2 + imag^2)` (complex magnitude).
    - Otherwise return the magnitude series from `dce_path`.

    Returns:
    - (reference_img, data_4d) where `reference_img` is the magnitude NIfTI loaded
      from `dce_path` (used for affine/shape checks by callers).
    """

    if nib is None:
        raise ImportError("Loading DCE NIfTI requires 'nibabel'.")

    dce_img = nib.load(dce_path)
    dce_mag = np.asarray(dce_img.dataobj, dtype=dtype)

    if not prefer_complex_mag:
        return dce_img, dce_mag

    real_path = _nifti_peer_path(dce_path, "_real")
    imag_path = _nifti_peer_path(dce_path, "_imaginary")
    if not (os.path.exists(real_path) and os.path.exists(imag_path)):
        return dce_img, dce_mag

    try:
        real_img = nib.load(real_path)
        imag_img = nib.load(imag_path)

        # If the converters stored different proxy scaling for real/imag than for magnitude,
        # computing complex magnitude in scaled space will not match the magnitude units.
        mag_slope = getattr(dce_img.dataobj, "slope", None)
        real_slope = getattr(real_img.dataobj, "slope", None)
        imag_slope = getattr(imag_img.dataobj, "slope", None)
        if (
            mag_slope is None
            or real_slope is None
            or imag_slope is None
            or not np.isfinite(float(mag_slope))
            or not np.isfinite(float(real_slope))
            or not np.isfinite(float(imag_slope))
            or abs(float(real_slope) - float(mag_slope)) > 1e-6
            or abs(float(imag_slope) - float(mag_slope)) > 1e-6
        ):
            return dce_img, dce_mag

        real = np.asarray(real_img.dataobj, dtype=dtype)
        imag = np.asarray(imag_img.dataobj, dtype=dtype)
        if real.shape != imag.shape or real.shape != dce_mag.shape:
            return dce_img, dce_mag

        cmag = np.sqrt(real * real + imag * imag, dtype=dtype)
        return dce_img, cmag
    except Exception:
        return dce_img, dce_mag


def load_roi_voxels_from_mat(mat_path: str):
    """Load a 2D ROI mask + slice index from a .mat file and return (z, vox_xy).

    Returns:
    - z: 0-based slice index
    - vox_xy: (N,2) array of (x,y) voxel indices in the NIfTI frame
    """

    import scipy.io as sio

    d = sio.loadmat(mat_path, squeeze_me=True)

    mask_key = getattr(settings, "ROI_MASK_KEY", "BW_input_big")
    m = d.get(mask_key, None)
    if m is None:
        # Fallback to common alternate key.
        m = d.get("BW_input", None)
    if m is None:
        raise KeyError(f"ROI .mat file missing mask key '{mask_key}' (or BW_input): {mat_path}")

    mask = np.asarray(m)
    if mask.ndim != 2:
        raise ValueError(f"ROI mask must be 2D, got shape {mask.shape} in {mat_path}")

    # 1-based slice index in exported artifacts.
    slice_1based = d.get("slice_c_input", None)
    if slice_1based is None:
        slice_1based = d.get("slice_input", None)
    if slice_1based is None:
        raise KeyError(f"ROI .mat file missing slice index (slice_c_input): {mat_path}")
    z = int(np.asarray(slice_1based).squeeze()) - 1
    if z < 0:
        z = 0

    rc = np.argwhere(mask != 0)
    if rc.size == 0:
        return z, np.zeros((0, 2), dtype=np.int16)

    r = rc[:, 0].astype(np.int64)
    c = rc[:, 1].astype(np.int64)

    xform = getattr(settings, "ROI_XFORM", "r90_inv")
    if str(xform).strip().lower() == "identity":
        x = r
        y = c
    else:
        # r90_inv: (r,c) -> (c, H-1-r)
        h = int(mask.shape[0])
        x = c
        y = (h - 1) - r

    vox = np.stack([x, y], axis=1).astype(np.int16, copy=False)
    return z, vox


def _nifti_sidecar_json_path(nifti_path: str) -> str:
    """Return the expected dcm2niix JSON sidecar path for a NIfTI file."""

    if nifti_path.endswith('.nii.gz'):
        return nifti_path[:-7] + '.json'
    if nifti_path.endswith('.nii'):
        return nifti_path[:-4] + '.json'
    base, _ext = os.path.splitext(nifti_path)
    return base + '.json'


def read_nifti_sidecar_json(nifti_path: str):
    """Load the JSON sidecar produced by dcm2niix for a NIfTI file.

    Returns the parsed dict, or None when the sidecar is missing/unreadable.
    """

    if not nifti_path:
        return None
    json_path = _nifti_sidecar_json_path(nifti_path)
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def read_flip_angle_deg_from_sidecar(nifti_path: str, default=None):
    """Return excitation flip angle (degrees) from the NIfTI JSON sidecar."""

    data = read_nifti_sidecar_json(nifti_path)
    if not isinstance(data, dict):
        return default
    for key in (
        'FlipAngle',
        'FlipAngleDeg',
        'FlipAngleDegrees',
        'FlipAngle_deg',
        'FlipAngle(deg)',
    ):
        if key not in data:
            continue
        try:
            return float(data[key])
        except (TypeError, ValueError):
            return default
    return default


def resolve_flip_angle_deg(nifti_path: str, default=None):
    """Resolve flip angle (degrees) from config override or metadata.

    Resolution order:
    1) `P_BRAIN_FLIP_ANGLE=<number>` override.
    2) NIfTI JSON sidecar `FlipAngle` (or equivalent keys).
    3) If still missing: attempt PAR/REC metadata enrichment (FlipAngle from `.PAR`).
    """

    def _strip_nii_ext(name: str) -> str:
        lower = name.lower()
        if lower.endswith(".nii.gz"):
            return name[:-7]
        if lower.endswith(".nii"):
            return name[:-4]
        return name

    def _guess_par_for_nifti(nifti_path_: str) -> str | None:
        if not nifti_path_:
            return None
        try:
            p = Path(nifti_path_)
        except Exception:
            return None

        nifti_base = _strip_nii_ext(p.name)
        target = sanitize_dcm2niix_basename(nifti_base)

        search_dirs: list[Path] = []
        try:
            search_dirs.append(p.parent)
        except Exception:
            pass
        try:
            search_dirs.append(p.parent.parent)
        except Exception:
            pass

        candidates: list[str] = []
        for d in search_dirs:
            try:
                if not d or not d.exists() or not d.is_dir():
                    continue
            except Exception:
                continue
            try:
                for par in list(d.glob("*.PAR")) + list(d.glob("*.par")):
                    candidates.append(str(par))
            except Exception:
                continue

        if not candidates:
            return None

        # Prefer a PAR whose protocol name sanitizes to the NIfTI basename.
        for par_path in candidates:
            proto = read_protocol_name_from_par(par_path, default=None)
            if not proto:
                continue
            if sanitize_dcm2niix_basename(str(proto)) == target:
                return par_path

        # Otherwise, if the NIfTI basename itself matches the PAR basename, use it.
        for par_path in candidates:
            try:
                par_base = _strip_nii_ext(Path(par_path).name)
            except Exception:
                continue
            if sanitize_dcm2niix_basename(par_base) == target:
                return par_path

        # Last resort: only safe when there's exactly one PAR around.
        if len(candidates) == 1:
            return candidates[0]
        return None

    override = getattr(settings, "FLIP_ANGLE_DEG", None)
    if override is not None:
        try:
            return float(override)
        except (TypeError, ValueError):
            pass
    v = read_flip_angle_deg_from_sidecar(nifti_path, default=None)
    if v is not None:
        return v

    # Optional PAR/REC enrichment path (useful when JSON sidecars are missing FlipAngle).
    par_path = (os.environ.get("P_BRAIN_DCE_PAR") or os.environ.get("P_BRAIN_PAR_PATH") or "").strip()
    if not par_path:
        par_path = _guess_par_for_nifti(nifti_path) or ""
    if par_path:
        v_par = read_flip_angle_deg_from_par(par_path, default=None)
        if v_par is not None:
            try:
                inject_parrec_metadata_into_sidecar_json(nifti_path, par_path)
            except Exception:
                pass
            return v_par

    if bool(getattr(settings, "STRICT_METADATA", False)) and default is None:
        raise ValueError(
            "Missing FlipAngle metadata (and no P_BRAIN_FLIP_ANGLE override). "
            "Either add FlipAngle to the JSON sidecar or set P_BRAIN_FLIP_ANGLE=<deg>."
        )
    return default


def read_repetition_time_s_from_sidecar(nifti_path: str, default=None):
    """Return repetition time (seconds) from the NIfTI JSON sidecar."""

    data = read_nifti_sidecar_json(nifti_path)
    if not isinstance(data, dict):
        return default
    for key in (
        "RepetitionTimeExcitation",
        "RepetitionTime",
        "RepetitionTime_s",
        "RepetitionTimeSeconds",
    ):
        if key not in data:
            continue
        try:
            value = float(data[key])
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return default


def read_inversion_time_s_from_sidecar(nifti_path: str, default=None):
    """Return inversion time / TI (seconds) from the NIfTI JSON sidecar.

    dcm2niix may store TI as seconds (typical) but some pipelines store
    milliseconds. When the value looks like milliseconds (>10), it is
    converted to seconds.
    """

    data = read_nifti_sidecar_json(nifti_path)
    if not isinstance(data, dict):
        return default
    for key in (
        "InversionTime",
        "InversionTime_s",
        "InversionTimeSeconds",
        "TI",
        "TI_s",
        "TI_dyn",
        "TI_dyn_s",
        "TI_dynamic",
        "TI_dynamic_s",
    ):
        if key not in data:
            continue
        try:
            v = float(data[key])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v) or v <= 0:
            continue
        # Heuristic: values >10 are almost certainly milliseconds.
        # In strict-metadata mode, do not guess units.
        if v > 10:
            if bool(getattr(settings, "STRICT_METADATA", False)):
                continue
            v = v * 1e-3
        return v
    return default


def resolve_turboflash_ti_s(nifti_path: str, default=0.12):
    """Resolve TurboFLASH TI during bolus passage (seconds).

    MATLAB menu 5 case 12 uses a user-entered TI_dyn (default 0.12s).
    In p-brain we keep it automatic:
    1) `P_BRAIN_TURBOFLASH_TI_S` / `P_BRAIN_TURBOFLASH_TI_MS` override.
    2) JSON sidecar TI keys (see `read_inversion_time_s_from_sidecar`).
    3) Provided default (0.12s).
    """

    raw_s = (os.environ.get("P_BRAIN_TURBOFLASH_TI_S") or os.environ.get("P_BRAIN_TI_DYN_S") or "").strip()
    raw_ms = (os.environ.get("P_BRAIN_TURBOFLASH_TI_MS") or os.environ.get("P_BRAIN_TI_DYN_MS") or "").strip()
    try:
        if raw_s:
            v = float(raw_s)
            if np.isfinite(v) and v > 0:
                return v
        if raw_ms:
            v = float(raw_ms)
            if np.isfinite(v) and v > 0:
                return v * 1e-3
    except Exception:
        pass

    v = read_inversion_time_s_from_sidecar(nifti_path, default=None)
    if v is not None:
        try:
            v = float(v)
            if np.isfinite(v) and v > 0:
                return v
        except Exception:
            pass

    if bool(getattr(settings, "STRICT_METADATA", False)) and default is None:
        raise ValueError(
            "Missing TurboFLASH TI metadata (and no override). "
            "Either add InversionTime/TI to the JSON sidecar (in seconds) or set P_BRAIN_TURBOFLASH_TI_DYN_S."
        )

    try:
        v = float(default)
        return v if np.isfinite(v) and v > 0 else 0.12
    except Exception:
        return 0.12


def _infer_n_slices_from_sidecar_or_nifti(nifti_path: str, sidecar: dict | None = None) -> int | None:
    """Best-effort inference of slice count for a NIfTI series."""

    if isinstance(sidecar, dict):
        st = sidecar.get("SliceTiming")
        if isinstance(st, (list, tuple)) and len(st) > 0:
            return int(len(st))

    try:
        import nibabel as nib

        img = nib.load(nifti_path)
        shape = getattr(img, "shape", None)
        if shape and len(shape) >= 3:
            n = int(shape[2])
            return n if n > 0 else None
    except Exception:
        pass

    return None


def resolve_turboflash_tr_s(nifti_path: str, default=None, *, mode: str = "excitation"):
    """Resolve TurboFLASH TR (seconds).

    Prefer:
    1) `settings.TURBOFLASH_TR_S` override.
    2) JSON sidecar `RepetitionTimeExcitation`.
    3) JSON sidecar `RepetitionTime` *only* when it looks like an excitation TR
       (i.e., small; not a per-volume DCE sampling interval).

    Parameters
    - mode:
        - "excitation": return excitation TR (time between RF pulses).
        - "slice": return effective per-slice TR (excitation TR * n_slices).
          This matches MATLAB-style TurboFLASH DCE conversion on multi-slice 2D
          data, where each slice is excited once every n_slices RF pulses.
    """

    override = getattr(settings, "TURBOFLASH_TR_S", None)
    if override is not None:
        try:
            v = float(override)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass

    data = read_nifti_sidecar_json(nifti_path)
    if isinstance(data, dict):
        try:
            v = float(data.get("RepetitionTimeExcitation"))
            if v > 0:
                tr = v
                if str(mode).strip().lower() == "slice":
                    n_slices = _infer_n_slices_from_sidecar_or_nifti(nifti_path, data)
                    if n_slices and n_slices > 1:
                        tr = tr * float(n_slices)
                return tr
        except Exception:
            pass

        # In strict-metadata mode, do not fall back to `RepetitionTime`.
        if bool(getattr(settings, "STRICT_METADATA", False)):
            return default

        # dcm2niix typically writes `RepetitionTime` as seconds, but for DCE it
        # is often the *volume* sampling interval (seconds per dynamic). Guard
        # against misusing that as excitation TR.
        try:
            v = float(data.get("RepetitionTime"))
            if v > 0 and v <= 0.1:
                tr = v
                if str(mode).strip().lower() == "slice":
                    n_slices = _infer_n_slices_from_sidecar_or_nifti(nifti_path, data)
                    if n_slices and n_slices > 1:
                        tr = tr * float(n_slices)
                return tr
        except Exception:
            pass

    return default


def resolve_turboflash_nph(nifti_path: str, default=None):
    """Resolve TurboFLASH nph (turbo factor / ky=0 line index).

    Prefer:
    1) `settings.TURBOFLASH_NPH` override (env: P_BRAIN_TURBOFLASH_NPH / P_BRAIN_TURBO_NPH).
    2) JSON sidecar fields when available.
    3) Provided default (fallback to 1).
    """

    override = getattr(settings, "TURBOFLASH_NPH", None)
    if override is not None:
        try:
            v = int(float(override))
            if v > 0:
                return v
        except Exception:
            pass

    data = read_nifti_sidecar_json(nifti_path)
    if isinstance(data, dict):
        # Prefer explicit ky=0 index fields when present.
        for key in ("nph", "NPH"):
            if key not in data:
                continue
            try:
                v = int(float(data.get(key)))
                if v > 0:
                    return v
            except Exception:
                continue

    try:
        v = int(float(default))
        return v if v > 0 else 1
    except Exception:
        return 1


def resolve_dce_time_step_s(nifti_path: str, default=None):
    """Resolve DCE sampling interval (seconds per volume).

    Compatibility shim for older modules that import this from utils.loading.

    Resolution order:
    1) `P_BRAIN_DCE_TIME_STEP_S` / `settings.DCE_TIME_STEP_S` override.
    2) NIfTI JSON sidecar fields (common dcm2niix keys).
    3) NIfTI header pixdim[4] with unit conversion.
    """

    override = getattr(settings, "DCE_TIME_STEP_S", None)
    if override is not None:
        try:
            v = float(override)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass

    data = read_nifti_sidecar_json(nifti_path)
    if isinstance(data, dict):
        for key in (
            "RepetitionTime",
            "RepetitionTime_s",
            "TemporalResolution",
            "TemporalResolution_s",
            "VolumeTiming",
            "dt",
            "dt_s",
        ):
            if key not in data:
                continue
            try:
                raw = data[key]
                if isinstance(raw, (list, tuple)) and raw:
                    # Some tools store per-volume timings; infer dt as median diff.
                    arr = np.asarray(raw, dtype=float)
                    if arr.size >= 2:
                        diffs = np.diff(arr)
                        dt = float(np.median(diffs[np.isfinite(diffs)]))
                    else:
                        dt = float(arr[0])
                else:
                    dt = float(raw)
            except Exception:
                continue
            if dt > 0:
                return dt

    # Header fallback.
    try:
        import nibabel as nib

        img = nib.load(nifti_path)
        hdr = img.header
        dt = float(hdr.get_zooms()[3]) if len(hdr.get_zooms()) >= 4 else None
        if dt is not None and dt > 0:
            try:
                _xyz, t_unit = hdr.get_xyzt_units()
            except Exception:
                t_unit = "sec"
            if t_unit == "msec":
                dt = dt / 1000.0
            elif t_unit == "usec":
                dt = dt / 1_000_000.0
            return dt
    except Exception:
        pass

    return default


def _load_matlab_mat_dict(mat_path: str) -> dict:
    """Load a MATLAB .mat file into a python dict.

    Supports classic v5/v7.2 MAT files via scipy.io.loadmat, and v7.3 MAT
    files (HDF5) via h5py.
    """

    mat_path = str(mat_path)
    try:
        import scipy.io as sio  # type: ignore

        return sio.loadmat(mat_path, squeeze_me=True)
    except Exception as e:
        # v7.3 commonly raises NotImplementedError in scipy.
        try:
            import h5py  # type: ignore
            import numpy as _np

            out: dict = {}
            with h5py.File(mat_path, "r") as f:
                # Flatten only datasets at the root; callers can extend as needed.
                for k in f.keys():
                    obj = f.get(k)
                    if obj is None:
                        continue
                    if isinstance(obj, h5py.Dataset):
                        arr = _np.array(obj)
                        # MATLAB stores column vectors as (N,1) often.
                        out[k] = arr.squeeze()
            return out
        except Exception:
            raise e


def load_matlab_roi_voxels(
    mat_path: str,
    *,
    mask_key: str | None = None,
    xform: str | None = None,
) -> tuple[int, np.ndarray]:
    """Load a 2D MATLAB ROI mask and return (slice_index_0based, voxels[N,2]).

    The returned voxels are (x,y) indices suitable for indexing p-brain DCE
    arrays as data[x, y, z, t].

    Expected MATLAB fields:
      - slice_c_input (1-based)
      - BW_input or BW_input_big (uint8 mask)
    """

    d = _load_matlab_mat_dict(mat_path)
    mask_key = (mask_key or getattr(settings, "MATLAB_ROI_MASK_KEY", "BW_input_big") or "BW_input_big").strip()
    xform = (xform or getattr(settings, "MATLAB_ROI_XFORM", "matlab_to_nifti_r90_inv") or "matlab_to_nifti_r90_inv").strip().lower()

    if "slice_c_input" not in d:
        raise KeyError(f"{mat_path}: missing slice_c_input")
    slice_index = int(np.asarray(d["slice_c_input"]).squeeze()) - 1
    if slice_index < 0:
        raise ValueError(f"{mat_path}: invalid slice_c_input={d['slice_c_input']}")

    if mask_key not in d:
        # Allow fallback between BW_input and BW_input_big.
        fallback = "BW_input" if mask_key != "BW_input" else "BW_input_big"
        if fallback in d:
            mask_key = fallback
        else:
            raise KeyError(f"{mat_path}: missing mask key {mask_key} (and {fallback})")

    mask = np.asarray(d[mask_key])
    if mask.ndim != 2:
        raise ValueError(f"{mat_path}: {mask_key} expected 2D mask, got shape {mask.shape}")
    mask_bool = mask.astype(bool)
    coords = np.argwhere(mask_bool)
    if coords.size == 0:
        return slice_index, np.zeros((0, 2), dtype=np.int16)

    # coords are (row, col) in MATLAB mask frame.
    if xform == "identity":
        vox = coords
    else:
        # matlab_to_nifti_r90_inv: (r,c) -> (c, N-1-r)
        n = int(mask.shape[0])
        r = coords[:, 0].astype(np.int64)
        c = coords[:, 1].astype(np.int64)
        x = c
        y = (n - 1) - r
        vox = np.stack([x, y], axis=1)

    return slice_index, vox.astype(np.int16, copy=False)


def build_time_points_s(n_volumes: int, dt_s, *, t0_s: float = 0.0):
    """Build time axis (seconds) for a DCE series.

    Compatibility shim for older modules that import this from utils.loading.
    """

    try:
        n = int(n_volumes)
    except Exception:
        n = 0
    if n <= 0:
        return np.asarray([], dtype=np.float32)

    try:
        dt = float(dt_s)
    except Exception:
        dt = 0.0
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0

    try:
        t0 = float(t0_s)
    except Exception:
        t0 = 0.0
    if not np.isfinite(t0):
        t0 = 0.0

    return (t0 + np.arange(n, dtype=np.float32) * np.float32(dt)).astype(np.float32)


def discover_vfa_series(
    nifti_directory: str,
    patterns: Optional[str] = None,
    *,
    min_series: int = 2,
):
    """Discover VFA spoiled GRE series within ``nifti_directory``.

    Returns a list of dicts: {"nifti": path, "flip_angle_deg": float, "tr_s": float}
    sorted by flip angle.

    Patterns are glob expressions (comma-separated). When omitted, uses
    ``settings.VFA_FILE_GLOB``.
    """

    if not nifti_directory or not os.path.isdir(nifti_directory):
        return []

    raw_patterns = patterns if patterns is not None else getattr(settings, "VFA_FILE_GLOB", "*VFA*.nii*")
    pat_list = [p.strip() for p in str(raw_patterns).split(",") if p.strip()]
    if not pat_list:
        pat_list = ["*VFA*.nii*"]

    candidates: set[str] = set()
    for pat in pat_list:
        for match in glob.glob(os.path.join(nifti_directory, pat)):
            if os.path.isfile(match):
                candidates.add(match)

    series = []
    for nifti_path in sorted(candidates):
        fa = read_flip_angle_deg_from_sidecar(nifti_path, default=None)
        tr = read_repetition_time_s_from_sidecar(nifti_path, default=None)
        if fa is None or tr is None:
            continue
        series.append({"nifti": nifti_path, "flip_angle_deg": float(fa), "tr_s": float(tr)})

    series.sort(key=lambda item: item.get("flip_angle_deg", 0.0))
    if len(series) < int(min_series):
        return []
    return series


def discover_ir_series(nifti_directory: str):
    """Discover inversion-recovery NIfTIs expected by the released pipeline.

    Returns a list of 7 NIfTI paths (one per TI) when complete; otherwise []
    so callers can fall back to VFA or "none".

    Overrides:
    - ``P_BRAIN_IR_TI``: comma-separated TI values (e.g. "00120,00300,...").
    - ``P_BRAIN_IR_PREFIXES``: comma-separated filename prefixes (default: "WIPTI_,WIPDelRec-TI_").
    """

    if not nifti_directory or not os.path.isdir(nifti_directory):
        return []

    raw_ti = (os.environ.get("P_BRAIN_IR_TI") or "").strip()
    if raw_ti:
        TI = [t.strip() for t in raw_ti.split(",") if t.strip()]
    else:
        TI = ["00120", "00300", "00600", "01000", "02000", "04000", "10000"]

    raw_prefixes = (os.environ.get("P_BRAIN_IR_PREFIXES") or "").strip()
    if raw_prefixes:
        patterns = [p.strip() for p in raw_prefixes.split(",") if p.strip()]
    else:
        patterns = ["WIPTI_", "WIPDelRec-TI_"]

    if not TI or not patterns:
        return []
    paths = []
    for ti in TI:
        hit = None
        for suf in (".nii", ".nii.gz"):
            hit = first_existing_file(nifti_directory, patterns, ti, suf)
            if hit:
                break
        if not hit:
            return []
        paths.append(hit)
    return paths


def _sanitize_dcm2niix_basename(name: str) -> str:
    """Approximate dcm2niix filename sanitisation for matching outputs."""

    if name is None:
        return ""
    text = str(name).strip()
    # Replace whitespace with underscores, drop obvious path separators.
    text = re.sub(r"\s+", "_", text)
    text = text.replace(os.sep, "_")
    # Keep alphanumerics, underscore, dash.
    text = re.sub(r"[^0-9A-Za-z_\-]+", "", text)
    return text


sanitize_dcm2niix_basename = _sanitize_dcm2niix_basename


def read_protocol_name_from_par(par_path: str, default=None):
    """Extract protocol name from a Philips PAR header (used by dcm2niix %p)."""

    if not par_path or not os.path.exists(par_path):
        return default
    try:
        with open(par_path, "r", errors="ignore") as f:
            for line in f:
                if "protocol name" in line.lower():
                    # Typical: "Protocol name                         : WIPhperf120long"
                    if ":" in line:
                        value = line.split(":", 1)[1].strip()
                        return value or default
        return default
    except OSError:
        return default


def read_flip_angle_deg_from_par(par_path: str, default=None):
    """Extract excitation flip angle (degrees) from a Philips PAR file."""

    if not par_path or not os.path.exists(par_path):
        return default

    try:
        with open(par_path, "r", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return default

    col_index_1based = None
    # Variant A: numbered column definitions.
    for line in lines:
        if not line.lstrip().startswith("#"):
            continue
        lower = line.lower()
        if "image_flip_angle" not in lower:
            continue
        # Example: "#  10. image_flip_angle (in degrees)        (float)"
        m = re.search(r"#\s*(\d+)\s*\.\s*image_flip_angle", lower)
        if m:
            try:
                col_index_1based = int(m.group(1))
                break
            except ValueError:
                col_index_1based = None

    # Variant B: unnumbered definitions; derive the column index by walking the
    # IMAGE INFORMATION DEFINITION list and summing field widths (e.g. 3*float).
    if not col_index_1based:
        in_def = False
        col = 1
        for line in lines:
            lower = line.lower()
            if line.lstrip().startswith("#") and "=== image information" in lower and "definition" in lower:
                in_def = True
                col = 1
                continue
            if line.lstrip().startswith("#") and "=== image information" in lower and "definition" not in lower:
                # reached data section
                break
            if not in_def:
                continue
            if not line.lstrip().startswith("#"):
                continue

            # Skip explanatory/comment-only lines that are not field definitions.
            # Field definitions include a trailing type spec like (integer)/(float)/(string)
            # optionally with multiplicity (e.g. 3*float).
            if not re.search(r"\((?:\s*\d+\s*\*)?\s*(?:integer|float|string)\s*\)", lower):
                continue

            # Extract multiplicity from the final (...) type spec.
            mult = 1
            m_mult = re.search(r"\((\s*\d+)\s*\*", line)
            if m_mult:
                try:
                    mult = int(m_mult.group(1))
                except Exception:
                    mult = 1
            if "image_flip_angle" in lower and col_index_1based is None:
                col_index_1based = col
            col += max(1, mult)

    if not col_index_1based:
        return default

    # Find IMAGE INFORMATION section (not DEFINITION) and parse data lines.
    in_image_info = False
    values = []
    for line in lines:
        lower = line.lower()
        if line.lstrip().startswith("#") and "=== image information" in lower and "definition" not in lower:
            in_image_info = True
            continue
        if not in_image_info:
            continue
        if line.lstrip().startswith("#"):
            continue
        tokens = line.strip().split()
        if len(tokens) < col_index_1based:
            continue
        try:
            values.append(float(tokens[col_index_1based - 1]))
        except ValueError:
            continue

    if not values:
        return default
    try:
        return float(np.median(np.asarray(values, dtype=float)))
    except Exception:
        return default


def read_repetition_time_excitation_s_from_par(par_path: str, default=None):
    """Extract excitation repetition time (seconds) from a Philips PAR header."""

    if not par_path or not os.path.exists(par_path):
        return default
    try:
        with open(par_path, "r", errors="ignore") as f:
            for line in f:
                lower = line.lower()
                if "repetition time" not in lower:
                    continue
                # Typical:
                #   "Repetition time [ms]           : 3.997"
                m = re.search(r"repetition\s+time\s*\[ms\]\s*:\s*([0-9.+\-eE]+)", lower)
                if not m:
                    continue
                try:
                    ms = float(m.group(1))
                except ValueError:
                    continue
                if ms > 0:
                    return ms * 1e-3
        return default
    except OSError:
        return default


def read_turbo_factor_from_par(par_path: str, default=None):
    """Extract Turbo factor (a.k.a. TFE/Turbo factor) from a Philips PAR file.

    This is used as TurboFLASH `nph` (lines at ky=0 within the readout train)
    in the MATLAB pipeline.
    """

    if not par_path or not os.path.exists(par_path):
        return default

    try:
        with open(par_path, "r", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return default

    col_index_1based = None

    # Variant A: numbered column definitions.
    for line in lines:
        if not line.lstrip().startswith("#"):
            continue
        lower = line.lower()
        if "turbo factor" not in lower:
            continue
        # Example: "#  31. TURBO factor  <0=no turbo>               (integer)"
        m = re.search(r"#\s*(\d+)\s*\.\s*turbo\s+factor", lower)
        if m:
            try:
                col_index_1based = int(m.group(1))
                break
            except ValueError:
                col_index_1based = None

    # Variant B: unnumbered definitions (walk IMAGE INFORMATION DEFINITION).
    if not col_index_1based:
        in_def = False
        col = 1
        for line in lines:
            lower = line.lower()
            if line.lstrip().startswith("#") and "=== image information" in lower and "definition" in lower:
                in_def = True
                col = 1
                continue
            if line.lstrip().startswith("#") and "=== image information" in lower and "definition" not in lower:
                break
            if not in_def:
                continue
            if not line.lstrip().startswith("#"):
                continue
            if not re.search(r"\((?:\s*\d+\s*\*)?\s*(?:integer|float|string)\s*\)", lower):
                continue

            mult = 1
            m_mult = re.search(r"\((\s*\d+)\s*\*", line)
            if m_mult:
                try:
                    mult = int(m_mult.group(1))
                except Exception:
                    mult = 1

            if "turbo factor" in lower and col_index_1based is None:
                col_index_1based = col
            col += max(1, mult)

    if not col_index_1based:
        return default

    in_image_info = False
    values = []
    for line in lines:
        lower = line.lower()
        if line.lstrip().startswith("#") and "=== image information" in lower and "definition" not in lower:
            in_image_info = True
            continue
        if not in_image_info:
            continue
        if line.lstrip().startswith("#"):
            continue
        tokens = line.strip().split()
        if len(tokens) < col_index_1based:
            continue
        try:
            values.append(float(tokens[col_index_1based - 1]))
        except ValueError:
            continue

    if not values:
        return default
    try:
        v = float(np.median(np.asarray(values, dtype=float)))
    except Exception:
        return default
    if not np.isfinite(v) or v <= 0:
        return default
    try:
        return int(round(v))
    except Exception:
        return default


def inject_repetition_time_excitation_into_sidecar_json(nifti_path: str, tr_s: float) -> bool:
    """Ensure NIfTI JSON sidecar contains repetition time metadata.

    Historically p-brain stored the excitation TR under `RepetitionTimeExcitation`.
    The Hemisure validator (and dcm2niix/BIDS conventions) use `RepetitionTime`.

    This function now ensures BOTH keys exist (seconds) when missing.
    Returns True when the file was written/updated.
    """

    if not nifti_path:
        return False
    json_path = _nifti_sidecar_json_path(nifti_path)
    if not os.path.exists(json_path):
        return False
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    changed = False
    try:
        v = float(tr_s)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(v) or v <= 0:
        return False

    if "RepetitionTimeExcitation" not in data:
        data["RepetitionTimeExcitation"] = v
        changed = True
    if "RepetitionTime" not in data:
        data["RepetitionTime"] = v
        changed = True

    if not changed:
        return False
    try:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        return True
    except OSError:
        return False


def inject_flip_angle_into_sidecar_json(nifti_path: str, flip_angle_deg: float) -> bool:
    """Ensure NIfTI JSON sidecar includes `FlipAngle`.

    Returns True when the file was written/updated.
    """

    if not nifti_path:
        return False
    json_path = _nifti_sidecar_json_path(nifti_path)
    if not os.path.exists(json_path):
        return False
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    if "FlipAngle" in data:
        return False
    try:
        data["FlipAngle"] = float(flip_angle_deg)
    except (TypeError, ValueError):
        return False
    try:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        return True
    except OSError:
        return False


def inject_turbo_factor_into_sidecar_json(nifti_path: str, turbo_factor: int) -> bool:
    """Ensure NIfTI JSON sidecar includes `TurboFactor`.

    Returns True when the file was written/updated.
    """

    if not nifti_path:
        return False
    json_path = _nifti_sidecar_json_path(nifti_path)
    if not os.path.exists(json_path):
        return False
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    if "TurboFactor" in data:
        return False
    try:
        v = int(turbo_factor)
    except Exception:
        return False
    if v <= 0:
        return False
    data["TurboFactor"] = v
    try:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        return True
    except OSError:
        return False


def _infer_inversion_time_s_from_protocol(protocol_name: str | None) -> float | None:
    """Infer TurboFLASH TI (seconds) from protocol/series naming.

    Examples:
    - "WIP TI_00120" -> 0.12
    - "WIPTI_00300" -> 0.30
    """

    if not protocol_name:
        return None
    s = str(protocol_name)
    m = re.search(r"\bTI[_\s-]*(\d{3,6})\b", s, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bTI(\d{3,6})\b", s, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        ms = float(m.group(1))
    except Exception:
        return None
    if not np.isfinite(ms) or ms <= 0:
        return None
    return ms * 1e-3


def _parrec_header_fields(par_path: str) -> dict:
    """Extract key metadata from a Philips PAR/REC file.

    Returned fields are designed to complement (not replace) dcm2niix JSON.
    """

    if nib is None:
        return {}

    try:
        img = nib.parrec.load(par_path)
        hdr = img.header
    except Exception:
        return {}

    out: dict = {}

    # Protocol name is most reliable for TI inference.
    proto = None
    try:
        proto = hdr.general_info.get("protocol_name")
    except Exception:
        proto = None
    if isinstance(proto, (list, tuple, np.ndarray)):
        try:
            proto = proto[0]
        except Exception:
            proto = None
    if proto:
        out["ProtocolNamePAR"] = str(proto)

    defs = getattr(hdr, "image_defs", None)
    if defs is None or not hasattr(defs, "dtype"):
        return out

    names = set(defs.dtype.names or ())

    def _median(name: str) -> float | None:
        if name not in names:
            return None
        try:
            arr = np.asarray(defs[name], dtype=float).reshape(-1)
        except Exception:
            return None
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return float(np.median(arr))

    # Geometry
    ps = _median("pixel spacing")
    if ps is not None and ps > 0:
        out["PixelSpacing"] = [float(ps), float(ps)]

    st = _median("slice thickness")
    if st is not None and st > 0:
        out["SliceThickness"] = float(st)

    sg = _median("slice gap")
    if st is not None and sg is not None and st > 0 and sg >= 0:
        out["SpacingBetweenSlices"] = float(st + sg)

    # Timing
    tr_ms = None
    try:
        tr = hdr.general_info.get("repetition_time")
        if isinstance(tr, (list, tuple, np.ndarray)):
            tr_ms = float(np.asarray(tr, dtype=float).reshape(-1)[0])
        elif tr is not None:
            tr_ms = float(tr)
    except Exception:
        tr_ms = None
    if tr_ms is not None and np.isfinite(tr_ms) and tr_ms > 0:
        out["RepetitionTimeExcitation"] = float(tr_ms) * 1e-3

    # dcm2niix often omits per-volume start times for PAR/REC; fill with dyn_scan_begin_time.
    if "dyn_scan_begin_time" in names:
        try:
            times = np.unique(np.asarray(defs["dyn_scan_begin_time"], dtype=float).reshape(-1))
            times = times[np.isfinite(times)]
            times.sort()
            if times.size:
                out["FrameTimesStart"] = [float(v) for v in times.tolist()]
        except Exception:
            pass

    # Sequence parameters
    fa = _median("image_flip_angle")
    if fa is not None and fa > 0:
        out["FlipAngle"] = float(fa)

    tf = _median("TURBO factor")
    if tf is not None and tf > 0:
        try:
            out["TurboFactor"] = int(round(float(tf)))
        except Exception:
            pass

    # Best-effort TI inference (usually missing in dcm2niix PAR/REC JSON for this dataset).
    ti = _infer_inversion_time_s_from_protocol(str(proto) if proto else None)
    if ti is not None and ti > 0:
        out["InversionTime"] = float(ti)

    return out


def inject_parrec_metadata_into_sidecar_json(nifti_path: str, par_path: str) -> bool:
    """Enrich an existing dcm2niix JSON sidecar using PAR/REC metadata.

    Only fills missing keys (never overwrites existing values).
    Returns True when it wrote updates.
    """

    if not nifti_path or not par_path:
        return False
    json_path = _nifti_sidecar_json_path(nifti_path)
    if not os.path.exists(json_path):
        return False
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False

    fields = _parrec_header_fields(par_path)
    if not fields:
        return False

    changed = False
    for k, v in fields.items():
        # Never stomp existing values.
        if k in data:
            continue
        data[k] = v
        changed = True

    if not changed:
        return False
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        return True
    except OSError:
        return False


def resolve_turboflash_nph(nifti_path: str, default=None):
    """Resolve TurboFLASH `nph` (ky=0 line index).

    MATLAB case-3 (`extract_t1_philips6_NewParRec2015`) defaults to `nph=1`
    ("low-high sampling") and does *not* derive it from TurboFactor.

    Resolution order:
    1) Env override: P_BRAIN_TURBOFLASH_NPH / P_BRAIN_TURBO_NPH
    2) JSON sidecar explicit keys: nph/NPH
    3) Fallback: `default` (or 1)
    """

    for key in ("P_BRAIN_TURBOFLASH_NPH", "P_BRAIN_TURBO_NPH"):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            try:
                v = int(float(raw))
                if v > 0:
                    return v
            except Exception:
                pass

    if nifti_path:
        data = read_nifti_sidecar_json(nifti_path)
        if isinstance(data, dict):
            for key in ("nph", "NPH"):
                if key in data:
                    try:
                        v = int(float(data.get(key)))
                        if v > 0:
                            return v
                    except Exception:
                        pass

    try:
        v = int(float(default))
        return v if v > 0 else 1
    except Exception:
        return 1

def list_addons():
    addon_folders = [f for f in os.listdir("addons") if os.path.isdir(os.path.join("addons", f))]
    print('--------------------------------')
    for i, addon in enumerate(addon_folders):
        print(f"{i+1}. {addon}")
    print('--------------------------------')    
    choice = int(input("[!] Choose addon: ")) - 1
    return addon_folders[choice]

import sys
def load_addon(addon_folder_name, *args):
    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
        base_dir = os.path.dirname(current_script_dir)
        addon_path = os.path.join(base_dir, 'addons')
        sys.path.append(addon_path)  

        addon = importlib.import_module(f'{addon_folder_name}.{addon_folder_name}')
        addon.run(*args)
    except ModuleNotFoundError as e:
        print(f"Addon {addon_folder_name} not available! Error: {str(e)}")
        print("Make sure the addon is correctly placed in the 'addons' directory and named correctly.")
        print(f"Attempted to import from {addon_path}")
    except Exception as e:
        print(f"An error occurred while loading the addon: {str(e)}")

def replace_max_with_artery_type_and_delete(values_json_path, max_info_json_path):
    # Read and parse max_info.json
    with open(max_info_json_path, 'r') as f:
        max_info_json = json.load(f)
    
    # Check if 'Max artery type' exists in max_info.json
    artery_type_list = [entry.split(": ")[1] for entry in max_info_json if "Max artery type" in entry]
    
    if not artery_type_list:
        raise ValueError("No 'Max artery type' found in max_info.json")
    
    artery_type = artery_type_list[0]
    
    # Read and parse values.json
    with open(values_json_path, 'r') as f:
        values_json = json.load(f)
    
    # Replace "Max" with the extracted "Artery type" in values.json
    updated_values_json = []
    
    for entry in values_json:
        new_entry = {k: v.replace("Max", artery_type) if "Max" in v else v for k, v in entry.items()}
        updated_values_json.append(new_entry)
    
    # Update values.json with the modified data
    with open(values_json_path, 'w') as f:
        json.dump(updated_values_json, f)
    
    # Uncomment the following line to remove max_info.json after processing
    # os.remove(max_info_json_path)

    return updated_values_json


def find_matching_file(directory, pattern):
    regex = re.compile(pattern, re.IGNORECASE)
    for filename in os.listdir(directory):
        if regex.match(filename):
            return os.path.join(directory, filename)
    return None

def save_values(Ki, SD_Ki, lambda_, P, P_std, subtype_tissue, slice_tissue,
                subtype_artery, venous_slice, arterial_slice,
                analysis_directory, suffix=""):
    """Save permeability results to ``values{suffix}.json``."""
    values_file_path = os.path.join(analysis_directory, f'values{suffix}.json')
    
    if os.path.exists(values_file_path):
        with open(values_file_path, 'r') as f:
            existing_values = json.load(f)
    else:
        existing_values = []
    
    new_entry = {
        "Ki": f"{Ki*6000} (+- {SD_Ki*6000})",
        "Lambda": f"{lambda_*100}",
        "Tissue": f"{subtype_tissue} (Slice {slice_tissue})",
        "Artery": f"{subtype_artery} (Venous Slice {venous_slice}, Arterial Slice {arterial_slice})",
        "Ki_f": f"{P} (+- {P_std})"
    }
    
    updated_values = []
    found = False
    
    for entry in existing_values:
        if entry.get("Tissue") == f"{subtype_tissue} (Slice {slice_tissue})":
            updated_values.append(new_entry)
            found = True
        else:
            updated_values.append(entry)
    
    if not found:
        updated_values.append(new_entry)
    
    with open(values_file_path, 'w') as f:
        json.dump(updated_values, f)




def leaver():
    leave = input('[!] Return to main menu? (y/n): ')    
    if leave == 'y' or leave=='':
        return
    if leave == 'n':
        leaver()
    else:
        print('This was not an option...')
        time.sleep(2)
        leaver()      

def find_matching_file(directory, pattern):
    regex = re.compile(pattern, re.IGNORECASE)
    for filename in os.listdir(directory):
        if regex.match(filename):
            return os.path.join(directory, filename)
    return None


def load_curves(venous_slice, arterial_slice, artery_choice, analysis_directory):
    shifted_vein_file = os.path.join(analysis_directory, 'CTC Data', 'Vein', 'Sinus Sagittalis', f'CTC_shifted_slice_{venous_slice}.npy')
    vein_file = os.path.join(analysis_directory, 'CTC Data', 'Vein', 'Sinus Sagittalis', f'CTC_slice_{venous_slice}.npy')
    
    shifted_artery_file = os.path.join(analysis_directory, 'CTC Data', 'Artery', artery_choice, f'CTC_shifted_slice_{arterial_slice}.npy')
    artery_file = os.path.join(analysis_directory, 'CTC Data', 'Artery', artery_choice, f'CTC_slice_{arterial_slice}.npy')

    if os.path.exists(shifted_vein_file):
        vein_curve = np.load(shifted_vein_file)
    else:
        vein_curve = np.load(vein_file)
    
    if os.path.exists(shifted_artery_file):
        artery_curve = np.load(shifted_artery_file)
    else:
        artery_curve = np.load(artery_file)
    
    return vein_curve, artery_curve



def save_as_pickle(matrix, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(matrix, file)


def load_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        matrix = pickle.load(file)
    return matrix


def first_existing_file(directory, patterns, time, suffix):
    for pattern in patterns:
        file_path = os.path.join(directory, f"{pattern}{time}{suffix}")
        if os.path.exists(file_path):
            return file_path
    return None


def _read_max_artery_type(analysis_directory):
    info_path = os.path.join(analysis_directory, 'max_info.json')
    if not os.path.exists(info_path):
        return None

    try:
        with open(info_path, 'r') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    for entry in data:
        if isinstance(entry, str) and entry.startswith('Max artery type:'):
            return entry.split(':', 1)[1].strip()
    return None


def _parse_tscc_filename(filename):
    match = re.match(r'TSCC_slice_(\d+)_(\d+)\.npy$', filename)
    if not match:
        raise ValueError(f"Unrecognised TSCC filename format: {filename}")
    venous_slice, arterial_slice = map(int, match.groups())
    return venous_slice, arterial_slice


def _load_shifted_input_function(analysis_directory, subtype, venous_slice, arterial_slice):
    tscc_root = os.path.join(analysis_directory, 'TSCC Data')

    if subtype is None or subtype == 'Max':
        max_dir = os.path.join(tscc_root, 'Max')
        npy_files = [
            f for f in os.listdir(max_dir)
            if f.endswith('.npy') and not f.startswith('.')
        ]
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {max_dir}.")
        npy_files.sort()
        filename = npy_files[0]
        venous_slice, arterial_slice = _parse_tscc_filename(filename)
        artery_type = _read_max_artery_type(analysis_directory) or 'Max'
        path = os.path.join(max_dir, filename)
    else:
        if venous_slice is None or arterial_slice is None:
            raise ValueError(
                "Both venous and arterial slice indices are required for TSCC input functions."
            )
        filename = f'TSCC_slice_{venous_slice}_{arterial_slice}.npy'
        path = os.path.join(tscc_root, subtype, filename)
        artery_type = subtype
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input function file not found: {path}")

    curve = np.load(path)
    metadata = {
        'source': 'SSS',
        'path': path,
        'artery_subtype': artery_type,
        'venous_slice': venous_slice,
        'arterial_slice': arterial_slice,
    }
    return curve, metadata


def _load_pure_input_function(analysis_directory, subtype, venous_slice, arterial_slice):
    artery_root = os.path.join(analysis_directory, 'CTC Data', 'Artery')
    if not os.path.isdir(artery_root):
        raise FileNotFoundError(f"Artery directory not found: {artery_root}")

    def available_slices(artery_dir):
        slices = []
        for fname in os.listdir(artery_dir):
            match = re.match(r'CTC_slice_(\d+)\.npy$', fname)
            if match:
                slices.append((int(match.group(1)), fname))
        return sorted(slices)

    best_curve = None
    best_metadata = None
    best_truncated_curve = None
    best_truncated_metadata = None

    if subtype is None or subtype == 'Max':
        for current_subtype in os.listdir(artery_root):
            artery_dir = os.path.join(artery_root, current_subtype)
            if not os.path.isdir(artery_dir):
                continue
            for slice_idx, fname in available_slices(artery_dir):
                path = os.path.join(artery_dir, fname)
                curve = np.load(path)
                peak = float(np.max(curve))
                is_trunc = False
                try:
                    is_trunc, _details = detect_truncated_bolus(
                        curve,
                        num_peaks=None,
                        peak_fraction=0.99,
                        plateau_min_points=3,
                    )
                    is_trunc = bool(is_trunc)
                except Exception:
                    is_trunc = False

                if is_trunc:
                    if best_truncated_curve is None or peak > best_truncated_metadata['peak']:
                        best_truncated_curve = curve
                        best_truncated_metadata = {
                            'source': 'RICA',
                            'path': path,
                            'artery_subtype': current_subtype,
                            'venous_slice': None,
                            'arterial_slice': slice_idx,
                            'peak': peak,
                        }
                else:
                    if best_curve is None or peak > best_metadata['peak']:
                        best_curve = curve
                        best_metadata = {
                            'source': 'RICA',
                            'path': path,
                            'artery_subtype': current_subtype,
                            'venous_slice': None,
                            'arterial_slice': slice_idx,
                            'peak': peak,
                        }

        if best_curve is None and best_truncated_curve is not None:
            best_curve = best_truncated_curve
            best_metadata = best_truncated_metadata

        if best_curve is None:
            raise FileNotFoundError(
                f"No arterial concentration curves found in {artery_root}."
            )
        metadata = best_metadata.copy()
        metadata.pop('peak', None)
        return best_curve, metadata

    artery_dir = os.path.join(artery_root, subtype)
    if not os.path.isdir(artery_dir):
        raise FileNotFoundError(f"Artery subtype directory not found: {artery_dir}")

    slices = available_slices(artery_dir)
    if not slices:
        raise FileNotFoundError(
            f"No pure arterial curves available for subtype '{subtype}'."
        )

    if arterial_slice is None or str(arterial_slice).lower() == 'auto':
        best_slice = None
        best_curve = None
        best_peak = None
        best_truncated_slice = None
        best_truncated_curve = None
        best_truncated_peak = None
        for slice_idx, fname in slices:
            path = os.path.join(artery_dir, fname)
            curve = np.load(path)
            peak = float(np.max(curve))
            is_trunc = False
            try:
                is_trunc, _details = detect_truncated_bolus(
                    curve,
                    num_peaks=None,
                    peak_fraction=0.99,
                    plateau_min_points=3,
                )
                is_trunc = bool(is_trunc)
            except Exception:
                is_trunc = False

            if is_trunc:
                if best_truncated_peak is None or peak > best_truncated_peak:
                    best_truncated_slice = slice_idx
                    best_truncated_curve = curve
                    best_truncated_peak = peak
                    best_truncated_path = path
            else:
                if best_peak is None or peak > best_peak:
                    best_slice = slice_idx
                    best_curve = curve
                    best_peak = peak
                    best_path = path

        if best_curve is None and best_truncated_curve is not None:
            best_slice = best_truncated_slice
            best_curve = best_truncated_curve
            best_path = best_truncated_path

        if best_curve is None:
            raise FileNotFoundError(
                f"No pure arterial curves available for subtype '{subtype}'."
            )
        metadata = {
            'source': 'RICA',
            'path': best_path,
            'artery_subtype': subtype,
            'venous_slice': None,
            'arterial_slice': best_slice,
        }
        return best_curve, metadata

    try:
        slice_idx = int(arterial_slice)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid arterial slice index '{arterial_slice}' for pure input function."
        )

    filename = f'CTC_slice_{slice_idx}.npy'
    path = os.path.join(artery_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input function file not found: {path}")

    curve = np.load(path)
    metadata = {
        'source': 'RICA',
        'path': path,
        'artery_subtype': subtype,
        'venous_slice': None,
        'arterial_slice': slice_idx,
    }
    return curve, metadata


def _load_vif_input_function(analysis_directory: str, venous_slice=None):
    """Load a pure venous input function (VIF) from SSS CTC outputs."""

    vein_dir = os.path.join(analysis_directory, 'CTC Data', 'Vein', 'Sinus Sagittalis')
    if not os.path.isdir(vein_dir):
        raise FileNotFoundError(f"Vein directory not found: {vein_dir}")

    def available_slices():
        out = []
        for fname in os.listdir(vein_dir):
            m = re.match(r'CTC_slice_(\d+)\.npy$', fname)
            if m:
                out.append((int(m.group(1)), fname))
        return sorted(out)

    slices = available_slices()
    if not slices:
        raise FileNotFoundError(f"No venous concentration curves found in {vein_dir}.")

    # Auto-pick: highest peak (prefer non-truncated).
    if venous_slice is None or str(venous_slice).lower() == 'auto':
        best = None
        best_meta = None
        best_trunc = None
        best_trunc_meta = None
        for slice_idx, fname in slices:
            path = os.path.join(vein_dir, fname)
            curve = np.load(path)
            peak = float(np.max(curve))
            is_trunc = False
            try:
                is_trunc, _details = detect_truncated_bolus(
                    curve,
                    num_peaks=None,
                    peak_fraction=0.99,
                    plateau_min_points=3,
                )
                is_trunc = bool(is_trunc)
            except Exception:
                is_trunc = False

            meta = {
                'source': 'VIF',
                'path': path,
                'artery_subtype': None,
                'venous_slice': slice_idx,
                'arterial_slice': None,
                'peak': peak,
            }
            if is_trunc:
                if best_trunc is None or peak > float(best_trunc_meta['peak']):
                    best_trunc = curve
                    best_trunc_meta = meta
            else:
                if best is None or peak > float(best_meta['peak']):
                    best = curve
                    best_meta = meta

        if best is None and best_trunc is not None:
            best = best_trunc
            best_meta = best_trunc_meta

        if best is None:
            raise FileNotFoundError(f"No venous concentration curves found in {vein_dir}.")

        meta = dict(best_meta)
        meta.pop('peak', None)
        return best, meta

    try:
        slice_idx = int(venous_slice)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid venous slice index '{venous_slice}' for VIF input function.")

    shifted_path = os.path.join(vein_dir, f'CTC_shifted_slice_{slice_idx}.npy')
    path = shifted_path if os.path.exists(shifted_path) else os.path.join(vein_dir, f'CTC_slice_{slice_idx}.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(f"VIF input function file not found: {path}")

    curve = np.load(path)
    metadata = {
        'source': 'VIF',
        'path': path,
        'artery_subtype': None,
        'venous_slice': slice_idx,
        'arterial_slice': None,
    }
    return curve, metadata


def get_input_function_curve(analysis_directory, subtype='Max',
                             venous_slice=None, arterial_slice=None):
    """Load the selected arterial input function as configured.

    Parameters
    ----------
    analysis_directory : str
        Root analysis directory for the current dataset.
    subtype : str, optional
        Artery subtype requested by the user.  ``'Max'`` selects the default
        time-shifted curve in SSS mode or the highest peak pure curve when the
        pure arterial option is enabled.
    venous_slice : str or int, optional
        Venous slice index (used for time-shifted SSS curves).
    arterial_slice : str or int, optional
        Arterial slice index.  When ``None`` or ``'auto'`` in pure mode, the
        slice with the highest peak is selected automatically.

    Returns
    -------
    tuple
        ``(curve, metadata)`` where *curve* is a NumPy array containing the
        input function and *metadata* a dictionary describing the selection.
    """

    mode = str(getattr(settings, 'MODELLING_INPUT_FUNCTION', '') or '').strip().lower()
    if mode == 'vif':
        return _load_vif_input_function(analysis_directory, venous_slice=venous_slice)

    source = settings.INPUT_FUNCTION_SOURCE
    # Explicit AIF mode forces pure arterial, regardless of legacy toggle.
    if mode == 'aif' or source in {'RICA', 'LICA'}:
        return _load_pure_input_function(
            analysis_directory,
            subtype,
            venous_slice,
            arterial_slice,
        )

    try:
        return _load_shifted_input_function(
            analysis_directory,
            subtype,
            venous_slice,
            arterial_slice,
        )
    except FileNotFoundError as exc:
        curve, metadata = _load_pure_input_function(
            analysis_directory,
            subtype,
            venous_slice,
            arterial_slice,
        )
        metadata = dict(metadata)
        metadata['fallback_from'] = source
        metadata['fallback_reason'] = str(exc)
        return curve, metadata

