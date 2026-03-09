import numpy as np
import os
import sys
import re
import time 
import json
from typing import Optional

import utils.settings as settings

def _get_first_existing_file(filenames, nifti_directory):
    """Return the first filename from ``filenames`` that exists."""

    for filename in filenames:
        if not filename:
            continue
        if os.path.exists(os.path.join(nifti_directory, filename)):
            return filename
    return None


def get_dce_filename(primary, fallback, nifti_directory):
    return _get_first_existing_file((fallback, primary), nifti_directory)


def get_diffusion_filename(candidates, nifti_directory):
    """Return the configured diffusion volume if present in ``nifti_directory``."""

    if isinstance(candidates, str):
        candidates = (candidates,)
    return _get_first_existing_file(candidates, nifti_directory)


# Diffusion acquisition configuration -------------------------------------------------

_LEGACY_DIFFUSION_FILENAMES = (
    "WIPDTI_RSI_P.nii",
    "WIPDTI_RSI_P.nii.gz",
    "WIPDTI_RSI_A.nii",
    "WIPDTI_RSI_A.nii.gz",
    "WIPDWI_RSI_P.nii",
    "WIPDWI_RSI_P.nii.gz",
)


DIFFUSION_FILE_GROUPS: dict[str, tuple[str, ...]] = {
    "dti": _LEGACY_DIFFUSION_FILENAMES,
    # Drift-corrected registered diffusion volume (preferred when available)
    "dwi_reg": (
        "Reg-DWInySENSE.nii",
        "Reg-DWInySENSE.nii.gz",
        "Reg-DWInySENSE_ADC.nii",
        "Reg-DWInySENSE_ADC.nii.gz",
    ),
    # Legacy isoDWI volumes retained for compatibility
    "dwi_iso": (
        "isoDWIb-1000.nii",
        "isoDWIb-1000.nii.gz",
    ),
}


_DEFAULT_DIFFUSION_PRIORITY = (
    "dti",
    "dwi_reg",
    "dwi_iso",
    "dwi",
)

_SUPPORTED_DIFFUSION_MODELS = {"DTI", "CSD"}

_DIFFUSION_MODEL_BY_GROUP = {
    "dti": "DTI",
    "dwi": "CSD",
    "dwi_reg": "CSD",
    "dwi_iso": "CSD",
}


def _parse_priority_list(raw: str) -> tuple[str, ...]:
    entries = []
    for token in raw.split(","):
        cleaned = token.strip().lower()
        if cleaned:
            entries.append(cleaned)
    return tuple(entries)


def diffusion_file_priority() -> tuple[str, ...]:
    env_value = os.environ.get("P_BRAIN_DIFFUSION_PRIORITY", "")
    if env_value:
        parsed = _parse_priority_list(env_value)
        if parsed:
            return parsed
    return _DEFAULT_DIFFUSION_PRIORITY


def diffusion_file_groups() -> dict[str, tuple[str, ...]]:
    return DIFFUSION_FILE_GROUPS


def diffusion_model_map() -> dict[str, str]:
    model_map: dict[str, str] = {}
    for group, default_model in _DIFFUSION_MODEL_BY_GROUP.items():
        env_key = f"P_BRAIN_DIFFUSION_MODEL_{group.upper()}"
        override = os.environ.get(env_key)
        if override:
            override_value = override.strip().upper()
            if override_value in _SUPPORTED_DIFFUSION_MODELS:
                model_map[group] = override_value
                continue
        model_map[group] = default_model
    return model_map


def ordered_diffusion_filenames() -> tuple[str, ...]:
    seen = set()
    ordered: list[str] = []
    groups = diffusion_file_priority()
    for group in groups:
        for pattern in DIFFUSION_FILE_GROUPS.get(group, ()):  # type: ignore[index]
            normalized = pattern.strip()
            if not normalized:
                continue
            if normalized in seen:
                continue
            ordered.append(normalized)
            seen.add(normalized)
    if not ordered:
        ordered.extend(_LEGACY_DIFFUSION_FILENAMES)
    return tuple(ordered)

# Global parameters: 


def get_freesurfer_version(fs_home: Optional[str] = None) -> Optional[str]:
    """Return the FreeSurfer version string (e.g. '7.4.1' or '8.1.0').

    Resolution order:
    1. ``build-stamp.txt`` inside *fs_home*.
    2. ``recon-all --version`` on PATH.
    Returns ``None`` when FreeSurfer cannot be found.
    """
    import shutil
    import subprocess as _sp

    if not fs_home:
        fs_home = os.environ.get("FREESURFER_HOME", "").strip()

    # Try build-stamp.txt first (fastest, no subprocess)
    if fs_home:
        stamp = os.path.join(fs_home, "build-stamp.txt")
        if os.path.isfile(stamp):
            try:
                with open(stamp, "r") as fh:
                    text = fh.read().strip()
                # e.g. "freesurfer-macOS-darwin_x86_64-7.4.1-20230614-7eb8460"
                m = re.search(r"(\d+\.\d+\.\d+)", text)
                if m:
                    return m.group(1)
            except Exception:
                pass

    # Fallback: recon-all --version
    recon = shutil.which("recon-all")
    if recon:
        try:
            r = _sp.run(
                [recon, "--version"],
                stdout=_sp.PIPE, stderr=_sp.STDOUT,
                text=True, timeout=10,
            )
            m = re.search(r"(\d+\.\d+\.\d+)", r.stdout or "")
            if m:
                return m.group(1)
        except Exception:
            pass

    return None


def _resolve_segmentation_method(explicit: Optional[str] = None,
                                  fs_home: Optional[str] = None) -> str:
    """Choose the right segmentation backend.

    Priority:
    1. If *explicit* is set (user override via settings) use that directly.
    2. Default is ``"fastsurfer"``.

    When the caller passes ``"freesurfer"`` the function auto-selects the
    concrete FreeSurfer tool:
    - FreeSurfer >= 8.0 → ``"synthseg"``  (mri_synthseg)
    - FreeSurfer <  8.0 → ``"recon-all"``
    """
    method = (explicit or "fastsurfer").strip().lower()

    # pbrain uses its own lightweight model – pass through directly.
    if method == "pbrain":
        return method

    if method == "freesurfer":
        ver = get_freesurfer_version(fs_home)
        if ver:
            try:
                major = int(ver.split(".")[0])
            except (ValueError, IndexError):
                major = 0
            method = "synthseg" if major >= 8 else "recon-all"
        else:
            # Can't determine version – assume old install
            method = "recon-all"

    return method


def global_parameters():
    t1_mode = getattr(settings, "T1_FIT_MODE", "auto")
    if t1_mode not in {"auto", "ir", "vfa", "none"}:
        t1_mode = "auto"

    IsVFA = t1_mode == "vfa"  # Variable flip angle for the T1/M0 fit
    IsIR = t1_mode in {"auto", "ir"}  # Inversion recovery method
    if t1_mode == "none":
        IsVFA = False
        IsIR = False
    apple_metal = True # Enable if running on apple M1/M2/M3...
    boundary = True #compute boundary mask from GM/WM masks and plot/compute patlak values alongside wm/gm
    RERUN_SEGMENTATION = False  # Force rerun of FastSurfer segmentation

    # Resolve segmentation method.  An explicit override stored in settings
    # takes priority; otherwise default to fastsurfer.  When the value is
    # "freesurfer" it is further resolved to "synthseg" (FS >= 8) or
    # "recon-all" (FS < 8) by _resolve_segmentation_method.
    raw_method = getattr(settings, "SEGMENTATION_METHOD", "fastsurfer")
    SEGMENTATION_METHOD = _resolve_segmentation_method(raw_method)

    COMPUTE_FA = False  # Compute fractional anisotropy from DWI
    return (
        IsVFA,
        IsIR,
        apple_metal,
        boundary,
        RERUN_SEGMENTATION,
        SEGMENTATION_METHOD,
        COMPUTE_FA,
    )

def refresh_nifti_directory(nifti_directory):
    return os.listdir(nifti_directory)


def _iter_nifti_filenames(nifti_directory: str) -> list[str]:
    try:
        entries = os.listdir(nifti_directory)
    except Exception:
        return []

    out: list[str] = []
    for name in entries:
        lower = name.lower()
        if lower.endswith(".nii") or lower.endswith(".nii.gz"):
            full_path = os.path.join(nifti_directory, name)
            if os.path.isfile(full_path):
                out.append(name)
    return out


def _strip_nii_ext(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        return filename[: -len(".nii.gz")]
    if lower.endswith(".nii"):
        return filename[: -len(".nii")]
    return filename


def _sidecar_path(nifti_directory: str, nifti_filename: str, ext: str) -> str:
    base = _strip_nii_ext(nifti_filename)
    return os.path.join(nifti_directory, f"{base}{ext}")


def _has_diffusion_sidecars(nifti_directory: str, nifti_filename: str) -> bool:
    return os.path.isfile(_sidecar_path(nifti_directory, nifti_filename, ".bval")) or os.path.isfile(
        _sidecar_path(nifti_directory, nifti_filename, ".bvec")
    )


def _read_text_sidecar(nifti_directory: str, nifti_filename: str) -> str:
    json_path = _sidecar_path(nifti_directory, nifti_filename, ".json")
    if not os.path.isfile(json_path):
        return ""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return ""

    pieces: list[str] = []
    for key in (
        "SeriesDescription",
        "ProtocolName",
        "SequenceName",
        "ImageType",
        "ScanningSequence",
        "SequenceVariant",
    ):
        value = payload.get(key)
        if isinstance(value, str):
            pieces.append(value)
        elif isinstance(value, list):
            pieces.extend([str(v) for v in value if v is not None])
    return " ".join(pieces).lower()


def _nifti_shape(nifti_path: str) -> Optional[tuple[int, ...]]:
    try:
        import nibabel as nib  # local import to keep module import light

        img = nib.load(nifti_path)
        shape = img.shape
        try:
            return tuple(int(s) for s in shape)
        except Exception:
            return None
    except Exception:
        return None


def _infer_dce_filename(nifti_directory: str) -> Optional[str]:
    """Best-effort: choose a DCE-like NIfTI from directory contents.

    Heuristics:
    - Prefer 4D files with t>1
    - Exclude diffusion-like files (bval/bvec sidecars)
    - Prefer highest t, then largest file size
    """

    candidates = []
    for name in _iter_nifti_filenames(nifti_directory):
        if _has_diffusion_sidecars(nifti_directory, name):
            continue
        full_path = os.path.join(nifti_directory, name)
        shape = _nifti_shape(full_path)
        if shape and len(shape) >= 4 and int(shape[3]) > 1:
            t = int(shape[3])
            size = int(os.path.getsize(full_path))
            candidates.append((t, size, name))

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]

    # Fallback: try keyword match from JSON sidecars.
    keyword_hits = []
    for name in _iter_nifti_filenames(nifti_directory):
        if _has_diffusion_sidecars(nifti_directory, name):
            continue
        text = _read_text_sidecar(nifti_directory, name)
        if not text:
            continue
        score = 0
        for kw in ("dce", "dynamic", "dyn", "perfusion", "hperf", "bolus"):
            if kw in text:
                score += 1
        if score:
            full_path = os.path.join(nifti_directory, name)
            size = int(os.path.getsize(full_path))
            keyword_hits.append((score, size, name))
    if keyword_hits:
        keyword_hits.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return keyword_hits[0][2]

    return None


def _infer_structural_filename(
    nifti_directory: str,
    *,
    include_keywords: tuple[str, ...],
    exclude_keywords: tuple[str, ...] = (),
) -> Optional[str]:
    """Best-effort modality selection from available NIfTI + JSON sidecars."""

    hits = []
    for name in _iter_nifti_filenames(nifti_directory):
        if _has_diffusion_sidecars(nifti_directory, name):
            continue
        text = _read_text_sidecar(nifti_directory, name)
        if not text:
            continue
        if any(ex in text for ex in exclude_keywords):
            continue
        score = sum(1 for kw in include_keywords if kw in text)
        if not score:
            continue
        full_path = os.path.join(nifti_directory, name)
        shape = _nifti_shape(full_path)
        is_3d = bool(shape and len(shape) == 3)
        size = int(os.path.getsize(full_path))
        # Prefer 3D anatomicals when available
        hits.append((score, 1 if is_3d else 0, size, name))

    if hits:
        hits.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return hits[0][3]

    return None


def _largest_3d_nifti(
    nifti_directory: str,
    *,
    exclude: set[str] | None = None,
    min_slices: int = 32,
    include_keywords: tuple[str, ...] = (),
    exclude_keywords: tuple[str, ...] = (),
) -> Optional[str]:
    """Return the largest plausible 3D NIfTI in the directory.

    This is used only as a *fallback* when explicit modality detection fails.
    We keep it conservative to avoid treating small axial stacks (e.g. T2 localizers)
    as structural anatomicals.
    """

    exclude = exclude or set()
    best: tuple[int, str] | None = None
    for name in _iter_nifti_filenames(nifti_directory):
        if name in exclude:
            continue
        if _has_diffusion_sidecars(nifti_directory, name):
            continue

        lower_name = name.lower()
        text = lower_name
        sidecar = _read_text_sidecar(nifti_directory, name)
        if sidecar:
            text = f"{text} {sidecar}"

        if exclude_keywords and any(ex in text for ex in exclude_keywords):
            continue
        if include_keywords and not any(kw in text for kw in include_keywords):
            continue

        full_path = os.path.join(nifti_directory, name)
        shape = _nifti_shape(full_path)
        if not shape or len(shape) != 3:
            continue
        try:
            if int(shape[2]) < int(min_slices):
                continue
        except Exception:
            continue

        size = int(os.path.getsize(full_path))
        if best is None or size > best[0]:
            best = (size, name)
    return best[1] if best else None


def _resolve_filename_or_infer(
    *,
    nifti_directory: str,
    default_candidates: tuple[str, ...],
    infer_fn,
) -> Optional[str]:
    existing = _get_first_existing_file(default_candidates, nifti_directory)
    if existing:
        return existing
    return infer_fn(nifti_directory)

# Global filenames:
def global_filenames(nifti_directory):
    refresh_nifti_directory(nifti_directory)

    t1_3D_defaults = (
        "WIPcs_T1W_3D_TFE_32channel.nii",
        "WIPcs_T1W_3D_TFE_32channel.nii.gz",
    )
    t1_3D_filename = _resolve_filename_or_infer(
        nifti_directory=nifti_directory,
        default_candidates=t1_3D_defaults,
        infer_fn=lambda d: _infer_structural_filename(
            d,
            include_keywords=("t1", "t1w", "mprage", "tfe", "spgr", "bravo", "vibe"),
            exclude_keywords=("dce", "dynamic", "dyn", "perfusion"),
        )
        # Conservative fallback: only accept likely T1 candidates and require
        # a reasonable number of slices to avoid picking small axial stacks.
        or _largest_3d_nifti(
            d,
            min_slices=32,
            include_keywords=("t1", "t1w", "mprage", "tfe", "spgr", "bravo", "vibe"),
            exclude_keywords=("t2", "tse", "flair", "dce", "dynamic", "dyn", "perfusion"),
        ),
    )
    axial_t1_3D_filename = r'ax([-_ ])?vwipcs_t1w_3d_tfe_32channel\.nii'

    t2_3D_defaults = (
        "WIPcs_3D_Brain_VIEW_T2_32chSHC.nii",
        "WIPcs_3D_Brain_VIEW_T2_32chSHC.nii.gz",
    )
    t2_3D_filename = _resolve_filename_or_infer(
        nifti_directory=nifti_directory,
        default_candidates=t2_3D_defaults,
        infer_fn=lambda d: _infer_structural_filename(
            d,
            include_keywords=("t2", "tse"),
            exclude_keywords=("flair", "dce", "dynamic", "dyn"),
        )
        or _largest_3d_nifti(d, min_slices=32, exclude_keywords=("dce", "dynamic", "dyn", "perfusion")),
    )
    axial_t2_3D_filename = r'ax([-_ ])?vwipcs_3D_Brain_VIEW_T2_32chSHC\.nii'

    flair_3D_defaults = (
        "WIPcs_3D_Brain_VIEW_FLAIR_SHC.nii",
        "WIPcs_3D_Brain_VIEW_FLAIR_SHC.nii.gz",
    )
    flair_3D_filename = _resolve_filename_or_infer(
        nifti_directory=nifti_directory,
        default_candidates=flair_3D_defaults,
        infer_fn=lambda d: (
            _infer_structural_filename(
                d,
                include_keywords=("flair",),
                exclude_keywords=("dce", "dynamic", "dyn"),
            )
            or t2_3D_filename
            or t1_3D_filename
            or _largest_3d_nifti(d, min_slices=32, exclude_keywords=("dce", "dynamic", "dyn", "perfusion"))
        ),
    )
    axial_flair_3D_filename = r'ax([-_ ])?VWIPcs_3D_Brain_VIEW_FLAIR_SHC\.nii'

    axial_t2_2D_defaults = (
        "WIPAxT2TSEmatrix.nii",
        "WIPAxT2TSEmatrix.nii.gz",
    )
    axial_t2_2D_filename = _resolve_filename_or_infer(
        nifti_directory=nifti_directory,
        default_candidates=axial_t2_2D_defaults,
        infer_fn=lambda d: _infer_structural_filename(
            d,
            include_keywords=("t2", "tse"),
            exclude_keywords=("flair", "dce", "dynamic", "dyn"),
        )
        or t2_3D_filename,
    )

    dce_filename_primary = 'WIPhperf120long.nii'
    dce_filename_fallback = 'WIPDelRec-hperf120long.nii'
    dce_defaults = (
        dce_filename_fallback,
        dce_filename_primary,
        f"{dce_filename_fallback}.gz",
        f"{dce_filename_primary}.gz",
        dce_filename_fallback.replace(".nii", ".nii.gz"),
        dce_filename_primary.replace(".nii", ".nii.gz"),
    )
    diffusion_candidates = ordered_diffusion_filenames()

    diffusion_filename = get_diffusion_filename(diffusion_candidates, nifti_directory)
    dce_filename = _resolve_filename_or_infer(
        nifti_directory=nifti_directory,
        default_candidates=dce_defaults,
        infer_fn=_infer_dce_filename,
    )

    return (
        t1_3D_filename,
        axial_t1_3D_filename,
        t2_3D_filename,
        axial_t2_3D_filename,
        flair_3D_filename,
        axial_flair_3D_filename,
        axial_t2_2D_filename,
        diffusion_filename,
        dce_filename,
    )

# Separate filenames for control datasets used by the AI pipeline
def control_filenames(nifti_directory):
    refresh_nifti_directory(nifti_directory)
    t1_3D_defaults = (
        "WIPT1W_3D_TFE.nii",
        "WIPT1W_3D_TFE.nii.gz",
    )
    t1_3D_filename = _resolve_filename_or_infer(
        nifti_directory=nifti_directory,
        default_candidates=t1_3D_defaults,
        infer_fn=lambda d: _infer_structural_filename(
            d,
            include_keywords=("t1", "t1w", "mprage", "tfe", "spgr", "bravo", "vibe"),
            exclude_keywords=("dce", "dynamic", "dyn", "perfusion"),
        )
        or _largest_3d_nifti(
            d,
            min_slices=32,
            include_keywords=("t1", "t1w", "mprage", "tfe", "spgr", "bravo", "vibe"),
            exclude_keywords=("t2", "tse", "flair", "dce", "dynamic", "dyn", "perfusion"),
        ),
    )
    axial_t1_3D_filename = r'ax([-_ ])?vwipcs_t1w_3d_tfe_32channel\.nii'

    t2_3D_defaults = (
        "WIPcs_3D_Brain_VIEW_T2_32chSHC.nii",
        "WIPcs_3D_Brain_VIEW_T2_32chSHC.nii.gz",
    )
    t2_3D_filename = _resolve_filename_or_infer(
        nifti_directory=nifti_directory,
        default_candidates=t2_3D_defaults,
        infer_fn=lambda d: _infer_structural_filename(
            d,
            include_keywords=("t2", "tse"),
            exclude_keywords=("flair", "dce", "dynamic", "dyn"),
        )
        or _largest_3d_nifti(d, min_slices=32, exclude_keywords=("dce", "dynamic", "dyn", "perfusion")),
    )
    axial_t2_3D_filename = r'ax([-_ ])?vwipcs_3D_Brain_VIEW_T2_32chSHC\.nii'

    flair_3D_defaults = (
        "WIPcs_3D_Brain_VIEW_FLAIR_SHC.nii",
        "WIPcs_3D_Brain_VIEW_FLAIR_SHC.nii.gz",
    )
    flair_3D_filename = _resolve_filename_or_infer(
        nifti_directory=nifti_directory,
        default_candidates=flair_3D_defaults,
        infer_fn=lambda d: (
            _infer_structural_filename(
                d,
                include_keywords=("flair",),
                exclude_keywords=("dce", "dynamic", "dyn"),
            )
            or t2_3D_filename
            or t1_3D_filename
            or _largest_3d_nifti(d, min_slices=32, exclude_keywords=("dce", "dynamic", "dyn", "perfusion"))
        ),
    )
    axial_flair_3D_filename = 'Ax_VWIPcs_3D_Brain_VIEW_FLAIR_SHC.nii'

    axial_t2_2D_defaults = (
        "WIPAxT2TSEmatrix.nii",
        "WIPAxT2TSEmatrix.nii.gz",
    )
    axial_t2_2D_filename = _resolve_filename_or_infer(
        nifti_directory=nifti_directory,
        default_candidates=axial_t2_2D_defaults,
        infer_fn=lambda d: _infer_structural_filename(
            d,
            include_keywords=("t2", "tse"),
            exclude_keywords=("flair", "dce", "dynamic", "dyn"),
        )
        or t2_3D_filename,
    )

    dce_filename_primary = 'WIPhperf120long.nii'
    dce_filename_fallback = 'WIPDelRec-hperf120long.nii'
    dce_defaults = (
        dce_filename_fallback,
        dce_filename_primary,
        f"{dce_filename_fallback}.gz",
        f"{dce_filename_primary}.gz",
        dce_filename_fallback.replace(".nii", ".nii.gz"),
        dce_filename_primary.replace(".nii", ".nii.gz"),
    )
    diffusion_candidates = ordered_diffusion_filenames()

    diffusion_filename = get_diffusion_filename(diffusion_candidates, nifti_directory)
    dce_filename = _resolve_filename_or_infer(
        nifti_directory=nifti_directory,
        default_candidates=dce_defaults,
        infer_fn=_infer_dce_filename,
    )

    return (
        t1_3D_filename,
        axial_t1_3D_filename,
        t2_3D_filename,
        axial_t2_3D_filename,
        flair_3D_filename,
        axial_flair_3D_filename,
        axial_t2_2D_filename,
        diffusion_filename,
        dce_filename,
    )

