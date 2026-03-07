"""
Pure-Python neuroimaging utilities for Windows.

Replaces FreeSurfer CLI tools (mri_convert, mri_binarize) and FSL tools
(flirt, fslmaths) with nibabel / numpy / scipy equivalents.

These functions produce identical outputs to their CLI counterparts when
given the same inputs.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Union

import nibabel as nib
import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
#  FreeSurfer label ID sets
#  (sourced from FreeSurferColorLUT.txt)
# ---------------------------------------------------------------------------

# --all-wm:  Left/Right Cerebral WM + Left/Right Cerebellar WM + CC segments
ALL_WM_LABELS: set[int] = {
    2, 41,           # Left / Right Cerebral WM
    7, 46,           # Left / Right Cerebellar WM
    77, 78, 79,      # WM-hypointensities, old labels kept for compat
    251, 252, 253, 254, 255,  # Corpus Callosum (Posterior → Anterior)
}

# --gm:  All grey-matter labels (cortex + subcortical + cerebellum + brainstem)
GM_LABELS: set[int] = {
    3, 42,           # Left / Right Cerebral Cortex
    8, 47,           # Left / Right Cerebellar Cortex (GM)
    10, 49,          # Left / Right Thalamus
    11, 50,          # Left / Right Caudate
    12, 51,          # Left / Right Putamen
    13, 52,          # Left / Right Pallidum
    17, 53,          # Left / Right Hippocampus
    18, 54,          # Left / Right Amygdala
    26, 58,          # Left / Right Accumbens area
    28, 60,          # Left / Right Ventral DC
    16,              # Brain-Stem
}

# --subcort-gm:  Subcortical grey-matter (thalamus, caudate, putamen, pallidum,
#                hippocampus, amygdala, accumbens, ventral DC, brainstem,
#                cerebellum GM)
SUBCORT_GM_LABELS: set[int] = {
    10, 49,          # Thalamus
    11, 50,          # Caudate
    12, 51,          # Putamen
    13, 52,          # Pallidum
    17, 53,          # Hippocampus
    18, 54,          # Amygdala
    26, 58,          # Accumbens
    28, 60,          # Ventral DC
    16,              # Brain-Stem
    8, 47,           # Cerebellar Cortex
}


# ---------------------------------------------------------------------------
#  mri_convert  ➜  mgz_to_nifti / nifti_to_nifti
# ---------------------------------------------------------------------------

def mgz_to_nifti(input_path: str, output_path: str) -> str:
    """Convert a FreeSurfer .mgz file to NIfTI (.nii.gz).

    Uses nibabel which handles MGZ natively.  The output is identical to
    ``mri_convert <in.mgz> <out.nii.gz>``.
    """
    img = nib.load(input_path)
    # Preserve data type (int32 for label maps, float for intensity volumes)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    nib.save(img, output_path)
    print(f"[neuroimaging] Converted {input_path} → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
#  mri_binarize  ➜  binarize_labels
# ---------------------------------------------------------------------------

def binarize_labels(
    input_path: str,
    output_path: str,
    *,
    match_labels: Optional[Sequence[int]] = None,
    all_wm: bool = False,
    gm: bool = False,
    subcort_gm: bool = False,
) -> str:
    """Create a binary mask from a label volume, mirroring ``mri_binarize``.

    Exactly one of the keyword groups must be used:
      - ``match_labels=[16]``   →  ``--match 16``
      - ``all_wm=True``         →  ``--all-wm``
      - ``gm=True``             →  ``--gm``
      - ``subcort_gm=True``     →  ``--subcort-gm``
    """
    img = nib.load(input_path)
    data = np.asarray(img.dataobj)

    if match_labels is not None:
        labels = set(int(l) for l in match_labels)
    elif all_wm:
        labels = ALL_WM_LABELS
    elif gm:
        labels = GM_LABELS
    elif subcort_gm:
        labels = SUBCORT_GM_LABELS
    else:
        raise ValueError("Specify match_labels, all_wm, gm, or subcort_gm")

    mask = np.isin(data, list(labels)).astype(np.uint8)

    out_img = nib.Nifti1Image(mask, img.affine, img.header)
    out_img.set_data_dtype(np.uint8)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    nib.save(out_img, output_path)
    print(f"[neuroimaging] Binarized {len(labels)} labels → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
#  fslmaths  ➜  nifti_math
# ---------------------------------------------------------------------------

def nifti_sub(a_path: str, b_path: str, output_path: str) -> str:
    """``fslmaths <A> -sub <B> <out>``"""
    a_img = nib.load(a_path)
    b_img = nib.load(b_path)
    result = a_img.get_fdata() - b_img.get_fdata()
    out = nib.Nifti1Image(result, a_img.affine, a_img.header)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    nib.save(out, output_path)
    return output_path


def nifti_thr_bin(input_path: str, output_path: str, threshold: float = 0.5) -> str:
    """``fslmaths <in> -thr <t> -bin <out>``"""
    img = nib.load(input_path)
    data = img.get_fdata()
    mask = (data >= threshold).astype(np.uint8)
    out = nib.Nifti1Image(mask, img.affine, img.header)
    out.set_data_dtype(np.uint8)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    nib.save(out, output_path)
    return output_path


def fslmaths_chain(
    input_path: str,
    operations: list,
    output_path: str,
) -> str:
    """Execute a chain of fslmaths-style operations.

    ``operations`` is a list of (op, arg) tuples:
      - ("-sub", "/path/to/other.nii.gz")
      - ("-thr", 0.5)
      - ("-bin", None)
    """
    img = nib.load(input_path)
    data = img.get_fdata().astype(np.float64)

    for op, arg in operations:
        if op == "-sub":
            other = nib.load(arg).get_fdata().astype(np.float64)
            data = data - other
        elif op == "-add":
            other = nib.load(arg).get_fdata().astype(np.float64)
            data = data + other
        elif op == "-mul":
            if isinstance(arg, str) and os.path.exists(arg):
                other = nib.load(arg).get_fdata().astype(np.float64)
                data = data * other
            else:
                data = data * float(arg)
        elif op == "-div":
            if isinstance(arg, str) and os.path.exists(arg):
                other = nib.load(arg).get_fdata().astype(np.float64)
                data = np.divide(data, other, out=np.zeros_like(data),
                                 where=(other != 0))
            else:
                data = data / float(arg)
        elif op == "-thr":
            data[data < float(arg)] = 0
        elif op == "-bin":
            data = (data > 0).astype(np.float64)
        elif op == "-abs":
            data = np.abs(data)
        else:
            raise ValueError(f"Unsupported fslmaths operation: {op}")

    # Determine output dtype
    if all(op in ("-bin", "-thr") for op, _ in operations[-2:] if op in ("-bin",)):
        out_dtype = np.uint8
    else:
        out_dtype = np.float32
    out_data = data.astype(out_dtype)
    out_img = nib.Nifti1Image(out_data, img.affine, img.header)
    out_img.set_data_dtype(out_dtype)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    nib.save(out_img, output_path)
    return output_path


# ---------------------------------------------------------------------------
#  flirt  ➜  affine_coregister
# ---------------------------------------------------------------------------

def affine_coregister(
    input_path: str,
    reference_path: str,
    output_path: str,
    *,
    interp: str = "nearestneighbour",
    use_sform: bool = True,
    dof: int = 6,
    omat_path: Optional[str] = None,
) -> str:
    """Align ``input`` to ``reference`` space using affine transforms.

    This replaces two FLIRT invocations:
      1. ``flirt -in <in> -ref <ref> -applyxfm -usesqform -interp nn -out <out>``
      2. ``flirt -in <in> -ref <ref> -omat <mat> -dof 6`` followed by
         ``flirt -in <in> -ref <ref> -applyxfm -init <mat> -interp nn -out <out>``

    When ``use_sform=True`` (default) the sform/qform affines embedded in the
    NIfTI headers are used — no iterative registration.  This matches the
    macOS pipeline's default (``use_flirt_registration = False``).
    """
    in_img = nib.load(input_path)
    ref_img = nib.load(reference_path)

    in_data = np.asarray(in_img.dataobj)
    in_affine = in_img.affine
    ref_affine = ref_img.affine
    ref_shape = ref_img.shape[:3]

    # Compute the voxel-space transform: in_vox → world → ref_vox
    # ref_vox = inv(ref_affine) @ in_affine @ in_vox
    vox_transform = np.linalg.inv(ref_affine) @ in_affine

    # Choose interpolation order
    order = 0 if interp == "nearestneighbour" else 1

    # Apply the affine transform
    # scipy.ndimage.affine_transform expects the *inverse* map from output to input
    # output_vox → input_vox  =  inv(vox_transform)
    inv_transform = np.linalg.inv(vox_transform)

    # Handle 3D or 4D data
    if in_data.ndim == 3:
        out_data = ndimage.affine_transform(
            in_data.astype(np.float64),
            inv_transform[:3, :3],
            offset=inv_transform[:3, 3],
            output_shape=ref_shape,
            order=order,
            mode="constant",
            cval=0.0,
        )
    elif in_data.ndim == 4:
        n_vols = in_data.shape[3]
        out_data = np.zeros((*ref_shape, n_vols), dtype=np.float64)
        for v in range(n_vols):
            out_data[..., v] = ndimage.affine_transform(
                in_data[..., v].astype(np.float64),
                inv_transform[:3, :3],
                offset=inv_transform[:3, 3],
                output_shape=ref_shape,
                order=order,
                mode="constant",
                cval=0.0,
            )
    else:
        raise ValueError(f"Expected 3D or 4D input, got {in_data.ndim}D")

    # Cast back to input dtype for label maps
    if order == 0:
        out_data = np.round(out_data).astype(in_data.dtype)

    out_img = nib.Nifti1Image(out_data, ref_affine)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    nib.save(out_img, output_path)
    print(f"[neuroimaging] Coregistered {input_path} → {output_path}  "
          f"(ref={os.path.basename(reference_path)}, interp={interp})")

    # Optionally write a .mat-style text matrix
    if omat_path:
        np.savetxt(omat_path, vox_transform, fmt="%.8f")

    return output_path


# ---------------------------------------------------------------------------
#  High-level helpers (match the macOS pipeline's calling patterns)
# ---------------------------------------------------------------------------

def create_all_masks(
    aseg_nii_path: str,
    output_dir: Optional[str] = None,
    *,
    force: bool = False,
) -> dict[str, str]:
    """Create all brain tissue masks from a FreeSurfer-compatible aseg NIfTI.

    Mirrors the mask-creation block inside ``segmentation()`` in
    ``AI_tissue_functions.py``.

    Returns a dict of mask-name → file-path.
    """
    if output_dir is None:
        output_dir = os.path.dirname(aseg_nii_path)

    def _out(name: str) -> str:
        return os.path.join(output_dir, name)

    paths: dict[str, str] = {}

    # ---- Individual label masks ----
    masks_spec: list[tuple[str, dict]] = [
        ("temp_wm.nii.gz", dict(all_wm=True)),
        ("temp_subcortical_gm.nii.gz", dict(subcort_gm=True)),
        ("gm.nii.gz", dict(gm=True)),
        ("gm_brainstem.nii.gz", dict(match_labels=[16])),
        ("gm_cerebellum.nii.gz", dict(match_labels=[8, 47])),
        ("wm_cerebellum.nii.gz", dict(match_labels=[7, 46])),
        ("wm_cc.nii.gz", dict(match_labels=[251, 252, 253, 254, 255])),
    ]

    for name, kwargs in masks_spec:
        p = _out(name)
        if force or not os.path.exists(p):
            binarize_labels(aseg_nii_path, p, **kwargs)
        paths[name.replace(".nii.gz", "")] = p

    # ---- Composite masks (fslmaths equivalent) ----

    # cortical_gm = gm - subcort_gm - brainstem - cerebellum_gm  →  thr 0.5 → bin
    cortical_p = _out("cortical_gm.nii.gz")
    if force or not os.path.exists(cortical_p):
        fslmaths_chain(
            _out("gm.nii.gz"),
            [
                ("-sub", _out("temp_subcortical_gm.nii.gz")),
                ("-sub", _out("gm_brainstem.nii.gz")),
                ("-sub", _out("gm_cerebellum.nii.gz")),
                ("-thr", 0.5),
                ("-bin", None),
            ],
            cortical_p,
        )
    paths["cortical_gm"] = cortical_p

    # subcortical_gm = temp_subcort - brainstem - cerebellum → thr 0.5 → bin
    subcort_p = _out("subcortical_gm.nii.gz")
    if force or not os.path.exists(subcort_p):
        fslmaths_chain(
            _out("temp_subcortical_gm.nii.gz"),
            [
                ("-sub", _out("gm_brainstem.nii.gz")),
                ("-sub", _out("gm_cerebellum.nii.gz")),
                ("-thr", 0.5),
                ("-bin", None),
            ],
            subcort_p,
        )
    paths["subcortical_gm"] = subcort_p

    # wm = temp_wm - cerebellum_wm - cc → thr 0.5 → bin
    wm_p = _out("wm.nii.gz")
    if force or not os.path.exists(wm_p):
        fslmaths_chain(
            _out("temp_wm.nii.gz"),
            [
                ("-sub", _out("wm_cerebellum.nii.gz")),
                ("-sub", _out("wm_cc.nii.gz")),
                ("-thr", 0.5),
                ("-bin", None),
            ],
            wm_p,
        )
    paths["wm"] = wm_p

    # ---- Clean up temp files ----
    for tmp in ("temp_wm.nii.gz", "temp_subcortical_gm.nii.gz", "gm.nii.gz"):
        p = _out(tmp)
        if os.path.exists(p):
            os.remove(p)
            # Also remove from paths dict (keep composite versions)
            key = tmp.replace(".nii.gz", "")
            paths.pop(key, None)

    return paths


def create_coregistered_masks(
    aseg_nii_path: str,
    ref_path: str,
    space_tag: str,
    *,
    force: bool = False,
) -> dict[str, str]:
    """Coregister aseg → ref space, then create all masks in that space.

    ``space_tag`` should be 'DCE' or 'T2'.

    Returns a dict of mask-name → file-path.
    """
    out_dir = os.path.dirname(aseg_nii_path)
    aseg_in_ref = aseg_nii_path.replace(".nii.gz", f"_in_{space_tag}.nii.gz")

    if force or not os.path.exists(aseg_in_ref):
        affine_coregister(
            aseg_nii_path, ref_path, aseg_in_ref,
            interp="nearestneighbour",
        )

    # Now create masks from the coregistered aseg
    # We'll build individual masks + composite masks, mirroring the macOS code
    def _out(name: str) -> str:
        return aseg_in_ref.replace(".nii.gz", f"_{name}.nii.gz")

    paths: dict[str, str] = {}

    masks_spec = [
        ("wm", dict(all_wm=True)),
        ("subcortical_gm", dict(subcort_gm=True)),
        ("gm", dict(gm=True)),
        ("gm_brainstem", dict(match_labels=[16])),
        ("gm_cerebellum", dict(match_labels=[8, 47])),
        ("wm_cerebellum", dict(match_labels=[7, 46])),
        ("wm_cc", dict(match_labels=[251, 252, 253, 254, 255])),
    ]

    for name, kwargs in masks_spec:
        p = _out(name)
        if force or not os.path.exists(p):
            binarize_labels(aseg_in_ref, p, **kwargs)
        paths[name] = p

    # cortical_gm = gm - subcort - brainstem - cerebellum → thr → bin
    cortical_p = _out("cortical_gm")
    if force or not os.path.exists(cortical_p):
        fslmaths_chain(
            _out("gm"),
            [
                ("-sub", _out("subcortical_gm")),
                ("-sub", _out("gm_brainstem")),
                ("-sub", _out("gm_cerebellum")),
                ("-thr", 0.5),
                ("-bin", None),
            ],
            cortical_p,
        )
    paths["cortical_gm"] = cortical_p

    # subcortical_gm (refined) = subcort - brainstem - cerebellum → thr → bin
    subcort_refined = _out("subcortical_gm")
    if force or not os.path.exists(subcort_refined):
        fslmaths_chain(
            paths["subcortical_gm"],
            [
                ("-sub", _out("gm_brainstem")),
                ("-sub", _out("gm_cerebellum")),
                ("-thr", 0.5),
                ("-bin", None),
            ],
            subcort_refined,
        )
    paths["subcortical_gm"] = subcort_refined

    # wm (refined) = wm - cerebellum_wm - cc → thr → bin
    wm_refined = _out("wm")
    if force or not os.path.exists(wm_refined):
        fslmaths_chain(
            paths["wm"],
            [
                ("-sub", _out("wm_cerebellum")),
                ("-sub", _out("wm_cc")),
                ("-thr", 0.5),
                ("-bin", None),
            ],
            wm_refined,
        )
    paths["wm"] = wm_refined

    # Remove temp gm mask
    gm_tmp = _out("gm")
    if os.path.exists(gm_tmp):
        os.remove(gm_tmp)
        paths.pop("gm", None)

    return paths
