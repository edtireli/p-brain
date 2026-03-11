"""p-brain tissue segmentation using a locally trained UNet3D.

Replaces FastSurfer / SynthSeg with a lightweight 6-class tissue model
that produces binary masks directly — no FreeSurfer CLI required.

Tissue classes:
    0 = background
    1 = cerebral white matter
    2 = cortical gray matter
    3 = subcortical gray matter
    4 = brainstem
    5 = cerebellum (WM + GM merged)

Usage inside p-brain:
    --segmentation pbrain

The model checkpoint is resolved from (in priority order):
    1. ``settings.PBRAIN_TISSUE_MODEL`` (if set)
    2. ``P_BRAIN_TISSUE_MODEL`` environment variable
    3. ``~/.pbrain/models/tissue_seg.pt``
    4. ``<p-brain-root>/AI/tissue_seg.pt``
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

# ---------------------------------------------------------------------------
# Tissue class IDs — must match training label order
# ---------------------------------------------------------------------------
TISSUE_BG = 0
TISSUE_CEREBRAL_WM = 1
TISSUE_CORTICAL_GM = 2
TISSUE_SUBCORTICAL_GM = 3
TISSUE_BRAINSTEM = 4
TISSUE_CEREBELLUM = 5
NUM_TISSUE = 6


def _resolve_model_path() -> str:
    """Find the tissue segmentation checkpoint."""
    import utils.settings as settings

    # 1. Explicit setting
    explicit = getattr(settings, "PBRAIN_TISSUE_MODEL", None)
    if explicit and os.path.isfile(explicit):
        return explicit

    # 2. Environment variable
    env = os.environ.get("P_BRAIN_TISSUE_MODEL", "").strip()
    if env and os.path.isfile(env):
        return env

    # 3. ~/.pbrain/models/tissue_seg.pt
    home_path = os.path.join(os.path.expanduser("~"), ".pbrain", "models", "tissue_seg.pt")
    if os.path.isfile(home_path):
        return home_path

    # 4. <p-brain-root>/AI/tissue_seg.pt
    pbrain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_path = os.path.join(pbrain_root, "AI", "tissue_seg.pt")
    if os.path.isfile(local_path):
        return local_path

    raise FileNotFoundError(
        "p-brain tissue model not found.  Place the checkpoint at one of:\n"
        f"  • {home_path}\n"
        f"  • {local_path}\n"
        "  • Set P_BRAIN_TISSUE_MODEL=/path/to/tissue_seg.pt\n"
    )


def _load_model(checkpoint_path: str, device: str = "cpu"):
    """Load the UNet3D tissue model from a training checkpoint."""
    import torch

    # Import model factory from the seg_dce package.
    # The package should be installed or on sys.path.
    try:
        from seg_dce.models import make_model
    except ImportError:
        # Fallback: try adding the DCE_segmentation directory
        dce_seg_dir = os.path.join(os.path.expanduser("~"), "DCE_segmentation")
        if os.path.isdir(dce_seg_dir) and dce_seg_dir not in sys.path:
            sys.path.insert(0, dce_seg_dir)
        from seg_dce.models import make_model

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    in_channels = int(ck["in_channels"])
    num_classes = int(ck["num_classes"])

    # Reconstruct model config from checkpoint
    model_cfg = ck.get("config", {}).get("model_cfg", {})
    base_channels = int(model_cfg.get("base_channels", 32))
    deep_supervision = bool(model_cfg.get("deep_supervision", False))
    dropout = float(model_cfg.get("dropout", 0.0))
    pool_z = bool(ck.get("config", {}).get("pool_z", False))

    model = make_model(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        model="unet3d",
        deep_supervision=deep_supervision,
        dropout=dropout,
        pool_z=pool_z,
    )
    model.load_state_dict(ck["model"], strict=True)
    model = model.to(device)
    model.eval()
    return model, num_classes


def _sliding_window_inference(
    model,
    volume: np.ndarray,
    patch_size: tuple[int, int, int] = (32, 128, 128),
    overlap: float = 0.5,
    device: str = "cpu",
) -> np.ndarray:
    """Run sliding-window inference on a 3D volume.

    Args:
        model: UNet3D in eval mode.
        volume: (D, H, W) float32 array, already normalised.
        patch_size: (pD, pH, pW).
        overlap: fractional overlap between patches.
        device: torch device string.

    Returns:
        (D, H, W) int16 array of predicted class labels.
    """
    import torch

    D, H, W = volume.shape
    pD, pH, pW = patch_size
    step_d = max(1, int(pD * (1 - overlap)))
    step_h = max(1, int(pH * (1 - overlap)))
    step_w = max(1, int(pW * (1 - overlap)))

    # Pad volume so it's at least patch_size in every dim, and dims are
    # multiples of 16 (UNet requirement for 4 pooling stages).
    pad_d = max(pD - D, 0)
    total_d = D + pad_d
    if total_d % 16 != 0:
        pad_d += 16 - (total_d % 16)
    pad_h = max(pH - H, 0)
    # Ensure H+pad_h is multiple of 16
    total_h = H + pad_h
    if total_h % 16 != 0:
        pad_h += 16 - (total_h % 16)
    pad_w = max(pW - W, 0)
    total_w = W + pad_w
    if total_w % 16 != 0:
        pad_w += 16 - (total_w % 16)

    vol = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
    vD, vH, vW = vol.shape

    # Accumulate softmax probabilities and counts
    accum = np.zeros((NUM_TISSUE, vD, vH, vW), dtype=np.float32)
    counts = np.zeros((vD, vH, vW), dtype=np.float32)

    # Generate patch positions
    d_starts = list(range(0, vD - pD + 1, step_d))
    if d_starts[-1] + pD < vD:
        d_starts.append(vD - pD)
    h_starts = list(range(0, vH - pH + 1, step_h))
    if h_starts[-1] + pH < vH:
        h_starts.append(vH - pH)
    w_starts = list(range(0, vW - pW + 1, step_w))
    if w_starts[-1] + pW < vW:
        w_starts.append(vW - pW)

    with torch.no_grad():
        for d0 in d_starts:
            for h0 in h_starts:
                for w0 in w_starts:
                    patch = vol[d0:d0 + pD, h0:h0 + pH, w0:w0 + pW]
                    x = torch.from_numpy(patch[np.newaxis, np.newaxis]).float().to(device)
                    logits = model(x)
                    if isinstance(logits, list):
                        logits = logits[0]  # full-res output
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (C, pD, pH, pW)
                    accum[:, d0:d0 + pD, h0:h0 + pH, w0:w0 + pW] += probs
                    counts[d0:d0 + pD, h0:h0 + pH, w0:w0 + pW] += 1.0

    # Average and argmax
    counts = np.maximum(counts, 1.0)
    accum /= counts[np.newaxis]
    pred = accum.argmax(axis=0).astype(np.int16)

    # Remove padding
    pred = pred[:D, :H, :W]
    return pred


def _normalise_volume(data: np.ndarray) -> np.ndarray:
    """Normalise a 3D volume to zero-mean unit-variance (brain region only)."""
    data = data.astype(np.float32)
    mask = data > 0
    if mask.any():
        mu = data[mask].mean()
        sd = data[mask].std()
        if sd > 1e-8:
            data = (data - mu) / sd
    return data


def _write_mask(data: np.ndarray, affine: np.ndarray, path: str) -> None:
    """Save a boolean mask as a NIfTI."""
    img = nib.Nifti1Image(data.astype(np.uint8), affine)
    nib.save(img, path)


def _write_label_volume(data: np.ndarray, affine: np.ndarray, path: str) -> None:
    """Save an integer label volume as a NIfTI."""
    img = nib.Nifti1Image(data.astype(np.int16), affine)
    nib.save(img, path)


def _register_and_apply(
    t1_path: str,
    ref_path: str,
    seg_path: str,
    out_seg_path: str,
    out_mat_path: str,
    *,
    reuse_mat: bool = True,
) -> None:
    """Compute rigid registration from T1→ref, apply to seg volume.

    Step 1: flirt -in T1 -ref ref -omat mat -dof 6
    Step 2: flirt -in seg -ref ref -applyxfm -init mat -interp nearestneighbour -out out
    """
    # Reuse existing transform if available
    if reuse_mat and os.path.isfile(out_mat_path):
        print(f"  Reusing existing registration matrix: {out_mat_path}")
    else:
        print(f"  Computing rigid registration: T1 → {os.path.basename(ref_path)}")
        cmd_reg = [
            "flirt",
            "-in", t1_path,
            "-ref", ref_path,
            "-omat", out_mat_path,
            "-dof", "6",
        ]
        result = subprocess.run(cmd_reg, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FLIRT registration failed (exit {result.returncode}): "
                f"{(result.stderr or result.stdout or '').strip()[-500:]}"
            )

    print(f"  Applying transform to tissue segmentation → {os.path.basename(out_seg_path)}")
    cmd_apply = [
        "flirt",
        "-in", seg_path,
        "-ref", ref_path,
        "-applyxfm",
        "-init", out_mat_path,
        "-interp", "nearestneighbour",
        "-out", out_seg_path,
    ]
    result = subprocess.run(cmd_apply, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FLIRT apply failed (exit {result.returncode}): "
            f"{(result.stderr or result.stdout or '').strip()[-500:]}"
        )


def _extract_masks_from_labels(
    label_vol: np.ndarray,
    affine: np.ndarray,
    output_dir: str,
) -> dict[str, np.ndarray]:
    """Extract binary tissue masks from a 6-class label volume and save them.

    The model predicts a single 'cerebellum' class (merged WM+GM).  For backward
    compatibility with the 14-tuple interface, we emit both ``gm_cerebellum`` and
    ``wm_cerebellum`` pointing to the same unified cerebellum mask.

    Returns a dict of {mask_name: bool_array}.
    """
    cerebellum_mask = (label_vol == TISSUE_CEREBELLUM)
    masks = {
        "wm":               (label_vol == TISSUE_CEREBRAL_WM),
        "cortical_gm":      (label_vol == TISSUE_CORTICAL_GM),
        "subcortical_gm":   (label_vol == TISSUE_SUBCORTICAL_GM),
        "gm_brainstem":     (label_vol == TISSUE_BRAINSTEM),
        "gm_cerebellum":    cerebellum_mask,   # unified cerebellum
        "wm_cerebellum":    cerebellum_mask,   # same mask for compat
        "wm_cc":            np.zeros_like(label_vol, dtype=bool),  # empty — CC is inside cerebral_wm
    }

    for name, data in masks.items():
        path = os.path.join(output_dir, f"{name}.nii.gz")
        _write_mask(data, affine, path)

    return masks


def run_pbrain_segmentation(
    t1_path: str,
    dce_path: str,
    t2_path: Optional[str],
    seg_dir: str,
    *,
    rerun: bool = False,
    apple_metal: bool = True,
) -> tuple:
    """Run tissue segmentation and produce all masks p-brain expects.

    This replaces both ``segmentation()`` and ``coregistration()`` when
    ``--segmentation pbrain`` is used.

    Returns the same 14-tuple as ``coregistration()``:
        (wm_mask_t2, wm_mask_dce, cortical_gm_mask_t2, cortical_gm_mask_dce,
         subcortical_gm_mask_t2, subcortical_gm_mask_dce,
         gm_brainstem_mask_t2, gm_brainstem_mask_dce,
         gm_cerebellum_mask_t2, gm_cerebellum_mask_dce,
         wm_cerebellum_mask_t2, wm_cerebellum_mask_dce,
         wm_cc_mask_t2, wm_cc_mask_dce)
    """
    import torch

    mri_dir = os.path.join(seg_dir, "segmentation", "mri")
    os.makedirs(mri_dir, exist_ok=True)

    # Output paths
    tissue_t1_path = os.path.join(mri_dir, "tissue_seg.nii.gz")
    tissue_dce_path = os.path.join(mri_dir, "tissue_seg_in_DCE.nii.gz")
    tissue_t2_path = os.path.join(mri_dir, "tissue_seg_in_T2.nii.gz")
    mat_dce_path = os.path.join(mri_dir, "T1_to_DCE.mat")
    mat_t2_path = os.path.join(mri_dir, "T1_to_T2.mat")

    # Determine device
    if apple_metal and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # ── Step 1: Inference on T1 ──────────────────────────────────────────
    if rerun or not os.path.isfile(tissue_t1_path):
        print("[pbrain-seg] Loading tissue segmentation model...")
        model_path = _resolve_model_path()
        print(f"  Checkpoint: {model_path}")
        model, num_classes = _load_model(model_path, device=device)
        assert num_classes == NUM_TISSUE, (
            f"Model has {num_classes} classes, expected {NUM_TISSUE}"
        )

        # Load and reconstruct checkpoint patch_size
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        patch_size = tuple(ck.get("config", {}).get("patch_size", [32, 128, 128]))

        print(f"[pbrain-seg] Running inference on T1: {t1_path}")
        print(f"  Device: {device}, patch_size: {patch_size}")
        t1_img = nib.load(t1_path)
        t1_data = np.asanyarray(t1_img.dataobj, dtype=np.float32)
        t1_norm = _normalise_volume(t1_data)

        pred = _sliding_window_inference(
            model, t1_norm,
            patch_size=patch_size,
            overlap=0.5,
            device=device,
        )

        _write_label_volume(pred, t1_img.affine, tissue_t1_path)
        voxel_counts = {i: int((pred == i).sum()) for i in range(NUM_TISSUE)}
        total_brain = sum(v for k, v in voxel_counts.items() if k > 0)
        print(f"[pbrain-seg] Inference complete — {total_brain:,} brain voxels")
        for i, name in enumerate(["bg", "wm", "cortical_gm", "subcortical_gm",
                                   "brainstem", "cerebellum"]):
            pct = 100 * voxel_counts[i] / max(pred.size, 1)
            print(f"  {name:18s}: {voxel_counts[i]:>10,}  ({pct:.1f}%)")

        del model, ck
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    else:
        print(f"[pbrain-seg] Tissue segmentation already exists: {tissue_t1_path}")

    # ── Step 2: Extract masks in T1 space ────────────────────────────────
    t1_masks_dir = mri_dir
    t1_seg_img = nib.load(tissue_t1_path)
    t1_labels = np.asarray(t1_seg_img.dataobj, dtype=np.int16)
    t1_affine = t1_seg_img.affine
    _extract_masks_from_labels(t1_labels, t1_affine, t1_masks_dir)
    print("[pbrain-seg] T1-space masks written.")

    # ── Step 3: Register to DCE space ────────────────────────────────────
    print("[pbrain-seg] Registering to DCE space...")
    _register_and_apply(
        t1_path=t1_path,
        ref_path=dce_path,
        seg_path=tissue_t1_path,
        out_seg_path=tissue_dce_path,
        out_mat_path=mat_dce_path,
        reuse_mat=(not rerun),
    )
    dce_seg_img = nib.load(tissue_dce_path)
    dce_labels = np.asarray(dce_seg_img.dataobj, dtype=np.int16)
    # Re-round after FLIRT (NN interp should keep integers but just in case)
    dce_labels = np.round(dce_labels).astype(np.int16)
    dce_affine = dce_seg_img.affine
    dce_masks = _extract_masks_from_labels(dce_labels, dce_affine, mri_dir)
    # Rename DCE masks to match p-brain naming convention
    for name in list(dce_masks.keys()):
        src = os.path.join(mri_dir, f"{name}.nii.gz")
        dst = os.path.join(mri_dir, f"tissue_seg_in_DCE_{name}.nii.gz")
        if os.path.isfile(src):
            os.replace(src, dst)
    print("[pbrain-seg] DCE-space masks written.")

    # ── Step 4: Register to T2 space (if available) ──────────────────────
    has_t2 = bool(t2_path and os.path.isfile(t2_path))
    if has_t2:
        print("[pbrain-seg] Registering to T2 space...")
        _register_and_apply(
            t1_path=t1_path,
            ref_path=t2_path,
            seg_path=tissue_t1_path,
            out_seg_path=tissue_t2_path,
            out_mat_path=mat_t2_path,
            reuse_mat=(not rerun),
        )
        t2_seg_img = nib.load(tissue_t2_path)
        t2_labels = np.round(np.asarray(t2_seg_img.dataobj, dtype=np.float32)).astype(np.int16)
        t2_affine = t2_seg_img.affine
        t2_masks = _extract_masks_from_labels(t2_labels, t2_affine, mri_dir)
        for name in list(t2_masks.keys()):
            src = os.path.join(mri_dir, f"{name}.nii.gz")
            dst = os.path.join(mri_dir, f"tissue_seg_in_T2_{name}.nii.gz")
            if os.path.isfile(src):
                os.replace(src, dst)
        print("[pbrain-seg] T2-space masks written.")

    # ── Step 5: Load and return masks ────────────────────────────────────
    def _load_bool(path):
        return nib.load(path).get_fdata().astype(bool)

    mask_names = ["wm", "cortical_gm", "subcortical_gm",
                  "gm_brainstem", "gm_cerebellum", "wm_cerebellum", "wm_cc"]

    dce_mask_arrays = {}
    for name in mask_names:
        dce_mask_arrays[name] = _load_bool(
            os.path.join(mri_dir, f"tissue_seg_in_DCE_{name}.nii.gz")
        )

    if has_t2:
        t2_mask_arrays = {}
        for name in mask_names:
            t2_mask_arrays[name] = _load_bool(
                os.path.join(mri_dir, f"tissue_seg_in_T2_{name}.nii.gz")
            )
    else:
        # When T2 is missing, return DCE masks as stand-ins.
        # The main pipeline will use voxelwise_only when T2 is absent anyway.
        t2_mask_arrays = dce_mask_arrays

    # Return the same 14-tuple as coregistration()
    return (
        t2_mask_arrays["wm"],               dce_mask_arrays["wm"],
        t2_mask_arrays["cortical_gm"],      dce_mask_arrays["cortical_gm"],
        t2_mask_arrays["subcortical_gm"],   dce_mask_arrays["subcortical_gm"],
        t2_mask_arrays["gm_brainstem"],     dce_mask_arrays["gm_brainstem"],
        t2_mask_arrays["gm_cerebellum"],    dce_mask_arrays["gm_cerebellum"],
        t2_mask_arrays["wm_cerebellum"],    dce_mask_arrays["wm_cerebellum"],
        t2_mask_arrays["wm_cc"],            dce_mask_arrays["wm_cc"],
    )
