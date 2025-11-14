"""Utilities to compute and visualise diffusion tractography streamlines."""

from __future__ import annotations
from functools import reduce, wraps

import inspect
import multiprocessing as mp
import os
import queue
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

try:  # ``fury`` is optional – fall back to matplotlib when unavailable.
    from dipy.viz import actor as dipy_actor
    from dipy.viz import colormap as dipy_colormap
    from dipy.viz import window as dipy_window

    _FURY_AVAILABLE = True
except Exception:  # pragma: no cover - fury is optional
    _FURY_AVAILABLE = False

import nibabel as nib
from nibabel import streamlines as nib_streamlines

try:  # ``resample_from_to`` is only needed for anatomical overlays.
    from nibabel.processing import resample_from_to
except ImportError:  # pragma: no cover - helper ships with supported nibabel
    resample_from_to = None

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.dti import TensorModel
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.utils import seeds_from_mask

from modules.opt08_fa import find_dwi_files


def _canonical_affine(affine: np.ndarray) -> np.ndarray:
    """Return a 4x4 voxel-to-world affine derived from ``affine``."""

    arr = np.asarray(affine, dtype=np.float64)
    if arr.shape == (4, 4):
        return np.array(arr, copy=True)

    if arr.ndim != 2 or arr.shape[0] < 4 or arr.shape[1] < 4:
        raise ValueError("Expected affine with at least 4x4 elements")

    canonical = np.eye(4, dtype=arr.dtype)
    canonical[:3, :3] = arr[:3, :3]
    canonical[:3, 3] = arr[:3, 3]
    return canonical

@dataclass
class TractographyOutputs:
    """Container describing generated tractography artefacts."""

    tract_path: str
    render_path: Optional[str] = None
    montage_path: Optional[str] = None


# ---------------------------------------------------------------------------
# DWI normalisation helpers
# ---------------------------------------------------------------------------


def _as_4d_dwi(data: np.ndarray, bvals: np.ndarray) -> np.ndarray:
    """
    Return a 4-D array shaped (X, Y, Z, N) from potentially 5-D inputs.
    We pick the gradient dimension by matching the axis whose length equals
    len(bvals). Any remaining singleton axes beyond Z are squeezed.
    """

    arr = np.asarray(data)
    if arr.ndim == 4:
        return arr
    if arr.ndim < 4:
        raise ValueError(f"DWI must be >=4-D, got {arr.ndim}D")

    grad_len = int(bvals.size)
    grad_axis = None
    for ax in range(3, arr.ndim):
        if arr.shape[ax] == grad_len:
            grad_axis = ax
            break
    if grad_axis is None:
        grad_axis = arr.ndim - 1

    arr = np.moveaxis(arr, grad_axis, -1)

    if arr.ndim > 4:
        to_squeeze: list[int] = []
        for ax in range(3, arr.ndim - 1):
            if arr.shape[ax] == 1:
                to_squeeze.append(ax)
        if to_squeeze:
            arr = np.squeeze(arr, axis=tuple(to_squeeze))

    if arr.ndim != 4:
        raise ValueError(
            f"Failed to coerce DWI to 4-D. Shape after coercion: {arr.shape}"
        )
    return arr


def _shape_str(a: np.ndarray) -> str:
    try:
        return "x".join(map(str, a.shape))
    except Exception:
        return "<unknown>"


# ---------------------------------------------------------------------------
# Streamline sanitisation helpers
# ---------------------------------------------------------------------------


def _to_xyz(points: np.ndarray) -> np.ndarray:
    """
    Return Nx3 float32 coordinates from points with possible extra columns.
    If a 4th column exists, treat it as homogeneous w and dehomogenise.
    """

    P = np.asarray(points)
    if P.ndim != 2:
        return P
    if P.shape[1] == 3:
        return P.astype(np.float32)
    if P.shape[1] >= 4:
        w = P[:, 3]
        w = np.where((w == 0) | (~np.isfinite(w)), 1.0, w)
        out = (P[:, :3] / w[:, None]).astype(np.float32)
        return out
    # fewer than 3 columns, pass through
    return P.astype(np.float32)


def _coerce_streamlines_xyz(sls: Streamlines) -> Streamlines:
    """Ensure every streamline is Nx3 and has at least two points."""

    out = Streamlines()
    for sl in sls:
        xyz = _to_xyz(sl)
        if xyz.shape[0] >= 2 and xyz.shape[1] == 3:
            out.append(xyz)
    return out


def _load_background_volume(
    nifti_path: str,
    reference_img: nib.Nifti1Image,
) -> nib.Nifti1Image:
    """Return a background volume resampled into ``reference_img`` space."""

    img = nib.load(nifti_path)
    if img.shape[:3] == reference_img.shape[:3]:
        return img

    if resample_from_to is None:
        raise RuntimeError(
            "nibabel.processing.resample_from_to is required for anatomical overlays"
        )

    resampled = resample_from_to(img, reference_img, order=1)
    return resampled


def _ensure_image_directory(
    image_directory: Optional[str], analysis_directory: str
) -> Optional[str]:
    """Return a usable image directory, falling back to ``analysis_directory``."""

    if image_directory:
        os.makedirs(image_directory, exist_ok=True)
        return image_directory

    parent = os.path.dirname(os.path.abspath(analysis_directory))
    fallback = os.path.join(parent, "Images")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def _render_with_fury(
    streamlines: Streamlines,
    output_path: str,
    *,
    background_color: tuple[float, float, float] = (0.02, 0.02, 0.05),
) -> None:
    """Render colourful streamlines using ``fury`` for rich shading and lighting."""

    scene = dipy_window.Scene()
    scene.background(background_color)

    colours = dipy_colormap.line_colors(streamlines)
    stream_actor = dipy_actor.line(
        streamlines,
        colors=colours,
        linewidth=1.0,
    )
    scene.add(stream_actor)

    # Gently rotate the scene so fibres are rendered with depth cues.
    scene.reset_camera()
    scene.pitch(-15)
    scene.yaw(20)
    scene.zoom(1.2)

    snapshot_kwargs = {"fname": output_path, "size": (1600, 1600)}
    try:
        signature = inspect.signature(dipy_window.snapshot)
    except (TypeError, ValueError):  # pragma: no cover - rare introspection issues
        signature = None
    if signature is not None and "offscreen" in signature.parameters:
        snapshot_kwargs["offscreen"] = True

    try:
        dipy_window.snapshot(scene, **snapshot_kwargs)
    except Exception as exc:  # noqa: BLE001 - propagate to caller for fallback
        raise RuntimeError("FURY snapshot failed") from exc
    finally:
        clear = getattr(scene, "clear", None)
        if callable(clear):  # pragma: no branch - safety guard for older fury
            clear()


def _fury_render_worker(
    tract_path: str,
    output_path: str,
    background_color: tuple[float, float, float],
    result_queue,
) -> None:
    """Helper executed in a subprocess to isolate FURY crashes."""

    try:
        tractogram = nib_streamlines.load(tract_path).tractogram
        streamlines = _coerce_streamlines_xyz(Streamlines(tractogram.streamlines))
        _render_with_fury(
            streamlines,
            output_path,
            background_color=background_color,
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 - propagate message to parent process
        result_queue.put((False, str(exc)))
    else:
        result_queue.put((True, None))


def _env_flag_disabled(value: str) -> bool:
    """Return ``True`` when ``value`` represents a disabled boolean flag."""

    return value.strip().lower() in {"0", "false", "no", "off"}


def _should_use_fury() -> bool:
    """Determine whether FURY rendering should be attempted."""

    if not _FURY_AVAILABLE:
        return False

    flag = os.environ.get("P_BRAIN_ENABLE_FURY", "")
    if flag:
        return not _env_flag_disabled(flag)

    disable_flag = os.environ.get("P_BRAIN_DISABLE_FURY", "")
    if disable_flag:
        return not disable_flag.strip().lower() in {"1", "true", "yes", "on"}

    return True


def _render_with_fury_isolated(
    tract_path: str,
    output_path: str,
    *,
    background_color: tuple[float, float, float] = (0.02, 0.02, 0.05),
) -> bool:
    """Render streamlines via FURY inside a subprocess, returning success."""

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_fury_render_worker,
        args=(tract_path, output_path, background_color, result_queue),
    )
    process.start()
    process.join()

    success = False
    message: Optional[str] = None

    if process.exitcode == 0:
        try:
            success, message = result_queue.get_nowait()
        except queue.Empty:  # pragma: no cover - unexpected but tolerable
            success = True
    else:
        success = False
        if process.exitcode is not None:
            if process.exitcode < 0:
                message = f"terminated by signal {-process.exitcode}"
            elif process.exitcode > 0:
                message = f"exited with status {process.exitcode}"

    if not success:
        prefix = "[tracks] FURY rendering failed"
        if message:
            print(f"{prefix}: {message}")
        else:
            print(f"{prefix} – falling back to matplotlib")

    return success


def _render_with_matplotlib(
    streamlines: Streamlines,
    output_path: str,
) -> None:
    """Fallback renderer relying on matplotlib when ``fury`` is unavailable."""

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - side effect enables 3D plots

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    for sl in streamlines:
        sl = _to_xyz(sl)
        if len(sl) < 2:
            continue
        segments = np.diff(sl, axis=0)
        norms = np.linalg.norm(segments, axis=1, keepdims=True)
        directions = np.zeros_like(segments)
        valid = norms[:, 0] > 0
        directions[valid] = segments[valid] / norms[valid]
        colour = np.abs(directions.mean(axis=0)) if valid.any() else np.array([0.5, 0.5, 0.5])
        colour = np.clip(colour, 0.1, 1.0)
        ax.plot(sl[:, 0], sl[:, 1], sl[:, 2], color=colour, linewidth=0.6, alpha=0.8)

    ax.set_facecolor("black")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        axis._axinfo["grid"]["color"] = (0.3, 0.3, 0.3, 0.15)

    ax.view_init(elev=75, azim=90)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor="black")
    plt.close(fig)


def _aggregate_streamline_colours(
    streamlines: Iterable[np.ndarray],
    affine: np.ndarray,
    volume_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-voxel colour averages and streamline densities."""

    inv_affine = np.linalg.inv(_canonical_affine(affine))
    counts = np.zeros(volume_shape, dtype=np.float32)
    colour_sum = np.zeros(volume_shape + (3,), dtype=np.float32)

    for streamline in streamlines:
        if streamline.shape[0] < 2:
            continue
        streamline = _to_xyz(streamline)

        world_a = streamline[:-1]
        world_b = streamline[1:]
        midpoints_world = 0.5 * (world_a + world_b)

        directions = world_b - world_a
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        valid = norms[:, 0] > 0
        if not np.any(valid):
            continue

        directions = directions[valid] / norms[valid]
        midpoints_world = midpoints_world[valid]

        colours = np.abs(directions)

        midpoints_vox = nib.affines.apply_affine(inv_affine, midpoints_world)
        indices = np.round(midpoints_vox).astype(int)

        valid_mask = (
            (indices[:, 0] >= 0)
            & (indices[:, 0] < volume_shape[0])
            & (indices[:, 1] >= 0)
            & (indices[:, 1] < volume_shape[1])
            & (indices[:, 2] >= 0)
            & (indices[:, 2] < volume_shape[2])
        )

        if not np.any(valid_mask):
            continue

        indices = indices[valid_mask]
        colours = colours[valid_mask]

        counts[indices[:, 0], indices[:, 1], indices[:, 2]] += 1.0
        colour_sum[indices[:, 0], indices[:, 1], indices[:, 2]] += colours

    with np.errstate(invalid="ignore", divide="ignore"):
        colour_avg = np.zeros_like(colour_sum)
        mask = counts > 0
        colour_avg[mask] = colour_sum[mask] / counts[mask][..., None]

    return colour_avg, counts


def _render_montage(
    streamlines: Streamlines,
    reference_img: nib.Nifti1Image,
    background_img: nib.Nifti1Image,
    output_path: str,
    *,
    title: str,
) -> None:
    """Create a multi-slice axial montage showing streamline overlays."""

    import matplotlib.pyplot as plt

    background = np.asarray(background_img.get_fdata(), dtype=np.float32)
    if background.ndim > 3:
        background = background[..., 0]

    background = np.nan_to_num(background)
    bg_min, bg_max = np.percentile(background, (1, 99))
    if bg_max <= bg_min:
        bg_min, bg_max = float(background.min()), float(background.max())

    colours, counts = _aggregate_streamline_colours(
        streamlines, _canonical_affine(reference_img.affine), background.shape
    )

    z_slices = np.linspace(0, background.shape[2] - 1, 6, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()

    for ax, z_index in zip(axes, z_slices):
        slice_bg = background[:, :, z_index]
        slice_colours = colours[:, :, z_index]
        slice_counts = counts[:, :, z_index]

        display_bg = np.rot90(slice_bg)
        ax.imshow(
            display_bg,
            cmap="gray",
            vmin=bg_min,
            vmax=bg_max,
            interpolation="bicubic",
        )

        if np.any(slice_counts):
            display_colours = np.rot90(slice_colours)
            display_alpha = np.rot90(slice_counts)
            display_alpha = display_alpha / float(display_alpha.max())
            display_alpha = np.clip(display_alpha, 0.05, 1.0)

            ax.imshow(
                display_colours,
                interpolation="bilinear",
                alpha=display_alpha,
            )

        ax.set_title(f"z = {z_index}")
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def generate_tractography(
    nifti_directory: str,
    analysis_directory: str,
    image_directory: Optional[str] = None,
    diffusion_filename: Optional[str] = None,
    *,
    anatomical_overlay: Optional[str] = None,
    create_montage: bool = False,
    montage_title: str = "Tractography overlay",
) -> TractographyOutputs:
    """Compute deterministic tractography and associated visualisations."""

    diffusion_dir = os.path.join(analysis_directory, "diffusion")
    os.makedirs(diffusion_dir, exist_ok=True)

    tract_path = os.path.join(diffusion_dir, "tractography.trk")
    preferred = (diffusion_filename,) if diffusion_filename else None
    found = find_dwi_files(nifti_directory, preferred_filenames=preferred)
    if not found:
        raise FileNotFoundError(
            "No diffusion dataset found – configure utils/parameters.py for DWI filenames."
        )

    dwi_path, bval_path, bvec_path = found

    img = nib.load(dwi_path)
    # Force a 4x4 voxel->world affine even if header carries higher-dim transforms.
    voxel_to_world = _canonical_affine(getattr(img, "affine", np.eye(4)))

    streamlines: Optional[Streamlines] = None
    if os.path.exists(tract_path):
        try:
            tractogram = nib_streamlines.load(tract_path).tractogram
            loaded = tractogram.streamlines
            streamlines = _coerce_streamlines_xyz(Streamlines(loaded))
        except Exception:
            streamlines = None

    fa_volume: Optional[np.ndarray] = None

    if streamlines is None:
        raw = img.get_fdata(dtype=np.float32)

        bvals = np.loadtxt(bval_path)
        bvecs = np.loadtxt(bvec_path)
        if bvecs.ndim == 2 and bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
            bvecs = bvecs.T
        data = _as_4d_dwi(raw, bvals)

        if data.shape[-1] != bvals.size or bvecs.shape != (bvals.size, 3):
            raise ValueError(
                f"bvals/bvecs mismatch. volumes={data.shape[-1]}, "
                f"bvals={bvals.size}, bvecs={bvecs.shape}"
            )
        print(
            f"[tracks] DWI shape={_shape_str(data)} | "
            f"bvals={bvals.size} | "
            f"affine(vox->mm) shape={voxel_to_world.shape}"
        )

        gtab = gradient_table(bvals=bvals, bvecs=bvecs)
        tensor_model = TensorModel(gtab)
        tensor_fit = tensor_model.fit(data)

        fa_volume = tensor_fit.fa.astype(np.float32)
        fa_volume = np.nan_to_num(fa_volume, nan=0.0, posinf=0.0, neginf=0.0)

        wm_mask = fa_volume > 0.2
        if not np.any(wm_mask):
            raise RuntimeError(
                "Diffusion data does not contain voxels above FA threshold (0.2)"
            )

        stopping_criterion = ThresholdStoppingCriterion(fa_volume, 0.15)

        peaks = peaks_from_model(
            tensor_model,
            data,
            default_sphere,
            relative_peak_threshold=0.5,
            min_separation_angle=25,
            mask=wm_mask,
            return_sh=False,
            # Make the direction-getter live in the same explicit 4x4 space.
            affine=voxel_to_world,
        )

        # Keep seeds in voxel space; LocalTracking will lift to mm using affine.
        seeds = seeds_from_mask(wm_mask, density=1)
        streamline_generator = LocalTracking(
            peaks,
            stopping_criterion,
            seeds,
            affine=voxel_to_world,
            step_size=0.5,
            return_all=False,
        )
        streamlines = _coerce_streamlines_xyz(Streamlines(streamline_generator))

        if len(streamlines) == 0:
            raise RuntimeError(
                "Tractography produced no streamlines – check diffusion quality"
            )

        tractogram = nib_streamlines.Tractogram(
            streamlines,
            # Streamlines are in world (mm) because we provided affine to LocalTracking.
            affine_to_rasmm=voxel_to_world,
        )
        nib_streamlines.save(tractogram, tract_path)
    else:
        fa_candidates = (
            os.path.join(diffusion_dir, "fa_map_native_debug.nii.gz"),
            os.path.join(diffusion_dir, "fa_map.nii.gz"),
        )
        for candidate in fa_candidates:
            if not os.path.isfile(candidate):
                continue
            try:
                fa_img = nib.load(candidate)
            except Exception:
                continue
            fa_data = np.asarray(fa_img.get_fdata(), dtype=np.float32)
            if fa_data.shape[:3] == img.shape[:3]:
                fa_volume = fa_data
                break
        # Streamlines loaded from file can carry 4 columns in rare cases
        try:
            tractogram = nib_streamlines.load(tract_path).tractogram
            streamlines = _coerce_streamlines_xyz(Streamlines(tractogram.streamlines))
        except Exception:
            pass

    image_directory = _ensure_image_directory(image_directory, analysis_directory)
    tract_image_dir = os.path.join(image_directory, "tractography")
    os.makedirs(tract_image_dir, exist_ok=True)

    render_path = os.path.join(tract_image_dir, "tractography_render.png")
    if _should_use_fury():
        if not _render_with_fury_isolated(tract_path, render_path):
            _render_with_matplotlib(streamlines, render_path)
    else:  # pragma: no cover - fallback depends on runtime environment
        _render_with_matplotlib(streamlines, render_path)

    montage_path: Optional[str] = None
    if create_montage:
        if anatomical_overlay:
            background_img = _load_background_volume(anatomical_overlay, img)
        else:
            if fa_volume is None:
                # When tractography was precomputed we may not have FA in-memory.
                raw = img.get_fdata(dtype=np.float32)
                bvals = np.loadtxt(bval_path)
                bvecs = np.loadtxt(bvec_path)
                if bvecs.ndim == 2 and bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
                    bvecs = bvecs.T
                data = _as_4d_dwi(raw, bvals)
                gtab = gradient_table(bvals=bvals, bvecs=bvecs)
                tensor_model = TensorModel(gtab)
                tensor_fit = tensor_model.fit(data)
                fa_volume = tensor_fit.fa.astype(np.float32)
                fa_volume = np.nan_to_num(fa_volume, nan=0.0, posinf=0.0, neginf=0.0)
            background_img = nib.Nifti1Image(
                fa_volume, voxel_to_world, img.header
            )
        montage_path = os.path.join(tract_image_dir, "tractography_montage.png")
        _render_montage(streamlines, img, background_img, montage_path, title=montage_title)

    return TractographyOutputs(
        tract_path=tract_path,
        render_path=render_path,
        montage_path=montage_path,
    )

