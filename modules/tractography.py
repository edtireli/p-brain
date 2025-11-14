"""Utilities to compute and visualise diffusion tractography streamlines."""

from __future__ import annotations

import os
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


@dataclass
class TractographyOutputs:
    """Container describing generated tractography artefacts."""

    tract_path: str
    render_path: Optional[str] = None
    montage_path: Optional[str] = None


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
    stream_actor = dipy_actor.line(streamlines, colours, linewidth=1.0)
    scene.add(stream_actor)

    # Gently rotate the scene so fibres are rendered with depth cues.
    scene.reset_camera()
    scene.pitch(-15)
    scene.yaw(20)
    scene.zoom(1.2)

    dipy_window.snapshot(scene, fname=output_path, size=(1600, 1600))


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

    inv_affine = np.linalg.inv(affine)
    counts = np.zeros(volume_shape, dtype=np.float32)
    colour_sum = np.zeros(volume_shape + (3,), dtype=np.float32)

    for streamline in streamlines:
        if streamline.shape[0] < 2:
            continue

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
        streamlines, reference_img.affine, background.shape
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

    streamlines: Optional[Streamlines] = None
    if os.path.exists(tract_path):
        try:
            tractogram = nib_streamlines.load(tract_path).tractogram
            loaded = tractogram.streamlines
            streamlines = Streamlines(loaded)
        except Exception:
            streamlines = None

    fa_volume: Optional[np.ndarray] = None

    if streamlines is None:
        data = img.get_fdata(dtype=np.float32)

        bvals = np.loadtxt(bval_path)
        bvecs = np.loadtxt(bvec_path)
        if bvecs.ndim == 2 and bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
            bvecs = bvecs.T

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
        )

        seeds = seeds_from_mask(wm_mask, density=1, affine=img.affine)
        streamline_generator = LocalTracking(
            peaks,
            stopping_criterion,
            seeds,
            affine=img.affine,
            step_size=0.5,
            return_all=False,
        )
        streamlines = Streamlines(streamline_generator)

        if len(streamlines) == 0:
            raise RuntimeError(
                "Tractography produced no streamlines – check diffusion quality"
            )

        tractogram = nib_streamlines.Tractogram(streamlines, affine_to_rasmm=img.affine)
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

    image_directory = _ensure_image_directory(image_directory, analysis_directory)
    tract_image_dir = os.path.join(image_directory, "tractography")
    os.makedirs(tract_image_dir, exist_ok=True)

    render_path = os.path.join(tract_image_dir, "tractography_render.png")
    if _FURY_AVAILABLE:
        _render_with_fury(streamlines, render_path)
    else:  # pragma: no cover - fallback depends on runtime environment
        _render_with_matplotlib(streamlines, render_path)

    montage_path: Optional[str] = None
    if create_montage:
        if anatomical_overlay:
            background_img = _load_background_volume(anatomical_overlay, img)
        else:
            if fa_volume is None:
                # When tractography was precomputed we may not have FA in-memory.
                data = img.get_fdata(dtype=np.float32)
                bvals = np.loadtxt(bval_path)
                bvecs = np.loadtxt(bvec_path)
                if bvecs.ndim == 2 and bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
                    bvecs = bvecs.T
                gtab = gradient_table(bvals=bvals, bvecs=bvecs)
                tensor_model = TensorModel(gtab)
                tensor_fit = tensor_model.fit(data)
                fa_volume = tensor_fit.fa.astype(np.float32)
                fa_volume = np.nan_to_num(fa_volume, nan=0.0, posinf=0.0, neginf=0.0)
            background_img = nib.Nifti1Image(fa_volume, img.affine, img.header)
        montage_path = os.path.join(tract_image_dir, "tractography_montage.png")
        _render_montage(streamlines, img, background_img, montage_path, title=montage_title)

    return TractographyOutputs(
        tract_path=tract_path,
        render_path=render_path,
        montage_path=montage_path,
    )

