import os
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest


os.environ.setdefault("MPLBACKEND", "Agg")

# Some test environments attempt to switch backends; no-op it.
import matplotlib  # noqa: E402

matplotlib.use = lambda *args, **kwargs: None


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import montage  # noqa: E402


def test_parcel_means_resample_atlas_to_map(tmp_path: Path) -> None:
    atlas_data = np.zeros((2, 2, 2), dtype=np.int32)
    atlas_data[0, 0, 0] = 1
    atlas_img = nib.Nifti1Image(atlas_data.astype(np.int16), affine=np.diag([2, 2, 2, 1]))
    atlas_labels = np.array([1], dtype=np.int32)

    map_data = np.ones((4, 4, 4), dtype=np.float32)
    map_data[0:2, 0:2, 0:2] = 5.0
    map_img = nib.Nifti1Image(map_data, affine=np.diag([1, 1, 1, 1]))
    map_path = tmp_path / "map.nii.gz"
    nib.save(map_img, str(map_path))

    means = montage._parcel_label_means(
        str(map_path),
        atlas_data,
        atlas_labels,
        atlas_img=atlas_img,
    )
    assert means[1] == pytest.approx(5.0)

    projected = montage._parcel_mean_projection(
        str(map_path),
        atlas_data,
        atlas_labels,
        atlas_img=atlas_img,
    )
    assert projected is not None
    assert projected.shape == map_data.shape
    assert np.nanmean(projected) == pytest.approx(5.0)


def test_render_projection_montage_colorbar_compat(tmp_path: Path) -> None:
    reference = np.ones((4, 4, 4), dtype=np.float32)
    ref_info = montage._build_reference(reference, tiles=4, mask=np.ones_like(reference, dtype=bool))
    assert ref_info is not None

    data = np.linspace(0, 1, 4 * 4 * 4, dtype=np.float32).reshape((4, 4, 4))
    reference_img = nib.Nifti1Image(reference, affine=np.eye(4))
    job = montage.MAP_JOB_LOOKUP["Ki_map_atlas"]

    out_path = tmp_path / "projection.png"
    montage._render_projection_montage(
        data,
        ref_info,
        job,
        str(out_path),
        rows=2,
        cols=2,
        dpi=50,
        reference_img=reference_img,
        transparent_background=True,
    )
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_render_projection_montage_keeps_low_values_visible(tmp_path: Path) -> None:
    reference = np.ones((4, 4, 4), dtype=np.float32)
    ref_info = montage._build_reference(reference, tiles=4, mask=np.ones_like(reference, dtype=bool))
    assert ref_info is not None

    # All values are low but valid; historically, percentile-based masking could
    # incorrectly drop these values and render a blank/washed-out montage.
    data = np.ones((4, 4, 4), dtype=np.float32)
    reference_img = nib.Nifti1Image(reference, affine=np.eye(4))
    job = montage.MAP_JOB_LOOKUP["Ki_map_atlas"]

    out_path = tmp_path / "projection_low_values.png"
    montage._render_projection_montage(
        data,
        ref_info,
        job,
        str(out_path),
        rows=2,
        cols=2,
        dpi=60,
        reference_img=reference_img,
        transparent_background=False,
    )

    import matplotlib.image as mpimg

    img = mpimg.imread(out_path)
    assert img.ndim == 3
    rgb = img[..., :3]
    alpha = img[..., 3] if img.shape[-1] >= 4 else np.ones(rgb.shape[:2], dtype=rgb.dtype)

    # Default facecolor for montages is #e0e0e0.
    bg = np.array([224 / 255, 224 / 255, 224 / 255], dtype=rgb.dtype)
    diff = np.max(np.abs(rgb - bg), axis=-1)
    assert np.any((alpha > 0.9) & (diff > 0.05))


def test_projection_montage_has_transparent_tile_gaps(tmp_path: Path) -> None:
    reference = np.ones((6, 6, 6), dtype=np.float32)
    ref_info = montage._build_reference(reference, tiles=4, mask=np.ones_like(reference, dtype=bool))
    assert ref_info is not None

    data = np.ones((6, 6, 6), dtype=np.float32)
    reference_img = nib.Nifti1Image(reference, affine=np.eye(4))
    job = montage.MAP_JOB_LOOKUP["Ki_map_atlas"]

    out_path = tmp_path / "projection_tile_gaps.png"
    montage._render_projection_montage(
        data,
        ref_info,
        job,
        str(out_path),
        rows=2,
        cols=2,
        dpi=60,
        reference_img=reference_img,
        transparent_background=True,
    )

    import matplotlib.image as mpimg

    img = mpimg.imread(out_path)
    assert img.ndim == 3
    assert img.shape[-1] >= 4
    alpha = img[..., 3]
    # With a transparent figure background and nonzero wspace/hspace, there should be
    # fully transparent pixels corresponding to tile gaps.
    assert np.min(alpha) <= 0.01


def test_projection_montage_masks_diffusion_zeros_to_background(tmp_path: Path) -> None:
    reference = np.ones((6, 6, 6), dtype=np.float32)
    ref_info = montage._build_reference(reference, tiles=4, mask=np.ones_like(reference, dtype=bool))
    assert ref_info is not None

    # Simulate diffusion-style volumes where background is exactly zero.
    data = np.zeros((6, 6, 6), dtype=np.float32)
    data[2:4, 2:4, 2:4] = 0.6

    reference_img = nib.Nifti1Image(reference, affine=np.eye(4))
    job = montage.MAP_JOB_LOOKUP["fa_map_atlas"]
    job = montage.replace(job, mask_zero=True)

    out_path = tmp_path / "projection_diffusion_zero_bg.png"
    montage._render_projection_montage(
        data,
        ref_info,
        job,
        str(out_path),
        rows=2,
        cols=2,
        dpi=60,
        reference_img=reference_img,
        transparent_background=False,
    )

    from PIL import Image

    img = np.asarray(Image.open(out_path).convert("RGBA"))
    rgb = img[..., :3]
    alpha = img[..., 3]

    # Background is #e0e0e0; ensure some fully-opaque pixels match it.
    bg = np.array([224, 224, 224], dtype=np.uint8)
    bg_match = np.all(rgb == bg[None, None, :], axis=-1)
    assert np.any(bg_match & (alpha == 255))


def test_pillow_projection_colorbar_width_expands_for_long_tick_labels(tmp_path: Path) -> None:
    reference = np.ones((6, 6, 6), dtype=np.float32)
    ref_info = montage._build_reference(reference, tiles=4, mask=np.ones_like(reference, dtype=bool))
    assert ref_info is not None

    # Force extremely long tick labels (many digits) to regress label clipping.
    data = np.linspace(0.0, 1.0e20, 6 * 6 * 6, dtype=np.float32).reshape((6, 6, 6))
    reference_img = nib.Nifti1Image(reference, affine=np.eye(4))
    job = montage.MAP_JOB_LOOKUP["Ki_map_atlas"]

    norm, tick_values = montage._build_projection_normalizer(data, job)
    units = montage._units_for_job(job)
    cb_w = montage._pillow_colorbar_required_width(
        norm=norm,
        tick_values=tick_values,
        units=units,
        units_position=montage.PILLOW_COLORBAR_UNITS_POSITION,
        units_gap_px=montage.PILLOW_COLORBAR_UNITS_GAP_PX,
        tick_font_size=montage.PILLOW_COLORBAR_TICK_FONT_SIZE,
        units_font_size=montage.PILLOW_COLORBAR_UNITS_FONT_SIZE,
    )
    assert cb_w >= 150

    rows, cols = 2, 2
    pad = int(montage.PILLOW_OUTER_MARGIN_PX)
    tile_w = 260
    gap = int(montage.PILLOW_TILE_GAP_PX)
    expected_w = pad * 2 + cols * tile_w + (cols - 1) * gap + cb_w + gap

    out_path = tmp_path / "projection_long_ticks.png"
    montage._render_projection_montage(
        data,
        ref_info,
        job,
        str(out_path),
        rows=rows,
        cols=cols,
        dpi=60,
        reference_img=reference_img,
        transparent_background=False,
    )

    from PIL import Image

    img = Image.open(out_path)
    assert img.size[0] == expected_w


def test_find_available_maps_discovers_t1_m0_in_fitting(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "Analysis"
    fitting_dir = analysis_dir / "Fitting"
    fitting_dir.mkdir(parents=True, exist_ok=True)

    # Minimal NIfTI volumes written under Analysis/Fitting.
    data = np.zeros((3, 3, 3), dtype=np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    t1_path = fitting_dir / "t1_map.nii.gz"
    m0_path = fitting_dir / "m0_map.nii.gz"
    nib.save(img, str(t1_path))
    nib.save(img, str(m0_path))

    t1_job = montage.MAP_JOB_LOOKUP["t1_map"]
    m0_job = montage.MAP_JOB_LOOKUP["m0_map"]

    t1_found = montage._find_available_maps(t1_job, str(analysis_dir))
    m0_found = montage._find_available_maps(m0_job, str(analysis_dir))

    assert "" in t1_found
    assert "" in m0_found
    assert Path(t1_found[""]).name == "t1_map.nii.gz"
    assert Path(m0_found[""]).name == "m0_map.nii.gz"


def test_tick_label_formatter_avoids_all_zero_for_small_ranges() -> None:
    labels = montage._format_tick_labels([0.0, 0.0005, 0.0011, 0.0016, 0.0021])
    assert len(labels) == 5
    assert len(set(labels)) == 5
    assert any(lbl not in {"0", "0.0", "0.00"} for lbl in labels[1:])


def test_render_parametric_montage_uses_pillow_and_draws_units(tmp_path: Path) -> None:
    reference = np.ones((6, 6, 6), dtype=np.float32)
    ref_info = montage._build_reference(reference, tiles=4, mask=np.ones_like(reference, dtype=bool))
    assert ref_info is not None

    # Create a Ki-like volume so we get a units label.
    data = np.zeros((6, 6, 6), dtype=np.float32)
    data[2:4, 2:4, 2:4] = 2.0
    img = nib.Nifti1Image(data, affine=np.eye(4))
    map_path = tmp_path / "Ki_per_voxel.nii.gz"
    nib.save(img, str(map_path))

    reference_img = nib.Nifti1Image(reference, affine=np.eye(4))
    out_path = tmp_path / "ki_voxel_montage.png"
    job = montage.MAP_JOB_LOOKUP["Ki_per_voxel"]

    montage._render_montage(
        str(map_path),
        str(out_path),
        job,
        ref_info,
        reference_img=reference_img,
        rows=2,
        cols=2,
        dpi=60,
        overlay=None,
        brain_mask=None,
        segmentation_img=None,
        transparent_background=False,
    )
    assert out_path.is_file()
    assert out_path.stat().st_size > 0

    from PIL import Image

    img_rgba = np.asarray(Image.open(out_path).convert("RGBA"))
    # Background is #e0e0e0.
    bg = np.array([224, 224, 224], dtype=np.uint8)
    rgb = img_rgba[..., :3]
    # Heuristic: the colorbar (and units label) live on the right and are
    # vertically centered; check the upper-right region for non-background pixels.
    h, w = rgb.shape[:2]
    roi = rgb[0 : max(1, int(h * 0.4)), int(w * 0.7) : w, :]
    diff = np.max(np.abs(roi.astype(np.int16) - bg.astype(np.int16)), axis=-1)
    assert np.any(diff > 15)


def test_montage_has_transparent_gutters_when_opaque_background(tmp_path: Path) -> None:
    reference = np.ones((6, 6, 6), dtype=np.float32)
    ref_info = montage._build_reference(reference, tiles=4, mask=np.ones_like(reference, dtype=bool))
    assert ref_info is not None

    data = np.zeros((6, 6, 6), dtype=np.float32)
    data[2:4, 2:4, 2:4] = 2.0
    img = nib.Nifti1Image(data, affine=np.eye(4))
    map_path = tmp_path / "Ki_per_voxel.nii.gz"
    nib.save(img, str(map_path))

    reference_img = nib.Nifti1Image(reference, affine=np.eye(4))
    out_path = tmp_path / "ki_montage_gutters.png"
    job = montage.MAP_JOB_LOOKUP["Ki_per_voxel"]

    montage._render_montage(
        str(map_path),
        str(out_path),
        job,
        ref_info,
        reference_img=reference_img,
        rows=2,
        cols=2,
        dpi=60,
        overlay=None,
        brain_mask=None,
        segmentation_img=None,
        transparent_background=False,
    )

    from PIL import Image

    rgba = np.asarray(Image.open(out_path).convert("RGBA"))
    alpha = rgba[..., 3]

    pad = int(montage.PILLOW_OUTER_MARGIN_PX)
    tile_w = 260
    tile_h = 260
    gap = int(montage.PILLOW_TILE_GAP_PX)
    # Sample a pixel in the vertical gutter between the two columns.
    x_gutter = pad + tile_w + gap // 2
    y_mid = pad + tile_h // 2
    assert alpha[y_mid, x_gutter] == 0
    # Sample a pixel well inside the first tile.
    assert alpha[y_mid, pad + tile_w // 2] == 255


def test_projection_has_transparent_gutters_when_opaque_background(tmp_path: Path) -> None:
    reference = np.ones((6, 6, 6), dtype=np.float32)
    ref_info = montage._build_reference(reference, tiles=4, mask=np.ones_like(reference, dtype=bool))
    assert ref_info is not None

    data = np.ones((6, 6, 6), dtype=np.float32)
    reference_img = nib.Nifti1Image(reference, affine=np.eye(4))
    job = montage.MAP_JOB_LOOKUP["Ki_map_atlas"]

    out_path = tmp_path / "projection_gutters.png"
    montage._render_projection_montage(
        data,
        ref_info,
        job,
        str(out_path),
        rows=2,
        cols=2,
        dpi=60,
        reference_img=reference_img,
        transparent_background=False,
    )

    from PIL import Image

    rgba = np.asarray(Image.open(out_path).convert("RGBA"))
    alpha = rgba[..., 3]

    pad = int(montage.PILLOW_OUTER_MARGIN_PX)
    tile_w = 260
    tile_h = 260
    gap = int(montage.PILLOW_TILE_GAP_PX)
    x_gutter = pad + tile_w + gap // 2
    y_mid = pad + tile_h // 2
    assert alpha[y_mid, x_gutter] == 0
    assert alpha[y_mid, pad + tile_w // 2] == 255


def test_colorbar_region_background_is_transparent_for_montage(tmp_path: Path) -> None:
    reference = np.ones((6, 6, 6), dtype=np.float32)
    ref_info = montage._build_reference(reference, tiles=4, mask=np.ones_like(reference, dtype=bool))
    assert ref_info is not None

    data = np.zeros((6, 6, 6), dtype=np.float32)
    data[2:4, 2:4, 2:4] = 2.0
    img = nib.Nifti1Image(data, affine=np.eye(4))
    map_path = tmp_path / "Ki_per_voxel.nii.gz"
    nib.save(img, str(map_path))

    reference_img = nib.Nifti1Image(reference, affine=np.eye(4))
    out_path = tmp_path / "ki_montage_colorbar_transparent_bg.png"
    job = montage.MAP_JOB_LOOKUP["Ki_per_voxel"]

    montage._render_montage(
        str(map_path),
        str(out_path),
        job,
        ref_info,
        reference_img=reference_img,
        rows=2,
        cols=2,
        dpi=60,
        overlay=None,
        brain_mask=None,
        segmentation_img=None,
        transparent_background=False,
    )

    from PIL import Image

    rgba = np.asarray(Image.open(out_path).convert("RGBA"))
    alpha = rgba[..., 3]

    pad = int(montage.PILLOW_OUTER_MARGIN_PX)
    tile_w = 260
    tile_h = 260
    gap = int(montage.PILLOW_TILE_GAP_PX)
    grid_w = 2 * tile_w + (2 - 1) * gap
    cb_x0 = pad + grid_w + gap
    cb_h = int(tile_h * 1.05)
    canvas_h = pad * 2 + 2 * tile_h + (2 - 1) * gap
    cb_y0 = pad + (canvas_h - 2 * pad - cb_h) // 2
    # Sample a point in the colorbar slot padding (left of the bar) that should be transparent.
    assert alpha[cb_y0 + cb_h // 2, cb_x0 + 1] == 0
    # Sample a point in the tile grid background that should be opaque.
    assert alpha[pad + 5, pad + 5] == 255


def test_colorbar_region_background_is_transparent_for_projection(tmp_path: Path) -> None:
    reference = np.ones((6, 6, 6), dtype=np.float32)
    ref_info = montage._build_reference(reference, tiles=4, mask=np.ones_like(reference, dtype=bool))
    assert ref_info is not None

    data = np.ones((6, 6, 6), dtype=np.float32)
    reference_img = nib.Nifti1Image(reference, affine=np.eye(4))
    job = montage.MAP_JOB_LOOKUP["Ki_map_atlas"]

    out_path = tmp_path / "projection_colorbar_transparent_bg.png"
    montage._render_projection_montage(
        data,
        ref_info,
        job,
        str(out_path),
        rows=2,
        cols=2,
        dpi=60,
        reference_img=reference_img,
        transparent_background=False,
    )

    from PIL import Image

    rgba = np.asarray(Image.open(out_path).convert("RGBA"))
    alpha = rgba[..., 3]

    pad = int(montage.PILLOW_OUTER_MARGIN_PX)
    tile_w = 260
    tile_h = 260
    gap = int(montage.PILLOW_TILE_GAP_PX)
    grid_w = 2 * tile_w + (2 - 1) * gap
    cb_x0 = pad + grid_w + gap
    cb_h = int(tile_h * 1.05)
    canvas_h = pad * 2 + 2 * tile_h + (2 - 1) * gap
    cb_y0 = pad + (canvas_h - 2 * pad - cb_h) // 2

    assert alpha[cb_y0 + cb_h // 2, cb_x0 + 1] == 0
    assert alpha[pad + 5, pad + 5] == 255


def test_discover_atlas_maps_prefers_known_bases(tmp_path: Path) -> None:
    analysis = tmp_path / "Analysis" / "diffusion"
    analysis.mkdir(parents=True)

    # Known base + suffix
    nib.save(
        nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), affine=np.eye(4)),
        str(analysis / "Ki_map_atlas_patlak.nii.gz"),
    )
    # Unknown atlas map
    nib.save(
        nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), affine=np.eye(4)),
        str(analysis / "my_custom_atlas_metric.nii.gz"),
    )

    # AppleDouble/resource-fork file (should be ignored even if it ends with .nii.gz)
    (analysis / "._Ki_map_atlas_patlak.nii.gz").write_bytes(b"not a nifti")

    discovered = montage._discover_atlas_maps(str(tmp_path / "Analysis"))
    assert "Ki_map_atlas" in discovered
    assert "_patlak" in discovered["Ki_map_atlas"]
    assert "my_custom_atlas_metric" in discovered
    assert "._Ki_map_atlas_patlak" not in discovered


def test_montage_import_forces_agg_backend_when_unset(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env.pop("MPLBACKEND", None)
    env["PYTHONPATH"] = str(repo_root)

    code = (
        "import os; "
        "import utils.montage; "
        "import matplotlib; "
        "print(os.environ.get('MPLBACKEND','')); "
        "print(matplotlib.get_backend())"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    out = proc.stdout.strip().splitlines()
    assert out, proc.stdout
    assert out[0].lower() == "agg"
    assert "agg" in out[1].lower()


def test_build_normalizer_honors_explicit_vmin_when_vmax_missing() -> None:
    data = np.linspace(1.0, 3.0, 100, dtype=np.float32).reshape((10, 10))
    job = montage.MapJob("cth_map", "cth_montage", vmin=0.0)
    norm, _ticks = montage._build_normalizer(data, job)
    assert float(norm.vmin) == pytest.approx(0.0)
    assert float(norm.vmax) > 1.0
