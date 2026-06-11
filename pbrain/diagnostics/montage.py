"""Independent parameter-map montage generator.

Model-agnostic: give it any 3-D map (and optionally a parcellation) and it
renders the publication-style axial montage — the same look used in the
p-Brain article. It couples onto *any* model's outputs, present or future:
the pipeline calls :func:`render_levels` for every map a model produces, so a
new model dropped into ``pbrain/models/`` gets voxel / tissue / parcel
montages for free.

Style (matches the article figures):

* **specthl** colormap (black → blue → magenta → orange → white);
* **symmetric grid** — for the usual 10-slice DCE volume this is 2×5; the
  layout is the most balanced factorisation of the slice count, so it adapts
  to any volume without a hardcoded slice number;
* white page, no per-slice titles, tight spacing, a single right-hand
  colorbar with rounded ticks;
* radiological axial orientation (``rot90`` + ``origin='lower'``).

Three levels per map:

* ``voxel``  — the raw voxelwise map;
* ``tissue`` — each tissue class painted with its in-class median;
* ``parcel`` — each parcel painted with its in-parcel median.

Tissue / parcel paints are computed inline from the map + parcellation, so the
generator needs nothing from the aggregation stage.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _specthl():
    """The article colormap (spectral-hot)."""
    from matplotlib.colors import LinearSegmentedColormap
    anchors = [
        (0.00, (0, 0, 0)), (0.10, (0, 0, 40)), (0.22, (0, 0, 120)),
        (0.35, (60, 0, 170)), (0.50, (130, 0, 180)), (0.62, (200, 0, 120)),
        (0.73, (230, 30, 60)), (0.83, (255, 120, 0)), (0.92, (255, 200, 0)),
        (1.00, (255, 255, 255)),
    ]
    xs, cols = zip(*anchors)
    cols = np.array(cols, dtype=float) / 255.0
    return LinearSegmentedColormap.from_list("specthl", list(zip(xs, cols)), N=256)


def balanced_grid(n: int) -> tuple[int, int]:
    """Most-balanced (rows, cols) for *n* tiles, landscape (cols ≥ rows).

    Prefers an exact factorisation (10 → 2×5, 9 → 3×3, 12 → 3×4); for a prime
    count it falls back to the near-square grid that fits all tiles.
    """
    if n <= 0:
        return (1, 1)
    best = (1, n)
    for r in range(1, int(math.isqrt(n)) + 1):
        if n % r == 0:
            best = (r, n // r)          # largest divisor ≤ √n → most balanced
    rows, cols = best
    if cols > 3 * max(rows, 1) and n > 3:   # prime/awkward → near-square instead
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    return rows, cols


def _select_slices(arr: np.ndarray, max_slices: int) -> list[int]:
    """Slices that actually contain data (so an empty edge slice — e.g. one
    outside the T1/M0 acquisition FOV — is not shown as a blank tile).

    If the data-bearing slices fit within ``max_slices`` they are all shown
    (a full 10-slice volume → 2×5, a 9-slice-covered one → 3×3); otherwise a
    balanced subset spanning their extent is taken."""
    Z = arr.shape[2]
    data_z = [z for z in range(Z)
              if np.isfinite(arr[:, :, z]).any() and np.any(np.nan_to_num(arr[:, :, z]) != 0)]
    if not data_z:
        data_z = list(range(Z))
    if len(data_z) <= max_slices:
        return data_z
    z0, z1 = data_z[0], data_z[-1]
    return sorted(set(np.linspace(z0, z1, num=max_slices, dtype=int).tolist()))


_TILE_GREY = (0.86, 0.87, 0.89)        # subtle grey square behind each map


def montage_volume(arr: np.ndarray, out_path: Path, *, mask: np.ndarray | None = None,
                   title: str = "", units: str = "", max_slices: int = 16,
                   vlims: tuple[float, float] = (2.0, 96.0), cmap=None) -> Path:
    """Render one article-style montage of ``arr`` (X, Y, Z) to ``out_path``.

    * ``mask`` (3-D bool): restrict display to brain voxels — voxels outside
      the mask become transparent so only segmented tissue is shown.
    * ``vlims`` (lo, hi) percentiles of the displayed tissue set the colour
      window. The default (2, 96) scales to the bulk of the tissue so bright
      outliers (vessels, partial-volume) saturate at the top instead of
      squashing the main tissue into the dark end.
    * Each slice sits on a subtle grey square; anterior points up.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"montage expects a 3-D volume, got {arr.shape}")
    if mask is not None:                           # keep only brain voxels
        arr = np.where(np.asarray(mask, bool), arr, np.nan)
    cmap = (cmap or _specthl()).copy()
    cmap.set_bad((0, 0, 0, 0))                     # NaN → transparent → grey shows
    cmap.set_over(cmap(1.0))                        # outliers saturate, not clip-white

    slices = _select_slices(arr, max_slices)
    rows, cols = balanced_grid(len(slices))

    # Colour window from the displayed-tissue distribution only (brain voxels),
    # so a few bright outliers don't darken everything else.
    finite = arr[np.isfinite(arr) & (arr != 0)]
    lo, hi = vlims
    vmin, vmax = (np.percentile(finite, [lo, hi]) if finite.size else (0.0, 1.0))
    if vmin == vmax:
        vmax = vmin + 1e-6

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2),
                             facecolor="white", squeeze=False)
    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.01, right=0.91,
                        top=0.95 if title else 0.99, bottom=0.01)
    im = None
    for idx, ax in enumerate(axes.ravel()):
        ax.set_xticks([]); ax.set_yticks([])
        sl_raw = arr[:, :, slices[idx]] if idx < len(slices) else None
        has_data = sl_raw is not None and np.any(np.isfinite(sl_raw) & (np.nan_to_num(sl_raw) != 0))
        if has_data:
            ax.set_facecolor(_TILE_GREY)           # grey square behind the brain
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(_TILE_GREY); sp.set_linewidth(0.5)
            # anterior up: rot90 then default (upper) origin
            sl = np.rot90(np.where(np.isfinite(sl_raw), sl_raw, np.nan))
            im = ax.imshow(sl, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax.axis("off")                         # absent slice → blank tile

    if title:
        fig.suptitle(title, fontsize=12)
    if im is not None:
        cax = fig.add_axes([0.925, 0.15, 0.015, 0.7])
        cb = fig.colorbar(im, cax=cax)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=9)
        if units and units not in ("(unitless)", "RGB"):   # real units only
            cb.set_label(units, fontsize=10, rotation=90, labelpad=2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path


def paint_by_label(arr: np.ndarray, parc: np.ndarray,
                   label_ids: dict[str, list[int]] | None = None) -> np.ndarray:
    """Median-per-label painted map.

    Without ``label_ids``: paint every parcel (each unique label > 0) with its
    in-parcel median (the *parcel* level). With ``label_ids`` (``{region:
    [labels]}``): paint each tissue class with its pooled median (the *tissue*
    level). Voxels outside any group are NaN.
    """
    arr = np.asarray(arr, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if label_ids is None:
        groups = {int(L): [int(L)] for L in np.unique(parc) if int(L) > 0}
    else:
        groups = {k: list(v) for k, v in label_ids.items()}
    for _, ids in groups.items():
        m = np.isin(parc, ids)
        vals = arr[m]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[m] = float(np.median(vals))
    return out


def render_levels(arr: np.ndarray, out_dir: Path, stem: str, *,
                  parc: np.ndarray | None = None,
                  region_map: dict[str, list[int]] | None = None,
                  title: str = "", units: str = "", max_slices: int = 16,
                  vlims: tuple[float, float] = (2.0, 96.0),
                  levels: tuple[str, ...] = ("voxel", "tissue", "parcel"),
                  ) -> dict[str, Path]:
    """Write the requested montage levels for one map. Returns ``{level: path}``.

    ``levels`` selects which to render — a parcel-painted map (from an
    average-then-fit model) passes ``("tissue", "parcel")`` so it is never
    shown as a (misleading) per-voxel montage. ``vlims`` are the colour-window
    percentiles (default scales to the bulk tissue, not outliers).
    """
    out_dir = Path(out_dir)
    written: dict[str, Path] = {}
    brain = (parc > 0) if parc is not None else None    # restrict to segmented brain
    if "voxel" in levels:
        written["voxel"] = montage_volume(
            arr, out_dir / f"{stem}_voxel.png", mask=brain,
            title=title, units=units, max_slices=max_slices, vlims=vlims)
    if parc is not None:
        if "tissue" in levels and region_map:
            written["tissue"] = montage_volume(
                paint_by_label(arr, parc, region_map),
                out_dir / f"{stem}_tissue.png",
                title=title, units=units, max_slices=max_slices, vlims=vlims)
        if "parcel" in levels:
            written["parcel"] = montage_volume(
                paint_by_label(arr, parc),
                out_dir / f"{stem}_parcel.png",
                title=title, units=units, max_slices=max_slices, vlims=vlims)
    return written
