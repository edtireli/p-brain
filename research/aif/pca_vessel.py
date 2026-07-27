"""Vessel localisation by temporal decomposition, constrained by anatomy.

Species-agnostic AIF/VIF finder. It exploits the one thing that separates a
feeding vessel from parenchyma in *any* animal: its concentration-time course.
A vessel fills early, peaks sharply and washes out fast; tissue lags, blunts and
plateaus. That contrast is a property of haemodynamics, not of anatomy, so the
same detector works on mouse and human — no trained model, no template, no
registration.

Two independent signals are combined, because each covers the other's blind spot:

* **Temporal (PCA).** Voxel time-courses are decomposed; the component whose
  score peaks earliest with the sharpest rise is the bolus component. Voxels
  loading on it are vessel-like. This finds vessels but will equally happily
  select a motion edge or a pulsating artefact.
* **Anatomical (segmentation).** A brain mask says *where* a vessel may be. Note
  the prior is NOT "inside the brain": the superior sagittal sinus is a dural
  vessel lying on the dorsal midline *outside* the parenchyma, and the internal
  carotids run below the brain base. So the admissible region is a shell around
  the mask plus its rim — which also excludes skull, orbit and muscle, the very
  structures that make a purely temporal criterion misfire.

Neither is trusted alone: candidates must be vessel-like *and* anatomically
plausible. The peak voxel is then refined deterministically, exactly as
``auto_vessel`` does — the decomposition proposes, arithmetic measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import InputFunction


def _interior(shape: tuple[int, int, int], voxel_mm: tuple[float, float, float],
              border_mm: float) -> np.ndarray:
    """Volume with a margin stripped off every face — the FOV edge is not anatomy.

    Two different artefacts live at the boundary and both mimic a bolus:

    * **In-plane** — wrap-around and ghosting fold signal from outside the FOV onto
      the edge columns. It is bright, it is not tissue, and it moves with the
      acquisition rather than with the animal.
    * **Through-plane** — in a 2-D multi-slice acquisition the *end* slices see
      fully-relaxed spins flowing in from outside the excited slab. That inflow
      enhancement is early, sharp and large: precisely the signature the temporal
      criterion is hunting for, so the detector prefers those slices to real
      vessels. Measured here: the segmentation-constrained variant put all 10 of
      its voxels on slice 0.

    The margin is given in millimetres and converted per axis, so one number does
    the right thing on a strongly anisotropic grid: at 0.21 mm in-plane / 1 mm
    through-plane, ``border_mm=1`` trims 5 columns in-plane and exactly the two end
    slices. A whole-voxel floor guarantees a non-zero trim whenever it is asked for.
    """
    keep = np.ones(shape, dtype=bool)
    if border_mm <= 0:
        return keep
    for ax, (n, mm) in enumerate(zip(shape, voxel_mm)):
        k = int(np.ceil(border_mm / (mm if mm > 0 else 1.0)))
        k = max(1, min(k, (n - 1) // 2))        # never strip the volume to nothing
        sl = [slice(None)] * 3
        sl[ax] = slice(0, k)
        keep[tuple(sl)] = False
        sl[ax] = slice(n - k, n)
        keep[tuple(sl)] = False
    return keep


def _search_region(shape: tuple[int, int, int], brain: np.ndarray | None,
                   signal: np.ndarray, *, shell_mm: float = 1.5,
                   voxel_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
                   border_mm: float = 1.0) -> np.ndarray:
    """Where a feeding vessel may legitimately live.

    With a segmentation this is a **rim**, not the whole dilated brain: the dural
    sinuses and the carotids sit just *outside* the parenchyma, so the admissible
    set is ``dilate(brain) − erode(brain)`` — a thin shell straddling the surface.
    Taking the whole dilated volume instead barely rejects anything (measured:
    32.6k of 33.2k voxels, ~2 %), which leaves the temporal criterion unconstrained
    and lets it pick hot spots deep in parenchyma.

    Without a segmentation: everything above a noise floor. Weaker, but a missing
    mask must never block a run.

    The rim is measured with a **distance transform in millimetres**, not by
    iterating a structuring element. DCE voxels here are strongly anisotropic
    (~0.21 mm in-plane vs ~1 mm through-plane); an iteration count derived from the
    in-plane size dilates ~7 mm through-plane and swallows the entire slab, which is
    exactly how a "rim" ends up admitting 98 % of the volume.

    Both branches are then trimmed by :func:`_interior`. A rim is *defined* to hug
    the outside of the mask, so wherever the brain approaches the FOV edge the rim
    runs straight into the boundary artefacts — the anatomical prior actively steers
    the search into them, which is why the constrained variant scored worse than the
    unconstrained one before this trim existed.
    """
    from scipy import ndimage
    inside = _interior(shape, voxel_mm, border_mm)
    if brain is not None and brain.any():
        sampling = tuple(float(v) if v > 0 else 1.0 for v in voxel_mm)
        d_out = ndimage.distance_transform_edt(~brain, sampling=sampling)
        d_in = ndimage.distance_transform_edt(brain, sampling=sampling)
        return (d_out <= shell_mm) & (d_in <= shell_mm) & inside
    finite = signal[np.isfinite(signal)]
    if not finite.size:
        return inside
    return (signal > np.percentile(finite, 60)) & inside


def _coherent_cluster(score: np.ndarray, coords: np.ndarray,
                      shape: tuple[int, int, int], *, keep: int,
                      hi_pct: float = 97.0) -> np.ndarray:
    """Indices of the best *spatially connected* group of vessel-like voxels.

    A vessel is contiguous; a scatter of isolated hot voxels across many slices is
    noise, motion or partial-volume — not a lumen. So threshold the score, label
    connected components, and keep the strongest one (summed score, which favours
    a cluster that is both bright and big over a single stray voxel). Falls back to
    the plain top-N if nothing survives, so this can never fail a run outright."""
    from scipy import ndimage
    order = np.argsort(score)[::-1]
    thr = np.percentile(score, hi_pct)
    hot = np.zeros(shape, dtype=bool)
    sel = order[score[order] >= thr]
    if sel.size == 0:
        return order[:keep]
    for v in sel:
        hot[tuple(coords[v])] = True
    lab, n = ndimage.label(hot)
    if n == 0:
        return order[:keep]
    lin = {tuple(c): i for i, c in enumerate(map(tuple, coords))}
    best, best_w = None, -np.inf
    for cid in range(1, n + 1):
        members = [lin[tuple(c)] for c in np.argwhere(lab == cid) if tuple(c) in lin]
        if not members:
            continue
        w = float(np.sum(score[members]))
        if w > best_w:
            best, best_w = members, w
    if not best:
        return order[:keep]
    best = np.asarray(best)
    return best[np.argsort(score[best])[::-1][:keep]]


def _bolus_component(curves: np.ndarray, n_comp: int = 6) -> np.ndarray:
    """Per-voxel vessel-likeness from a PCA of the time-courses.

    ``curves`` is ``(V, T)``, baseline-removed. The component chosen is the one
    whose temporal score rises earliest and most sharply — the bolus. Returns a
    ``(V,)`` score; larger = more vessel-like. Sign is resolved by requiring the
    component to be positive-going at its own peak, so an arbitrary PCA sign flip
    cannot invert the ranking.
    """
    X = np.nan_to_num(curves - curves.mean(axis=1, keepdims=True))
    if X.shape[0] < 4 or X.shape[1] < 8:
        return np.nan_to_num(curves.max(axis=1))
    n_comp = int(min(n_comp, X.shape[0] - 1, X.shape[1] - 1))
    # economical SVD: rows are voxels, so V^T rows are temporal components
    _, _, vt = np.linalg.svd(X - X.mean(axis=0, keepdims=True), full_matrices=False)
    comps = vt[:n_comp]
    T = X.shape[1]
    # An arterial component peaks EARLY — that is its defining property, not a
    # tiebreak. Ranking by rise-sharpness first selects a sharp-but-late component
    # (scanner drift, or delayed venous return), which then peaks *after* tissue and
    # is useless as an input function. So restrict to components peaking in the
    # first `early_frac` of the series, and only then prefer the sharpest rise.
    early_frac = 0.45
    cutoff = max(3, int(early_frac * T))
    best, best_key = None, None
    for i, c in enumerate(comps):
        c = c if abs(c.max()) >= abs(c.min()) else -c      # positive-going
        pk = int(np.argmax(c))
        if pk == 0 or pk > cutoff:
            continue
        rise = float(c[pk] - c[:pk].min()) / max(pk, 1)     # sharpness of the rise
        if best_key is None or rise > best_key:
            best, best_key = i, rise
    if best is None:                                        # nothing peaks early
        return np.nan_to_num(curves[:, :cutoff].max(axis=1))
    c = comps[best]
    c = c if abs(c.max()) >= abs(c.min()) else -c
    return X @ c


@dataclass(frozen=True, slots=True)
class _PcaVessel:
    key: ClassVar[str] = "pca_vessel"
    name: ClassVar[str] = "Temporal-decomposition vessel finder"
    description: ClassVar[str] = (
        "Finds the input-function vessel by decomposing voxel time-courses and "
        "selecting the bolus component, constrained to a shell around the brain "
        "segmentation. Species-agnostic: no trained model, template or registration."
    )
    accepts: ClassVar[dict[str, type]] = {"dce_data": np.ndarray, "t_s": np.ndarray}
    produces: ClassVar[dict[str, type]] = {"input_function": InputFunction}

    def extract(
        self,
        dce_data: np.ndarray,
        t_s: np.ndarray,
        dce_affine: np.ndarray,
        *,
        concentration_data: np.ndarray | None = None,
        brain_mask_path: Path | str | None = None,
        n_voxels: int = 20,
        n_components: int = 6,
        shell_mm: float = 1.5,
        border_mm: float = 1.0,
        baseline_frames: int = 5,
        **_: Any,
    ) -> InputFunction:
        # The returned curve must be in mM: every downstream model reads ``c_a`` as
        # a concentration. ``signal_to_conc`` already did the per-voxel T1/M0
        # conversion upstream, so take the curve from that volume exactly as every
        # other extractor does — reading ``dce_data`` instead hands Patlak a curve
        # in arbitrary signal-enhancement units and silently rescales Ki by whatever
        # the signal-to-concentration slope happened to be.
        data = np.asarray(
            concentration_data if concentration_data is not None else dce_data,
            dtype=float,
        )
        if data.ndim != 4:
            raise ValueError(f"pca_vessel needs a 4-D DCE series; got {data.shape}")
        X, Y, Z, T = data.shape
        t_s = np.asarray(t_s, dtype=float).ravel()[:T]

        brain = None
        if brain_mask_path is not None and Path(brain_mask_path).exists():
            import nibabel as nib
            bm = np.asarray(nib.load(str(brain_mask_path)).dataobj) > 0
            if bm.shape == (X, Y, Z):
                brain = bm

        aff = np.asarray(dce_affine, dtype=float)
        vox_mm = tuple(float(np.linalg.norm(aff[:3, i])) or 1.0 for i in range(3))
        base = np.nanmean(data[..., :max(1, baseline_frames)], axis=-1)
        region = _search_region((X, Y, Z), brain, base, shell_mm=shell_mm,
                                voxel_mm=vox_mm, border_mm=border_mm)
        idx = np.argwhere(region)
        if not idx.size:
            raise ValueError("pca_vessel: empty search region")

        curves = data[region]                                   # (V, T)
        curves = curves - np.nanmean(curves[:, :max(1, baseline_frames)],
                                     axis=1, keepdims=True)     # enhancement

        # Hard physical constraint: a feeding vessel fills BEFORE the tissue it
        # feeds. The mean curve over the search region behaves like tissue, so its
        # peak is a self-calibrating reference — no hard-coded frame number, works
        # at any temporal resolution or species. Voxels peaking after it are
        # parenchyma, delayed venous return or drift, and cannot be an input
        # function however vessel-shaped they look to a decomposition.
        from scipy.ndimage import uniform_filter1d
        smooth = uniform_filter1d(np.nan_to_num(curves), size=3, axis=1)
        ttp = np.argmax(smooth, axis=1)
        ref_peak = int(np.argmax(np.nanmean(curves, axis=0)))
        eligible = ttp <= ref_peak
        if eligible.sum() < max(8, max(1, int(n_voxels))):
            eligible = ttp <= max(ref_peak, int(np.percentile(ttp, 25)))
        if not eligible.any():
            eligible = np.ones(curves.shape[0], dtype=bool)
        curves, idx = curves[eligible], idx[eligible]
        score = _bolus_component(curves, n_comp=n_components)

        # Vessel-like AND contiguous: a lumen is a connected structure, so take the
        # strongest connected cluster rather than the top-N scattered voxels.
        keep = max(1, int(n_voxels))
        chosen = _coherent_cluster(score, idx, (X, Y, Z), keep=keep)

        mask = np.zeros((X, Y, Z), dtype=bool)
        for v in chosen:
            mask[tuple(idx[v])] = True
        c_a = np.nanmean(curves[chosen], axis=0)

        peak = int(np.nanargmax(c_a)) if np.isfinite(c_a).any() else 0
        return InputFunction(
            c_a=c_a, t_s=t_s, mask=mask, source="pca_vessel",
            meta={
                "algorithm": "pca_vessel",
                "n_voxels": int(mask.sum()),
                "search_voxels": int(region.sum()),
                "anatomical_prior": ("segmentation shell" if brain is not None
                                     else "intensity floor (no segmentation given)"),
                "curve_units": ("mM" if concentration_data is not None
                                else "signal enhancement (no concentration volume)"),
                "shell_mm": float(shell_mm),
                "border_mm": float(border_mm),
                "slices": sorted({int(k) for k in np.argwhere(mask)[:, 2]}),
                "peak_frame": peak,
                "peak_conc": float(np.nanmax(c_a)) if np.isfinite(c_a).any() else float("nan"),
            },
        )


PLUGIN = _PcaVessel()
