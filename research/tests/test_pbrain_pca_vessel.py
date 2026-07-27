"""Tests for the species-agnostic temporal-decomposition AIF finder.

Two properties matter and neither is about ranking quality: the curve must be in
concentration units, and the FOV boundary must never be searched.
"""

from __future__ import annotations

import numpy as np

from pbrain.aif import REGISTRY
from pbrain.aif.pca_vessel import _interior, _search_region


def _series(shape=(24, 24, 7), n_t=40, peak=12, rng_seed=0):
    """A DCE-like series: flat baseline, one gamma-ish bolus, mild noise."""
    rng = np.random.default_rng(rng_seed)
    t = np.arange(n_t)
    bolus = np.exp(-((t - peak) ** 2) / (2 * 3.0 ** 2))
    data = 100.0 + rng.normal(0, 0.5, size=(*shape, n_t))
    return data, bolus, t


def test_interior_strips_a_margin_from_every_face():
    keep = _interior((10, 10, 10), (1.0, 1.0, 1.0), border_mm=1.0)
    assert not keep[0].any() and not keep[-1].any()
    assert not keep[:, 0].any() and not keep[:, -1].any()
    assert not keep[:, :, 0].any() and not keep[:, :, -1].any()
    assert keep[1:-1, 1:-1, 1:-1].all()


def test_interior_margin_is_anisotropy_correct():
    """One millimetre trims many columns in-plane and one slice through-plane."""
    keep = _interior((96, 96, 9), (0.208, 0.208, 1.0), border_mm=1.0)
    kept_i = np.where(keep.any(axis=(1, 2)))[0]
    kept_k = np.where(keep.any(axis=(0, 1)))[0]
    assert kept_i[0] == 5 and kept_i[-1] == 90       # ceil(1/0.208) = 5
    assert kept_k[0] == 1 and kept_k[-1] == 7        # ceil(1/1.0)   = 1


def test_interior_never_empties_a_thin_axis():
    """A margin wider than the axis must still leave something to search."""
    keep = _interior((96, 96, 3), (0.208, 0.208, 1.0), border_mm=5.0)
    assert keep.any()


def test_interior_disabled_by_zero_border():
    assert _interior((8, 8, 8), (1.0, 1.0, 1.0), border_mm=0.0).all()


def test_search_region_excludes_boundary_even_when_the_mask_touches_it():
    """The failure this guards: a rim hugs the outside of the mask, so a brain that
    reaches the FOV edge steers the search straight into the edge artefacts."""
    shape = (20, 20, 7)
    brain = np.zeros(shape, dtype=bool)
    brain[2:18, 2:18, 0:5] = True            # touches slice 0
    region = _search_region(shape, brain, np.zeros(shape),
                            shell_mm=1.5, voxel_mm=(1.0, 1.0, 1.0), border_mm=1.0)
    assert not region[:, :, 0].any(), "slice 0 is the FOV boundary and must be excluded"
    assert region.any(), "the rim must not be emptied entirely"


def test_border_keeps_the_detector_off_the_boundary():
    """An early, sharp, bright artefact on the edge slice must not be selected."""
    data, bolus, _ = _series()
    shape = data.shape[:3]
    brain = np.zeros(shape, dtype=bool)
    brain[4:20, 4:20, 1:6] = True
    data[6:10, 6:10, 1] += 30.0 * bolus                     # a real early source
    data[:, :, 0] += 80.0 * np.exp(-((np.arange(40) - 4) ** 2) / 8.0)  # edge inflow

    ifn = REGISTRY["pca_vessel"].extract(
        data, np.arange(40, dtype=float) * 5.0, np.eye(4),
        concentration_data=data, brain_mask_path=None, baseline_frames=4,
    )
    picked = np.argwhere(np.asarray(ifn.mask))
    assert picked.size, "detector returned nothing"
    assert not (picked[:, 2] == 0).any(), "selected a voxel on the boundary slice"


def test_curve_is_taken_from_the_concentration_volume():
    """c_a must be in mM: every downstream model reads it as a concentration."""
    data, bolus, _ = _series()
    conc = (data - 100.0) * 0.01                 # a distinct, known scale

    ifn = REGISTRY["pca_vessel"].extract(
        data, np.arange(40, dtype=float) * 5.0, np.eye(4),
        concentration_data=conc, baseline_frames=4,
    )
    assert ifn.meta["curve_units"] == "mM"
    # Signal sits near 100; concentration near 0. Reading the wrong volume is a
    # ~4-order-of-magnitude error, so a loose bound is a decisive test.
    assert np.nanmax(np.abs(ifn.c_a)) < 1.0


def test_falls_back_to_signal_and_says_so_when_no_concentration_given():
    data, _, _ = _series()
    ifn = REGISTRY["pca_vessel"].extract(
        data, np.arange(40, dtype=float) * 5.0, np.eye(4), baseline_frames=4,
    )
    assert "signal" in ifn.meta["curve_units"]
