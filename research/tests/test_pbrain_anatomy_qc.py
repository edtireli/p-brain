"""Tests for the parcellation orientation check.

The bug this exists to catch: a 90°-transposed affine that produced a full, plausible
label volume scoring 27/28 slices covered, while painting labels into empty air. Overlap
metrics endorsed it. Structure positions do not.
"""

from __future__ import annotations

import numpy as np
import pytest

from pbrain.tissue_roi._anatomy_qc import check_orientation

# Voxel axes here are already RAS-ordered: i = L→R, j = P→A, k = I→S.
SHAPE = (10, 24, 16)
RAS = np.eye(4)


def _brain() -> tuple[np.ndarray, dict]:
    """A toy parcellation with the mammalian layout built in.

    Proportioned like a real brain — clearly longer A-P (j) than tall S-I (k) — and
    deliberately NOT symmetric under swapping those two axes, which an earlier version
    of this fixture was. That symmetry let a transposed affine pass every check.
    """
    parc = np.zeros(SHAPE, dtype=np.int32)
    parc[3:7, 19:23, 5:9] = 1       # olfactory    — most anterior, mid height
    parc[3:7, 1:5, 4:8] = 2         # cerebellum   — most posterior, mid height
    parc[2:8, 5:20, 9:14] = 3       # cortex       — spans A-P, dorsal
    parc[4:6, 10:14, 1:4] = 4       # hypothalamus — ventral, mid A-P
    region_map = {"Olfactory": [1], "Cerebellum": [2], "Cortex": [3],
                  "Hypothalamus": [4], "Other": []}
    return parc, region_map


def test_correct_layout_passes():
    parc, rm = _brain()
    out = check_orientation(parc, RAS, rm)
    assert out["status"] == "ok"
    assert len(out["checks"]) == 4          # 3 structure relations + the extent check
    assert all(c["passed"] for c in out["checks"])


def test_catches_the_ap_si_transposition():
    """THE regression test. A Quadruped's affine reported in the magnet frame swaps
    anterior-posterior with superior-inferior — exactly this."""
    parc, rm = _brain()
    transposed = RAS.copy()
    transposed[[1, 2]] = transposed[[2, 1]]         # swap the A-P and S-I world rows

    out = check_orientation(parc, transposed, rm)
    assert out["status"] == "warn"
    assert any(not c["passed"] for c in out["checks"])
    assert "mm along" in out["summary"]


def test_catches_an_anterior_posterior_flip():
    parc, rm = _brain()
    flipped = RAS.copy()
    flipped[1, 1] = -1.0                            # A-P mirrored

    out = check_orientation(parc, flipped, rm)
    assert out["status"] == "warn"
    failed = [c["relation"] for c in out["checks"] if not c["passed"]]
    assert any("olfactory" in r for r in failed)


def test_catches_a_superior_inferior_flip():
    parc, rm = _brain()
    flipped = RAS.copy()
    flipped[2, 2] = -1.0

    out = check_orientation(parc, flipped, rm)
    assert out["status"] == "warn"
    failed = [c["relation"] for c in out["checks"] if not c["passed"]]
    assert any("superior" in r for r in failed)


def test_a_pure_translation_is_not_a_failure():
    """Position in the scanner is irrelevant; only relative layout matters."""
    parc, rm = _brain()
    shifted = RAS.copy()
    shifted[:3, 3] = [123.0, -45.0, 67.0]
    assert check_orientation(parc, shifted, rm)["status"] == "ok"


def test_left_right_mirror_is_not_detected_and_that_is_documented():
    """A pure L-R mirror preserves every A-P and S-I relation, so these checks cannot
    see it. Recorded deliberately: the affine determinant is what guards against it."""
    parc, rm = _brain()
    mirrored = RAS.copy()
    mirrored[0, 0] = -1.0
    assert check_orientation(parc, mirrored, rm)["status"] == "ok"


def test_region_names_match_loosely():
    """Providers name groups differently; the check must not depend on an exact LUT."""
    parc, rm = _brain()
    renamed = {"olfactory bulb": rm["Olfactory"], "CEREBELLAR CORTEX": rm["Cerebellum"],
               "Cerebral cortex": rm["Cortex"], "hypothalamic region": rm["Hypothalamus"]}
    assert check_orientation(parc, renamed and RAS, renamed)["status"] == "ok"


def test_unrecognisable_regions_fall_back_to_the_shape_check_only():
    """With unnameable groups the structure relations cannot run, but the extent check
    still can — so report honestly on reduced coverage rather than claiming a full
    pass or staying silent."""
    parc, _ = _brain()
    out = check_orientation(parc, RAS, {"label_1": [1], "label_2": [2]})
    assert [c["axis"] for c in out["checks"]] == ["extent"]
    assert out["status"] == "ok"
    assert "1 anatomical relation" in out["summary"], out["summary"]


def test_absent_structures_are_skipped_not_failed():
    """A parcellation that simply lacks a structure is not evidence of mis-orientation."""
    parc, rm = _brain()
    parc[parc == 1] = 0                              # no olfactory in the FOV
    out = check_orientation(parc, RAS, rm)
    assert out["status"] == "ok"                     # the remaining relation still holds
    assert all("olfactory" not in c["relation"] for c in out["checks"])


def test_margin_requires_real_separation():
    parc, rm = _brain()
    assert check_orientation(parc, RAS, rm, margin_mm=1000.0)["status"] == "warn"


def test_empty_parcellation_is_skipped():
    out = check_orientation(np.zeros(SHAPE, dtype=np.int32), RAS,
                            {"Cortex": [3], "Cerebellum": [2]})
    assert out["status"] == "skipped"
