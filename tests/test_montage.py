import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from utils import montage


def test_map_z_from_ref_includes_endpoints():
    # Fractions that would previously round away from the extrema should now
    # clamp to the first and last slices of the target volume.
    z_fracs = np.linspace(0.1, 0.94, num=10)
    mapped = montage._map_z_from_ref(z_fracs, 10)

    assert mapped[0] == 0
    assert mapped[-1] == 9


def test_map_z_from_ref_is_monotonic():
    # Ensure the mapped indices remain non-decreasing so the montage traversal
    # continues across the slab even when rounding collapses nearby fractions.
    z_fracs = np.linspace(0.0, 1.0, num=10)
    mapped = montage._map_z_from_ref(z_fracs, 5)

    assert np.all(mapped[:-1] <= mapped[1:])
    assert mapped[0] == 0
    assert mapped[-1] == 4


def test_map_z_from_ref_respects_valid_bounds():
    z_fracs = np.linspace(0.0, 1.0, num=8)
    mapped = montage._map_z_from_ref(z_fracs, 12, zmin=2, zmax=9)

    assert mapped[0] == 2
    assert mapped[-1] == 9
    assert np.all(mapped >= 2)
    assert np.all(mapped <= 9)
