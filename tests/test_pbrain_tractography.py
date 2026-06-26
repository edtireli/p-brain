"""Tractography plug-in: synthetic-field tracking + .tck sidecar.

Exercises the ``diffusion.tractography`` plug-in on a tiny synthetic
single-fibre DWI volume and asserts (a) non-empty streamlines come back,
(b) a non-trivial track-density map is produced, and (c) the ``.tck``
byte sidecar round-trips through nibabel. Also confirms the plug-in is
discovered in the diffusion REGISTRY.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from pbrain.diffusion import REGISTRY, DWIInputs


def _synthetic_single_fibre_dwi(shape=(8, 8, 8), n_dirs=30, bval=1000.0):
    """A homogeneous single-fibre-along-x DWI on one b-value shell."""
    from dipy.core.gradients import gradient_table
    from dipy.core.sphere import HemiSphere, disperse_charges
    from dipy.sims.voxel import multi_tensor

    rng = np.random.default_rng(0)
    hsph = HemiSphere(theta=np.pi * rng.random(n_dirs),
                      phi=2 * np.pi * rng.random(n_dirs))
    hsph, _ = disperse_charges(hsph, 50)
    bvecs = np.vstack([[0, 0, 0], hsph.vertices])
    bvals = np.hstack([0.0, np.full(n_dirs, bval)])
    gtab = gradient_table(bvals, bvecs=bvecs)

    mevals = np.array([[0.0015, 0.0003, 0.0003]])     # prolate → high FA
    sig, _ = multi_tensor(gtab, mevals, S0=100, angles=[(0, 0)],
                          fractions=[100], snr=None)
    data = np.tile(sig, shape + (1,)).astype(np.float32)
    mask = np.ones(shape, bool)
    return data, bvals, bvecs, mask


def test_tractography_registered():
    assert "tractography" in REGISTRY
    plug = REGISTRY["tractography"]
    assert plug.outputs == ("track_density",)
    assert callable(getattr(plug, "fit", None))


@pytest.mark.parametrize("recon", ["dti", "csd"])
def test_tractography_produces_streamlines_and_tck(recon):
    from nibabel.streamlines import TckFile

    data, bvals, bvecs, mask = _synthetic_single_fibre_dwi()
    # CSD reads a chosen shell; the synthetic data is on b=1000.
    extra = {"fit_bval": 1000.0} if recon == "csd" else {}
    res = REGISTRY["tractography"].fit(
        DWIInputs(signal=data, bvals=bvals, bvecs=bvecs,
                  affine=np.eye(4), mask=mask),
        recon=recon, stop_thr=0.05, seed_density=1, step_size=0.5, **extra,
    )

    # ── streamlines were generated ──
    assert res.aux["n_streamlines"] > 0, "tracking produced no streamlines"
    assert res.aux["recon"] == recon

    # ── density map is a populated scalar volume on the DWI grid ──
    density = res.maps["track_density"]
    assert density.shape == data.shape[:3]
    assert np.count_nonzero(density) > 0
    assert float(density.max()) > 0

    # ── the .tck sidecar round-trips and is non-empty ──
    ext, payload = res.aux["sidecars"]["streamlines"]
    assert ext == "tck"
    assert isinstance(payload, (bytes, bytearray)) and len(payload) > 0
    loaded = TckFile.load(io.BytesIO(bytes(payload)))
    assert len(loaded.tractogram.streamlines) == res.aux["n_streamlines"]


def test_tractography_tck_writes_to_disk(tmp_path):
    """The byte payload writes a valid, loadable .tck file on disk —
    mirrors how DiffusionStage persists ``aux['sidecars']``."""
    from nibabel.streamlines import load as load_streamlines

    data, bvals, bvecs, mask = _synthetic_single_fibre_dwi()
    res = REGISTRY["tractography"].fit(
        DWIInputs(signal=data, bvals=bvals, bvecs=bvecs,
                  affine=np.eye(4), mask=mask),
        recon="dti", stop_thr=0.05,
    )
    _ext, payload = res.aux["sidecars"]["streamlines"]
    tck_path = tmp_path / "streamlines.tck"
    tck_path.write_bytes(bytes(payload))

    assert tck_path.exists() and tck_path.stat().st_size > 0
    tg = load_streamlines(str(tck_path))
    assert len(tg.streamlines) == res.aux["n_streamlines"] > 0
