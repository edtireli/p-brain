import json
from pathlib import Path

import nibabel as nib
import numpy as np


def test_compute_connectome_writes_outputs(tmp_path: Path):
    # Arrange a tiny diffusion reference grid.
    nifti_dir = tmp_path / "NIfTI"
    analysis_dir = tmp_path / "Analysis"
    nifti_dir.mkdir()
    analysis_dir.mkdir()

    affine = np.eye(4)
    ref = nib.Nifti1Image(np.zeros((6, 6, 6), dtype=np.float32), affine)
    dwi_path = nifti_dir / "dwi.nii.gz"
    nib.save(ref, str(dwi_path))

    # Atlas: label 1 in one corner, label 2 in the opposite.
    atlas = np.zeros((6, 6, 6), dtype=np.int16)
    atlas[1:3, 1:3, 1:3] = 1
    atlas[3:5, 3:5, 3:5] = 2

    atlas_path = nifti_dir / "segmentation" / "segmentation" / "mri" / "aparc.DKTatlas+aseg.deep.nii.gz"
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(atlas, affine), str(atlas_path))

    # Two streamlines connecting the two parcels.
    s1 = np.array([[1.5, 1.5, 1.5], [3.5, 3.5, 3.5]], dtype=np.float32)
    s2 = np.array([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]], dtype=np.float32)

    from modules.connectome import compute_connectome

    outputs = compute_connectome(
        str(nifti_dir),
        str(analysis_dir),
        diffusion_filename=str(dwi_path),
        streamlines=[s1, s2],
        min_streamlines=1,
        small_world_random=5,
        seed=0,
    )

    # Assert files exist.
    assert Path(outputs.matrix_csv).exists()
    assert Path(outputs.labels_csv).exists()
    assert Path(outputs.metrics_json).exists()

    with open(outputs.metrics_json, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    # Basic sanity.
    assert metrics["nodes"] == 2
    assert metrics["edges"] == 1
    assert "density" in metrics
    assert "small_worldness" in metrics
