from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import enumerator


@pytest.fixture
def dataset_root(tmp_path):
    # Create patient datasets
    (tmp_path / "001").mkdir()
    (tmp_path / "002").mkdir()

    # Create controls directory with two entries
    controls = tmp_path / "controls"
    controls.mkdir()
    (controls / "ctrl_a").mkdir()
    (controls / "ctrl_b").mkdir()

    return tmp_path


def test_collect_all_patients(dataset_root):
    datasets = enumerator.collect_datasets(dataset_root, use_all=True, ids=[], use_controls=False)
    assert datasets == [("001", False), ("002", False)]


def test_collect_all_controls(dataset_root):
    datasets = enumerator.collect_datasets(dataset_root, use_all=True, ids=[], use_controls=True)
    assert datasets == [("ctrl_a", True), ("ctrl_b", True)]


def test_collect_specific_ids(dataset_root):
    datasets = enumerator.collect_datasets(dataset_root, use_all=False, ids=["001"], use_controls=False)
    assert datasets == [("001", False)]


def test_collect_specific_ids_marked_as_controls(dataset_root):
    datasets = enumerator.collect_datasets(dataset_root, use_all=False, ids=["ctrl_a"], use_controls=True)
    assert datasets == [("ctrl_a", True)]


def test_collect_ids_detect_control_directory(dataset_root):
    datasets = enumerator.collect_datasets(dataset_root, use_all=False, ids=["ctrl_a"], use_controls=False)
    assert datasets == [("ctrl_a", True)]


def test_run_montage_for_dataset_generates_images(tmp_path, monkeypatch):
    dataset = tmp_path / "001"
    (dataset / "Analysis").mkdir(parents=True)
    (dataset / "Images").mkdir()
    nifti_dir = dataset / "NIfTI"
    nifti_dir.mkdir()
    dce_file = nifti_dir / "WIPhperf120long.nii"
    dce_file.write_bytes(b"")

    called = {}

    def fake_loader():
        def fake_generate(
            analysis_directory,
            image_directory,
            dce_path,
            *,
            anatomical_overlay=None,
        ):
            called["args"] = (analysis_directory, image_directory, dce_path, anatomical_overlay)

        class DummyParameters:
            @staticmethod
            def global_filenames(_):
                return ("", "", "", "", "", "", "", "", "WIPhperf120long.nii")

            @staticmethod
            def control_filenames(_):
                raise AssertionError("control_filenames should not be used for patient data")

        return fake_generate, DummyParameters

    monkeypatch.setattr(enumerator, "_load_montage_dependencies", fake_loader)

    assert enumerator._run_montage_for_dataset(tmp_path, "001", False) is True
    assert called["args"] == (
        str(dataset / "Analysis"),
        str(dataset / "Images"),
        str(dce_file),
        None,
    )


def test_run_montage_for_dataset_with_anatomical_overlay(tmp_path, monkeypatch):
    dataset = tmp_path / "001"
    (dataset / "Analysis").mkdir(parents=True)
    (dataset / "Images").mkdir()
    nifti_dir = dataset / "NIfTI"
    nifti_dir.mkdir()
    dce_file = nifti_dir / "WIPhperf120long.nii"
    dce_file.write_bytes(b"")

    overlay_dir = nifti_dir / "segmentation" / "segmentation" / "mri"
    overlay_dir.mkdir(parents=True)
    overlay_file = overlay_dir / "T1w_conformed_in_DCE.nii.gz"
    overlay_file.write_bytes(b"")

    called = {}

    def fake_loader():
        def fake_generate(
            analysis_directory,
            image_directory,
            dce_path,
            *,
            anatomical_overlay=None,
        ):
            called["args"] = (
                analysis_directory,
                image_directory,
                dce_path,
                anatomical_overlay,
            )

        class DummyParameters:
            @staticmethod
            def global_filenames(_):
                return ("", "", "", "", "", "", "", "", "WIPhperf120long.nii")

            @staticmethod
            def control_filenames(_):
                raise AssertionError("control_filenames should not be used for patient data")

        return fake_generate, DummyParameters

    monkeypatch.setattr(enumerator, "_load_montage_dependencies", fake_loader)

    assert (
        enumerator._run_montage_for_dataset(
            tmp_path, "001", False, use_anatomical=True
        )
        is True
    )
    assert called["args"] == (
        str(dataset / "Analysis"),
        str(dataset / "Images"),
        str(dce_file),
        str(overlay_file),
    )


def test_run_montage_for_dataset_with_nested_anatomical_overlay(tmp_path, monkeypatch):
    dataset = tmp_path / "001"
    (dataset / "Analysis").mkdir(parents=True)
    (dataset / "Images").mkdir()
    nifti_dir = dataset / "NIfTI"
    nifti_dir.mkdir()
    dce_file = nifti_dir / "WIPhperf120long.nii"
    dce_file.write_bytes(b"")

    overlay_dir = nifti_dir / "segmentation" / "segmentation" / "mri" / "transforms"
    overlay_dir.mkdir(parents=True)
    overlay_file = overlay_dir / "T1w_conformed_in_DCE.nii.gz"
    overlay_file.write_bytes(b"")

    called = {}

    def fake_loader():
        def fake_generate(
            analysis_directory,
            image_directory,
            dce_path,
            *,
            anatomical_overlay=None,
        ):
            called["args"] = (
                analysis_directory,
                image_directory,
                dce_path,
                anatomical_overlay,
            )

        class DummyParameters:
            @staticmethod
            def global_filenames(_):
                return ("", "", "", "", "", "", "", "", "WIPhperf120long.nii")

            @staticmethod
            def control_filenames(_):
                raise AssertionError("control_filenames should not be used for patient data")

        return fake_generate, DummyParameters

    monkeypatch.setattr(enumerator, "_load_montage_dependencies", fake_loader)

    assert (
        enumerator._run_montage_for_dataset(
            tmp_path, "001", False, use_anatomical=True
        )
        is True
    )
    assert called["args"] == (
        str(dataset / "Analysis"),
        str(dataset / "Images"),
        str(dce_file),
        str(overlay_file),
    )


def test_run_montage_for_dataset_rebuilds_missing_anatomical_overlay(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "001"
    (dataset / "Analysis").mkdir(parents=True)
    (dataset / "Images").mkdir()
    nifti_dir = dataset / "NIfTI"
    nifti_dir.mkdir()

    dce_file = nifti_dir / "WIPhperf120long.nii"
    dce_image = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4))
    dce_image.to_filename(str(dce_file))

    overlay_dir = nifti_dir / "segmentation" / "segmentation" / "mri"
    overlay_dir.mkdir(parents=True)
    t1_candidate = overlay_dir / "T1w_conformed.nii.gz"
    t1_image = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4))
    t1_image.to_filename(str(t1_candidate))

    called = {}

    def fake_loader():
        def fake_generate(
            analysis_directory,
            image_directory,
            dce_path,
            *,
            anatomical_overlay=None,
        ):
            called["args"] = (
                analysis_directory,
                image_directory,
                dce_path,
                anatomical_overlay,
            )

        class DummyParameters:
            @staticmethod
            def global_filenames(_):
                return ("", "", "", "", "", "", "", "", "WIPhperf120long.nii")

            @staticmethod
            def control_filenames(_):
                raise AssertionError("control_filenames should not be used for patient data")

        return fake_generate, DummyParameters

    monkeypatch.setattr(enumerator, "_load_montage_dependencies", fake_loader)
    monkeypatch.setattr(enumerator.shutil, "which", lambda _: None)

    assert (
        enumerator._run_montage_for_dataset(
            tmp_path, "001", False, use_anatomical=True
        )
        is True
    )

    overlay_path = overlay_dir / "T1w_conformed_in_DCE.nii.gz"
    assert overlay_path.exists()
    assert called["args"] == (
        str(dataset / "Analysis"),
        str(dataset / "Images"),
        str(dce_file),
        str(overlay_path),
    )


def test_run_montage_for_dataset_missing_dce(tmp_path, monkeypatch):
    dataset = tmp_path / "001"
    (dataset / "Analysis").mkdir(parents=True)
    (dataset / "Images").mkdir()
    (dataset / "NIfTI").mkdir()

    def fake_loader():
        def fake_generate(*args, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("generate_parametric_montages should not be invoked")

        class DummyParameters:
            @staticmethod
            def global_filenames(_):
                return ("", "", "", "", "", "", "", None, None)

            @staticmethod
            def control_filenames(_):
                return DummyParameters.global_filenames(_)

        return fake_generate, DummyParameters

    monkeypatch.setattr(enumerator, "_load_montage_dependencies", fake_loader)

    assert enumerator._run_montage_for_dataset(tmp_path, "001", False) is False


def _create_dataset(root, name, *, control=False):
    if control:
        base = root / "controls" / name
    else:
        base = root / name
    base.mkdir(parents=True)
    return base


def test_diffusion_only_processes_patients(monkeypatch, tmp_path):
    _create_dataset(tmp_path, "001")
    _create_dataset(tmp_path, "ctrl_a", control=True)

    calls = []

    def fake_diffusion(data_root, dataset_id, is_control):
        calls.append((data_root, dataset_id, is_control))
        return True

    monkeypatch.setattr(enumerator, "_run_diffusion_for_dataset", fake_diffusion)

    def fake_montage(*args, **kwargs):  # pragma: no cover - should not run
        raise AssertionError("Montage rendering should not run without --montage")

    monkeypatch.setattr(enumerator, "_run_montage_for_dataset", fake_montage)

    def fail_run(*args, **kwargs):  # pragma: no cover - should not be invoked
        raise AssertionError("Pipeline should not execute in diffusion-only mode")

    monkeypatch.setattr(enumerator.subprocess, "run", fail_run)

    argv = [
        "enumerator.py",
        "--diffusion_only",
        "--data-dir",
        str(tmp_path),
        "001",
        "ctrl_a",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc:
        enumerator.main()

    assert exc.value.code == 0
    assert calls == [(str(tmp_path), "001", False)]


def test_diffusion_only_with_montage_runs_once(monkeypatch, tmp_path):
    _create_dataset(tmp_path, "001")
    _create_dataset(tmp_path, "ctrl_a", control=True)

    diffusion_calls = []
    montage_calls = []

    def fake_diffusion(data_root, dataset_id, is_control):
        diffusion_calls.append((data_root, dataset_id, is_control))
        return True

    def fake_montage(
        data_root,
        dataset_id,
        is_control,
        *,
        use_projection=False,
        projection_stats=None,
        use_anatomical=False,
    ):
        montage_calls.append(
            (
                data_root,
                dataset_id,
                is_control,
                use_projection,
                projection_stats,
                use_anatomical,
            )
        )
        return True

    monkeypatch.setattr(enumerator, "_run_diffusion_for_dataset", fake_diffusion)
    monkeypatch.setattr(enumerator, "_run_montage_for_dataset", fake_montage)

    def fail_run(*args, **kwargs):  # pragma: no cover - should not be invoked
        raise AssertionError("Pipeline should not execute in diffusion-only mode")

    monkeypatch.setattr(enumerator.subprocess, "run", fail_run)

    argv = [
        "enumerator.py",
        "--diffusion_only",
        "--montage",
        "--data-dir",
        str(tmp_path),
        "001",
        "ctrl_a",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc:
        enumerator.main()

    assert exc.value.code == 0
    assert diffusion_calls == [(str(tmp_path), "001", False)]
    assert montage_calls == [
        (
            str(tmp_path),
            "001",
            False,
            False,
            None,
            False,
        )
    ]


def test_montage_anatomical_implies_montage(monkeypatch, tmp_path):
    _create_dataset(tmp_path, "001")

    montage_calls = []

    def fake_montage(
        data_root,
        dataset_id,
        is_control,
        *,
        use_projection=False,
        projection_stats=None,
        use_anatomical=False,
    ):
        montage_calls.append(
            (
                data_root,
                dataset_id,
                is_control,
                use_projection,
                projection_stats,
                use_anatomical,
            )
        )
        return True

    monkeypatch.setattr(enumerator, "_run_montage_for_dataset", fake_montage)

    def fail_run(*args, **kwargs):  # pragma: no cover - should not be invoked
        raise AssertionError("Pipeline should not execute when only montages are requested")

    monkeypatch.setattr(enumerator.subprocess, "run", fail_run)

    argv = [
        "enumerator.py",
        "--montage_anatomical",
        "--data-dir",
        str(tmp_path),
        "001",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc:
        enumerator.main()

    assert exc.value.code == 0
    assert montage_calls == [
        (
            str(tmp_path),
            "001",
            False,
            False,
            None,
            True,
        )
    ]
