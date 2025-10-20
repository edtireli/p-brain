from pathlib import Path
import sys

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
