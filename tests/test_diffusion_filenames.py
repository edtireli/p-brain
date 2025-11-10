import importlib.util
import os
import sys
import types
from pathlib import Path

import matplotlib
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use = lambda *args, **kwargs: None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

modules_pkg = sys.modules.get("modules")
if modules_pkg is None:
    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = [os.path.join(ROOT, "modules")]
    sys.modules["modules"] = modules_pkg
elif not hasattr(modules_pkg, "__path__"):
    modules_pkg.__path__ = [os.path.join(ROOT, "modules")]
opt08_spec = importlib.util.spec_from_file_location(
    "modules.opt08_fa",
    os.path.join(ROOT, "modules", "opt08_fa.py"),
    submodule_search_locations=[os.path.join(ROOT, "modules")],
)
opt08_module = importlib.util.module_from_spec(opt08_spec)
sys.modules["modules.opt08_fa"] = opt08_module
opt08_spec.loader.exec_module(opt08_module)
find_dwi_files = opt08_module.find_dwi_files

utils_pkg = sys.modules.get("utils")
if utils_pkg is None:
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = [os.path.join(ROOT, "utils")]
    sys.modules["utils"] = utils_pkg
elif not hasattr(utils_pkg, "__path__"):
    utils_pkg.__path__ = [os.path.join(ROOT, "utils")]
parameters_spec = importlib.util.spec_from_file_location(
    "utils.parameters",
    os.path.join(ROOT, "utils", "parameters.py"),
    submodule_search_locations=[os.path.join(ROOT, "utils")],
)
parameters_module = importlib.util.module_from_spec(parameters_spec)
sys.modules["utils.parameters"] = parameters_module
parameters_spec.loader.exec_module(parameters_module)
get_diffusion_filename = parameters_module.get_diffusion_filename


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_get_diffusion_filename_prefers_dti(tmp_path):
    diffusion_dir = tmp_path / "NIfTI"
    diffusion_dir.mkdir()

    _touch(diffusion_dir / "WIPDTI_RSI_P.nii")
    _touch(diffusion_dir / "WIPDTI_RSI_A.nii")

    selected = get_diffusion_filename(
        (
            "WIPDTI_RSI_P.nii",
            "WIPDTI_RSI_A.nii",
            "WIPDWI_RSI_P.nii",
        ),
        diffusion_dir,
    )

    assert selected == "WIPDTI_RSI_P.nii"


def test_find_dwi_files_uses_preferred_dti_with_gradients(tmp_path):
    diffusion_dir = tmp_path / "NIfTI"
    diffusion_dir.mkdir()

    # Hidden macOS resource fork should be ignored.
    _touch(diffusion_dir / "._WIPDTI_RSI_P.nii")

    stem = diffusion_dir / "WIPDTI_RSI_P"
    _touch(stem.with_suffix(".nii"))

    bvals = np.zeros(5, dtype=float)
    bvecs = np.zeros((5, 3), dtype=float)
    np.savetxt(stem.with_suffix(".bval"), bvals)
    np.savetxt(stem.with_suffix(".bvec"), bvecs)

    # Provide an anterior volume without matching gradients – should be skipped.
    _touch(diffusion_dir / "WIPDTI_RSI_A.nii")

    dwi_path, bval_path, bvec_path = find_dwi_files(
        str(diffusion_dir),
        preferred_filenames=(
            "._WIPDTI_RSI_P.nii",
            "WIPDTI_RSI_A.nii",
            "WIPDTI_RSI_P.nii",
        ),
    )

    assert dwi_path == str(stem.with_suffix(".nii"))
    assert bval_path == str(stem.with_suffix(".bval"))
    assert bvec_path == str(stem.with_suffix(".bvec"))


def test_find_dwi_files_accepts_extensionless_preference(tmp_path):
    diffusion_dir = tmp_path / "NIfTI"
    diffusion_dir.mkdir()

    stem = diffusion_dir / "WIPDTI_RSI_P"
    _touch(stem.with_suffix(".nii.gz"))

    bvals = np.zeros(5, dtype=float)
    bvecs = np.zeros((5, 3), dtype=float)
    np.savetxt(stem.with_suffix(".bval"), bvals)
    np.savetxt(stem.with_suffix(".bvec"), bvecs)

    dwi_path, _, _ = find_dwi_files(
        str(diffusion_dir),
        preferred_filenames=("WIPDTI_RSI_P",),
    )

    assert dwi_path == str(stem.with_suffix(".nii.gz"))
