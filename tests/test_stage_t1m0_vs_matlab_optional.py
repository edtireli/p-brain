import json
import os
import subprocess
import sys
from pathlib import Path


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    p = Path(raw).expanduser().resolve()
    return p


def test_t1m0_stage_matches_matlab_when_dataset_available():
    """Optional integration test.

    Skips unless:
      PBRAIN_TEST_DATA_DIR=/path/to/data_root
      PBRAIN_TEST_SUBJECT_ID=20240618x2_flot

    Optionally:
      PBRAIN_TEST_T1M0_MAT=/path/to/T1_M0_plusError_maps_.mat
      PBRAIN_TEST_DEFAULTS_JSON=/path/to/defaults.json
    """

    data_dir = _env_path("PBRAIN_TEST_DATA_DIR")
    subject_id = (os.environ.get("PBRAIN_TEST_SUBJECT_ID") or "").strip()

    if data_dir is None or not subject_id:
        return  # intentionally silent skip for CI/dev convenience

    subject_root = data_dir / subject_id
    if not subject_root.exists():
        return

    repo_root = Path(__file__).resolve().parents[1]
    main_py = repo_root / "main.py"

    defaults_json = _env_path("PBRAIN_TEST_DEFAULTS_JSON")
    mat_path = _env_path("PBRAIN_TEST_T1M0_MAT")

    env = dict(os.environ)
    env["P_BRAIN_COMPARE_MATLAB_USE_VALIDATOR_ORIENT"] = "1"
    env.setdefault("P_BRAIN_VALIDATOR_NIFTI_ROT90_K", "1")
    env.setdefault("P_BRAIN_VALIDATOR_NIFTI_FLIP_LR", "0")
    env.setdefault("P_BRAIN_VALIDATOR_NIFTI_FLIP_UD", "0")

    cmd = [sys.executable, str(main_py), "--mode", "auto", "--data-dir", str(data_dir), "--id", subject_id, "--t1m0-only", "--compare-matlab", "--t1m0-force"]
    if defaults_json and defaults_json.exists():
        cmd += ["--defaults-json", str(defaults_json)]
    if mat_path and mat_path.exists():
        cmd += ["--compare-matlab-path", str(mat_path)]

    proc = subprocess.run(cmd, env=env)
    assert proc.returncode == 0

    compare_json = subject_root / "Analysis" / "Fitting" / "compare_matlab_t1m0.json"
    assert compare_json.exists()

    d = json.loads(compare_json.read_text(encoding="utf-8"))
    t1_corr = float((d.get("metrics", {}) or {}).get("t1", {}).get("corr") or 0.0)
    m0_corr = float((d.get("metrics", {}) or {}).get("m0", {}).get("corr") or 0.0)

    # Require near-perfect agreement.
    assert t1_corr > 0.9999
    assert m0_corr > 0.9999

    # Validator-like figure should exist.
    fit_dir = subject_root / "Images" / "Fit"
    assert any(p.name.startswith("t1_m0_compare_slice") and p.suffix == ".png" for p in fit_dir.glob("t1_m0_compare_slice*.png"))
