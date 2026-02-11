#!/usr/bin/env python3
"""Stage-by-stage parity runner against MATLAB reference outputs.

This is an integration harness intended to mirror the validation script style,
but driven via the production `p-brain` CLI entrypoint (`main.py`).

Usage example (T1/M0 stage only):

  python tools/stage_validate_matlab.py \
    --data-dir /Volumes/T5_EVO_EDT/hemisure \
    --id 20240618x2_flot \
    --stage t1m0 \
    --defaults-json tools/pbrain_defaults_example.json \
    --mat-t1m0 /Volumes/T5_EVO_EDT/hemisure/20240618x2_flot/T1_M0_plusError_maps_.mat

Outputs are written under the subject's `Analysis/` and `Images/` folders.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run p-brain stage(s) and compare to MATLAB.")
    p.add_argument("--data-dir", required=True, help="Directory containing subject folder(s).")
    p.add_argument("--id", required=True, help="Subject folder name (e.g. 20240618x2_flot).")
    p.add_argument(
        "--stage",
        required=True,
        choices=["t1m0", "ctc"],
        help="Which stage to run and validate (start with: t1m0).",
    )
    p.add_argument(
        "--defaults-json",
        default=None,
        help="Optional p-brain-web Defaults JSON to apply (passed to main.py).",
    )
    p.add_argument(
        "--mat-t1m0",
        default=None,
        help="Optional explicit MATLAB T1/M0 .mat path (passed to main.py).",
    )
    p.add_argument(
        "--mat-ctc",
        default=None,
        help="MATLAB CTC reference .mat containing s_input/c_input/time/BW_input/slice_c_input.",
    )
    p.add_argument(
        "--dce-nifti",
        default=None,
        help="Optional DCE NIfTI path for ROI signal extraction check (default: subject/NIfTI/WIPhperf120long.nii).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force recompute stage outputs (clears cached outputs for the stage).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands but do not execute.",
    )
    return p.parse_args()


def _run(cmd: list[str], *, dry_run: bool) -> int:
    sys.stdout.write("\n$ " + " ".join(cmd) + "\n")
    sys.stdout.flush()
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def _t1m0_command(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(MAIN)]
    cmd += ["--mode", "auto"]
    cmd += ["--data-dir", str(args.data_dir)]
    cmd += ["--id", str(args.id)]
    cmd += ["--t1m0-only"]
    cmd += ["--compare-matlab"]
    if args.defaults_json:
        cmd += ["--defaults-json", str(args.defaults_json)]
    if args.mat_t1m0:
        cmd += ["--compare-matlab-path", str(args.mat_t1m0)]
    if args.force:
        cmd += ["--t1m0-force"]
    return cmd


def _load_compare_json(subject_root: Path) -> dict:
    p = subject_root / "Analysis" / "Fitting" / "compare_matlab_t1m0.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing compare JSON: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    args = _parse_args()

    subject_root = Path(args.data_dir).expanduser().resolve() / str(args.id)
    if not subject_root.exists():
        sys.stderr.write(f"Subject not found: {subject_root}\n")
        return 2

    # Ensure validator-style NIfTI orientation in comparisons.
    os.environ.setdefault("P_BRAIN_COMPARE_MATLAB_USE_VALIDATOR_ORIENT", "1")
    os.environ.setdefault("P_BRAIN_VALIDATOR_NIFTI_ROT90_K", "1")
    os.environ.setdefault("P_BRAIN_VALIDATOR_NIFTI_FLIP_LR", "0")
    os.environ.setdefault("P_BRAIN_VALIDATOR_NIFTI_FLIP_UD", "0")

    if args.stage == "t1m0":
        cmd = _t1m0_command(args)
        rc = _run(cmd, dry_run=bool(args.dry_run))
        if rc != 0:
            return rc

        if args.dry_run:
            return 0

        d = _load_compare_json(subject_root)
        t1_corr = (d.get("metrics", {}) or {}).get("t1", {}).get("corr")
        m0_corr = (d.get("metrics", {}) or {}).get("m0", {}).get("corr")
        slice_metrics = (d.get("metrics", {}) or {}).get("slice", {}) or {}
        z1 = slice_metrics.get("z1")
        t1_slice = (slice_metrics.get("t1") or {})
        m0_slice = (slice_metrics.get("m0") or {})
        sys.stdout.write("\n")
        try:
            t1_corr_f = float(t1_corr)
        except Exception:
            t1_corr_f = float("nan")
        try:
            m0_corr_f = float(m0_corr)
        except Exception:
            m0_corr_f = float("nan")

        # Match validator-style printing precision (corr is shown with 9 decimals elsewhere).
        sys.stdout.write(f"T1 corr: {t1_corr_f:.15f}\n")
        sys.stdout.write(f"M0 corr: {m0_corr_f:.15f}\n")

        if z1:
            try:
                sys.stdout.write(
                    f"slice z={int(z1)} T1 max_abs={t1_slice.get('max_abs')} mean_abs={t1_slice.get('mean_abs')} n={t1_slice.get('n')}\n"
                )
                sys.stdout.write(
                    f"slice z={int(z1)} M0 max_abs={m0_slice.get('max_abs')} mean_abs={m0_slice.get('mean_abs')} n={m0_slice.get('n')}\n"
                )
            except Exception:
                pass
        sys.stdout.write(f"Compare JSON: {subject_root / 'Analysis' / 'Fitting' / 'compare_matlab_t1m0.json'}\n")
        sys.stdout.write(f"Compare PNGs: {subject_root / 'Images' / 'Fit'}\n")
        return 0

    if args.stage == "ctc":
        if not args.mat_ctc:
            sys.stderr.write("--mat-ctc is required for --stage ctc\n")
            return 2

        # Ensure fitted maps exist (CTC ROI uses T1/M0 maps).
        cmd = _t1m0_command(args)
        rc = _run(cmd, dry_run=bool(args.dry_run))
        if rc != 0:
            return rc
        if args.dry_run:
            return 0

        # Run CTC comparison.
        sys.path.insert(0, str(ROOT))
        from utils.compare_matlab_ctc import compare_matlab_ctc

        payload = compare_matlab_ctc(
            subject_root,
            mat_ctc_path=str(args.mat_ctc),
            dce_nifti_path=str(args.dce_nifti) if args.dce_nifti else None,
            write_outputs=True,
        )

        c_met = (payload.get("metrics") or {}).get("c") or {}
        s_met = (payload.get("metrics") or {}).get("s_from_nifti")
        c_nifti_met = (payload.get("metrics") or {}).get("c_from_nifti")
        c_nifti_met = (payload.get("metrics") or {}).get("c_from_nifti")

        sys.stdout.write("\n")
        sys.stdout.write(f"CTC rmse: {c_met.get('rmse')}\n")
        sys.stdout.write(f"CTC max_abs: {c_met.get('max_abs')}\n")
        sys.stdout.write(f"CTC corr: {c_met.get('corr')}\n")
        if s_met is not None:
            sys.stdout.write(f"ROI signal (NIfTI vs MATLAB) rmse: {s_met.get('rmse')}\n")
            sys.stdout.write(f"ROI signal (NIfTI vs MATLAB) corr: {s_met.get('corr')}\n")
        if c_nifti_met is not None:
            sys.stdout.write(f"CTC from NIfTI rmse: {c_nifti_met.get('rmse')}\n")
            sys.stdout.write(f"CTC from NIfTI max_abs: {c_nifti_met.get('max_abs')}\n")
            sys.stdout.write(f"CTC from NIfTI corr: {c_nifti_met.get('corr')}\n")
        if c_nifti_met is not None:
            sys.stdout.write(f"CTC from NIfTI signal rmse: {c_nifti_met.get('rmse')}\n")
            sys.stdout.write(f"CTC from NIfTI signal max_abs: {c_nifti_met.get('max_abs')}\n")
            sys.stdout.write(f"CTC from NIfTI signal corr: {c_nifti_met.get('corr')}\n")

        sys.stdout.write(f"Compare JSON: {subject_root / 'Analysis' / 'Fitting' / 'compare_matlab_ctc.json'}\n")
        sys.stdout.write(f"Compare PNGs: {subject_root / 'Images' / 'Fit'}\n")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
