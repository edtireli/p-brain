#!/usr/bin/env python3
"""CLI helper to generate T1/M0 comparison artifacts against MATLAB outputs.

Outputs the same JSON + montage PNG as the library helper `compare_t1m0_to_matlab`,
so you can run a single command instead of dropping into a Python REPL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is importable when invoked from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.compare_matlab import compare_t1m0_to_matlab  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate T1/M0 comparison metrics and montage vs MATLAB exports.",
    )
    subject = parser.add_mutually_exclusive_group(required=True)
    subject.add_argument(
        "--subject-root",
        help="Path to the subject directory (expects Analysis/ and Images/ under it).",
    )
    subject.add_argument(
        "--id",
        help="Subject ID (e.g. 20240618x2_flot) when used with --data-dir.",
    )
    parser.add_argument(
        "--data-dir",
        help="Data root that contains the subject directory; used with --id.",
    )
    parser.add_argument(
        "--analysis-dir",
        help="Override Analysis directory path; defaults to <subject-root>/Analysis.",
    )
    parser.add_argument(
        "--images-dir",
        help="Override Images directory path; defaults to <subject-root>/Images.",
    )
    parser.add_argument(
        "--mat-path",
        help="Explicit MATLAB .mat path (defaults to auto-detecting T1_M0_plusError_maps_.mat).",
    )
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.subject_root:
        subject_root = Path(args.subject_root).expanduser().resolve()
    else:
        if not args.data_dir:
            raise SystemExit("--data-dir is required when using --id")
        subject_root = Path(args.data_dir).expanduser().resolve() / args.id

    analysis_dir = (
        Path(args.analysis_dir).expanduser().resolve()
        if args.analysis_dir
        else subject_root / "Analysis"
    )
    images_dir = (
        Path(args.images_dir).expanduser().resolve()
        if args.images_dir
        else subject_root / "Images"
    )
    return subject_root, analysis_dir, images_dir


def main() -> int:
    args = _parse_args()
    subject_root, analysis_dir, images_dir = _resolve_paths(args)
    mat_path = Path(args.mat_path).expanduser().resolve() if args.mat_path else None

    result = compare_t1m0_to_matlab(
        subject_root=str(subject_root),
        analysis_directory=str(analysis_dir),
        image_directory=str(images_dir),
        mat_path=str(mat_path) if mat_path else None,
    )

    if result is None:
        sys.stderr.write("Compare failed: missing inputs or MATLAB reference.\n")
        return 1

    summary = {
        "mat_path": result.mat_path,
        "t1_units": {
            "matlab": result.t1_units_matlab,
            "nifti": result.t1_units_nifti,
        },
        "t1_scale_applied_to_nifti": result.t1_scale_applied_to_nifti,
        "m0_scale_applied_to_nifti": result.m0_scale_applied_to_nifti,
        "metrics": result.metrics,
        "paths": result.paths,
    }
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")

    png_path = result.paths.get("png", "")
    if png_path:
        sys.stdout.write(f"Wrote montage PNG to {png_path}\n")
    else:
        sys.stdout.write("Montage PNG not generated (matplotlib missing or error).\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
