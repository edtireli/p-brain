"""
p-brain Windows CLI.

Command-line interface for running the p-brain neuroimaging pipeline on
Windows.  No FreeSurfer, FastSurfer, or FSL required — all neuroimaging
operations are handled by pure-Python equivalents.

Usage:
    python -m windows --id <subject_id> [options]
    python -m windows --help
"""

from __future__ import annotations

import argparse
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pbrain-windows",
        description=(
            "p-brain Windows CLI — pure-Python neuroimaging analysis.\n"
            "Produces identical outputs to the macOS version."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Required ----
    parser.add_argument(
        "--id", type=str, required=True,
        help="Subject / patient ID (folder name under --data-dir).",
    )

    # ---- Data layout ----
    parser.add_argument(
        "--data-dir", dest="data_dir", type=str,
        default=os.environ.get("P_BRAIN_DATA_DIR", "./Data"),
        help="Root directory containing subject folders (default: ./Data).",
    )

    # ---- Segmentation ----
    parser.add_argument(
        "--synthseg-home", dest="synthseg_home", type=str, default=None,
        help="Path to cloned SynthSeg repository (default: SYNTHSEG_HOME env or vendor/SynthSeg).",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU inference for SynthSeg (no GPU).",
    )

    # ---- PK modelling ----
    parser.add_argument(
        "--pk-model", dest="pk_model", type=str,
        default="both",
        help=(
            "PK model(s): patlak | tikhonov | both | etofts | all.  "
            "Combine with '+': e.g. patlak+etofts.  Default: both (patlak + tikhonov)."
        ),
    )
    parser.add_argument(
        "--lambda", dest="tikhonov_lambda", type=float, default=None,
        help="Tikhonov regularisation weight (disables auto L-curve).",
    )

    # ---- T1 fitting ----
    parser.add_argument(
        "--t1-fit", dest="t1_fit", type=str,
        choices=["auto", "ir", "vfa", "none"],
        default="auto",
        help="T1/M0 fitting source: auto (default) | ir | vfa | none.",
    )

    # ---- ROI method ----
    parser.add_argument(
        "--roi-method", dest="roi_method", type=str,
        choices=["deterministic", "geometry", "file"],
        default="deterministic",
        help="ROI extraction method (default: deterministic).  "
             "'ai' is not available on Windows.",
    )

    # ---- Tissue ROI ----
    parser.add_argument(
        "--tissue-roi", dest="tissue_roi", type=str,
        choices=["automatic", "manual"],
        default="automatic",
        help="Tissue ROI method: automatic (default) uses SynthSeg, "
             "manual lets the user draw ROIs on DCE slices.",
    )

    # ---- Flip angle ----
    parser.add_argument(
        "--flip-angle", dest="flip_angle", type=str, default=None,
        help="Flip angle in degrees (number) or 'auto' (default: from metadata).",
    )

    # ---- p-brain-web defaults ----
    parser.add_argument(
        "--defaults-json", dest="defaults_json", type=str, default=None,
        help="Path to a p-brain-web Defaults JSON file.",
    )

    # ---- Misc ----
    parser.add_argument(
        "--force-masks", dest="force_masks", action="store_true",
        help="Force re-creation of all tissue masks even if they exist.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Normalise ROI method
    if args.roi_method == "geometry":
        args.roi_method = "deterministic"

    print(f"p-brain Windows CLI  v{_version()}")
    print(f"  Subject:   {args.id}")
    print(f"  Data dir:  {args.data_dir}")
    print(f"  PK model:  {args.pk_model}")
    print(f"  T1 fit:    {args.t1_fit}")
    print(f"  ROI:       {args.roi_method}")
    print(f"  Tissue ROI:{args.tissue_roi}")
    print()

    from windows.pipeline import run_pipeline

    run_pipeline(
        subject_id=args.id,
        data_dir=args.data_dir,
        synthseg_home=args.synthseg_home,
        pk_model=args.pk_model,
        t1_fit=args.t1_fit,
        roi_method=args.roi_method,
        tissue_roi_method=args.tissue_roi,
        defaults_json=args.defaults_json,
        force_recreate_masks=args.force_masks,
        cpu=args.cpu,
        tikhonov_lambda=args.tikhonov_lambda,
        flip_angle=args.flip_angle,
    )


def _version() -> str:
    try:
        from windows import __version__
        return __version__
    except Exception:
        return "dev"


if __name__ == "__main__":
    main()
