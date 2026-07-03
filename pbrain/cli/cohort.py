"""``python -m pbrain run-cohort`` — run the pipeline over many subjects.

Replaces ad-hoc batch shell scripts with a first-class command: a subject
list or glob, a per-subject **process pool**, and **continue-on-failure** with
a summary report. Resume (cached stages) makes re-running a cohort after a
crash or a code tweak cheap.

Two input modes:

* ``--cohort DIR`` — **raw-scanner mode**. Every immediate sub-directory of
  ``DIR`` is a subject of raw Philips PAR/REC. Inputs are auto-discovered by
  protocol name (DCE = ``hperf*``; the ``TI_*`` saturation-recovery series is
  assembled into an IR; a 3-D T1 anatomical for SynthSeg) and the full pipeline
  runs with all kinetic models at tissue (region) + parcel level. No config
  needed — this is the one-flag "do the whole study" path::

      python -m pbrain run-cohort --cohort /data/patients --workers 4 --force

* ``--config + --data-dir/--subjects-glob/--subjects`` — **config mode**. Each
  subject runs exactly as ``pbrain run`` with inputs resolved from the config
  (relative to each subject dir, or templated). For pre-converted NIfTI cohorts.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pbrain.core import configure_logging, get_logger, level_from_flags

# Raw-scanner cohort defaults (overridable with --models / --aggregations).
COHORT_MODELS = "patlak,tikhonov,gamma,mittag_leffler,inverse_gaussian,stieltjes,extended_tofts"
COHORT_AGGS = "region,parcel"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pbrain run-cohort", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--cohort", nargs="+", metavar="DIR",
                   help="One or more raw-scanner cohort roots: run every sub-directory "
                        "as a subject with auto-discovered inputs (DCE/IR/T1) and all "
                        "models at tissue+parcel level. No --config needed. "
                        "e.g. --cohort /data/patients /data/controls")
    g.add_argument("--data-dir", help="Run every immediate sub-directory of this "
                                      "folder as a subject (inputs from --config).")
    g.add_argument("--subjects-glob", help="Glob of subject directories, e.g. '/data/sub-*'.")
    g.add_argument("--subjects", nargs="+", help="Explicit subject directories.")
    p.add_argument("--config", type=Path, default=None,
                   help="Shared config (.toml/.yaml). Required unless --cohort is used.")
    p.add_argument("--models", default=None,
                   help=f"[--cohort] kinetic models (default: {COHORT_MODELS}).")
    p.add_argument("--aggregations", default=None,
                   help=f"[--cohort] aggregation levels (default: {COHORT_AGGS}).")
    p.add_argument("--voxelwise", action="store_true",
                   help="[--cohort] fit models voxelwise instead of the default "
                        "parcel-level average-then-fit (slower).")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="[--cohort] subject names to skip (e.g. known-bad acquisitions).")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel subject processes (default 1 = serial).")
    p.add_argument("--force", action="store_true", help="Ignore cached stages (fresh run).")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-file", type=Path, default=None)
    return p


def _config_argv(subject_dir: Path, cfg: Path, force: bool,
                 quiet: bool, verbose: bool) -> list[str]:
    """``pbrain run`` argv for one subject in config mode."""
    argv = ["--config", str(cfg), "--subject-dir", str(subject_dir)]
    if force:
        argv.append("--force")
    if quiet:
        argv.append("--quiet")
    if verbose:
        argv.append("--verbose")
    return argv


def _cohort_argv(subject_root: Path, inputs: dict, *, models: str, aggregations: str,
                 parcel: bool, force: bool, verbose: bool) -> list[str]:
    """``pbrain run`` argv for one auto-discovered raw-scanner subject."""
    argv = ["--subject-dir", str(Path(subject_root) / "pbrain"),
            "--dce", str(inputs["dce"]),
            "--t1m0", "inversion_recovery",
            "--aif", "cnn_sss_shifted",
            "--tissue-roi", "synthseg",
            "--models", models,
            "--aggregations", aggregations]
    if inputs.get("ir"):
        argv += ["--relax", str(inputs["ir"])]
    if inputs.get("t1"):
        argv += ["--t1", str(inputs["t1"])]
    if parcel:
        for m in models.split(","):
            argv += ["--opt", f"models.{m.strip()}.level=parcel"]
    if force:
        argv.append("--force")
    argv.append("--quiet")
    if verbose:
        argv.append("--verbose")
    return argv


def _run_one(args_tuple) -> tuple[str, str, str]:
    """Worker: run one subject. Returns (subject, status, message)."""
    subject, params = args_tuple
    from pbrain.cli.run import main as run_main
    name = Path(subject).name
    try:
        if params["mode"] == "cohort":
            from pbrain.io.subject_discovery import discover_subject_inputs
            inputs = discover_subject_inputs(subject)
            if inputs["dce"] is None:
                return (name, "failed", "no DCE (hperf) scan found")
            argv = _cohort_argv(Path(subject), inputs, models=params["models"],
                                aggregations=params["aggregations"], parcel=params["parcel"],
                                force=params["force"], verbose=params["verbose"])
        else:
            argv = _config_argv(Path(subject), Path(params["config"]),
                                params["force"], True, params["verbose"])
        rc = run_main(argv)
        return (name, "ok" if rc == 0 else "failed", f"exit {rc}")
    except SystemExit as exc:
        return (name, "failed", f"SystemExit: {exc}")
    except Exception as exc:  # noqa: BLE001 — isolate per-subject failures
        return (name, "failed", f"{type(exc).__name__}: {exc}")


def main(argv: list[str]) -> int:
    """``pbrain cohort`` entry point — run the pipeline over many subjects.

    Parses ``argv``, discovers subjects under the cohort root, runs each
    through the pipeline, and writes a cohort-level summary. Returns the
    process exit code.
    """
    from pbrain._console import ensure_utf8_console
    ensure_utf8_console()

    args = _build_parser().parse_args(argv)
    configure_logging(level_from_flags(args.quiet, args.verbose), log_file=args.log_file)
    log = get_logger("cohort")

    cohort_mode = bool(args.cohort)
    if not cohort_mode and args.config is None:
        raise SystemExit("--config is required unless --cohort is used")

    if cohort_mode:
        excl = set(args.exclude or [])
        subjects = []
        for r in args.cohort:
            root = Path(r)
            if not root.is_dir():
                raise SystemExit(f"--cohort: not a directory: {root}")
            subjects += [str(p) for p in root.iterdir()
                         if p.is_dir() and not p.name.startswith((".", "_"))
                         and p.name not in excl]
        subjects = sorted(subjects)
        params = {
            "mode": "cohort",
            "models": args.models or COHORT_MODELS,
            "aggregations": args.aggregations or COHORT_AGGS,
            "parcel": not args.voxelwise,
            "force": args.force,
            "verbose": args.verbose,
        }
    else:
        if args.data_dir:
            root = Path(args.data_dir)
            subjects = sorted(str(p) for p in root.iterdir()
                              if p.is_dir() and not p.name.startswith("."))
        elif args.subjects_glob:
            import glob
            subjects = sorted(p for p in glob.glob(args.subjects_glob) if Path(p).is_dir())
        else:
            subjects = list(args.subjects)
        params = {"mode": "config", "config": str(args.config),
                  "force": args.force, "verbose": args.verbose}

    if not subjects:
        raise SystemExit("no subjects matched")

    log.info("cohort: %d subjects, %d worker(s), mode=%s%s", len(subjects), args.workers,
             params["mode"],
             f", models={params['models']}, aggs={params['aggregations']}, "
             f"level={'parcel' if params['parcel'] else 'voxelwise'}"
             if cohort_mode else f", config={args.config}")

    work = [(s, params) for s in subjects]
    results: list[tuple[str, str, str]] = []
    t0 = time.time()

    if args.workers <= 1:
        for w in work:
            r = _run_one(w)
            results.append(r)
            log.info("  [%d/%d] %-16s %s", len(results), len(subjects), r[0], r[1])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_one, w): w[0] for w in work}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                log.info("  [%d/%d] %-16s %s", len(results), len(subjects), r[0], r[1])

    ok = [r for r in results if r[1] == "ok"]
    bad = [r for r in results if r[1] != "ok"]
    log.info("cohort done in %.0fs — %d ok, %d failed", time.time() - t0, len(ok), len(bad))
    for name, _, msg in bad:
        log.warning("  FAILED %-16s %s", name, msg)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
