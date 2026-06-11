"""``python -m pbrain run-cohort`` — run the pipeline over many subjects.

Replaces ad-hoc batch shell scripts with a first-class command: a subject
list or glob, a shared config, a per-subject **process pool**, and
**continue-on-failure** with a summary report. Resume (cached stages) makes
re-running a cohort after a crash or a code tweak cheap.

Example::

    python -m pbrain run-cohort \\
        --config study.toml \\
        --subjects-glob '/data/sub-*' \\
        --workers 8

Each subject is run exactly as ``pbrain run`` would, with the config /
options from ``--config`` plus per-subject input resolution (the subject's
own dce/ir/t1/dwi found relative to its directory, or templated).
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pbrain.core import configure_logging, get_logger, level_from_flags


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pbrain run-cohort", description=__doc__)
    p.add_argument("--config", type=Path, required=True,
                   help="Shared config file (.toml/.yaml). Per-subject inputs "
                        "are resolved relative to each subject directory.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--data-dir", help="Run every immediate sub-directory of this "
                                      "folder as a subject (the simplest 'do all of them').")
    g.add_argument("--subjects-glob", help="Glob of subject directories, e.g. '/data/sub-*'.")
    g.add_argument("--subjects", nargs="+", help="Explicit subject directories.")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel subject processes (default 1 = serial).")
    p.add_argument("--force", action="store_true", help="Ignore cached stages.")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-file", type=Path, default=None)
    return p


def _subject_argv(subject_dir: Path, cfg: Path, force: bool,
                  quiet: bool, verbose: bool) -> list[str]:
    """Build the ``pbrain run`` argv for one subject.

    Inputs come from the config file; we only set --subject-dir here so the
    config's relative input names resolve against this subject. Absolute input
    paths in the config still win.
    """
    argv = ["--config", str(cfg), "--subject-dir", str(subject_dir)]
    if force:
        argv.append("--force")
    if quiet:
        argv.append("--quiet")
    if verbose:
        argv.append("--verbose")
    return argv


def _run_one(args_tuple) -> tuple[str, str, str]:
    """Worker: run one subject. Returns (subject, status, message)."""
    subject_dir, cfg, force, quiet, verbose = args_tuple
    from pbrain.cli.run import main as run_main
    name = Path(subject_dir).name
    try:
        rc = run_main(_subject_argv(Path(subject_dir), Path(cfg), force, quiet, verbose))
        return (name, "ok" if rc == 0 else "failed", f"exit {rc}")
    except SystemExit as exc:
        return (name, "failed", f"SystemExit: {exc}")
    except Exception as exc:  # noqa: BLE001 — isolate per-subject failures
        return (name, "failed", f"{type(exc).__name__}: {exc}")


def main(argv: list[str]) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging(level_from_flags(args.quiet, args.verbose), log_file=args.log_file)
    log = get_logger("cohort")

    if args.data_dir:
        root = Path(args.data_dir)
        subjects = sorted(str(p) for p in root.iterdir()
                          if p.is_dir() and not p.name.startswith("."))
    elif args.subjects_glob:
        import glob
        subjects = sorted(p for p in glob.glob(args.subjects_glob) if Path(p).is_dir())
    else:
        subjects = list(args.subjects)
    if not subjects:
        raise SystemExit("no subjects matched")

    log.info("cohort: %d subjects, %d worker(s), config=%s",
             len(subjects), args.workers, args.config)

    work = [(s, str(args.config), args.force, True, args.verbose) for s in subjects]
    results: list[tuple[str, str, str]] = []
    t0 = time.time()

    if args.workers <= 1:
        for w in work:
            r = _run_one(w)
            results.append(r)
            log.info("  [%d/%d] %-14s %s", len(results), len(subjects), r[0], r[1])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_one, w): w[0] for w in work}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                log.info("  [%d/%d] %-14s %s", len(results), len(subjects), r[0], r[1])

    ok = [r for r in results if r[1] == "ok"]
    bad = [r for r in results if r[1] != "ok"]
    log.info("cohort done in %.0fs — %d ok, %d failed", time.time() - t0, len(ok), len(bad))
    for name, _, msg in bad:
        log.warning("  FAILED %-14s %s", name, msg)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
