"""Centralised logging for p-Brain.

One configured logger (``pbrain``) replaces scattered ``print`` calls so
that pipeline output respects ``--quiet`` / ``--verbose`` and can be
mirrored to a log file. Plug-ins and stages obtain it via
:func:`get_logger`; the CLI configures verbosity once at startup with
:func:`configure`.

Levels:
  * ``--quiet``   → WARNING and above only
  * default       → INFO (stage progress, parameter summaries)
  * ``--verbose`` → DEBUG (per-voxel / per-iteration detail)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = "pbrain"
_CONFIGURED = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the ``pbrain`` logger (or a ``pbrain.<name>`` child)."""
    return logging.getLogger(_ROOT if not name else f"{_ROOT}.{name}")


def configure(level: int = logging.INFO, *, log_file: Path | str | None = None,
              fmt: str | None = None) -> logging.Logger:
    """Configure the root ``pbrain`` logger. Idempotent.

    Parameters
    ----------
    level
        ``logging.WARNING`` (quiet), ``INFO`` (default) or ``DEBUG`` (verbose).
    log_file
        If given, also append all records (always DEBUG) to this file.
    """
    global _CONFIGURED
    logger = logging.getLogger(_ROOT)
    logger.setLevel(logging.DEBUG)         # handlers gate the actual output
    logger.propagate = False

    # Reset handlers so re-configuration (e.g. per cohort subject) is clean.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = fmt or "%(message)s"
    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console)

    if log_file is not None:
        fh = logging.FileHandler(str(log_file), mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s | %(message)s"))
        logger.addHandler(fh)

    _CONFIGURED = True
    return logger


def level_from_flags(quiet: bool = False, verbose: bool = False) -> int:
    """Map ``--quiet``/``--verbose`` CLI flags to a ``logging`` level.

    ``quiet`` → ``WARNING``, ``verbose`` → ``DEBUG``, neither → ``INFO``.
    ``quiet`` wins if both are set.
    """
    if quiet:
        return logging.WARNING
    if verbose:
        return logging.DEBUG
    return logging.INFO
