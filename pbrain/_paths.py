"""User-writable locations for downloaded assets (CNN weights, example data).

A pip-installed p-Brain lives under ``site-packages`` where users cannot (and
should not) drop large model files. These helpers give a stable, per-user,
writable home for the Zenodo-hosted CNN weights and the example dataset, so
``pbrain setup`` / ``pbrain fetch-weights`` can download them once and every
later run finds them automatically.

Resolution order for the base directory:
    1. ``$PBRAIN_HOME``                    (explicit override)
    2. ``$XDG_DATA_HOME/p-brain``          (Linux XDG convention, if set)
    3. ``~/.p-brain``                      (default)
"""

from __future__ import annotations

import os
from pathlib import Path


def user_data_dir() -> Path:
    """Base directory for p-Brain's downloaded assets (see module docstring)."""
    override = os.environ.get("PBRAIN_HOME")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg).expanduser() / "p-brain"
    return Path.home() / ".p-brain"


def weights_dir() -> Path:
    """Directory holding the CNN AIF ``.keras`` weights (``<user_data_dir>/AI``)."""
    return user_data_dir() / "AI"
