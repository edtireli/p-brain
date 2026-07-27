"""Bundled acquisition profiles for ``pbrain run --profile <name>``.

A profile is a preset that pre-selects the plug-ins and acquisition parameters
for a whole class of data, so a user need not re-type a long ``--opt`` pile. Each
value has the SAME shape :func:`pbrain.cli._config_file.load_config_file` returns:
a flat dict of argparse *dests* plus an ``opt`` list of fully-qualified
``<plug-point>.<plugin>.<key>=<value>`` strings.

The overlay is applied only where the CLI is silent and *before* ``--config``, so
precedence is: explicit CLI flag > ``--config`` > ``--profile`` > argparse default.
A human run (no ``--profile``) never enters the overlay, so the paper defaults are
untouched.

Per-subject *paths* are deliberately excluded (they differ per subject) — supply
them via ``--opt`` or ``--config``.

No profiles ship in this release; the table is the extension point.
"""

from __future__ import annotations

from typing import Any

PROFILES: dict[str, dict[str, Any]] = {}


def resolve_profile(name: str) -> dict[str, Any]:
    """Return a fresh copy of the named profile's overlay dict.

    Raises ``SystemExit`` (listing the available profiles) on an unknown name.
    """
    try:
        prof = PROFILES[name]
    except KeyError:
        raise SystemExit(
            f"--profile {name!r}: unknown. Available: {sorted(PROFILES)}"
        )
    return {k: (list(v) if k == "opt" else v) for k, v in prof.items()}
