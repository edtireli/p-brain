"""Stage manifests — the file-based interface between stages.

Each stage writes a JSON ``manifest.json`` next to its outputs listing:

* ``stage``      — the stage's name (e.g. ``"t1_m0"``).
* ``plugin``     — which plug-in implementation was used
                   (e.g. ``"inversion_recovery"``).
* ``config``     — the resolved (post-defaults) plug-in options used.
* ``inputs``     — paths to upstream manifests this stage consumed.
* ``outputs``    — ``{name: path}`` for every artefact written.
* ``metadata``   — free-form provenance (timings, software versions, …).
* ``qc``         — status + messages from the stage's QC checks.
* ``started_at`` / ``finished_at`` — ISO-8601 timestamps.

Downstream stages read the manifest, not the directory listing — which
is the paper's "stable file-based interface" promise.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Manifest:
    """One stage's ``manifest.json`` — the file-based stage interface.

    See the module docstring for the field semantics. Construct with
    :meth:`now`, fill in ``outputs``/``metadata``/``qc`` during the run,
    call :meth:`finish` to stamp the end time, and :meth:`write` it next
    to the stage's outputs. Downstream stages :meth:`read` it back.
    """

    stage: str
    plugin: str
    config: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    qc: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""

    @classmethod
    def now(cls, stage: str, plugin: str) -> "Manifest":
        """Create a manifest with ``started_at`` stamped to now (UTC)."""
        m = cls(stage=stage, plugin=plugin)
        m.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return m

    def finish(self) -> "Manifest":
        """Stamp ``finished_at`` to now (UTC) and return ``self``."""
        self.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return self

    def write(self, path: Path) -> None:
        """Serialise to ``path`` as pretty-printed JSON (creating dirs)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=False, default=str)

    @classmethod
    def read(cls, path: Path) -> "Manifest":
        """Load a manifest previously written by :meth:`write`."""
        with Path(path).open() as f:
            data = json.load(f)
        return cls(**data)
