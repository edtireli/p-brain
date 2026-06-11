"""Output path scheme — the cross-cutting plug-point that decides where
stage outputs live on disk and what they are named.

A scheme is a Plugin implementing :class:`PathScheme`. The orchestrator
asks the scheme for a directory or file path for any combination of
``(subject_dir, stage_name, plugin_key, aggregation_key, output_name)``.
Two schemes ship with p-Brain:

* ``bids_like`` — ``derivatives/<stage>/<plugin>/<agg>/<name>.<ext>`` (default).
* ``legacy``    — the original ``Analysis/Fitting/``, ``Images/AI_patlak/``
                  layout used by ``main.py`` and the p-Brain Platform app.

Implementations live in ``pbrain/io/path_schemes/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from .plugin import Plugin


@runtime_checkable
class PathScheme(Plugin, Protocol):
    """Directory layout + file naming policy for a pipeline run."""

    def stage_dir(
        self,
        subject_dir: Path,
        stage_name: str,
        plugin_key: str | None = None,
    ) -> Path:
        """Directory holding a stage's manifest and primary artefacts."""
        ...

    def aggregation_dir(
        self,
        subject_dir: Path,
        stage_name: str,
        plugin_key: str,
        aggregation_key: str,
    ) -> Path:
        """Directory for one (model × aggregator) bundle of artefacts."""
        ...

    def output_path(
        self,
        subject_dir: Path,
        stage_name: str,
        plugin_key: str | None,
        aggregation_key: str | None,
        output_name: str,
        ext: str,
    ) -> Path:
        """Path for a named artefact (e.g. ``ki.nii.gz``)."""
        ...

    def manifest_path(
        self,
        subject_dir: Path,
        stage_name: str,
        plugin_key: str | None = None,
    ) -> Path:
        """Where the stage's ``manifest.json`` lives."""
        ...
