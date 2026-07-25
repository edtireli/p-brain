"""p-Brain modular framework.

A plug-and-play implementation of the nine-stage DCE-MRI pipeline
described in Tireli et al., "p-Brain: Advanced Neuroimaging platform".

Public surface:

    from pbrain.core import Plugin, discover, Stage, Pipeline, Config
    from pbrain.models import REGISTRY as MODELS, CurveInputs, ModelResult
    from pbrain.aif import REGISTRY as AIF_EXTRACTORS
    # ... etc., one REGISTRY per plug-point

Every plug-point is a folder containing:
    * ``base.py``     — the plug-point's typed Protocol (refines Plugin)
    * ``__init__.py`` — a two-line auto-discovery loop building ``REGISTRY``
    * one ``<key>.py`` per implementation, exposing a module-level ``PLUGIN``

Drop a new file in a plug-point folder and it appears in the registry on
next import. Delete or ``.gitignore`` the file and it silently disappears.

See ``docs/ARCHITECTURE.md`` for the design rationale.
"""

from pbrain.core import (
    Plugin,
    Stage,
    StageContext,
    StageOutput,
    Pipeline,
    Config,
    PathScheme,
    Manifest,
    discover,
)

__all__ = [
    "Plugin",
    "Stage",
    "StageContext",
    "StageOutput",
    "Pipeline",
    "Config",
    "PathScheme",
    "Manifest",
    "discover",
]

__version__ = "3.1.5"
