"""Kinetic-model registry. Auto-discovers ``pbrain/models/*.py``.

Drop a new file containing a module-level ``PLUGIN`` satisfying
:class:`KineticModel` and it appears in ``REGISTRY`` on next import.
Delete or ``.gitignore`` the file and it silently disappears. This
mechanism is how ``gamma.py`` stays off ``main`` without any special
casing — see ``docs/ARCHITECTURE.md``.
"""

from pbrain.core import discover
from .base import CurveInputs, KineticModel, ModelResult

REGISTRY: dict[str, KineticModel] = discover(
    __name__, __file__, expected_protocol=KineticModel
)

__all__ = ["REGISTRY", "KineticModel", "CurveInputs", "ModelResult"]
