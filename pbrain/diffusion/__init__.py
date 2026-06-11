"""Diffusion-model registry. Auto-discovers ``pbrain/diffusion/*.py``.

Drop a new file containing a module-level ``PLUGIN`` satisfying
:class:`DiffusionModel` and it appears in ``REGISTRY`` on next import.
Same mechanism as :mod:`pbrain.models` and every other plug-point.
"""

from pbrain.core import discover
from .base import DiffusionModel, DWIInputs, DiffusionResult

REGISTRY: dict[str, DiffusionModel] = discover(
    __name__, __file__, expected_protocol=DiffusionModel
)

__all__ = ["REGISTRY", "DiffusionModel", "DWIInputs", "DiffusionResult"]
