"""Curve-normaliser registry. Auto-discovers ``pbrain/normalisation/*.py``."""

from pbrain.core import discover
from .base import CurveNormaliser

REGISTRY: dict[str, CurveNormaliser] = discover(
    __name__, __file__, expected_protocol=CurveNormaliser
)

__all__ = ["REGISTRY", "CurveNormaliser"]
