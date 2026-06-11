"""Tissue ROI provider registry. Auto-discovers ``pbrain/tissue_roi/*.py``."""

from pbrain.core import discover
from .base import TissueROI, TissueROIProvider

REGISTRY: dict[str, TissueROIProvider] = discover(
    __name__, __file__, expected_protocol=TissueROIProvider
)

__all__ = ["REGISTRY", "TissueROIProvider", "TissueROI"]
