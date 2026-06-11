"""T1/M0 fitter registry. Auto-discovers ``pbrain/t1_m0/*.py``."""

from pbrain.core import discover
from .base import T1M0Fitter, T1M0Result

REGISTRY: dict[str, T1M0Fitter] = discover(
    __name__, __file__, expected_protocol=T1M0Fitter
)

__all__ = ["REGISTRY", "T1M0Fitter", "T1M0Result"]
