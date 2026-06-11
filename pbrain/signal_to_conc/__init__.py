"""Signal→concentration converter registry. Auto-discovers ``pbrain/signal_to_conc/*.py``."""

from pbrain.core import discover
from .base import SignalToConcConverter

REGISTRY: dict[str, SignalToConcConverter] = discover(
    __name__, __file__, expected_protocol=SignalToConcConverter
)

__all__ = ["REGISTRY", "SignalToConcConverter"]
