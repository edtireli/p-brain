"""AIF/VIF extractor registry. Auto-discovers ``pbrain/aif/*.py``."""

from pbrain.core import discover
from .base import AIFExtractor, InputFunction

REGISTRY: dict[str, AIFExtractor] = discover(
    __name__, __file__, expected_protocol=AIFExtractor
)

__all__ = ["REGISTRY", "AIFExtractor", "InputFunction"]
