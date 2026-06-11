"""Output path scheme registry. Auto-discovers ``pbrain/io/path_schemes/*.py``."""

from pbrain.core import discover
from pbrain.core.path_scheme import PathScheme

REGISTRY: dict[str, PathScheme] = discover(
    __name__, __file__, expected_protocol=PathScheme
)

__all__ = ["REGISTRY", "PathScheme"]
