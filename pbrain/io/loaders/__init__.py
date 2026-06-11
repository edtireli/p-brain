"""Input loader registry. Auto-discovers ``pbrain/io/loaders/*.py``."""

from pathlib import Path

from pbrain.core import discover
from .base import ImageLoader, Series4D

REGISTRY: dict[str, ImageLoader] = discover(
    __name__, __file__, expected_protocol=ImageLoader
)


def load_4d(path: Path, **opts) -> Series4D:
    """Dispatch ``path`` to the first loader whose ``detect`` returns True.

    Loaders are tried in deterministic alphabetical order of their
    ``key``. Raise ``ValueError`` if no loader claims the file.
    """
    path = Path(path)
    for key in sorted(REGISTRY):
        loader = REGISTRY[key]
        if loader.detect(path):
            return loader.load(path, **opts)
    raise ValueError(
        f"No loader can read {path!s}. "
        f"Registered: {sorted(REGISTRY.keys())}. "
        f"Add a new loader by dropping a file in pbrain/io/loaders/."
    )


__all__ = ["REGISTRY", "load_4d", "ImageLoader", "Series4D"]
