"""Auto-discovery for plug-point folders.

Every plug-point's ``__init__.py`` is two lines::

    from pbrain.core import discover
    REGISTRY = discover(__name__, __file__)

``discover()`` walks the package directory for ``*.py`` modules, imports
each, and indexes whichever expose a module-level ``PLUGIN`` attribute.

Files starting with ``_`` or named ``base.py`` are skipped (the
plug-point's own Protocol lives there). Duplicate ``PLUGIN.key`` values
within a plug-point are a hard error.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TypeVar

from .plugin import Plugin

P = TypeVar("P", bound=Plugin)


def discover(
    package_name: str,
    package_file: str,
    *,
    expected_protocol: type[P] | None = None,
) -> dict[str, P]:
    """Scan ``package_name``'s directory for plug-in modules.

    Parameters
    ----------
    package_name
        Pass ``__name__`` from the plug-point's ``__init__.py``.
    package_file
        Pass ``__file__`` from the plug-point's ``__init__.py``.
    expected_protocol
        Optional sub-Protocol (e.g. ``KineticModel``) that every
        discovered ``PLUGIN`` must satisfy. Verified via ``isinstance``
        (works because every plug-point Protocol is
        ``@runtime_checkable``).

    Returns
    -------
    dict
        ``{plugin.key: plugin}`` for every discovered module.

    Raises
    ------
    TypeError
        If ``expected_protocol`` is set and a discovered ``PLUGIN``
        fails the runtime check.
    ValueError
        If two modules expose ``PLUGIN``s with the same ``key``.
    """

    package_dir = Path(package_file).parent
    registry: dict[str, P] = {}

    for path in sorted(package_dir.glob("*.py")):
        stem = path.stem
        if stem.startswith("_") or stem == "base":
            continue

        module = importlib.import_module(f".{stem}", package=package_name)
        plugin = getattr(module, "PLUGIN", None)
        if plugin is None:
            continue

        if expected_protocol is not None and not isinstance(plugin, expected_protocol):
            raise TypeError(
                f"{package_name}.{stem}.PLUGIN does not satisfy "
                f"{expected_protocol.__name__} contract. "
                f"Got {type(plugin).__name__}; "
                f"expected attributes: {sorted(getattr(expected_protocol, '__annotations__', {}).keys())}"
            )

        if plugin.key in registry:
            existing = type(registry[plugin.key]).__name__
            new = type(plugin).__name__
            raise ValueError(
                f"Duplicate plugin key {plugin.key!r} in {package_name}: "
                f"{existing} vs {new} (both modules export PLUGIN.key={plugin.key!r})"
            )

        registry[plugin.key] = plugin

    return registry
