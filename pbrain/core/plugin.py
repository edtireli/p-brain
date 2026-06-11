"""The universal Plugin contract.

A *plug-point* is a directory under ``pbrain/`` containing single-file
plug-ins. Every plug-in module exports a module-level ``PLUGIN``
attribute that satisfies the :class:`Plugin` Protocol below — plus a
plug-point-specific sub-Protocol that adds typed entry points.

The Protocol is intentionally minimal — five class-level attributes:

* ``key``         — registry name (unique within plug-point), e.g.
                    ``"patlak"`` or ``"cnn_aif"``.
* ``name``        — short human-readable label.
* ``description`` — longer free-text describing what the plug-in does.
* ``accepts``     — ``{input_name: type}``; the orchestrator uses this
                    to introspect what data the plug-in needs.
* ``produces``    — ``{output_name: type}``; ditto for outputs.

The ``accepts``/``produces`` introspection enables dynamic wiring of
stages — plug-ins declare their I/O once, in code, in one place
(satisfies the paper's "dynamic number of variables" requirement).
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable


@runtime_checkable
class Plugin(Protocol):
    """Universal plug-in contract."""

    key: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str]
    accepts: ClassVar[dict[str, type]]
    produces: ClassVar[dict[str, type]]
