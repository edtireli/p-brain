"""Diagnostic-plotter registry. Auto-discovers ``pbrain/diagnostics/*.py``."""

from pbrain.core import discover
from .base import Diagnostic, DiagnosticContext

REGISTRY: dict[str, Diagnostic] = discover(
    __name__, __file__, expected_protocol=Diagnostic
)

__all__ = ["REGISTRY", "Diagnostic", "DiagnosticContext"]
