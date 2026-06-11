"""Aggregator registry. Auto-discovers ``pbrain/aggregation/*.py``."""

from pbrain.core import discover
from .base import Aggregator, AggregationResult

REGISTRY: dict[str, Aggregator] = discover(
    __name__, __file__, expected_protocol=Aggregator
)

__all__ = ["REGISTRY", "Aggregator", "AggregationResult"]
