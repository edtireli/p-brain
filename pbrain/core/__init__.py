"""Framework primitives — Plugin, discovery, Stage, Pipeline, Config."""

from .config import Config
from .devices import configure_tf_device, probe as probe_devices, resolve as resolve_device
from .discovery import discover
from .logs import configure as configure_logging, get_logger, level_from_flags
from .manifest import Manifest
from .path_scheme import PathScheme
from .pipeline import Pipeline, StageRunRecord
from .plugin import Plugin
from .stage import Stage, StageContext, StageOutput

__all__ = [
    "Plugin",
    "PathScheme",
    "Manifest",
    "Config",
    "Stage",
    "StageContext",
    "StageOutput",
    "Pipeline",
    "StageRunRecord",
    "discover",
    "resolve_device",
    "probe_devices",
    "configure_tf_device",
    "configure_logging",
    "get_logger",
    "level_from_flags",
]
