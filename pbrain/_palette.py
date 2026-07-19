"""p-Brain's colour identity — a small set of named palettes the whole CLI reads
from, so the banner, cockpit, logs and progress all move together when the user
switches theme.

Dependency-free (stdlib only) so the banner can share it without pulling rich.
Resolution order: ``PBRAIN_THEME`` env var → ``~/.config/pbrain/config.json`` →
default (**clay**). Each palette is (base, deep, lite) hex, mirroring spiral's
CLAY / CLAY_DEEP / CLAY_LITE trio.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# base, deep, lite
PALETTES: dict[str, tuple[str, str, str]] = {
    "clay":   ("#D97757", "#B8543A", "#EBA680"),   # spiral's warm signature — p-Brain default
    "teal":   ("#1098AD", "#0B7183", "#4FBECE"),
    "green":  ("#3FB950", "#2EA043", "#7EE787"),
    "red":    ("#F85149", "#DA3633", "#FF7B72"),
    "amber":  ("#E3B341", "#BB8009", "#F2CC60"),
    "violet": ("#A371F7", "#8957E5", "#D2A8FF"),
    "cyan":   ("#39C5CF", "#1B9AA3", "#7CE0E6"),
}
DEFAULT = "clay"

_CONFIG = Path.home() / ".config" / "pbrain" / "config.json"


def active_name() -> str:
    """The selected theme name — env wins, then the config file, then the default."""
    env = os.environ.get("PBRAIN_THEME", "").strip().lower()
    if env in PALETTES:
        return env
    try:
        if _CONFIG.is_file():
            name = json.loads(_CONFIG.read_text()).get("theme", "")
            if name in PALETTES:
                return name
    except Exception:
        pass
    return DEFAULT


def palette(name: str | None = None) -> tuple[str, str, str]:
    """(base, deep, lite) hex for ``name`` or the active theme."""
    return PALETTES.get(name or active_name(), PALETTES[DEFAULT])


TONES = ("single", "two", "three")


def active_tone() -> str:
    """Brain-glyph shading: single · two · three (default single/flat). Env then config."""
    env = os.environ.get("PBRAIN_TONE", "").strip().lower()
    if env in TONES:
        return env
    try:
        if _CONFIG.is_file():
            t = json.loads(_CONFIG.read_text()).get("tone", "")
            if t in TONES:
                return t
    except Exception:
        pass
    return "single"


def set_tone(name: str) -> bool:
    if name not in TONES:
        return False
    _CONFIG.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    try:
        if _CONFIG.is_file():
            data = json.loads(_CONFIG.read_text())
    except Exception:
        data = {}
    data["tone"] = name
    _CONFIG.write_text(json.dumps(data, indent=2))
    return True


def set_theme(name: str) -> bool:
    """Persist the chosen theme to the config file. Returns False for unknown names."""
    if name not in PALETTES:
        return False
    _CONFIG.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    try:
        if _CONFIG.is_file():
            data = json.loads(_CONFIG.read_text())
    except Exception:
        data = {}
    data["theme"] = name
    _CONFIG.write_text(json.dumps(data, indent=2))
    return True


def rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
