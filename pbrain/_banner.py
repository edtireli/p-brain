"""p-Brain's terminal identity: a side-profile brain in braille — traced from a
mid-sagittal anatomical drawing — beside the wordmark and tagline.

Dependency-free (raw ANSI truecolour, stdlib only) and TTY-gated. On an
interactive terminal the brain draws itself in with a reveal order picked at
random each launch (spiral, ripple, zig-zag, sparkle, …). Colour is
theme-aware and, optionally, two- or three-tone: each braille cell is shaded by
the anatomy's depth (deep structures dark, cortex light) using the active
palette's deep/base/light trio.
"""
from __future__ import annotations

import math
import random
import sys
import time

from pbrain._palette import active_tone, palette, rgb

# side-profile brain (facing left), 8x3 braille — traced from a sagittal drawing,
# then hand-finished by Edis in the paint editor and downsampled to its tightest
# still-legible size (below three rows it stops reading as a brain).
BRAIN = [
    "⢀⣴⣿⣿⣿⣷⣦⡀",
    "⠈⢻⣿⣿⣿⣿⣿⠿",
    "⠀⠀⠉⠉⠹⣿⠟⠀",
]
# per-cell depth level for shading: 0 = light, 1 = base, 2 = deep. Flat (all base)
# by default; `pbrain tone two|three` re-derives depth. Kept for future use.
BRAIN_TONE = [
    "11111111",
    "11111111",
    "11111111",
]

TAGLINE = "perfusion & permeability"

_RST = "\x1b[0m"
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_BLANK = "⠀"

_BRAILLE_BASE = 0x2800
_DOT = {(0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (0, 3): 0x40,
        (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80}
_ROWS, _COLS = len(BRAIN), len(BRAIN[0])
_H, _W = _ROWS * 4, _COLS * 2


def _ansi(hex_color: str) -> str:
    r, g, b = rgb(hex_color)
    return f"\x1b[38;2;{r};{g};{b}m"


def _cell_ansi(level: int, mode: str) -> str:
    base, deep, lite = palette()
    if mode == "single":
        c = base
    elif mode == "two":
        c = deep if level >= 1 else lite
    else:  # three
        c = (lite, base, deep)[max(0, min(2, level))]
    return _ansi(c)


def _colorize(rows: list[str], mode: str) -> list[str]:
    """Colour each braille cell by its depth level (blank cells stay uncoloured)."""
    out = []
    for ry, row in enumerate(rows):
        s = ""
        for cx, ch in enumerate(row):
            if ch == _BLANK:
                s += ch
            else:
                lvl = int(BRAIN_TONE[ry][cx]) if cx < len(BRAIN_TONE[ry]) else 1
                s += _cell_ansi(lvl, mode) + ch
        out.append(s + _RST)
    return out


def _decode() -> list[list[bool]]:
    grid = [[False] * _W for _ in range(_H)]
    for cy, row in enumerate(BRAIN):
        for cx, ch in enumerate(row):
            bits = ord(ch) - _BRAILLE_BASE
            for (dx, dy), bit in _DOT.items():
                if bits & bit:
                    grid[cy * 4 + dy][cx * 2 + dx] = True
    return grid


def _encode(grid: list[list[bool]]) -> list[str]:
    out = []
    for cy in range(_ROWS):
        s = ""
        for cx in range(_COLS):
            bits = 0
            for (dx, dy), bit in _DOT.items():
                if grid[cy * 4 + dy][cx * 2 + dx]:
                    bits |= bit
            s += chr(_BRAILLE_BASE + bits)
        out.append(s)
    return out


def _lines(brain_rows: list[str], tagline: str, mode: str) -> list[str]:
    base = _ansi(palette()[0])
    out = []
    for i, crow in enumerate(_colorize(brain_rows, mode)):
        line = "  " + crow
        if i == 1:
            line += "   " + _BOLD + base + "p-Brain" + _RST
        elif i == 2:
            line += "   " + _DIM + tagline + _RST
        out.append(line)
    return out


def render(tagline: str = TAGLINE, mode: str | None = None) -> str:
    return "\n".join(_lines(BRAIN, tagline, mode or active_tone()))


_CX, _CY = (_W - 1) / 2, (_H - 1) / 2
_MAXR = math.hypot(_CX, _CY)


def _rad(c: tuple[int, int]) -> float:
    return math.hypot(c[0] - _CX, c[1] - _CY) / _MAXR


def _ang(c: tuple[int, int]) -> float:
    return (math.atan2(c[1] - _CY, c[0] - _CX) + math.pi) / (2 * math.pi)


def _boust(c: tuple[int, int], row: int) -> float:
    """Boustrophedon scan value: sweep each row, alternating L->R / R->L."""
    pos = c[0] if row % 2 == 0 else (_W - 1 - c[0])
    return row * _W + pos


def _lit_cells(grid: list[list[bool]]) -> list[tuple[int, int]]:
    return [(x, y) for y in range(_H) for x in range(_W) if grid[y][x]]


def _value_map(cells, val) -> dict[tuple[int, int], float]:
    """Reveal order from a continuous field, normalised to [0,1]: equal values
    surface together (radial fields draw as rings, vertical ones as a wipe)."""
    vs = {c: val(c) for c in cells}
    lo, hi = min(vs.values()), max(vs.values())
    rng = (hi - lo) or 1.0
    return {c: (v - lo) / rng for c, v in vs.items()}


def _rank_map(cells) -> dict[tuple[int, int], float]:
    """Reveal order one cell at a time, in the given sequence."""
    n = max(1, len(cells) - 1)
    return {c: i / n for i, c in enumerate(cells)}


def _reveal_map(name: str, grid: list[list[bool]]) -> dict[tuple[int, int], float]:
    """A cell -> normalised reveal-time map for one named style. `converge` is
    defined but kept out of _MIX — Edis wanted the shuffle to be every style
    except that one."""
    cells = _lit_cells(grid)
    if name == "center":  # nearest-to-centre first, one dot at a time
        return _rank_map(sorted(cells, key=lambda c: (_rad(c), _ang(c))))
    if name == "sparkle":  # random dissolve-in
        cells = list(cells)
        random.shuffle(cells)
        return _rank_map(cells)
    field = {
        "spiral": lambda c: (_rad(c) + _ang(c)) / 2,
        "ripple": _rad,
        "split": lambda c: abs(c[0] - _CX),
        "bottomup": lambda c: _H - 1 - c[1],
        "zigtop": lambda c: _boust(c, c[1]),
        "zigbottom": lambda c: _boust(c, _H - 1 - c[1]),
        "converge": lambda c: 1.0 - _rad(c),
    }[name]
    return _value_map(cells, field)


# the probabilistic draw-in: one of these is chosen at random per launch.
_MIX = ("spiral", "center", "zigtop", "zigbottom", "ripple", "split", "sparkle", "bottomup")


def _reveal(grid: list[list[bool]], tmap: dict[tuple[int, int], float], p: float) -> list[list[bool]]:
    return [[grid[y][x] and tmap.get((x, y), 1.0) <= p for x in range(_W)] for y in range(_H)]


def print_banner(tagline: str = TAGLINE, stream=None, animate: bool = True) -> None:
    """Print the banner to an interactive terminal only; piped/redirected output
    gets nothing. On a TTY the brain draws itself in over ~0.7s."""
    stream = stream or sys.stdout
    try:
        if not stream.isatty():
            return
    except Exception:
        return
    mode = active_tone()
    stream.write("\n")
    if animate:
        grid = _decode()
        tmap = _reveal_map(random.choice(_MIX), grid)
        stream.write("\x1b[?25l")
        frames = 14
        for f in range(frames):
            rows = _encode(_reveal(grid, tmap, (f + 1) / frames))
            if f:
                stream.write(f"\x1b[{_ROWS}A")
            stream.write("\r" + "\n".join(_lines(rows, tagline, mode)) + "\n")
            stream.flush()
            time.sleep(0.7 / frames)
        stream.write(f"\x1b[{_ROWS}A")
    stream.write("\r" + render(tagline, mode) + "\n\n")
    if animate:
        stream.write("\x1b[?25h")
    stream.flush()


if __name__ == "__main__":
    class _TTY:
        def isatty(self): return True
        def write(self, s): sys.stdout.write(s)
        def flush(self): sys.stdout.flush()
    print_banner(stream=_TTY())
