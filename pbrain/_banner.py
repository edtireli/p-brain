"""p-Brain's terminal identity: a neuron in braille — traced from an icon and
hand-tuned — beside the wordmark and tagline.

Dependency-free (raw ANSI truecolour, stdlib only) and TTY-gated. On an
interactive terminal the neuron draws itself in with one of two reveal orders
picked at random each launch — a spiral, or a sparkle dissolve — both radiating
from the nucleus. Colour is theme-aware and, optionally, two- or three-tone using
the active palette's deep/base/light trio.
"""
from __future__ import annotations

import math
import random
import sys
import time

from pbrain._palette import active_tone, palette, rgb

# the p-Brain mark: a neuron (multipolar — radial dendrites with forked tips, a
# hollow nucleus, and a long axon), traced from a vector icon and hand-tuned by
# Edis in the braille editor to its tightest still-legible size, 10x5 braille.
NEURON = [
    "⠀⢀⠵⡄⠀⡸⠂⠀⠀⠀",
    "⠐⣄⣠⡞⢻⣧⢴⠉⠀⠀",
    "⠐⠁⣠⡽⠛⣇⠀⠉⠀⠀",
    "⠀⠈⠀⠃⠀⠘⢆⠀⠀⡀",
    "⠀⠀⠀⠀⠀⠀⠀⠑⠺⡂",
]
# per-cell depth level for shading: 0 = light, 1 = base, 2 = deep. Flat (all base)
# by default; `pbrain tone two|three` re-derives depth. Kept for future use.
NEURON_TONE = [
    "1111111111",
    "1111111111",
    "1111111111",
    "1111111111",
    "1111111111",
]

def art() -> list[str]:
    """The neuron braille rows (used by the banner and the web review)."""
    return NEURON


def _tone() -> list[str]:
    return NEURON_TONE


TAGLINE = "perfusion & permeability"

_RST = "\x1b[0m"
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_BLANK = "⠀"

_BRAILLE_BASE = 0x2800
_DOT = {(0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (0, 3): 0x40,
        (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80}
def _dims(art_rows: list[str]) -> tuple[int, int, int, int]:
    """(rows, cols, dot-height, dot-width) for the neuron braille art."""
    r, c = len(art_rows), len(art_rows[0])
    return r, c, r * 4, c * 2


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
                tone = _tone()
                lvl = int(tone[ry][cx]) if ry < len(tone) and cx < len(tone[ry]) else 1
                s += _cell_ansi(lvl, mode) + ch
        out.append(s + _RST)
    return out


def _decode(art_rows: list[str] | None = None) -> list[list[bool]]:
    a = art_rows if art_rows is not None else art()
    _, _, H, W = _dims(a)
    grid = [[False] * W for _ in range(H)]
    for cy, row in enumerate(a):
        for cx, ch in enumerate(row):
            bits = ord(ch) - _BRAILLE_BASE
            for (dx, dy), bit in _DOT.items():
                if bits & bit:
                    grid[cy * 4 + dy][cx * 2 + dx] = True
    return grid


def _encode(grid: list[list[bool]]) -> list[str]:
    rows, cols = len(grid) // 4, len(grid[0]) // 2
    out = []
    for cy in range(rows):
        s = ""
        for cx in range(cols):
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
    mid = max(1, (len(brain_rows) - 2) // 2)   # wordmark beside the soma (upper-middle)
    for i, crow in enumerate(_colorize(brain_rows, mode)):
        line = "  " + crow
        if i == mid:
            line += " " + _BOLD + base + "p-Brain" + _RST
        elif i == mid + 1:
            line += " " + _DIM + tagline + _RST
        out.append(line)
    return out


def render(tagline: str = TAGLINE, mode: str | None = None) -> str:
    return "\n".join(_lines(art(), tagline, mode or active_tone()))


def _lit_cells(grid: list[list[bool]]) -> list[tuple[int, int]]:
    H, W = len(grid), len(grid[0])
    return [(x, y) for y in range(H) for x in range(W) if grid[y][x]]


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


def _soma_center(grid) -> tuple[float, float]:
    """Where the radial reveals should centre: the nucleus. That's the enclosed
    empty region inside the soma — found by flood-filling the background in from
    the border and taking whatever off-cells it can't reach. If the silhouette has
    no enclosed hole (can happen at the smallest sizes), fall back to the deepest
    interior point (the lit cell furthest from any edge), which lands on the soma."""
    from collections import deque
    H, W = len(grid), len(grid[0])
    outside = [[False] * W for _ in range(H)]
    q = deque()
    for x in range(W):
        for y in (0, H - 1):
            if not grid[y][x] and not outside[y][x]:
                outside[y][x] = True; q.append((x, y))
    for y in range(H):
        for x in (0, W - 1):
            if not grid[y][x] and not outside[y][x]:
                outside[y][x] = True; q.append((x, y))
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and not grid[ny][nx] and not outside[ny][nx]:
                outside[ny][nx] = True; q.append((nx, ny))
    holes = [(x, y) for y in range(H) for x in range(W) if not grid[y][x] and not outside[y][x]]
    if holes:
        return (sum(x for x, _ in holes) / len(holes), sum(y for _, y in holes) / len(holes))
    lit = _lit_cells(grid)
    off = [(x, y) for y in range(H) for x in range(W) if not grid[y][x]]
    if not lit:
        return ((W - 1) / 2, (H - 1) / 2)
    if not off:
        return (sum(x for x, _ in lit) / len(lit), sum(y for _, y in lit) / len(lit))
    best = max(lit, key=lambda c: min((c[0] - ox) ** 2 + (c[1] - oy) ** 2 for ox, oy in off))
    return (float(best[0]), float(best[1]))


def _reveal_map(name: str, grid: list[list[bool]]) -> dict[tuple[int, int], float]:
    """A cell -> normalised reveal-time map for one named style, sized to the art.
    Only ``spiral`` and ``sparkle`` are in _MIX (the launch shuffle); the others
    stay defined and available but unused."""
    H, W = len(grid), len(grid[0])
    cells = _lit_cells(grid)
    cx, cy = _soma_center(grid)                          # radiate from the nucleus
    maxr = max((math.hypot(c[0] - cx, c[1] - cy) for c in cells), default=1.0) or 1.0
    rad = lambda c: math.hypot(c[0] - cx, c[1] - cy) / maxr
    ang = lambda c: (math.atan2(c[1] - cy, c[0] - cx) + math.pi) / (2 * math.pi)

    def boust(c, row):   # sweep each row, alternating L->R / R->L
        pos = c[0] if row % 2 == 0 else (W - 1 - c[0])
        return row * W + pos

    if name == "center":   # nearest-to-centre first, one dot at a time
        return _rank_map(sorted(cells, key=lambda c: (rad(c), ang(c))))
    if name == "sparkle":  # random dissolve-in
        cells = list(cells)
        random.shuffle(cells)
        return _rank_map(cells)
    field = {
        "spiral": lambda c: (rad(c) + ang(c)) / 2,
        "ripple": rad,
        "split": lambda c: abs(c[0] - cx),
        "bottomup": lambda c: H - 1 - c[1],
        "zigtop": lambda c: boust(c, c[1]),
        "zigbottom": lambda c: boust(c, H - 1 - c[1]),
        "converge": lambda c: 1.0 - rad(c),
    }[name]
    return _value_map(cells, field)


# the probabilistic draw-in: one of these is chosen at random per launch.
_MIX = ("spiral", "sparkle")


def _reveal(grid: list[list[bool]], tmap: dict[tuple[int, int], float], p: float) -> list[list[bool]]:
    H, W = len(grid), len(grid[0])
    return [[grid[y][x] and tmap.get((x, y), 1.0) <= p for x in range(W)] for y in range(H)]


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
    n_rows = len(art())
    stream.write("\n")
    if animate:
        grid = _decode()
        tmap = _reveal_map(random.choice(_MIX), grid)
        stream.write("\x1b[?25l")
        frames = 14
        for f in range(frames):
            rows = _encode(_reveal(grid, tmap, (f + 1) / frames))
            if f:
                stream.write(f"\x1b[{n_rows}A")
            stream.write("\r" + "\n".join(_lines(rows, tagline, mode)) + "\n")
            stream.flush()
            time.sleep(0.7 / frames)
        stream.write(f"\x1b[{n_rows}A")
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
