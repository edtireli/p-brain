"""Render assets/p-brain.gif — the CLI splash, faithfully braillised, transparent.

Uses the SAME brain and spiral draw-in the CLI calls (``pbrain._banner``), rendered
as text with the real Apple Braille font (what the terminal falls back to for
U+2800). Background is TRANSPARENT so the GIF sits on any page theme. The brain
winds itself in → holds → unwinds → loops.

    python scripts/make_gif.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))
from pbrain._banner import _COLS, _ROWS, _decode, _encode, _reveal  # noqa: E402 — CLI-identical

CLAY = (217, 119, 87)      # #D97757, p-Brain's default palette base
DIM = (140, 150, 138)      # tagline — readable on both light and dark pages
TRANSPARENT = (0, 0, 0, 0)

BRAILLE_FONT = "/System/Library/Fonts/Apple Braille.ttf"
BRAILLE_SIZE = 44
WORDMARK = "p-Brain"
TAGLINE = "perfusion & permeability"

_bf = ImageFont.truetype(BRAILLE_FONT, BRAILLE_SIZE)
_asc, _desc = _bf.getmetrics()
LINE_H = _asc + _desc
CELL_W = _bf.getbbox("⣿")[2]
H = _ROWS * LINE_H + 72
GRID_X, GRID_Y = 44, (H - _ROWS * LINE_H) // 2
TEXT_X = GRID_X + _COLS * CELL_W + 42
_GRID = _decode()


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def frame(progress: float, font_big, font_small, width: int) -> Image.Image:
    img = Image.new("RGBA", (width, H), TRANSPARENT)
    d = ImageDraw.Draw(img)
    for row, line in enumerate(_encode(_reveal(_GRID, progress))):
        d.text((GRID_X, GRID_Y + row * LINE_H), line, font=_bf, fill=CLAY)
    # wordmark on the brain's middle row, tagline below — matching the CLI banner
    d.text((TEXT_X, GRID_Y + 1 * LINE_H + 2), WORDMARK, font=font_big, fill=CLAY)
    d.text((TEXT_X, GRID_Y + 2 * LINE_H + 2), TAGLINE, font=font_small, fill=DIM)
    return img


def to_transparent_p(img: Image.Image) -> Image.Image:
    """RGBA → palette image with index 255 reserved as transparent, driven by the
    alpha channel (GIF has no alpha, only a transparent index)."""
    opaque = img.getchannel("A").point(lambda a: 255 if a >= 96 else 0)
    p = img.convert("RGB").quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    p.paste(255, mask=opaque.point(lambda v: 255 - v))
    p.info["transparency"] = 255
    return p


def main() -> None:
    font_big, font_small = load_font(34), load_font(15)
    width = int(TEXT_X + max(font_big.getlength(WORDMARK), font_small.getlength(TAGLINE))) + 44
    fps, draw_s, hold_s = 25, 1.5, 0.8

    def eased(steps: int, reverse: bool = False):
        for i in range(steps):
            p = (1 - math.cos(math.pi * (i + 1) / steps)) / 2
            yield (1 - p) if reverse else p

    rgba: list[Image.Image] = []
    for p in eased(int(draw_s * fps)):
        rgba.append(frame(p, font_big, font_small, width))
    rgba += [frame(1.0, font_big, font_small, width)] * int(hold_s * fps)
    for p in eased(int(draw_s * fps), reverse=True):
        rgba.append(frame(p, font_big, font_small, width))
    rgba += [frame(0.0, font_big, font_small, width)] * int(0.3 * fps)

    frames = [to_transparent_p(f) for f in rgba]
    out = Path(__file__).parent.parent / "assets" / "p-brain.gif"
    out.parent.mkdir(exist_ok=True)
    frames[0].save(
        out, save_all=True, append_images=frames[1:],
        duration=int(1000 / fps), loop=0, transparency=255, disposal=2, optimize=False,
    )
    print(f"wrote {out} · {len(frames)} frames · {out.stat().st_size // 1024} KB · transparent")


if __name__ == "__main__":
    main()
