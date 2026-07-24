"""Record the REAL p-Brain CLI into an animated GIF — a genuine terminal capture.

Each command runs in a real PTY; the raw ANSI byte stream is captured with its
true timing, replayed through a terminal emulator (pyte), and every screen state
is rasterized into a GIF frame. Nothing shown is simulated — the banner, the live
cockpit, the spinners are the program's actual output. Only the window chrome and
the typed prompt line are drawn by this script.

Two presets, matching the README:

    python scripts/record_demo.py install   # pip install p-brain, then the banner
    python scripts/record_demo.py run        # the live cockpit on the example subject

`run` needs the example subject on disk. Point at it with --data-dir (a folder
holding ``sub-01/`` and ``sub-01.toml``); the default is the one this script
lays down under the system temp dir from ``pbrain fetch-data`` conventions.

Requires: pyte (pip install pyte) and the Menlo / Apple Braille system fonts.
Adapted from the spiral recorder (same author); the replay/raster/GIF core is
shared, the driver here is p-Brain's.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time
from pathlib import Path

import pyte
import pyte.graphics
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent

# ---- palette (p-Brain "clay" theme) ------------------------------------------
BG = (13, 13, 15)
CHROME = (26, 26, 30)
FG = (224, 226, 223)
CLAY = (217, 119, 87)
ANSI = {
    "black": (40, 42, 46), "red": (255, 99, 92), "green": (78, 201, 108),
    "brown": (255, 176, 0), "blue": (108, 153, 255), "magenta": (198, 120, 221),
    "cyan": (86, 182, 194), "white": (224, 226, 223),
    "brightblack": (110, 116, 124), "brightred": (255, 121, 116),
    "brightgreen": (110, 220, 138), "brightyellow": (255, 200, 80),
    "brightblue": (140, 178, 255), "brightmagenta": (216, 148, 235),
    "brightcyan": (120, 200, 210), "brightwhite": (238, 240, 236),
}

MENLO = "/System/Library/Fonts/Menlo.ttc"
BRAILLE_TTF = "/System/Library/Fonts/Apple Braille.ttf"
FALLBACKS = ("/System/Library/Fonts/Apple Symbols.ttf", "/Library/Fonts/Arial Unicode.ttf")
SIZE = 20
PAD, TITLE_H = 26, 42


class FontKit:
    def __init__(self) -> None:
        self.reg = ImageFont.truetype(MENLO, SIZE, index=0)
        self.bold = self.reg
        for i in range(1, 8):
            try:
                f = ImageFont.truetype(MENLO, SIZE, index=i)
                if f.getname()[1] == "Bold":
                    self.bold = f
                    break
            except Exception:
                break
        self.fb = [ImageFont.truetype(p, SIZE) for p in FALLBACKS if Path(p).is_file()]
        self._notdef: dict[int, bytes] = {}
        self._cache: dict[tuple[str, bool], ImageFont.FreeTypeFont] = {}
        self.cell = round(self.reg.getlength("M"))
        asc, desc = self.reg.getmetrics()
        self.line_h = asc + desc + 2

    @staticmethod
    def _bitmap(font: ImageFont.FreeTypeFont, ch: str) -> bytes:
        im = Image.new("L", (SIZE * 2, SIZE * 2), 0)
        ImageDraw.Draw(im).text((0, 0), ch, font=font, fill=255)
        return im.tobytes()

    def _covers(self, font: ImageFont.FreeTypeFont, ch: str) -> bool:
        key = id(font)
        if key not in self._notdef:
            self._notdef[key] = self._bitmap(font, "￿")
        bm = self._bitmap(font, ch)
        return bm != self._notdef[key] and any(bm)

    def pick(self, ch: str, bold: bool) -> ImageFont.FreeTypeFont:
        got = self._cache.get((ch, bold))
        if got:
            return got
        primary = self.bold if bold else self.reg
        font = primary if (ch.isascii() or self._covers(primary, ch)) else next(
            (f for f in self.fb if self._covers(f, ch)), primary)
        self._cache[(ch, bold)] = font
        return font


FK = FontKit()

_BRAILLE_DOT = {0x01: (0, 0), 0x02: (0, 1), 0x04: (0, 2), 0x40: (0, 3),
                0x08: (1, 0), 0x10: (1, 1), 0x20: (1, 2), 0x80: (1, 3)}


def draw_char(d: ImageDraw.ImageDraw, cx: float, cy: float, ch: str, color, bold: bool) -> None:
    """Braille and rules are hand-drawn (crisp at terminal size); everything
    else renders through the font chain."""
    w, h = FK.cell, FK.line_h
    o = ord(ch)
    if 0x2800 <= o <= 0x28FF:
        bits = o - 0x2800
        r = max(1.6, w / 5.4)
        for bit, (bx, by) in _BRAILLE_DOT.items():
            if bits & bit:
                px = cx + w * (0.28 + 0.44 * bx)
                py = cy + h * (0.16 + 0.24 * by)
                d.ellipse((px - r, py - r, px + r, py + r), fill=color)
        return
    if ch in "─━":
        t = 3 if ch == "━" else 1
        my = cy + h * 0.52
        d.rectangle((cx, my - t / 2, cx + w, my + t / 2), fill=color)
        return
    d.text((cx, cy), ch, font=FK.pick(ch, bold), fill=color)


def _color(spec: str, default) -> tuple:
    if spec == "default":
        return default
    if spec in ANSI:
        return ANSI[spec]
    try:
        return tuple(int(spec[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return default


def _dim(c: tuple) -> tuple:
    return tuple(int(v * 0.55 + b * 0.45) for v, b in zip(c, BG))


# ---- recording ---------------------------------------------------------------
def record(argv: list[str], cwd: str, cols: int, rows: int, max_s: float,
           env_extra: dict | None = None, tail: float = 4.0) -> list[tuple[float, bytes]]:
    """Run the command in a real PTY and capture its byte stream with timing."""
    master, slave = os.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    # Hermetic env: drop the RECORDER's own venv markers so the recorded command
    # runs in its own interpreter, not ours. Without this, a freshly-installed
    # venv's `pbrain` silently imported (or mis-imported) through the outer
    # VIRTUAL_ENV/PYTHONPATH and produced no output at all.
    base = {k: v for k, v in os.environ.items()
            if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
    env = {**base, "TERM": "xterm-256color", "COLORTERM": "truecolor",
           "PYTHONUNBUFFERED": "1", "COLUMNS": str(cols), "LINES": str(rows),
           **(env_extra or {})}
    proc = subprocess.Popen(argv, cwd=cwd, stdin=slave, stdout=slave, stderr=slave,
                            env=env, close_fds=True)
    os.close(slave)
    chunks: list[tuple[float, bytes]] = []
    t0 = time.monotonic()
    ended_at: float | None = None
    try:
        while True:
            t = time.monotonic() - t0
            if t > max_s:
                break
            if ended_at is not None and t > ended_at + tail:
                break
            r, _, _ = select.select([master], [], [], 0.03)
            if r:
                try:
                    data = os.read(master, 65536)
                except OSError:
                    break
                if not data:
                    break
                chunks.append((t, data))
            elif proc.poll() is not None and ended_at is None:
                ended_at = t
    finally:
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        os.close(master)
    return chunks


# ---- replay -> frames --------------------------------------------------------
Grid = tuple  # rows of (char, fg, bg, bold, dim)


def replay(chunks: list[tuple[float, bytes]], cols: int, rows: int) -> list[tuple[Grid, int]]:
    pyte.graphics.TEXT[2] = "+blink"  # carry SGR 2 (dim) through pyte's blink flag
    screen = pyte.Screen(cols, rows)
    stream = pyte.ByteStream(screen)

    def snap() -> Grid:
        out = []
        for y in range(rows):
            line = screen.buffer[y]
            out.append(tuple(
                (line[x].data or " ", line[x].fg, line[x].bg, line[x].bold, line[x].blink)
                for x in range(cols)))
        return tuple(out)

    frames: list[tuple[Grid, int]] = []
    last_t = 0.0
    for t, data in chunks:
        gap = int((t - last_t) * 1000)
        grid = snap()
        if frames and grid == frames[-1][0]:
            frames[-1] = (grid, frames[-1][1] + gap)
        elif gap > 25 or not frames:  # sub-25ms bursts are one repaint, not a frame
            frames.append((grid, gap))
        else:  # burst of writes — replace the barely-shown frame
            frames[-1] = (frames[-1][0], frames[-1][1] + gap)
        stream.feed(data)
        last_t = t
    frames.append((snap(), 600))
    frames = [(g, min(max(ms, 40), 600)) for g, ms in frames if ms >= 20]

    # thin status-line-only ticks (the spinner) to ~4-5 fps; keep everything else
    thinned: list[tuple[Grid, int]] = []
    for g, ms in frames:
        if thinned:
            changed = [i for i, (a, b) in enumerate(zip(thinned[-1][0], g)) if a != b]
            if len(changed) <= 1 and thinned[-1][1] < 220:
                thinned[-1] = (g, thinned[-1][1] + ms)
                continue
        thinned.append((g, ms))
    return _squeeze(thinned)


def _squeeze(frames: list[tuple[Grid, int]], max_run: int = 10, fast_ms: int = 140) -> list[tuple[Grid, int]]:
    """Time-lapse the waits: a long run of status-only frames becomes a short
    fast-spin flourish instead of dead GIF time."""
    out: list[tuple[Grid, int]] = []
    run: list[tuple[Grid, int]] = []

    def flush() -> None:
        nonlocal run
        if len(run) > max_run:
            idx = [round(i * (len(run) - 1) / (max_run - 1)) for i in range(max_run)]
            run = [(run[i][0], fast_ms) for i in idx]
        out.extend(run)
        run = []

    prev: Grid | None = None
    for g, ms in frames:
        status_only = prev is not None and sum(1 for a, b in zip(prev, g) if a != b) <= 1
        if status_only:
            run.append((g, ms))
        else:
            flush()
            out.append((g, ms))
        prev = g
    flush()
    return _deflicker(out)


def _deflicker(frames: list[tuple[Grid, int]], passes: int = 3) -> list[tuple[Grid, int]]:
    """Drop mid-repaint frames. A pinned Live panel repaints by clearing and
    redrawing; a snapshot landing between the two shows a one-frame dip that
    immediately recovers. Detect the dip-and-recover shape and merge it away."""
    def ink(g: Grid) -> int:
        return sum(1 for row in g[len(g) // 2:] for cell in row if cell[0] != " ")

    for _ in range(passes):
        out = [frames[0]]
        changed = False
        i = 1
        while i < len(frames):
            if i + 1 < len(frames):
                a, b, c = ink(out[-1][0]), ink(frames[i][0]), ink(frames[i + 1][0])
                if a > 60 and b < 0.90 * a and c >= 0.97 * a:
                    out[-1] = (out[-1][0], out[-1][1] + frames[i][1])
                    changed = True
                    i += 1
                    continue
            out.append(frames[i])
            i += 1
        frames = out
        if not changed:
            break
    return frames


# ---- rasterize ---------------------------------------------------------------
def render(grid: Grid, cols: int, rows: int, title: str, prompt: str) -> Image.Image:
    W = PAD * 2 + FK.cell * cols
    H = TITLE_H + PAD + FK.line_h * (rows + 1) + PAD
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, W - 1, H - 1), radius=14, fill=BG)
    d.rounded_rectangle((0, 0, W - 1, TITLE_H), radius=14, fill=CHROME)
    d.rectangle((0, TITLE_H // 2, W - 1, TITLE_H), fill=CHROME)
    for i, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        d.ellipse((PAD - 6 + i * 22, 15, PAD + 6 + i * 22, 27), fill=c)
    d.text(((W - FK.reg.getlength(title)) / 2, 10), title, font=FK.reg, fill=_dim(FG))

    y0 = TITLE_H + PAD
    if prompt:
        draw_char(d, PAD, y0, "❯", CLAY, True)
        for i, ch in enumerate(prompt):
            if ch != " ":
                draw_char(d, PAD + (i + 2) * FK.cell, y0, ch, FG, False)
    for ry, row in enumerate(grid):
        y = y0 + (ry + 1) * FK.line_h
        for rx, (ch, fg, bg, bold, dim) in enumerate(row):
            if ch == " " and bg == "default":
                continue
            x = PAD + rx * FK.cell
            color = _color(fg, FG)
            if bg != "default":
                d.rectangle((x, y, x + FK.cell, y + FK.line_h), fill=_color(bg, BG))
            draw_char(d, x, y, ch, _dim(color) if dim else color, bold)
    return img


def save_gif(imgs: list[Image.Image], durs: list[int], out: Path) -> None:
    """Delta-encoded GIF: one shared palette, and every frame after the first
    keeps only its changed pixels (the rest transparent, disposal=keep)."""
    from PIL import ImageChops

    w, h = imgs[0].size
    step = max(1, len(imgs) // 6)
    sample = Image.new("RGB", (w, h * min(6, len(imgs))))
    for i, f in enumerate(imgs[::step][:6]):
        sample.paste(f.convert("RGB"), (0, i * h))
    pal = sample.quantize(colors=127, method=Image.Quantize.MEDIANCUT)
    qs = [f.convert("RGB").quantize(palette=pal, dither=Image.Dither.NONE) for f in imgs]

    first = qs[0].copy()
    outer = imgs[0].getchannel("A").point(lambda a: 255 if a < 96 else 0)
    first.paste(255, mask=outer)  # window corners stay transparent
    frames = [first]
    prev = qs[0].convert("RGB")
    for q in qs[1:]:
        cur = q.convert("RGB")
        same = ImageChops.difference(cur, prev).convert("L").point(lambda v: 255 if v == 0 else 0)
        fr = q.copy()
        fr.paste(255, mask=same)
        frames.append(fr)
        prev = cur
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durs,
                   loop=0, transparency=255, disposal=1, optimize=True)


# ---- scenes ------------------------------------------------------------------
class Scene:
    def __init__(self, prompt: str, argv: list[str], cwd: str,
                 env_extra: dict | None = None, max_s: float = 240.0,
                 max_frames: int = 0, warmup: bool = False):
        self.prompt = prompt
        self.argv = argv
        self.cwd = cwd
        self.env_extra = env_extra or {}
        self.max_s = max_s
        self.max_frames = max_frames  # 0 = keep all; else uniformly subsample
        self.warmup = warmup          # run once off-camera first (cold-start cost)


def _cap_frames(frames: list[tuple[Grid, int]], n: int) -> list[tuple[Grid, int]]:
    """Uniformly subsample to at most ``n`` frames, folding each dropped frame's
    duration into the kept frame before it, so total wall-time is preserved.
    Scrolling output (e.g. a pip install) produces full-frame diffs that the
    delta-encoder cannot shrink; capping the frame count is what keeps that GIF
    small without touching the in-place repaints of the cockpit."""
    if n <= 0 or len(frames) <= n:
        return frames
    keep = {round(i * (len(frames) - 1) / (n - 1)) for i in range(n)}
    out: list[tuple[Grid, int]] = []
    carry = 0
    for i, (g, ms) in enumerate(frames):
        if i in keep:
            out.append((g, ms + carry))
            carry = 0
        else:
            carry += ms
    if carry and out:
        out[-1] = (out[-1][0], out[-1][1] + carry)
    return out


def _redact(chunks: list[tuple[float, bytes]]) -> list[tuple[float, bytes]]:
    """Strip machine-specific paths from the captured stream — the recording
    machine's interpreter path and $HOME — so nothing local leaks into a
    committed asset. Purely cosmetic; the numbers are untouched."""
    subs = [(sys.executable.encode(), b"python3"),
            (str(REPO).encode(), b"."),
            (str(Path.home()).encode(), b"~")]
    out = []
    for t, d in chunks:
        for a, b in subs:
            d = d.replace(a, b)
        out.append((t, d))
    return out


def build(scenes: list[Scene], cols: int, rows: int, title: str,
          scale: float, dump: int, out: Path) -> None:
    imgs: list[Image.Image] = []
    durs: list[int] = []
    blank: Grid = tuple(tuple((" ", "default", "default", False, False) for _ in range(cols))
                        for _ in range(rows))
    for si, sc in enumerate(scenes):
        print(f"  scene {si + 1}/{len(scenes)}: {' '.join(sc.argv)}  (cwd={sc.cwd})")
        if sc.warmup:
            # Pay the cold-start cost (first import compiles bytecode; matplotlib
            # builds its font cache — ~35s) off-camera, so the recorded run is the
            # fast warm run and its output actually fits inside max_s.
            print("    warming up (off-camera) …")
            subprocess.run(sc.argv, cwd=sc.cwd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           env={**os.environ, "PYTHONUNBUFFERED": "1"}, timeout=180)
        chunks = record(sc.argv, sc.cwd, cols, rows, sc.max_s, env_extra=sc.env_extra)
        chunks = _redact(chunks)
        print(f"    captured {len(chunks)} chunks · {sum(len(c) for _, c in chunks)} bytes")
        frames = replay(chunks, cols, rows)
        if sc.max_frames:
            frames = _cap_frames(frames, sc.max_frames)
        print(f"    {len(frames)} frames after dedup")

        # typed-prompt intro (the one presentational flourish), then the capture
        for i in range(0, len(sc.prompt) + 1, 3):
            imgs.append(render(blank, cols, rows, title, sc.prompt[:i]))
            durs.append(52)
        imgs.append(render(blank, cols, rows, title, sc.prompt))
        durs.append(430)
        for grid, ms in frames:
            imgs.append(render(grid, cols, rows, title, sc.prompt))
            durs.append(ms)
        durs[-1] = 1600 if si < len(scenes) - 1 else 3200  # linger between/at end

    if scale != 1.0:
        size = (int(imgs[0].width * scale), int(imgs[0].height * scale))
        imgs = [f.resize(size, Image.LANCZOS) for f in imgs]
    if dump:
        for i in range(0, len(imgs), dump):
            imgs[i].convert("RGB").save(out.parent / f"_rec_{out.stem}_{i:03d}.png")
    save_gif(imgs, durs, out)
    print(f"wrote {out} · {len(imgs)} frames · {out.stat().st_size // 1024} KB")


# ---- presets -----------------------------------------------------------------
def _seed_data(data_dir: Path) -> Path:
    """Lay down the example-subject folder + weights-free config the `run`
    preset records against, if it is not already there. Expects the four
    sub-01 files reachable via $PBRAIN_EXAMPLE (a `pbrain fetch-data` folder)."""
    sub = data_dir / "sub-01"
    if (sub / "sub-01_dce.nii.gz").exists() and (data_dir / "sub-01.toml").exists():
        return data_dir
    src = Path(os.environ.get("PBRAIN_EXAMPLE", "")).expanduser()
    if not src.is_dir():
        raise SystemExit(
            "run preset needs the example subject. Set $PBRAIN_EXAMPLE to a folder "
            "holding sub-01_{dce,ir,aif,parcellation}.* (from `pbrain fetch-data`).")
    sub.mkdir(parents=True, exist_ok=True)
    for tag in ("dce", "ir", "parcellation"):
        shutil.copy(src / f"sub-01_{tag}.nii.gz", sub / f"sub-01_{tag}.nii.gz")
    shutil.copy(src / "sub-01_aif.npy", sub / "sub-01_aif.npy")
    (data_dir / "sub-01.toml").write_text(
        '# Weights-free reproduction of the p-Brain example subject (sub-01).\n'
        'subject_dir = "sub-01"\n\n'
        '[inputs]\n'
        'dce = "sub-01/sub-01_dce.nii.gz"\n'
        'ir  = "sub-01/sub-01_ir.nii.gz"\n\n'
        '[pipeline]\n'
        't1m0           = "inversion_recovery"\n'
        'signal_to_conc = "saturation_recovery"\n'
        'aif            = "curve_file"\n'
        'tissue_roi     = "preloaded"\n'
        'normaliser     = "identity"\n'
        'aggregations   = ["region", "parcel", "voxelwise"]\n\n'
        '[options]\n'
        '"aif.curve_file.curve_path" = "sub-01/sub-01_aif.npy"\n'
        '"tissue_roi.preloaded.parcellation_path" = "sub-01/sub-01_parcellation.nii.gz"\n')
    return data_dir


def preset_run(args) -> tuple[list[Scene], int, int, str]:
    data = _seed_data(Path(args.data_dir).expanduser().resolve())
    # Drive the repo's own interpreter so the recording is of THIS working tree.
    pb = [sys.executable, "-m", "pbrain"]
    scene = Scene(
        prompt="pbrain run --config sub-01.toml --models patlak",
        argv=[*pb, "run", "--config", "sub-01.toml", "--models", "patlak"],
        cwd=str(data), max_s=args.max)
    return [scene], 92, 30, "p-brain — sub-01"


def preset_install(args) -> tuple[list[Scene], int, int, str]:
    venv = Path(args.venv or (tempfile.mkdtemp(prefix="pbrain-demo-") + "/venv"))
    if not (venv / "bin" / "python").exists():
        print(f"  creating fresh venv at {venv} …")
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        subprocess.run([str(venv / "bin" / "python"), "-m", "pip", "install",
                        "--quiet", "--upgrade", "pip"], check=True)
    vpip = str(venv / "bin" / "pip")
    vpbrain = str(venv / "bin" / "pbrain")
    src = args.package  # "p-brain" (PyPI) or a local wheel path
    scenes = [
        # pip output scrolls (full-frame diffs the delta-encoder can't shrink),
        # so cap the frame count; the banner repaints in place and stays whole.
        Scene(prompt="pip install p-brain",
              argv=[vpip, "install", src], cwd=str(venv), max_s=args.max,
              max_frames=42),
        Scene(prompt="pbrain",
              argv=[vpbrain], cwd=str(venv), max_s=30.0, warmup=True),
    ]
    # 100 cols keeps most help lines off the wrap; 28 rows keeps the brain glyph
    # in frame under the command menu after the banner has drawn in.
    return scenes, 100, 28, "p-brain"


PRESETS = {"run": preset_run, "install": preset_install}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", choices=sorted(PRESETS), help="which demo to record")
    ap.add_argument("--data-dir", default=str(Path(tempfile.gettempdir()) / "pbrain-demo-data"),
                    help="run preset: folder holding sub-01/ and sub-01.toml")
    ap.add_argument("--venv", default="", help="install preset: reuse this venv instead of a fresh one")
    ap.add_argument("--package", default="p-brain",
                    help="install preset: what pip installs (PyPI name or a local wheel path)")
    ap.add_argument("--scale", type=float, default=0.70, help="output downscale factor")
    ap.add_argument("--max", type=float, default=300.0, help="hard per-scene cap in seconds")
    ap.add_argument("--dump", type=int, default=0, help="also write every Nth frame as PNG")
    ap.add_argument("--out", default="", help="output path (default assets/demo_<preset>.gif)")
    a = ap.parse_args()

    scenes, cols, rows, title = PRESETS[a.preset](a)
    out = Path(a.out) if a.out else REPO / "assets" / f"demo_{a.preset}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"recording preset {a.preset!r} → {out}  ({cols}x{rows})")
    build(scenes, cols, rows, title, a.scale, a.dump, out)


if __name__ == "__main__":
    main()
