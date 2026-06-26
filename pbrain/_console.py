"""Console encoding safety for non-UTF-8 terminals (e.g. Windows ``cp1252``).

p-Brain's CLI prints a handful of non-ASCII glyphs (arrows, box-drawing rules,
Greek symbols). On a legacy Windows console whose active code page is ``cp1252``,
writing any of those to ``stdout`` raises ``UnicodeEncodeError`` and aborts the
command before it does any work. Reconfiguring the standard streams to UTF-8
(with error-replacement as a last resort) makes every entry point tolerate them,
so the same code runs identically on POSIX and Windows.
"""

from __future__ import annotations

import sys


def ensure_utf8_console() -> None:
    """Best-effort: stop ``stdout``/``stderr`` aborting on non-UTF-8 consoles.

    Idempotent and exception-free, so it is safe to call at the very top of every
    entry-point ``main()``. Tries UTF-8 first (clean glyphs on modern terminals);
    if the stream refuses, falls back to replacing unencodable characters so a
    stray Unicode glyph can never crash a run. Streams that are not reconfigurable
    (captured/redirected pipes) are left untouched.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:        # noqa: BLE001
            try:
                reconfigure(errors="replace")   # keep native encoding, just don't abort
            except Exception:    # noqa: BLE001
                pass
