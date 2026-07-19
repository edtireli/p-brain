"""Cross-platform onboarding for the optional local model backend (Ollama).

First time someone uses ``--assist`` without a model, p-Brain should help them
get one — detect the OS, detect how much memory the machine has, recommend a
text model (and a vision model, for the AIF localiser) sized to that hardware,
and offer to pull them. Nothing is installed or downloaded without the user
saying yes; on a non-interactive run it just prints the commands.

Stdlib only. Works on macOS, Linux, and Windows.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys


# ---------------------------------------------------------------- detection
def installed() -> bool:
    """Is the ``ollama`` binary on PATH (installed, though maybe not running)?"""
    return shutil.which("ollama") is not None


def reachable() -> bool:
    from pbrain import _assist
    return bool(_assist._tags()) or _assist.available()


def total_ram_gb() -> float:
    """Physical RAM in GB — the constraint that decides model size."""
    try:
        if sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip()) / 1e9
        if sys.platform.startswith("linux"):
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        if sys.platform.startswith("win"):
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = _MS(); m.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 1e9
    except Exception:
        pass
    return 0.0


def has_gpu() -> bool:
    try:
        from pbrain.core.devices import probe
        p = probe()
        return bool(p.get("cuda_torch") or p.get("mps_torch") or p.get("gpu_tf"))
    except Exception:
        return sys.platform == "darwin"   # Apple Silicon has a usable GPU for Ollama


# ---------------------------------------------------------------- recommendation
def recommend(ram_gb: float | None = None) -> dict:
    """Models sized to the machine's memory.

    Text assist → Ollama (a good local text lineup). Vision assist (the AIF/VIF
    localiser) → HuggingFace, because Ollama's vision lineup is thin and HF has the
    capable Qwen-VL family: mlx-vlm on Apple Silicon (fast, native), transformers
    elsewhere. Bigger RAM → a stronger VLM."""
    r = ram_gb if ram_gb is not None else total_ram_gb()
    if r < 8:
        text = "llama3.2:1b"
    elif r < 16:
        text = "qwen2.5:3b"
    elif r < 32:
        text = "qwen3:8b"
    elif r < 64:
        text = "qwen2.5:14b"
    else:
        text = "qwen2.5:32b"

    mlx = sys.platform == "darwin"
    if r < 24:
        vrepo = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit" if mlx else "Qwen/Qwen2.5-VL-3B-Instruct"
    elif r < 48:
        vrepo = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit" if mlx else "Qwen/Qwen2.5-VL-7B-Instruct"
    else:
        vrepo = "mlx-community/Qwen2.5-VL-32B-Instruct-4bit" if mlx else "Qwen/Qwen2.5-VL-32B-Instruct"
    vision = {
        "backend": "mlx-vlm" if mlx else "transformers",
        "repo": vrepo,
        "install": "pip install mlx-vlm" if mlx else "pip install 'transformers>=4.49' accelerate torch pillow",
    }
    return {"ram_gb": round(r, 1), "text": text, "vision": vision}


def install_hint() -> str:
    if sys.platform == "darwin":
        return "brew install ollama    (or download: https://ollama.com/download)"
    if sys.platform.startswith("win"):
        return "winget install Ollama.Ollama    (or download: https://ollama.com/download)"
    return "curl -fsSL https://ollama.com/install.sh | sh"


# ---------------------------------------------------------------- onboarding
def _ask(prompt: str) -> bool:
    if not sys.stdin.isatty():
        return False
    return input(prompt).strip().lower() in ("y", "yes")


def guide(con, *, want_vision: bool = False) -> bool:
    """Walk the user from nothing to a working assist backend. Returns True if an
    assist model is available at the end. Prints commands; only runs installs/pulls
    the user explicitly approves. ``con`` is a p-Brain rich console."""
    rec = recommend()
    con.print(f"  [pb.accent]▸ assist setup[/]  [pb.mut]{rec['ram_gb']} GB RAM · "
              f"{'GPU' if has_gpu() else 'CPU'} · {sys.platform}[/]")

    if not installed():
        con.print("  [pb.warn]Ollama isn't installed.[/] It runs local models on your machine (free, offline).")
        con.print(f"    install:  [pb.ink]{install_hint()}[/]")
        if sys.platform != "win32" and _ask("    run the install now? [y/N] "):
            try:
                subprocess.run("curl -fsSL https://ollama.com/install.sh | sh" if sys.platform.startswith("linux")
                               else "brew install ollama", shell=True, check=True)
            except Exception as e:
                con.print(f"    [pb.fail]install failed[/] ({e}) — install manually, then re-run.")
                return False
        else:
            con.print("    [pb.dim]install it, then re-run with --assist.[/]")
            return False

    if not reachable():
        con.print("  [pb.warn]Ollama is installed but not running.[/]  start it:  [pb.ink]ollama serve[/]  "
                  "[pb.dim](or launch the Ollama app)[/]")
        return False

    from pbrain import _assist
    need = [] if _assist.available() else [("text", rec["text"])]
    if want_vision:
        need.append(("vision", rec["vision"]))
    if not need:
        con.print(f"  [pb.accent]●[/] ready · using [pb.ink]{_assist.model()}[/]")
        return True

    con.print(f"  recommended for your machine:  " +
              " · ".join(f"[pb.ink]{m}[/] ({role})" for role, m in need))
    for role, m in need:
        con.print(f"    pull:  [pb.ink]ollama pull {m}[/]")
        if _ask(f"    pull {m} now? [y/N] "):
            try:
                subprocess.run(["ollama", "pull", m], check=True)
            except Exception as e:
                con.print(f"    [pb.fail]pull failed[/] ({e})")
    return _assist.available()
