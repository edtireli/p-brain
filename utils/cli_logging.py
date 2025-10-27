"""Utilities for consistent logging in the fully automatic CLI mode."""
from __future__ import annotations

import builtins
import os
import sys
from datetime import datetime
from typing import Any, Callable, Dict, Optional

try:  # Matplotlib is optional when running in headless/test modes
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib may be unavailable in tests
    plt = None  # type: ignore

try:  # NumPy is a hard dependency but guarded for completeness
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # Nibabel is optional for some environments
    import nibabel as nib  # type: ignore
except Exception:  # pragma: no cover
    nib = None  # type: ignore

import json
import pickle
import shutil

_PREFIX = "[AUTO]"
_BASE_PRINT = builtins.print
_HOOK_STATE: Dict[str, Any] = {"installed": False}


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _format_path(path: os.PathLike[str] | str) -> str:
    try:
        return os.path.abspath(os.fspath(path))
    except TypeError:
        return str(path)


def log_auto(message: str) -> None:
    """Emit a standardised log message for automatic mode."""
    _BASE_PRINT(f"{_PREFIX} {_timestamp()} | {message}")


def log_process_start(name: str) -> None:
    log_auto(f"Starting process: {name}")


def log_process_end(name: str) -> None:
    log_auto(f"Completed process: {name}")


def log_generated_file(path: os.PathLike[str] | str) -> None:
    log_auto(f"Generated file: {_format_path(path)}")


def log_generated_image(path: os.PathLike[str] | str) -> None:
    log_auto(f"Generated image: {_format_path(path)}")


def log_existing_file(description: str, path: os.PathLike[str] | str) -> None:
    log_auto(f"Loaded {description}: {_format_path(path)}")


def _path_from_file_like(handle: Any) -> Optional[str]:
    if isinstance(handle, (str, os.PathLike)):
        return os.fspath(handle)
    name = getattr(handle, "name", None)
    if isinstance(name, (str, os.PathLike)):
        return os.fspath(name)
    return None


def _patch_print() -> None:
    if _HOOK_STATE.get("print") is not None:
        return

    original_print = builtins.print

    def patched_print(*args: Any, sep: str = " ", end: str = "\n", file: Any = None, flush: bool = False) -> None:  # type: ignore[override]
        if file not in (None, sys.stdout, sys.stderr):
            original_print(*args, sep=sep, end=end, file=file, flush=flush)
            return
        message = sep.join(str(arg) for arg in args)
        if end and end != "\n":
            message = f"{message}{end}"
        log_auto(message)

    _HOOK_STATE["print"] = original_print
    builtins.print = patched_print  # type: ignore[assignment]


def _patch_matplotlib() -> None:
    if plt is None:
        return
    if _HOOK_STATE.get("savefig") is not None:
        return

    original_savefig = plt.savefig

    def savefig_hook(*args: Any, **kwargs: Any) -> Any:
        path = None
        if args:
            path = args[0]
        elif "fname" in kwargs:
            path = kwargs["fname"]
        result = original_savefig(*args, **kwargs)
        if isinstance(path, (str, os.PathLike)):
            log_generated_image(path)
        return result

    _HOOK_STATE["savefig"] = original_savefig
    plt.savefig = savefig_hook  # type: ignore[assignment]


def _patch_numpy() -> None:
    if np is None:
        return

    if _HOOK_STATE.get("np_save") is None:
        original_save = np.save

        def save_hook(file: Any, arr: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_save(file, arr, *args, **kwargs)
            path = _path_from_file_like(file)
            if path:
                log_generated_file(path)
            return result

        _HOOK_STATE["np_save"] = original_save
        np.save = save_hook  # type: ignore[assignment]

    if _HOOK_STATE.get("np_savez") is None:
        original_savez = np.savez

        def savez_hook(file: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_savez(file, *args, **kwargs)
            path = _path_from_file_like(file)
            if path:
                log_generated_file(path)
            return result

        _HOOK_STATE["np_savez"] = original_savez
        np.savez = savez_hook  # type: ignore[assignment]

    if _HOOK_STATE.get("np_savez_compressed") is None and hasattr(np, "savez_compressed"):
        original_savez_compressed = np.savez_compressed

        def savez_compressed_hook(file: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_savez_compressed(file, *args, **kwargs)
            path = _path_from_file_like(file)
            if path:
                log_generated_file(path)
            return result

        _HOOK_STATE["np_savez_compressed"] = original_savez_compressed
        np.savez_compressed = savez_compressed_hook  # type: ignore[assignment]


def _patch_json_pickle() -> None:
    if _HOOK_STATE.get("json_dump") is None:
        original_json_dump = json.dump

        def json_dump_hook(obj: Any, fp: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_json_dump(obj, fp, *args, **kwargs)
            path = getattr(fp, "name", None)
            if isinstance(path, (str, os.PathLike)):
                log_generated_file(path)
            return result

        _HOOK_STATE["json_dump"] = original_json_dump
        json.dump = json_dump_hook  # type: ignore[assignment]

    if _HOOK_STATE.get("pickle_dump") is None:
        original_pickle_dump = pickle.dump

        def pickle_dump_hook(obj: Any, file: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_pickle_dump(obj, file, *args, **kwargs)
            path = _path_from_file_like(file)
            if path:
                log_generated_file(path)
            return result

        _HOOK_STATE["pickle_dump"] = original_pickle_dump
        pickle.dump = pickle_dump_hook  # type: ignore[assignment]


def _patch_nibabel() -> None:
    if nib is None or not hasattr(nib, "save"):
        return
    if _HOOK_STATE.get("nib_save") is not None:
        return

    original_nib_save = nib.save

    def nib_save_hook(img: Any, filename: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_nib_save(img, filename, *args, **kwargs)
        path = _path_from_file_like(filename)
        if path:
            log_generated_file(path)
        return result

    _HOOK_STATE["nib_save"] = original_nib_save
    nib.save = nib_save_hook  # type: ignore[assignment]


def _patch_shutil_os() -> None:
    if _HOOK_STATE.get("copy2") is None:
        original_copy2 = shutil.copy2

        def copy2_hook(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_copy2(src, dst, *args, **kwargs)
            path = _path_from_file_like(dst)
            if path:
                log_generated_file(path)
            return result

        _HOOK_STATE["copy2"] = original_copy2
        shutil.copy2 = copy2_hook  # type: ignore[assignment]

    if _HOOK_STATE.get("copy") is None:
        original_copy = shutil.copy

        def copy_hook(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_copy(src, dst, *args, **kwargs)
            path = _path_from_file_like(dst)
            if path:
                log_generated_file(path)
            return result

        _HOOK_STATE["copy"] = original_copy
        shutil.copy = copy_hook  # type: ignore[assignment]

    if _HOOK_STATE.get("move") is None:
        original_move = shutil.move

        def move_hook(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_move(src, dst, *args, **kwargs)
            path = _path_from_file_like(dst)
            if path:
                log_generated_file(path)
            return result

        _HOOK_STATE["move"] = original_move
        shutil.move = move_hook  # type: ignore[assignment]

    if _HOOK_STATE.get("rename") is None:
        original_rename = os.rename

        def rename_hook(src: Any, dst: Any) -> None:
            original_rename(src, dst)
            path = _path_from_file_like(dst)
            if path:
                log_generated_file(path)

        _HOOK_STATE["rename"] = original_rename
        os.rename = rename_hook  # type: ignore[assignment]


_PATCHERS: Dict[str, Callable[[], None]] = {
    "print": _patch_print,
    "matplotlib": _patch_matplotlib,
    "numpy": _patch_numpy,
    "json_pickle": _patch_json_pickle,
    "nibabel": _patch_nibabel,
    "shutil_os": _patch_shutil_os,
}


def auto_logging_enabled() -> bool:
    """Return ``True`` when automatic logging hooks are active."""
    return bool(_HOOK_STATE.get("installed"))


def install_auto_logging_hooks() -> None:
    """Install hooks that standardise logging and record file outputs."""
    if _HOOK_STATE.get("installed"):
        return

    for patcher in _PATCHERS.values():
        patcher()

    _HOOK_STATE["installed"] = True
    log_auto("Automatic logging hooks installed.")


def uninstall_auto_logging_hooks() -> None:
    """Restore patched functions to their original implementations."""
    if not _HOOK_STATE.get("installed"):
        return

    if "print" in _HOOK_STATE:
        builtins.print = _HOOK_STATE.pop("print")  # type: ignore[assignment]
    if plt is not None and "savefig" in _HOOK_STATE:
        plt.savefig = _HOOK_STATE.pop("savefig")  # type: ignore[assignment]
    if np is not None:
        if "np_save" in _HOOK_STATE:
            np.save = _HOOK_STATE.pop("np_save")  # type: ignore[assignment]
        if "np_savez" in _HOOK_STATE:
            np.savez = _HOOK_STATE.pop("np_savez")  # type: ignore[assignment]
        if "np_savez_compressed" in _HOOK_STATE:
            np.savez_compressed = _HOOK_STATE.pop("np_savez_compressed")  # type: ignore[assignment]
    if "json_dump" in _HOOK_STATE:
        json.dump = _HOOK_STATE.pop("json_dump")  # type: ignore[assignment]
    if "pickle_dump" in _HOOK_STATE:
        pickle.dump = _HOOK_STATE.pop("pickle_dump")  # type: ignore[assignment]
    if nib is not None and "nib_save" in _HOOK_STATE:
        nib.save = _HOOK_STATE.pop("nib_save")  # type: ignore[assignment]
    if "copy2" in _HOOK_STATE:
        shutil.copy2 = _HOOK_STATE.pop("copy2")  # type: ignore[assignment]
    if "copy" in _HOOK_STATE:
        shutil.copy = _HOOK_STATE.pop("copy")  # type: ignore[assignment]
    if "move" in _HOOK_STATE:
        shutil.move = _HOOK_STATE.pop("move")  # type: ignore[assignment]
    if "rename" in _HOOK_STATE:
        os.rename = _HOOK_STATE.pop("rename")  # type: ignore[assignment]

    _HOOK_STATE["installed"] = False
    log_auto("Automatic logging hooks removed.")
