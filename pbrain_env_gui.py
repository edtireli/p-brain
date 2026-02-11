#!/usr/bin/env python3
"""Small GUI applet for configuring p-brain environment variables.

- Lists common P_BRAIN_* settings with defaults.
- Lets you toggle / edit values.
- One-click "Apply" saves a .env file and updates this process env.
- Optional: launch p-brain (main.py) with the chosen env.

This does NOT modify your shell environment automatically; use the generated
export commands or source the saved .env file in your shell.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional


try:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import messagebox
    from tkinter.scrolledtext import ScrolledText
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Tkinter is required to run this UI. On some Python installs (macOS/Homebrew), "
        "tk may not be bundled.\n\n"
        f"Import error: {exc}"
    )


@dataclass(frozen=True)
class Setting:
    key: str
    kind: str  # bool|choice|str|int|float
    default: str
    help: str
    choices: tuple[str, ...] = ()


SETTINGS: tuple[Setting, ...] = (
    Setting(
        key="P_BRAIN_AIF_USE_SSS",
        kind="bool",
        default="1",
        help="Use TSCC (SSS time-shifted/rescaled curve) as modelling input.",
    ),
    Setting(
        key="P_BRAIN_AIF_ARTERY",
        kind="choice",
        default="RICA",
        choices=("RICA", "LICA"),
        help="Reference artery used for alignment (RICA or LICA).",
    ),
    Setting(
        key="P_BRAIN_TSCC_RESCALE",
        kind="bool",
        default="1",
        help="Enable amplitude rescaling when generating TSCC (disable = time-shift only).",
    ),
    Setting(
        key="P_BRAIN_TSCC_RESCALE_METHOD",
        kind="choice",
        default="peak",
        choices=("peak", "auc"),
        help="How TSCC is amplitude-matched: peak height or AUC.",
    ),
    Setting(
        key="P_BRAIN_VASCULAR_ROI_CURVE_METHOD",
        kind="choice",
        default="max",
        choices=("max", "mean"),
        help="How vascular ROI curves are extracted: max voxel vs ROI mean.",
    ),
    Setting(
        key="P_BRAIN_VASCULAR_ROI_ADAPTIVE_MAX",
        kind="bool",
        default="1",
        help="If enabled (and curve method=max), re-select brightest voxel per frame.",
    ),
    Setting(
        key="P_BRAIN_ROI_METHOD",
        kind="choice",
        default="ai",
        choices=("ai", "deterministic"),
        help="ROI extraction method.",
    ),
    Setting(
        key="P_BRAIN_ROI_NORMALIZE_CURVES",
        kind="bool",
        default="1",
        help="Normalize curves for PCA/diagnostic plots (baseline+robust scaling).",
    ),
    Setting(
        key="P_BRAIN_CTC_MODEL",
        kind="choice",
        default="saturation",
        choices=("saturation", "turboflash"),
        help="Signal-to-concentration conversion model.",
    ),
    Setting(
        key="P_BRAIN_TURBO_NPH",
        kind="int",
        default="1",
        help="TurboFLASH nph (used only if CTC_MODEL=turboflash).",
    ),
    Setting(
        key="P_BRAIN_NUMBER_OF_PEAKS",
        kind="int",
        default="2",
        help="Number of bolus peaks (alignment/TSCC utilities).",
    ),
)


def _bool_from_env(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _env_default(s: Setting) -> str:
    v = os.environ.get(s.key)
    if v is None or str(v).strip() == "":
        return s.default
    return str(v).strip()


def _serialize_env_line(key: str, value: str) -> str:
    # .env style: KEY=value (value may be quoted if needed)
    value = "" if value is None else str(value)
    if value == "" or any(ch.isspace() for ch in value) or any(ch in value for ch in ('"', "'", "#")):
        return f"{key}={shlex.quote(value)}"
    return f"{key}={value}"


class EnvGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("p-brain env settings")
        self.geometry("980x680")

        self._vars: dict[str, tk.Variable] = {}
        self._widgets: dict[str, tk.Widget] = {}

        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root)
        header.pack(fill=tk.X)

        ttk.Label(
            header,
            text="Configure p-brain environment variables",
            font=("TkDefaultFont", 14, "bold"),
        ).pack(side=tk.LEFT)

        btns = ttk.Frame(header)
        btns.pack(side=tk.RIGHT)

        ttk.Button(btns, text="Reload from env", command=self._load_from_env).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btns, text="Reset to defaults", command=self._reset_to_defaults).pack(side=tk.LEFT)

        ttk.Separator(root).pack(fill=tk.X, pady=10)

        body = ttk.Frame(root)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(body)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        self._build_settings_table(left)
        self._build_actions_panel(right)

        self._load_from_env()
        self._refresh_preview()

    def _build_settings_table(self, parent: ttk.Frame) -> None:
        canvas = tk.Canvas(parent)
        vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll = ttk.Frame(canvas)

        scroll.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=scroll, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Headers
        hdr = ttk.Frame(scroll)
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text="Env var", width=36).grid(row=0, column=0, sticky="w")
        ttk.Label(hdr, text="Value", width=20).grid(row=0, column=1, sticky="w")
        ttk.Label(hdr, text="Description").grid(row=0, column=2, sticky="w")

        ttk.Separator(scroll).pack(fill=tk.X, pady=6)

        for idx, s in enumerate(SETTINGS, start=1):
            row = ttk.Frame(scroll)
            row.pack(fill=tk.X, pady=2)

            ttk.Label(row, text=s.key, width=36).grid(row=0, column=0, sticky="w")

            if s.kind == "bool":
                v = tk.BooleanVar(value=_bool_from_env(_env_default(s)))
                cb = ttk.Checkbutton(row, variable=v, command=self._refresh_preview)
                cb.grid(row=0, column=1, sticky="w")
                self._vars[s.key] = v
                self._widgets[s.key] = cb
            elif s.kind == "choice":
                v = tk.StringVar(value=_env_default(s))
                combo = ttk.Combobox(row, textvariable=v, values=list(s.choices), width=18, state="readonly")
                combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_preview())
                combo.grid(row=0, column=1, sticky="w")
                self._vars[s.key] = v
                self._widgets[s.key] = combo
            else:
                v = tk.StringVar(value=_env_default(s))
                ent = ttk.Entry(row, textvariable=v, width=20)
                ent.bind("<KeyRelease>", lambda _e: self._refresh_preview())
                ent.grid(row=0, column=1, sticky="w")
                self._vars[s.key] = v
                self._widgets[s.key] = ent

            ttk.Label(row, text=s.help, wraplength=520, justify=tk.LEFT).grid(row=0, column=2, sticky="w")

        # Mouse wheel scrolling
        def _on_mousewheel(event: tk.Event) -> None:  # type: ignore[override]
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def _build_actions_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent)
        panel.pack(fill=tk.BOTH, expand=True)

        ttk.Label(panel, text="Apply / export", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")

        self._env_path_var = tk.StringVar(value=str(self._default_env_path()))
        path_row = ttk.Frame(panel)
        path_row.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(path_row, text=".env path:").pack(side=tk.LEFT)
        ttk.Entry(path_row, textvariable=self._env_path_var, width=44).pack(side=tk.LEFT, padx=(6, 0))

        btn_row = ttk.Frame(panel)
        btn_row.pack(fill=tk.X, pady=(6, 10))
        ttk.Button(btn_row, text="Apply (save .env)", command=self._apply).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Copy exports", command=self._copy_exports).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(panel, text="Generated exports:").pack(anchor="w")
        self._preview = ScrolledText(panel, height=16, width=52)
        self._preview.pack(fill=tk.BOTH, expand=False)

        ttk.Separator(panel).pack(fill=tk.X, pady=10)

        ttk.Label(panel, text="Run p-brain", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        self._run_args = tk.StringVar(value="")
        ttk.Label(panel, text="Args (optional):").pack(anchor="w", pady=(8, 0))
        ttk.Entry(panel, textvariable=self._run_args, width=52).pack(fill=tk.X)
        ttk.Button(panel, text="Launch main.py with these env", command=self._launch).pack(anchor="w", pady=(8, 0))

        ttk.Label(
            panel,
            text="Note: saving .env won’t change your current shell; you can `source` it in zsh.",
            wraplength=380,
            justify=tk.LEFT,
            foreground="#444",
        ).pack(anchor="w", pady=(10, 0))

    def _default_env_path(self) -> Path:
        # Prefer repo-local file next to main.py if we can find it.
        here = Path(__file__).resolve().parent
        return here / ".pbrain.env"

    def _collect_values(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for s in SETTINGS:
            v = self._vars.get(s.key)
            if v is None:
                continue
            if s.kind == "bool":
                out[s.key] = "1" if bool(v.get()) else "0"
            else:
                out[s.key] = str(v.get()).strip()
        return out

    def _refresh_preview(self) -> None:
        env = self._collect_values()
        lines = [f"export {k}={shlex.quote(v)}" for k, v in env.items()]
        text = "\n".join(lines) + "\n"
        self._preview.configure(state="normal")
        self._preview.delete("1.0", tk.END)
        self._preview.insert("1.0", text)
        self._preview.configure(state="disabled")

    def _load_from_env(self) -> None:
        for s in SETTINGS:
            raw = _env_default(s)
            v = self._vars.get(s.key)
            if v is None:
                continue
            if s.kind == "bool":
                v.set(_bool_from_env(raw))
            else:
                v.set(raw)
        self._refresh_preview()

    def _reset_to_defaults(self) -> None:
        for s in SETTINGS:
            v = self._vars.get(s.key)
            if v is None:
                continue
            if s.kind == "bool":
                v.set(_bool_from_env(s.default))
            else:
                v.set(s.default)
        self._refresh_preview()

    def _apply(self) -> None:
        path = Path(str(self._env_path_var.get()).strip()).expanduser().resolve()
        env = self._collect_values()

        # Basic validation for int/float kinds.
        for s in SETTINGS:
            if s.kind == "int":
                try:
                    int(env[s.key])
                except Exception:
                    messagebox.showerror("Invalid value", f"{s.key} must be an integer")
                    return
            if s.kind == "float":
                try:
                    float(env[s.key])
                except Exception:
                    messagebox.showerror("Invalid value", f"{s.key} must be a number")
                    return

        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(_serialize_env_line(k, v) for k, v in env.items()) + "\n"
        path.write_text(content, encoding="utf-8")

        # Apply to this process env as well.
        os.environ.update(env)

        messagebox.showinfo("Saved", f"Saved {len(env)} settings to:\n{path}")

    def _copy_exports(self) -> None:
        txt = self._preview.get("1.0", tk.END)
        self.clipboard_clear()
        self.clipboard_append(txt)
        self.update()
        messagebox.showinfo("Copied", "Export commands copied to clipboard")

    def _launch(self) -> None:
        # Launch p-brain main.py using this process's current env.
        here = Path(__file__).resolve().parent
        main_py = here / "main.py"
        if not main_py.exists():
            messagebox.showerror("Not found", f"Could not find {main_py}")
            return

        args = shlex.split(str(self._run_args.get() or "").strip())
        cmd = [sys.executable, str(main_py), *args]
        try:
            subprocess.Popen(cmd, cwd=str(here), env=dict(os.environ))
        except Exception as exc:
            messagebox.showerror("Launch failed", str(exc))
            return

        messagebox.showinfo("Launched", "p-brain launched in a new process")


def main() -> int:
    app = EnvGui()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
