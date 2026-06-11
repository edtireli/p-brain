"""Generic fallback diagnostic — used for any kinetic model that doesn't
ship its own ``.diagnose()`` method nor a ``pbrain/diagnostics/<key>.py``.

The pipeline is fully blind to model identity: it re-runs the model on the
single tissue curve in the :class:`DiagnosticContext`, renders the AIF +
measured Cₜ, and annotates whichever scalar/array maps the model produced.
No assumption about output names — works for Patlak's ``{ki, vp}``, Tikhonov's
``{cbf, mtt, cth, lambda_opt}``, or any new model's bespoke set.

The file is underscore-prefixed so :func:`pbrain.core.discovery.discover`
does NOT register it as a plug-in — it is called directly from the
diagnostics stage as the last resort in the fall-back chain.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .base import DiagnosticContext


def generic_diagnostic(ctx: DiagnosticContext, model: Any) -> None:
    """Render a model-agnostic diagnostic figure for one curve.

    Layout: AIF (left axis) + measured Cₜ (right axis) on the left panel;
    parameter table on the right panel listing every map the model emitted
    for this curve. If the model exposes a ``predict(maps, c_input, t_s)``
    callable, a fitted-Cₜ overlay is added.
    """
    from pbrain.models import CurveInputs

    c_t = np.asarray(ctx.c_tissue, dtype=float).ravel()
    c_a = np.asarray(ctx.c_input, dtype=float).ravel()
    t = np.asarray(ctx.t_s, dtype=float).ravel()
    n = int(min(c_t.size, c_a.size, t.size))
    c_t, c_a, t = c_t[:n], c_a[:n], t[:n]

    # Re-fit the model on this single curve to recover its scalar params.
    inputs = CurveInputs(c_tissue=c_t.reshape(-1, 1), c_input=c_a, t_s=t)
    try:
        res = model.fit(inputs, **dict(ctx.model_opts or {}))
    except Exception as exc:  # noqa: BLE001
        _render_error(ctx, f"{type(model).__name__}.fit raised: {exc}")
        return

    units = dict(getattr(res, "units", {}) or {})
    param_rows: list[tuple[str, str, str]] = []
    for name in (getattr(model, "outputs", None) or list(res.maps.keys())):
        arr = res.maps.get(name)
        if arr is None:
            continue
        arr = np.asarray(arr).reshape(-1)
        val = float(arr[0]) if arr.size else float("nan")
        param_rows.append((name, _fmt(val), units.get(name, "")))

    # Optional fit overlay via duck-typed Model.predict().
    fit_curve = None
    predict = getattr(model, "predict", None)
    if callable(predict):
        try:
            fit_curve = np.asarray(
                predict({k: np.asarray(v).reshape(-1)[0] for k, v in res.maps.items()},
                        c_a, t),
                dtype=float,
            ).ravel()
            if fit_curve.size != n:
                fit_curve = None
        except Exception:
            fit_curve = None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                              gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(t, c_a, "tab:blue", lw=1.4, label=f"Cₐ peak={c_a.max():.3g}")
    ax2 = ax.twinx()
    ax2.plot(t, c_t, ".", color="tab:green", ms=3.5, alpha=0.6, label="Cₜ measured")
    if fit_curve is not None:
        ax2.plot(t, fit_curve, "k-", lw=1.5, label=f"{model.key} fit")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Cₐ (mM)", color="tab:blue")
    ax2.set_ylabel("Cₜ (mM)", color="tab:green")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.set_title(f"AIF + tissue  —  {model.key}")

    ax = axes[1]
    ax.axis("off")
    if param_rows:
        table = ax.table(
            cellText=[[n, v, u] for (n, v, u) in param_rows],
            colLabels=["parameter", "value", "unit"],
            cellLoc="center", loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.6)
    else:
        ax.text(0.5, 0.5, f"{model.key} produced no maps", ha="center", va="center")
    ax.set_title("Parameters")

    fig.suptitle(
        f"{model.key} diagnostic — {ctx.label}",
        fontsize=12, y=0.995,
    )
    ctx.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ctx.out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "NaN"
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1e4 or av < 1e-3:
        return f"{v:.4g}"
    return f"{v:.4f}"


def _render_error(ctx: DiagnosticContext, msg: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=10,
            wrap=True, family="monospace")
    ax.axis("off")
    ctx.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ctx.out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
