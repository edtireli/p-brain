"""Shared builders for kinetic-model ``review()`` (--mode verify/manual).

Two shapes cover every model:

* :func:`curve_fit_review` — the whole-brain mean tissue Cₜ with the model's own
  ``predict`` overlay, plus the median parameter values. Fits every
  residue/compartment model (Tikhonov, Extended Tofts, gamma, inverse-Gaussian,
  Mittag-Leffler, Stieltjes, transit-spectrum, …).
* :func:`patlak_plot_review` — the Patlak x=∫Cₐ/Cₐ vs y=Cₜ/Cₐ scatter: the full x
  trajectory for context but y scaled to the fitted points (early high-leverage
  outliers clipped off-axis).

Both return the declarative spec dict the web review renders, and either may carry
manual-mode ``controls`` (editable fit params → a live re-fit). Kept dependency-light
and defensive: a model's review must never break a run.
"""

from __future__ import annotations

import numpy as np

from .base import CurveInputs, ModelResult


def mean_tissue(inputs: CurveInputs) -> np.ndarray:
    """The whole-brain mean tissue curve (averaged over the mask if voxel-wise)."""
    c_t = np.asarray(inputs.c_tissue, dtype=float)
    if c_t.ndim == 2:
        m = inputs.mask
        cols = (c_t[:, np.asarray(m, dtype=bool)]
                if m is not None and np.asarray(m).size == c_t.shape[1] else c_t)
        return np.nanmean(cols, axis=1) if cols.size else c_t.mean(axis=1)
    return c_t


def median_maps(result: ModelResult) -> dict[str, float | None]:
    """Median of each output map over its finite, non-zero voxels (representative
    scalars for the ``predict`` overlay and the values panel)."""
    out: dict[str, float | None] = {}
    for k, mp in (result.maps or {}).items():
        a = np.asarray(mp, dtype=float).ravel()
        f = a[np.isfinite(a)]
        f = f[f != 0]
        out[k] = float(np.median(f)) if f.size else None
    return out


def _fmt(v: float | None, unit: str) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    s = f"{v:.0f}" if abs(v) >= 100 else (f"{v:.2f}" if abs(v) >= 1 else f"{v:.3g}")
    return f"{s} {unit}".strip()


def curve_fit_review(model, inputs, result, *, title: str, ylabel: str = "Cₜ (mM)",
                     fit_label: str = "model fit", show=None, controls=None) -> dict:
    """Generic residue/compartment view: mean Cₜ + the model's ``predict`` overlay
    (from the median maps) + the median parameter endpoints. ``show`` restricts which
    outputs land in the values panel (default: all of ``model.outputs``)."""
    t = np.asarray(inputs.t_s, dtype=float)
    ca = np.asarray(inputs.c_input, dtype=float)
    m_ct = np.asarray(mean_tissue(inputs), dtype=float)
    meds = median_maps(result)

    series = [{"label": "tissue Cₜ (mean)", "x": t.tolist(), "y": m_ct.tolist()}]
    try:
        recon = np.asarray(model.predict(
            {k: (v if v is not None else 0.0) for k, v in meds.items()}, ca, t), dtype=float)
        if recon.shape == t.shape and np.isfinite(recon).any():
            series.append({"label": fit_label, "color": "--accent",
                           "x": t.tolist(), "y": recon.tolist()})
    except Exception:
        pass                                            # no overlay if predict can't run

    units = getattr(model, "units", {}) or {}
    keys = show or getattr(model, "outputs", tuple(meds.keys()))
    items = {f"{k} (median)": _fmt(meds.get(k), units.get(k, "")) for k in keys}
    spec = {"title": title, "panels": [
        {"kind": "curve", "title": "tissue curve · fit — whole-brain mean",
         "xlabel": "time (s)", "ylabel": ylabel, "series": series},
        {"kind": "values", "items": items},
    ]}
    if controls:
        spec["controls"] = controls
    return spec


def patlak_plot_review(inputs, result, *, title: str = "Patlak · whole-brain mean",
                       controls=None) -> dict:
    """The Patlak plot for the mean curve — full x context, y scaled to the fitted
    points; points not used in the fit are drawn dim and separate."""
    from .patlak import fit_patlak
    ca = np.asarray(inputs.c_input, dtype=float)
    t = np.asarray(inputs.t_s, dtype=float)
    m_ct = np.asarray(mean_tissue(inputs), dtype=float)
    fit = fit_patlak(m_ct, ca, t)
    xf = np.asarray(fit.x_patlak, dtype=float)
    yf = np.asarray(fit.y_patlak, dtype=float)
    good = np.asarray(fit.good_mask, dtype=bool)
    finite = np.isfinite(xf) & np.isfinite(yf)
    used, excl = finite & good, finite & ~good
    ki, vb = fit.ki_ml_per_100g_min, fit.vb_ml_per_100g

    lines = []
    xu = xf[used]
    if xu.size:
        x0, x1 = float(np.nanmin(xu)), float(np.nanmax(xu))
        slope, icpt = ki / 6000.0, vb / 100.0
        lines = [{"label": f"fit · Kᵢ={ki:.3f}", "x": [x0, x1],
                  "y": [icpt + slope * x0, icpt + slope * x1]}]
    panel = {"kind": "scatter", "title": "Patlak plot",
             "xlabel": "∫Cₐ dt / Cₐ  (s)", "ylabel": "Cₜ / Cₐ",
             "series": [
                 {"label": "not used in fit", "role": "muted",
                  "x": xf[excl].tolist(), "y": yf[excl].tolist()},
                 {"label": "fit points", "color": "--accent",
                  "x": xf[used].tolist(), "y": yf[used].tolist()},
             ], "lines": lines}
    xa = xf[finite]
    if xa.size:
        panel["xlim"] = [float(xa.min()), float(xa.max())]
    yu = yf[used]
    if yu.size:
        ylo, yhi = float(np.nanmin(yu)), float(np.nanmax(yu))
        if lines:
            ylo = min(ylo, min(lines[0]["y"]))
            yhi = max(yhi, max(lines[0]["y"]))
        pad = 0.15 * ((yhi - ylo) or abs(yhi) or 1.0)
        panel["ylim"] = [ylo - pad, yhi + pad]

    med = median_maps(result).get("ki")
    spec = {"title": title, "panels": [
        panel,
        {"kind": "values", "items": {
            "Kᵢ (mean curve)": f"{ki:.3f} mL/100g/min",
            "vₚ": f"{vb:.2f} mL/100g",
            "map Kᵢ median": (f"{med:.3f}" if med is not None else "—")}},
    ]}
    if controls:
        spec["controls"] = controls
    return spec
