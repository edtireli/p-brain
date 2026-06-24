"""Signal→concentration QC figure — rendered by default on every run.

Not model-specific (so not a ``Diagnostic`` plug-in): it summarises the
*conversion* itself so a researcher can eyeball whether the concentration
volume is sane before trusting any kinetic fit. One PNG per run, next to the
concentration file, with six panels on a representative slice:

1. DCE signal at the bolus-peak frame (the input);
2. C(t) at peak, windowed 0–2 mM (the tissue range);
3. C(t) at peak, autoscaled — exposes out-of-brain voxels that hit the
   saturation-recovery ceiling (the usual "looks bad in a viewer" culprit);
4. a sample of individual tissue-voxel C(t) curves (shape / noise check);
5. the mean tissue C(t) curve;
6. the AIF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def render_conversion_qc(conc: np.ndarray, dce: np.ndarray | None, aif: np.ndarray,
                         t_s: np.ndarray, brain_mask: np.ndarray, out_path: Path | str, *,
                         t1_map: np.ndarray | None = None,
                         baseline_frames: int = 5, title: str = "") -> Path:
    """Write the conversion-QC PNG. Returns the path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    C = np.asarray(conc, dtype=np.float32)
    if C.ndim != 4:
        raise ValueError(f"conc must be 4-D (X,Y,Z,T); got {C.shape}")
    bm = np.asarray(brain_mask, dtype=bool)
    T = C.shape[-1]
    t_s = np.asarray(t_s, dtype=float)
    if t_s.size != T:
        t_s = np.arange(T, dtype=float)

    # bolus-peak frame from the mean in-brain curve; representative slice = most brain
    brain_curves = C[bm]
    mean_curve = (np.nanmean(brain_curves, axis=0) if brain_curves.size else np.zeros(T))
    pf = int(np.nanargmax(mean_curve)) if np.isfinite(mean_curve).any() else 0
    z = (int(np.argmax([int(bm[:, :, k].sum()) for k in range(C.shape[2])]))
         if bm.any() else C.shape[2] // 2)
    peak = np.nanmax(C, axis=-1)
    sl_max = float(np.nanmax(C[:, :, z, pf])) if np.isfinite(C[:, :, z, pf]).any() else 0.0

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))

    # 1. DCE signal
    if dce is not None:
        D = np.asarray(dce, dtype=np.float32)
        df = min(pf, D.shape[-1] - 1)
        ax[0, 0].imshow(D[:, :, z, df].T, cmap="gray", origin="lower")
    ax[0, 0].set_title(f"DCE signal  z={z}  frame={pf}")
    ax[0, 0].axis("off")

    # 2. conc @peak, tissue window
    im = ax[0, 1].imshow(C[:, :, z, pf].T, cmap="magma", origin="lower", vmin=0.0, vmax=2.0)
    ax[0, 1].set_title("conc @peak  (clip 0–2 mM)")
    ax[0, 1].axis("off")
    fig.colorbar(im, ax=ax[0, 1], fraction=0.046, pad=0.04)

    # 3. conc @peak, autoscale (exposes saturated voxels)
    im = ax[0, 2].imshow(C[:, :, z, pf].T, cmap="magma", origin="lower")
    ax[0, 2].set_title(f"conc @peak  (autoscale, max={sl_max:.0f} mM)")
    ax[0, 2].axis("off")
    fig.colorbar(im, ax=ax[0, 2], fraction=0.046, pad=0.04)

    # 4. sample tissue-voxel curves
    cand = np.argwhere(bm & np.isfinite(peak) & (peak > 0.1) & (peak < 3.0))
    if len(cand):
        rng = np.random.default_rng(0)
        sel = cand[rng.choice(len(cand), min(12, len(cand)), replace=False)]
        for vx, vy, vz in sel:
            ax[1, 0].plot(t_s, C[vx, vy, vz], lw=0.8, alpha=0.6)
    ax[1, 0].set_title("12 random tissue-voxel curves")
    ax[1, 0].set_xlabel("time (s)")
    ax[1, 0].set_ylabel("C(t)  (mM)")

    # 5. mean tissue curve
    ax[1, 1].plot(t_s, mean_curve, color="#185FA5", lw=1.5)
    ax[1, 1].set_title(f"mean tissue $C_t$  (peak {np.nanmax(mean_curve):.3f} mM)")
    ax[1, 1].set_xlabel("time (s)")
    ax[1, 1].set_ylabel("C(t)  (mM)")

    # 6. AIF
    if aif is not None and np.size(aif):
        a = np.asarray(aif, dtype=float)
        ax[1, 2].plot(t_s[: a.size], a[:T], color="#BA7517", lw=1.3)
        ax[1, 2].set_title(f"AIF  (peak {np.nanmax(a):.2f} mM)")
    ax[1, 2].set_xlabel("time (s)")
    ax[1, 2].set_ylabel("C(t)  (mM)")

    base_med = (float(np.nanmedian(np.nanmean(C[..., : max(1, baseline_frames)], axis=-1)[bm]))
                if bm.any() else float("nan"))
    sat = int(np.nansum(peak > 20.0))
    t1_txt = ""
    if t1_map is not None:
        v = np.asarray(t1_map, float)
        v = v[(v > 0) & np.isfinite(v)]
        if v.size:
            t1_txt = f"   ·   T1 median {np.median(v):.0f} ms"
    sub = (f"baseline offset {base_med:+.3f} mM   ·   "
           f"{sat} saturated voxels >20 mM (out-of-brain vessels){t1_txt}")
    fig.suptitle((title or "signal → concentration QC") + "\n" + sub, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(out_path), dpi=80)
    plt.close(fig)
    return out_path
