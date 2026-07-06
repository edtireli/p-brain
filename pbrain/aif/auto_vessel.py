"""Data-driven vessel AIF — species-agnostic, no CNN and no atlas.

Finds the input function straight from the DCE data by locating the voxels with
the sharpest, earliest, strongest bolus passage — large vessels fill first and
highest — then robustly averaging their curves. Works where the CNN (trained on
the human ICA/SSS) and fixed slice-position heuristics do not, e.g. rodent-brain
DCE.

Two selection regimes (``select_by``):

* ``"sharpness"`` (default) — the literature-grounded arterial-cluster rule
  (Mouridsen 2006 MRM 55:524; Yin 2014 PLoS ONE 9:e85884): constrain candidates
  to the brain (``brain_mask_path``) with a raw-signal intensity floor, drop the
  low-AUC / low-upslope voxels, then rank by the sharpness metric
  ``M = peak / (TTP · FWHM)`` inside a tight *absolute* early-arrival window, keep
  a small coherent cluster, peak-align the members and take a trimmed mean. This
  is what recovers a sharp, early bolus rather than a blunted late curve.
* ``"peak"`` — the original brightest-fraction-then-median behaviour, kept for
  reproducibility.

For absolute quantification the AIF must be a *blood/plasma* curve, whose baseline
T1 differs from tissue. If ``blood_t1_ms``/``tr_s``/``flip_angle_deg`` are given,
the selected voxels' *signal* is re-converted with the blood T1 (ratio-SPGR, M0-
free) and divided by ``(1 − plasma_hct)`` to yield a plasma AIF — decoupled from
the tissue-concentration T1 used elsewhere.

Selectable with ``--aif auto_vessel``; tune with ``--opt aif.auto_vessel.<key>=<v>``.
Extends, does not modify, the existing extractors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import AIFExtractor, InputFunction


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """Shift a 1-D array by k frames (k>0 → right), zero-filling (baseline≈0)."""
    out = np.zeros_like(a)
    if k == 0:
        out[:] = a
    elif k > 0:
        out[k:] = a[:-k]
    else:
        out[:k] = a[-k:]
    return out


def _ratio_spgr_conc(sig: np.ndarray, *, t1_0_ms: float, tr_s: float,
                     flip_angle_deg: float, r1: float, b0: int) -> np.ndarray:
    """M0-free ratio SPGR signal→concentration for a stack of curves (N,T),
    anchored to a uniform baseline T1 (used for the *blood* AIF re-conversion)."""
    S = np.asarray(sig, dtype=float)
    a = np.deg2rad(float(flip_angle_deg))
    cos_a = float(np.cos(a))
    R1_0 = 1000.0 / float(t1_0_ms)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        E1_0 = np.exp(-tr_s * R1_0)
        f0 = (1.0 - E1_0) / (1.0 - cos_a * E1_0)
        S0 = np.nanmedian(S[:, :b0], axis=1, keepdims=True)
        ratio = S / np.where(S0 > 0, S0, np.nan)
        g = ratio * f0
        E1 = np.clip((1.0 - g) / (1.0 - cos_a * g), 1e-6, 1.0 - 1e-9)
        R1 = -np.log(E1) / tr_s
        C = (R1 - R1_0) / float(r1)
        C = C - np.nanmean(C[:, :b0], axis=1, keepdims=True)
    return C


@dataclass(frozen=True, slots=True)
class _AutoVessel:
    key: ClassVar[str] = "auto_vessel"
    name: ClassVar[str] = "Data-driven vessel AIF (sharpest, earliest bolus)"
    description: ClassVar[str] = (
        "Species-agnostic input function: constrains candidates to the brain, "
        "ranks by a sharpness metric M=peak/(TTP·FWHM) in a tight early window, "
        "and returns the peak-aligned trimmed-mean (optionally gamma-fitted) "
        "curve of a small vessel cluster. Optional blood-T1/Hct re-conversion "
        "yields a plasma AIF. No CNN, no atlas; works on rodent brain DCE."
    )
    accepts: ClassVar[dict[str, type]] = {"dce_data": np.ndarray, "t_s": np.ndarray}
    produces: ClassVar[dict[str, type]] = {"input_function": InputFunction}

    def extract(
        self,
        dce_data: np.ndarray,
        t_s: np.ndarray,
        dce_affine: np.ndarray,
        *,
        t1_map: np.ndarray | None = None,
        m0_map: np.ndarray | None = None,
        t1w_volume: np.ndarray | None = None,
        t1w_affine: np.ndarray | None = None,
        concentration_data: np.ndarray | None = None,
        # ---- selection ----
        select_by: str = "sharpness",
        aggregate: str = "trimmed_mean",
        brain_mask_path: Path | str | None = None,
        intensity_floor_pct: float = 60.0,
        baseline_frames: int = 6,
        bat_frame: int = -1,
        early_window_s: float = 45.0,
        top_fraction: float = 0.05,
        n_min: int = 15,
        n_max: int = 250,
        trim_pct: float = 10.0,
        align_peaks: bool = True,
        min_peak_mM: float = 0.05,
        max_peak_mM: float = 1.0e9,
        clip_reject_mM: float = 19.0,
        late_spike_frac: float = 1.3,
        smooth: int = 0,
        gamma_fit: bool = False,
        # ---- optional blood-T1 / plasma re-conversion for a true plasma AIF ----
        blood_t1_ms: float = 0.0,
        tr_s: float = 0.0,
        flip_angle_deg: float = 0.0,
        r1_per_s_mM: float = 4.0,
        plasma_hct: float = 0.0,
        # ---- legacy ----
        early_window_frac: float = 0.15,
        **opts: Any,
    ) -> InputFunction:
        src = concentration_data if concentration_data is not None else dce_data
        C = np.asarray(src, dtype=float)
        if C.ndim == 3:
            C = C[..., None]
        X, Y, Z, T = C.shape
        flat = C.reshape(-1, T)
        b0 = min(max(int(baseline_frames), 1), max(T - 2, 1))
        dt = float(np.median(np.diff(np.asarray(t_s, float)))) if T > 1 else 1.0

        finite = np.isfinite(flat).all(axis=1)

        # --- candidate restriction: in-brain + raw-signal intensity floor ---
        cand_base = finite.copy()
        in_brain_frac = None
        if brain_mask_path is not None:
            try:
                import nibabel as nib
                bm = np.asarray(nib.load(str(brain_mask_path)).dataobj) > 0
                if bm.shape == (X, Y, Z):
                    cand_base &= bm.reshape(-1)
            except Exception:
                pass
        if float(intensity_floor_pct) > 0 and dce_data is not None:
            sig = np.asarray(dce_data, dtype=float)
            if sig.ndim == 4 and sig.shape[:3] == (X, Y, Z):
                base_sig = np.nanmean(sig.reshape(-1, sig.shape[-1])[:, :b0], axis=1)
                ref = base_sig[cand_base] if cand_base.any() else base_sig
                thr = np.nanpercentile(ref, float(intensity_floor_pct))
                cand_base &= base_sig > thr

        # --- bolus-arrival anchor: the injection follows the fixed baseline, so
        # arrival is known a priori (~b0). An explicit ``bat_frame`` overrides. ---
        if int(bat_frame) >= 0:
            bat = int(bat_frame)
        else:
            mc = np.nanmean(flat[cand_base], axis=0) if cand_base.any() else np.zeros(T)
            mpk = float(np.nanmax(mc)) if mc.size else 0.0
            bat = int(min(max(int(np.argmax(mc > 0.2 * mpk)) if mpk > 0 else b0, 0), b0 + 1))
        win = (max(3, int(np.ceil(float(early_window_s) / dt))) if float(early_window_s) > 0
               else max(3, int(float(early_window_frac) * (T - bat))))
        w0, w1 = bat, min(T, bat + win + 1)                       # early bolus window
        clip_thr = 0.95 * float(clip_reject_mM) if float(clip_reject_mM) > 0 else np.inf

        # --- per-voxel descriptors, LOBE-LOCAL to the early window. Global-max
        # peak/TTP/FWHM is the confirmed break: a late noise/clip spike sets them,
        # inflates the sharpness metric, and (via align_peaks) sums into a spurious
        # late AIF peak. Restricting peak/TTP/FWHM to the early window + flagging
        # clipped and late-spiking voxels fixes it. ---
        nvox = flat.shape[0]
        peak = np.full(nvox, -np.inf); ttp = np.zeros(nvox, dtype=int)
        fwhm = np.full(nvox, dt); late_exc = np.zeros(nvox); clipped = np.zeros(nvox, dtype=bool)
        idx = np.where(cand_base)[0]
        if idx.size:
            sub = flat[idx]; wsub = sub[:, w0:w1]
            pk = wsub.max(axis=1); tp = wsub.argmax(axis=1) + w0
            peak[idx] = pk; ttp[idx] = tp
            fw = np.ones(len(idx))                                # contiguous half-max lobe
            for r in range(len(idx)):
                half = 0.5 * pk[r]; lo = hi = int(tp[r])
                while lo - 1 >= 0 and sub[r, lo - 1] > half:
                    lo -= 1
                while hi + 1 < sub.shape[1] and sub[r, hi + 1] > half:
                    hi += 1
                fw[r] = hi - lo + 1
            fwhm[idx] = np.maximum(fw, 1) * dt
            if w1 < sub.shape[1]:                                 # late excursion vs window peak
                late_exc[idx] = np.nanmax(sub[:, w1:], axis=1) / np.maximum(pk, 1e-6)
            clipped[idx] = np.nanmax(np.abs(sub), axis=1) >= clip_thr

        # --- shape gate BEFORE ranking: early, non-clipped, no dominant late spike.
        # Relax progressively if too few survive so the AIF never comes back empty. ---
        base_ok = (cand_base & (peak > float(min_peak_mM)) & (peak < float(max_peak_mM))
                   & (ttp >= bat) & (ttp <= bat + win))
        gate = base_ok & (~clipped) & (late_exc <= float(late_spike_frac))
        if int(gate.sum()) < int(n_min):
            gate = base_ok & (late_exc <= float(late_spike_frac))
        if int(gate.sum()) < int(n_min):
            gate = base_ok
        if int(gate.sum()) < int(n_min):
            gate = cand_base & (peak > float(min_peak_mM))

        # --- rank survivors by early-window sharpness M = peak / (rise * FWHM) ---
        if str(select_by) == "sharpness":
            with np.errstate(divide="ignore", invalid="ignore"):
                score = peak / (np.maximum((ttp - bat + 1) * dt, dt) * np.maximum(fwhm, dt))
        else:
            score = peak.copy()
        score = np.where(gate, score, -np.inf)
        n_take = int(np.clip(float(top_fraction) * max(int(gate.sum()), 1),
                             int(n_min), int(n_max)))
        sel_idx = np.argsort(score)[::-1][:n_take]
        sel = np.zeros(nvox, dtype=bool)
        sel[sel_idx] = True

        if brain_mask_path is not None:
            try:
                in_brain_frac = float((sel & cand_base).sum() / max(int(sel.sum()), 1))
            except Exception:
                in_brain_frac = None

        # --- member curves: optional blood-T1 re-conversion for a plasma AIF ---
        use_blood = (float(blood_t1_ms) > 0 and float(tr_s) > 0
                     and float(flip_angle_deg) > 0 and dce_data is not None)
        if use_blood:
            sig = np.asarray(dce_data, dtype=float).reshape(-1, T)
            members = _ratio_spgr_conc(sig[sel_idx], t1_0_ms=float(blood_t1_ms),
                                       tr_s=float(tr_s), flip_angle_deg=float(flip_angle_deg),
                                       r1=float(r1_per_s_mM), b0=b0)
        else:
            members = flat[sel_idx]

        # --- peak-align then robust aggregate ---
        if align_peaks and members.shape[0] > 1:
            tgt = int(np.median(ttp[sel_idx]))
            members = np.vstack([_shift(m, tgt - int(tp))
                                 for m, tp in zip(members, ttp[sel_idx])])
        if str(aggregate) == "median":
            c_a = np.nanmedian(members, axis=0)
        elif str(aggregate) == "mean":
            c_a = np.nanmean(members, axis=0)
        else:  # trimmed_mean
            lo, hi = np.nanpercentile(members, [float(trim_pct), 100 - float(trim_pct)], axis=0)
            keep = (members >= lo) & (members <= hi)
            c_a = np.nansum(np.where(keep, members, 0.0), axis=0) / np.maximum(keep.sum(0), 1)

        if float(plasma_hct) > 0:
            c_a = c_a / (1.0 - float(plasma_hct))

        # --- optional gamma-variate fit (clean, sharp, monotone rise) ---
        gamma_ok = False
        if gamma_fit:
            try:
                from scipy.optimize import curve_fit

                def _g(t, A, t0, al, be):
                    x = np.clip(t - t0, 1e-6, None)
                    return A * (x ** al) * np.exp(-x / be)

                t_arr = np.asarray(t_s, float)
                p0 = [float(np.nanmax(c_a)), bat * dt, 3.0, 25.0]
                popt, _ = curve_fit(_g, t_arr, np.nan_to_num(c_a), p0=p0, maxfev=40000)
                c_a = _g(t_arr, *popt)
                gamma_ok = True
            except Exception:
                pass

        if int(smooth) > 0 and not gamma_ok:  # smooth the tail only, never the peak
            k = 2 * int(smooth) + 1
            tp = int(np.nanargmax(c_a))
            tail = np.convolve(c_a, np.ones(k) / k, mode="same")
            c_a = np.concatenate([c_a[:tp + 1], tail[tp + 1:]])

        return InputFunction(
            c_a=np.asarray(c_a, dtype=float),
            t_s=np.asarray(t_s, dtype=float),
            mask=sel.reshape(X, Y, Z),
            source="auto_vessel",
            meta={
                "n_voxels": int(sel.sum()),
                "select_by": str(select_by),
                "aggregate": str(aggregate),
                "bat_frame": int(bat),
                "ttp_window_frames": [int(bat), int(bat + win)],
                "peak_mM": float(np.nanmax(c_a)) if c_a.size else 0.0,
                "ttp_s": float(np.asarray(t_s, float)[int(np.nanargmax(c_a))]) if c_a.size else 0.0,
                "in_brain_fraction": in_brain_frac,
                "blood_t1_ms": float(blood_t1_ms) if use_blood else None,
                "plasma_hct": float(plasma_hct) if float(plasma_hct) > 0 else None,
                "gamma_fitted": gamma_ok,
                "used_concentration": concentration_data is not None,
            },
        )


PLUGIN = _AutoVessel()
