"""CNN AIF: SSS time-shifted to rICA — paper §4.5.1 default.

> *"For kinetic modelling, p-Brain constructs an SSS-derived effective
>  input function by time-shifting the venous curve to the selected ICA
>  reference via cross-correlation, with optional amplitude rescaling;
>  the ICA-derived arterial input function can also be used directly."*
>  — Tireli et al., §4.5.1

This plug-in composes the existing `cnn` extractor:

1. Run the rICA CNN → rICA signal curve.
2. Run the SSS CNN → SSS signal curve.
3. Detect top-``num_peaks`` peaks in each (default 2 for two-bolus
   protocol), separated by ``peak_separation`` frames. Compute the
   per-peak frame offsets ``vein − artery`` and take the median →
   integer shift in samples.
4. Shift the SSS curve by ``−shift`` (vein arrives later, so we
   move it backward in time) with zero padding.
5. Optionally rescale the shifted SSS to match the rICA peak amplitude
   (median of peak-amplitude ratios). The legacy "never scale down"
   policy is preserved: if the SSS peak is already higher than rICA,
   keep the larger SSS (better SNR from a bigger vessel).

The returned :class:`InputFunction` carries the **SSS-shifted-to-rICA**
curve as ``c_a`` and the SSS spatial mask. ``meta`` records the shift,
rescale factor, and which peaks drove alignment.

To use rICA-direct instead (without the SSS shift), pass ``--aif cnn``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np

from .base import AIFExtractor, InputFunction
from .cnn import PLUGIN as _CNN


# ────────────────────────────────────────────────────────────────────────
# Curve-level alignment (ported from modules/time_shifting.py:454)
# ────────────────────────────────────────────────────────────────────────


def _baseline_to_zero(curve: np.ndarray, baseline_frames: int) -> np.ndarray:
    if curve.size == 0:
        return curve
    b = min(int(curve.size), int(baseline_frames))
    base = float(np.nanmean(curve[:b])) if b > 0 else 0.0
    if not np.isfinite(base):
        base = 0.0
    out = np.asarray(curve, dtype=np.float32).copy() - base
    if b > 0:
        out[:b] = 0.0
    return out


def _top_peaks_by_height(curve: np.ndarray, n_peaks: int, separation: int) -> list[int]:
    from scipy.signal import find_peaks
    if curve.size == 0:
        return []
    peaks, _ = find_peaks(curve)
    if peaks.size == 0:
        return []
    candidates = [int(p) for p in peaks if np.isfinite(curve[int(p)]) and float(curve[int(p)]) > 0]
    if not candidates:
        return []
    sorted_peaks = sorted(candidates, key=lambda x: float(curve[x]), reverse=True)
    chosen: list[int] = []
    for p in sorted_peaks:
        if all(abs(p - cp) > int(separation) for cp in chosen):
            chosen.append(p)
            if len(chosen) >= int(n_peaks):
                break
    return chosen


def _shift_with_zeros(curve: np.ndarray, shift_samples: int, target_len: int) -> np.ndarray:
    """Integer-sample shift with zero-padding (legacy semantics)."""
    if target_len <= 0:
        return np.asarray([], dtype=np.float32)
    curve = np.asarray(curve, dtype=np.float32).reshape(-1)
    shift = int(shift_samples)
    if curve.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if shift > 0:
        if shift >= curve.size:
            shifted = np.zeros(target_len, dtype=np.float32)
        else:
            shifted = np.concatenate([curve[shift:], np.zeros(shift, dtype=np.float32)])
    elif shift < 0:
        s = -shift
        if s >= curve.size:
            shifted = np.zeros(target_len, dtype=np.float32)
        else:
            shifted = np.concatenate([np.zeros(s, dtype=np.float32), curve[: curve.size - s]])
    else:
        shifted = curve.copy()
    if shifted.size < target_len:
        shifted = np.pad(shifted, (0, target_len - shifted.size), mode="constant")
    elif shifted.size > target_len:
        shifted = shifted[:target_len]
    return shifted.astype(np.float32)


def _align_first_peaks(
    vein_curve: np.ndarray,
    artery_curve: np.ndarray,
    *,
    num_peaks: int = 2,
    peak_separation: int = 10,
    baseline_frames: int = 5,
    rescale: bool = True,
    rescale_method: Literal["peak", "auc", "none"] = "peak",
) -> tuple[np.ndarray, int, float]:
    """Align vein peaks to artery peaks; returns (aligned_vein, shift, rescale).

    Ported from ``modules.time_shifting.align_first_peaks``. The shift is
    determined by the median of per-peak frame offsets (vein − artery).
    Rescale uses the median of per-peak amplitude ratios (artery/vein),
    but is clamped to >= 1.0 (never scale the venous curve *down*).
    """
    vein = np.asarray(vein_curve, dtype=np.float32).reshape(-1)
    artery = np.asarray(artery_curve, dtype=np.float32).reshape(-1)
    if vein.size == 0 or artery.size == 0:
        return vein.copy(), 0, 1.0

    # Baseline-match the venous curve to the arterial baseline mean
    base_a = float(np.nanmean(artery[: min(baseline_frames, artery.size)])) if artery.size else 0.0
    base_v = float(np.nanmean(vein[: min(baseline_frames, vein.size)])) if vein.size else 0.0
    if not np.isfinite(base_a):
        base_a = 0.0
    if not np.isfinite(base_v):
        base_v = 0.0
    vein_matched = vein - float(base_v - base_a)

    artery_zero = _baseline_to_zero(artery, baseline_frames)
    vein_zero = _baseline_to_zero(vein_matched, baseline_frames)

    vein_peaks = _top_peaks_by_height(vein_zero, num_peaks, peak_separation)
    artery_peaks = _top_peaks_by_height(artery_zero, num_peaks, peak_separation)

    k = min(len(vein_peaks), len(artery_peaks), int(num_peaks))
    if k > 0:
        diffs = [int(vein_peaks[i]) - int(artery_peaks[i]) for i in range(k)]
        shift = int(np.round(float(np.median(diffs))))
    else:
        # Broad cross-correlation fallback (rare)
        from scipy.signal import correlate
        cc = correlate(vein_zero, artery_zero)
        shift = int(np.argmax(cc) - len(vein_zero) + 1)

    target_len = int(artery_zero.size)
    aligned_vein = _shift_with_zeros(vein_zero, shift, target_len)

    rescale_factor = 1.0
    method = (rescale_method or "peak").strip().lower()
    if rescale and method == "peak" and k > 0:
        ratios: list[float] = []
        for i in range(k):
            v_p = int(vein_peaks[i]) - shift
            a_p = int(artery_peaks[i])
            if not (0 <= v_p < aligned_vein.size and 0 <= a_p < artery_zero.size):
                continue
            v = float(aligned_vein[v_p])
            a = float(artery_zero[a_p])
            if not (np.isfinite(v) and np.isfinite(a) and abs(v) > 1e-8):
                continue
            ratios.append(a / v)
        if ratios:
            factor = float(np.median(ratios))
            # Legacy policy: never scale VIF down.
            if np.isfinite(factor) and factor >= 1.0:
                aligned_vein = aligned_vein * factor
                rescale_factor = factor
    elif rescale and method == "auc":
        v_auc = float(np.trapezoid(np.clip(aligned_vein, 0, None))) if aligned_vein.size else 0.0
        a_auc = float(np.trapezoid(np.clip(artery_zero, 0, None))) if artery_zero.size else 0.0
        if np.isfinite(v_auc) and np.isfinite(a_auc) and v_auc > 1e-8 and a_auc > 1e-8:
            factor = float(a_auc / v_auc)
            if np.isfinite(factor) and factor >= 1.0:
                aligned_vein = aligned_vein * factor
                rescale_factor = factor

    return aligned_vein, int(shift), float(rescale_factor)


# ────────────────────────────────────────────────────────────────────────
# Plug-in
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _CnnSssShiftedAIF:
    key: ClassVar[str] = "cnn_sss_shifted"
    name: ClassVar[str] = "CNN SSS time-shifted to rICA (paper §4.5.1 default)"
    description: ClassVar[str] = (
        "Runs the CNN twice (rICA + SSS), time-shifts the SSS curve to "
        "align with the rICA reference via per-peak median offset, and "
        "optionally rescales SSS amplitude up to the rICA peak. The "
        "shifted SSS is the AIF used by downstream Patlak — matches "
        "paper §4.5.1 default. Use --aif cnn for rICA-direct."
    )
    accepts: ClassVar[dict[str, type]] = {
        "dce_data": np.ndarray, "t_s": np.ndarray,
    }
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
        num_peaks: int = 2,
        peak_separation: int = 10,
        baseline_frames: int = 5,
        rescale: bool = True,
        rescale_method: str = "peak",
        rot90_k: int = -1,
        rica_opts: dict | None = None,
        sss_opts: dict | None = None,
        **_: Any,
    ) -> InputFunction:
        rica_opts = dict(rica_opts or {})
        sss_opts = dict(sss_opts or {})
        rica_opts.setdefault("rot90_k", rot90_k)
        sss_opts.setdefault("rot90_k", rot90_k)

        # Both passes see the signal (for CNN) AND the mM volume
        # (for curve extraction). The returned curves are therefore
        # in mM with per-voxel T1/M0 already applied.
        ica = _CNN.extract(dce_data, t_s, dce_affine,
                            t1_map=t1_map, m0_map=m0_map,
                            concentration_data=concentration_data,
                            source="rica", **rica_opts)
        sss = _CNN.extract(dce_data, t_s, dce_affine,
                            t1_map=t1_map, m0_map=m0_map,
                            concentration_data=concentration_data,
                            source="sss", **sss_opts)

        aligned, shift, factor = _align_first_peaks(
            sss.c_a, ica.c_a,
            num_peaks=num_peaks, peak_separation=peak_separation,
            baseline_frames=baseline_frames,
            rescale=rescale, rescale_method=rescale_method,
        )

        return InputFunction(
            c_a=aligned,
            t_s=np.asarray(t_s, dtype=float),
            # SSS spatial mask — this is the *actual* anatomical ROI that
            # produced the curve. The pipeline operates on the per-voxel
            # mM concentration volume upstream of this stage, so each SSS
            # voxel kept its own T1/M0 through the conversion. No further
            # T1/M0 averaging is needed.
            mask=sss.mask,
            source="sss_shifted_to_rica",
            meta={
                "algorithm": "cnn_sss_shifted_to_rica",
                "shift_frames": int(shift),
                "rescale_factor": float(factor),
                "rica_meta": ica.meta,
                "sss_meta": sss.meta,
                "num_peaks": int(num_peaks),
                "rica_mask_voxels": int(ica.mask.sum()),
                "sss_mask_voxels": int(sss.mask.sum()),
            },
        )


PLUGIN = _CnnSssShiftedAIF()
