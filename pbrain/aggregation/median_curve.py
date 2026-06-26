"""Median-curve aggregator — the article's ROI-curve kinetic method.

Where :mod:`pbrain.aggregation.region` fits every voxel and then reduces
the parameter *map*, this aggregator pools the *concentration curves*
first and fits the kinetic model **once** per region:

1. within each slice, take the per-voxel median concentration curve over
   the region's voxels, detect its bolus-arrival baseline point and shift
   it to a zero baseline (``_median_then_shift``);
2. take the median of the per-slice curves → one pooled curve per region;
3. fit the kinetic model to that single curve against the AIF.

This reproduces the published ``make_patlak_plots`` pipeline exactly, so
it is the aggregator to use when bit-matching legacy / article results.

It needs three things the voxelwise aggregators do not — the **raw**
tissue concentration 4-D, the AIF curve and the time axis — which the
summary stage threads in via ``opts``:

    concentration_4d : (X, Y, Z, T)  raw mM volume (pre-normalisation)
    c_a              : (T,)          AIF curve (used verbatim)
    t_s              : (T,)          time axis (s)
    model            : KineticModel  the model to fit (e.g. Patlak)
    model_opts       : dict          options forwarded to ``model.fit``

The baseline-detection / shifter functions below are a verbatim port of
``article_bbb_nocth/code/make_patlak_plots.py`` so the pooled curves are
identical to the article's.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import Aggregator, AggregationResult


# ── Baseline / shifter functions (verbatim port of make_patlak_plots.py) ──

def _butter_lowpass_filter(data, cutoff=4.0, fs=15, order=3):
    from scipy.signal import butter, filtfilt
    b, a = butter(order, cutoff / (0.5 * fs), btype="low", analog=False)
    return filtfilt(b, a, data)


def _find_major_peaks(gradient, radius=10, num_peaks=2):
    from scipy.signal import argrelextrema
    peak_indices = argrelextrema(gradient, np.greater)[0]
    if len(peak_indices) == 0:
        return []
    peak_values = gradient[peak_indices]
    sorted_peak_indices = [x for _, x in sorted(zip(peak_values, peak_indices), reverse=True)]
    major_peaks: list[int] = []
    for peak in sorted_peak_indices:
        if all(abs(peak - mp) >= radius for mp in major_peaks):
            major_peaks.append(peak)
            if len(major_peaks) >= num_peaks:
                break
    return major_peaks


def _find_baseline_intersection(y_data, fs=15, cutoff=4.0, order=3, radius=10):
    try:
        y_filtered = _butter_lowpass_filter(y_data, cutoff, fs, order)
    except Exception:
        return 10
    bl_mean = np.mean(y_filtered[3:9])
    bl_std = np.std(y_filtered[3:9])
    gradient = np.gradient(y_filtered)
    major_peaks = _find_major_peaks(gradient, radius)
    grad_peak = min(major_peaks) if major_peaks else len(y_data) - 1
    bolus_peak = np.max(y_filtered[grad_peak:min(grad_peak + 10, len(y_filtered))])
    signal_range = abs(bolus_peak - bl_mean)
    threshold = max(bl_std * 4, signal_range * 0.05, 0.0005)
    bp = 8
    for f in range(8, min(grad_peak + 2, len(y_filtered) - 2)):
        d0 = abs(y_filtered[f] - bl_mean)
        d1 = abs(y_filtered[f + 1] - bl_mean)
        d2 = abs(y_filtered[f + 2] - bl_mean)
        if d0 > threshold and d1 > threshold and d2 > threshold:
            bp = max(f - 1, 3)
            break
    return bp


def _custom_shifter(array, baseline_point):
    array = np.asarray(array, dtype=float)
    if array.size == 0:
        return array
    baseline_point = max(0, min(int(baseline_point or 10), array.size - 1))
    after = array[baseline_point:]
    shifted = after - np.min(after)
    return np.concatenate([np.zeros(baseline_point), shifted])


def _median_then_shift(ctc_slice_3d, mask_2d, max_voxels=2000):
    coords = np.argwhere(mask_2d)
    if len(coords) < 3:
        return None
    if len(coords) > max_voxels:
        rng = np.random.default_rng(42)
        coords = coords[rng.choice(len(coords), max_voxels, replace=False)]
    raw_curves = []
    for x, y in coords:
        c = ctc_slice_3d[x, y, :]
        if not np.isnan(c).any() and not np.all(c == 0):
            raw_curves.append(c)
    if not raw_curves:
        return None
    median_raw = np.nanmedian(np.stack(raw_curves), axis=0)
    bp = _find_baseline_intersection(median_raw)
    return _custom_shifter(median_raw, bp)


def _pool_region_curve(conc4d, mask3d, c_a, t_s, max_voxels):
    """Per-slice median_then_shift → median across slices → one curve."""
    ml = int(min(c_a.size, t_s.size))
    stacked = []
    for sl in range(conc4d.shape[2]):
        md = _median_then_shift(conc4d[:, :, sl, :], mask3d[:, :, sl], max_voxels)
        if md is None:
            continue
        p = np.full(ml, np.nan)
        k = int(min(md.size, ml))
        p[:k] = md[:k]
        stacked.append(p)
    if not stacked:
        return None, 0
    return np.nanmedian(np.stack(stacked), axis=0), len(stacked)


@dataclass(frozen=True, slots=True)
class _MedianCurveAggregator:
    key: ClassVar[str] = "median_curve"
    name: ClassVar[str] = "Median-curve ROI aggregator"
    description: ClassVar[str] = (
        "Pools concentration curves per region (per-slice voxel median → "
        "baseline shift → median across slices) and fits the kinetic model "
        "once per region. Reproduces the article's make_patlak_plots method "
        "for bit-parity. Needs the raw concentration 4-D, AIF curve and "
        "time axis threaded in via opts."
    )
    accepts: ClassVar[dict[str, type]] = {
        "concentration_4d": np.ndarray, "c_a": np.ndarray, "t_s": np.ndarray,
        "parcellation": np.ndarray, "region_map": dict,
    }
    produces: ClassVar[dict[str, type]] = {"json_files": dict}

    def aggregate(
        self,
        maps: dict[str, np.ndarray],
        units: dict[str, str],
        *,
        reference_affine: np.ndarray | None = None,
        parcellation: np.ndarray | None,
        labels: dict[int, str] | None = None,
        region_map: dict[str, list[int]] | None,
        slice_axis: int = 2,
        out_dir: Path = Path("."),
        **opts: Any,
    ) -> AggregationResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        conc4d = opts.get("concentration_4d")
        c_a = opts.get("c_a")
        t_s = opts.get("t_s")
        model = opts.get("model")
        model_opts = dict(opts.get("model_opts") or {})
        max_voxels = int(opts.get("max_voxels", 2000))
        if conc4d is None or c_a is None or t_s is None or model is None:
            raise RuntimeError(
                "median_curve aggregator requires 'concentration_4d', 'c_a', "
                "'t_s' and 'model' via opts (threaded by the summary stage). "
                "Got: " + ", ".join(
                    k for k, v in
                    [("concentration_4d", conc4d), ("c_a", c_a),
                     ("t_s", t_s), ("model", model)] if v is None
                ) + " missing."
            )

        from pbrain.models import CurveInputs

        conc4d = np.moveaxis(np.asarray(conc4d, dtype=float), slice_axis, 2)
        c_a = np.asarray(c_a, dtype=float).ravel()
        t_s = np.asarray(t_s, dtype=float).ravel()
        out_names = tuple(getattr(model, "outputs", ("ki",)))

        # region → 3-D mask
        regions: dict[str, np.ndarray] = {}
        if parcellation is not None and region_map:
            pc = np.moveaxis(np.asarray(parcellation), slice_axis, 2)
            if pc.shape != conc4d.shape[:3]:
                raise ValueError(
                    f"parcellation shape {pc.shape} != concentration "
                    f"spatial shape {conc4d.shape[:3]}"
                )
            for region, label_ids in region_map.items():
                regions[region] = np.isin(pc, list(label_ids))
        else:
            regions["whole_brain"] = np.ones(conc4d.shape[:3], dtype=bool)

        per_region: dict[str, dict[str, Any]] = {}
        for region, mask3d in regions.items():
            c_t, n_slices = _pool_region_curve(conc4d, mask3d, c_a, t_s, max_voxels)
            if c_t is None:
                per_region[region] = {nm: float("nan") for nm in out_names}
                per_region[region]["n_slices"] = 0
                continue
            n = int(min(c_t.size, c_a.size, t_s.size))
            result = model.fit(
                CurveInputs(c_tissue=c_t[:n], c_input=c_a[:n], t_s=t_s[:n]),
                **model_opts,
            )
            vals: dict[str, Any] = {}
            for nm in out_names:
                if nm in result.maps:
                    vals[nm] = float(np.asarray(result.maps[nm]).ravel()[0])
            vals["n_slices"] = int(n_slices)
            per_region[region] = vals

        # One JSON + CSV per output name, region rows (mirrors region.py layout).
        files: dict[str, Path] = {}
        summary: dict[str, Any] = {}
        for nm in out_names:
            payload = {
                region: {"value": rec.get(nm, float("nan")),
                         "n_slices": rec.get("n_slices", 0)}
                for region, rec in per_region.items()
            }
            payload["__unit__"] = {"unit": units.get(nm, getattr(model, "units", {}).get(nm, ""))}
            jpath = out_dir / f"{nm}.json"
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            files[f"{nm}_json"] = jpath
            summary[nm] = {r: v["value"] for r, v in payload.items() if r != "__unit__"}

            cpath = out_dir / f"{nm}.csv"
            with open(cpath, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["region", "value", "n_slices"])
                w.writeheader()
                for region, rec in per_region.items():
                    w.writerow({"region": region,
                                "value": rec.get(nm, float("nan")),
                                "n_slices": rec.get("n_slices", 0)})
            files[f"{nm}_csv"] = cpath

        return AggregationResult(files=files, summary=summary)


PLUGIN = _MedianCurveAggregator()
