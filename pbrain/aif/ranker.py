"""Automatic input-function ranking and selection (paper supp. §AIF ranking).

Ranks candidate concentration-time curves on seven physiologically-motivated
shape-quality metrics, applies self-spike veto rules, and returns the
highest-scoring non-vetoed candidate. Roughness (plateau jitter) carries the
largest weight, then tail-CV — the strongest discriminators against the jagged
single-voxel curves that destabilise Patlak.

Candidate classes (paper): (i) pure arterial (rICA/lICA/basilar/MCA),
(ii) the same time-shifted to a reference slice, (iii) TSCC — SSS shifted to a
candidate artery and amplitude-rescaled to that artery's first-pass peak (tagged
with the parent artery so an artery's veto propagates to its TSCC).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AIFCandidate:
    name: str
    curve: np.ndarray
    artery: str = ""          # parent artery name (for TSCC veto propagation)
    is_tscc: bool = False
    meta: dict = field(default_factory=dict)


def _moving_avg3(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    if c.size < 3:
        return c.copy()
    return np.convolve(c, np.ones(3) / 3.0, mode="same")


def _curve_metrics(curve: np.ndarray, t: np.ndarray, baseline_frames: int) -> dict | None:
    """Seven shape-quality metrics. (1)-(4) on a 3-pt moving average so single
    spikes are not read as a high peak; (5)-(7) on the raw curve."""
    c = np.asarray(curve, dtype=float)
    n = c.size
    if n < 5:
        return None
    t = np.asarray(t, dtype=float)[:n]
    sm = _moving_avg3(c)
    pk = int(np.nanargmax(sm))
    peak = float(sm[pk])
    if not np.isfinite(peak) or peak <= 0:
        return None
    # (2) arrival time at 10% peak crossing
    cross = np.where(sm >= 0.10 * peak)[0]
    t_arr = float(t[cross[0]]) if cross.size else float(t[pk])
    # (3) FWHM of the first pass (frames + seconds)
    half = 0.5 * peak
    lo = pk
    while lo > 0 and sm[lo] >= half:
        lo -= 1
    hi = pk
    while hi < n - 1 and sm[hi] >= half:
        hi += 1
    fwhm_frames = float(hi - lo)
    fwhm_s = float(t[min(hi, n - 1)] - t[max(lo, 0)])
    # (4) AUC-to-peak ratio over the first pass (small => sharper bolus)
    dt = np.diff(t)
    dt = np.append(dt, dt[-1] if dt.size else 1.0)
    auc_to_peak = float(np.sum(np.clip(sm[: pk + 1], 0, None) * dt[: pk + 1]) / peak)
    # (5) tail coefficient of variation (raw, post-bolus)
    tail = c[pk + 1:] if pk + 1 < n else c[-max(1, n // 3):]
    tmean = float(np.nanmean(tail))
    tail_cv = float(np.nanstd(tail) / abs(tmean)) if tmean != 0 else np.inf
    # (6) baseline std over pre-bolus frames (raw)
    bf = max(1, int(baseline_frames))
    base_std = float(np.nanstd(c[:bf]))
    # (7) normalised total variation = (1/N) sum|Δc| / peak (raw roughness)
    roughness = float(np.nanmean(np.abs(np.diff(c))) / peak)
    return {
        "peak": peak, "t_arr": t_arr, "fwhm_frames": fwhm_frames, "fwhm_s": fwhm_s,
        "auc_to_peak": auc_to_peak, "tail_cv": tail_cv, "base_std": base_std,
        "roughness": roughness, "peak_idx": pk, "smoothed_peak": peak,
    }


def _self_spike_veto(curve: np.ndarray, m: dict) -> bool:
    """Flag single-frame spikes rather than boluses."""
    c = np.asarray(curve, dtype=float)
    pk, speak = m["peak_idx"], m["smoothed_peak"]
    if m["fwhm_frames"] < 2.5:                                  # <~6 s at 2.4 s TR
        return True
    if float(c[pk]) > 1.5 * speak:                              # raw apex >> smoothed
        return True
    fp = c[: pk + 1]
    if fp.size and float(np.nanmax(fp)) > 1.6 * speak:          # raw max in first pass
        return True
    return False


_WEIGHTS = {"peak": 0.7, "t_arr": 0.5, "fwhm_s": 0.6, "auc_to_peak": 0.7,
            "tail_cv": 1.5, "base_std": 0.7, "roughness": 2.0}
_LOW_IS_GOOD = {"peak": False, "t_arr": True, "fwhm_s": True, "auc_to_peak": True,
                "tail_cv": True, "base_std": True, "roughness": True}


def rank_candidates(cands: list[AIFCandidate], t_s: np.ndarray, *,
                    baseline_frames: int = 5) -> tuple[AIFCandidate | None, list[dict]]:
    """Return (best_candidate, ranking). ``ranking`` is a list of dicts sorted
    best-first: vetoed candidates always fall to the bottom band; within a band
    by composite score (high = good)."""
    t = np.asarray(t_s, dtype=float)
    recs: list[dict] = []
    for cand in cands:
        m = _curve_metrics(cand.curve, t, baseline_frames)
        vetoed = True if m is None else _self_spike_veto(cand.curve, m)
        recs.append({"cand": cand, "m": m, "vetoed": vetoed})

    # parent-artery veto propagation: a TSCC inherits its source artery's flag.
    artery_vetoed = {r["cand"].name: r["vetoed"] for r in recs if not r["cand"].is_tscc}
    for r in recs:
        c = r["cand"]
        if c.is_tscc and artery_vetoed.get(c.artery, False):
            r["vetoed"] = True

    valid = [r for r in recs if r["m"] is not None]
    if not valid:
        return None, [{"name": r["cand"].name, "score": float("-inf"), "vetoed": True} for r in recs]

    def _norm(key: str) -> dict[int, float]:
        vals = np.array([r["m"][key] for r in valid], dtype=float)
        finite = vals[np.isfinite(vals)]
        lo = float(np.min(finite)) if finite.size else 0.0
        hi = float(np.max(finite)) if finite.size else 1.0
        out = {}
        for r in valid:
            v = r["m"][key]
            if not np.isfinite(v) or hi <= lo:
                nv = 0.0 if (np.isfinite(v) and _LOW_IS_GOOD[key]) else 0.5
            else:
                nv = (v - lo) / (hi - lo)
                if _LOW_IS_GOOD[key]:
                    nv = 1.0 - nv
            out[id(r["cand"])] = float(nv)
        return out

    norms = {k: _norm(k) for k in _WEIGHTS}
    sw = sum(_WEIGHTS.values())
    for r in recs:
        if r["m"] is None:
            r["score"] = float("-inf")
            continue
        cid = id(r["cand"])
        r["score"] = sum(_WEIGHTS[k] * norms[k][cid] for k in _WEIGHTS) / sw

    recs.sort(key=lambda r: (r["vetoed"], -r["score"]))
    ranking = [{"name": r["cand"].name, "artery": r["cand"].artery,
                "is_tscc": r["cand"].is_tscc, "score": float(r["score"]),
                "vetoed": bool(r["vetoed"]), "metrics": r["m"]} for r in recs]
    return recs[0]["cand"], ranking


__all__ = ["AIFCandidate", "rank_candidates"]
