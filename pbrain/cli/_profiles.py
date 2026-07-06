"""Bundled acquisition profiles for ``pbrain run --profile <name>``.

A profile is a preset that pre-selects the plug-ins and acquisition parameters
for a whole class of data, so a user need not re-type a long ``--opt`` pile. Each
value has the SAME shape :func:`pbrain.cli._config_file.load_config_file` returns:
a flat dict of argparse *dests* plus an ``opt`` list of fully-qualified
``<plug-point>.<plugin>.<key>=<value>`` strings.

The overlay is applied only where the CLI is silent and *before* ``--config``, so
precedence is: explicit CLI flag > ``--config`` > ``--profile`` > argparse default.
A human run (no ``--profile``) never enters the overlay, so the paper defaults are
untouched.

Per-subject *paths* are deliberately excluded (they differ per subject) — supply
``aif.auto_vessel.brain_mask_path``, ``tissue_roi.mouse_atlas.labelmap_path`` and
``tissue_roi.mouse_atlas.lut_path`` via ``--opt`` or ``--config``.
"""

from __future__ import annotations

from typing import Any

# ``mouse`` — Bruker 7T rodent DCE (2D FLASH, VFA T1, separate-acquisition T1).
# Encodes the validated internal recipe: M0-free ratio SPGR conversion, the
# in-brain sharpness AIF with a blood-T1/Hct plasma re-conversion, an atlas
# parcellation, and the Patlak + Tikhonov (up-sampled deconvolution) models.
PROFILES: dict[str, dict[str, Any]] = {
    "mouse": {
        "t1m0": "vfa_spgr",
        "signal_to_conc": "spgr_ratio",
        "aif": "auto_vessel",
        "tissue_roi": "mouse_atlas",
        "normaliser": "identity",
        "models": "patlak,tikhonov",
        "aggregations": "voxelwise,region",
        "r1_per_s_mM": 2.8,  # Gd-DOTA (gadoterate) at 7T; Rohrer 2005
        "flip_angle_deg": 15.0,
        "tr_s": 0.07,
        "dt_s": 8.96,
        "baseline_frames": 6,
        "opt": [
            # T1: a UNIFORM PINNED prior for the [Gd] conversion, NOT the per-voxel
            # VFA map. VFA T1 here is B1-degenerate (no B1 map), so the per-voxel map
            # is B1/slice-profile noise that inflates + destabilises Ki (~10x too
            # high); a uniform T1 scales tissue and AIF the SAME way and cancels in
            # the Patlak ratio. 2000 ms = 7T mouse GM (van de Ven 2007). The vfa_spgr
            # fit (b1_scale~0.65 -> ~1750 ms) is kept only as a QC overlay.
            "t1_m0.vfa_spgr.b1_scale=0.65",
            "t1_m0.vfa_spgr.tr_s=0.07",
            "signal_to_conc.spgr_ratio.t1_0_ms=2000",
            "signal_to_conc.spgr_ratio.flip_angle_deg=15",
            "signal_to_conc.spgr_ratio.r1_per_s_mM=2.8",
            "signal_to_conc.spgr_ratio.baseline_method=fixed",
            "signal_to_conc.spgr_ratio.baseline_frames=6",
            "signal_to_conc.spgr_ratio.baseline_skip=2",
            "signal_to_conc.spgr_ratio.max_conc_mM=50",
            # AIF: peak-ranked (strongest = least partial-volumed) MEAN of a small
            # BRAIN-CORE cluster. Pass an eroded brain_mask_path per subject to drop
            # the pial rim. blood_t1_ms=0 -> use the pinned-T1 conc directly.
            "aif.auto_vessel.select_by=peak",
            "aif.auto_vessel.aggregate=mean",
            "aif.auto_vessel.top_fraction=0.02",
            "aif.auto_vessel.n_min=15",
            "aif.auto_vessel.n_max=20",
            "aif.auto_vessel.max_peak_mM=45",
            "aif.auto_vessel.clip_reject_mM=0",
            "aif.auto_vessel.align_peaks=true",
            "aif.auto_vessel.bat_frame=7",
            "aif.auto_vessel.early_window_s=45",
            "aif.auto_vessel.blood_t1_ms=0",
            "aif.auto_vessel.tr_s=0.07",
            "aif.auto_vessel.flip_angle_deg=15",
            "aif.auto_vessel.r1_per_s_mM=2.8",
            "aif.auto_vessel.plasma_hct=0",
            # PATLAK. tail_mode=legacy routes to the per-voxel fit_patlak loop
            # (the fast legacy_vectorised path ignores the single-curve knobs).
            # Ki is the slope of the LINEAR TAIL of the Patlak plot: the fit keeps
            # the last two thirds of the x = integral(Ca)/Ca range
            # (window_start_fraction=1/3, the canonical p-Brain window) and drops
            # the early vascular/mixing phase. The 5%-of-peak AIF floor
            # (aif_floor_double_bolus) removes near-zero late-AIF samples, which
            # otherwise land at large x inside the tail window and flip the slope
            # negative (seen on the sharp, low-AUC awake AIF: Ki<0 everywhere
            # without the floor). This pairing = the p-Brain tail plus its
            # documented Ca-floor robustification; it supersedes single_bolus,
            # whose post-peak TIME window did not take the x-tail. NO detrending
            # (model default detrend=none): a tail detrend also de-trends the AIF
            # and drives the denominator through zero.
            "models.patlak.tail_mode=legacy",
            "models.patlak.window_start_fraction=0.33333333",
            "models.patlak.aif_floor_double_bolus=true",
            # TIKHONOV (perfusion: CBF/CBV/MTT/CTH via model-free deconvolution).
            # The mouse DCE at dt=8.96 s samples the residue peak far too coarsely
            # for the default 5*dt knot spacing (~45 s): a closed-loop recovery test
            # (known CBF=200) showed native-grid loses ~70% of CBF (recovers 0.18-
            # 0.41x) and over-estimates MTT ~2-4x. h_factor=10 up-samples the AIF+
            # tissue to ~0.9 s (knots ~4.5 s) and recovers CBF to 0.98-1.04x across
            # MTT 2-6 s; with the central-volume MTT (= CBV/CBF, CBV being grid-
            # invariant) MTT recovers to 1.01-1.10x too. GCV lambda selection resists
            # L-curve endpoint bias on low-SNR rodent curves; hematocrit 0.45 = mouse;
            # the vessel AIF is whole-blood (plasma_derived_aif stays False).
            "models.patlak.single_bolus=false",
            "models.tikhonov.h_factor=10",
            "models.tikhonov.mtt_cth_method=central_volume",
            "models.tikhonov.hematocrit=0.45",
            "models.tikhonov.lambda_selection=gcv",
        ],
    },
}


def resolve_profile(name: str) -> dict[str, Any]:
    """Return a fresh copy of the named profile's overlay dict.

    Raises ``SystemExit`` (listing the available profiles) on an unknown name.
    """
    try:
        prof = PROFILES[name]
    except KeyError:
        raise SystemExit(
            f"--profile {name!r}: unknown. Available: {sorted(PROFILES)}"
        )
    return {k: (list(v) if k == "opt" else v) for k, v in prof.items()}
