"""Single immutable configuration object for a pipeline run.

Replaces the legacy ``utils/settings.py`` 1167-line module of mutable
globals. Every stage receives ``config: Config`` explicitly — no
``import settings`` anywhere. Per-plug-in options live in a free-form
``plugin_options`` dict keyed by ``"<plug-point>.<plugin-key>"`` so
research groups can add their own knobs without editing this file.

Defaults reproduce the paper's protocol (Tireli et al., §4.2): Philips
3T DCE-MRI with TR ≈ 11.18 ms, FA = 8°, r1 = 4 s⁻¹·mM⁻¹, dual-bolus
injection, baseline = 5 frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable, frozen configuration for one pipeline run.

    Holds every knob: which plug-in to use at each plug-point, the
    acquisition constants (paper §4.2 defaults), the T1/M0 and
    normalisation parameters, the data paths, and the free-form
    ``plugin_options`` dict. Stages receive an instance explicitly —
    there are intentionally no per-model fields baked in; model-specific
    knobs go in ``plugin_options`` keyed ``"<plug-point>.<plugin-key>"``.
    """

    # ── Plug-in selection ────────────────────────────────────────────
    loader: str = "nifti"
    t1_m0_fitter: str = "inversion_recovery"
    aif_extractor: str = "cnn_sss_shifted"
    tissue_roi_provider: str = "synthseg"
    signal_to_conc: str = "saturation_recovery"
    normaliser: str = "identity"
    kinetic_models: tuple[str, ...] = ("patlak", "tikhonov")
    diffusion_models: tuple[str, ...] = ()                # opt-in via --diffusion
    analysis_levels: tuple[str, ...] = ("median_curve", "voxelwise", "parcel", "region")
    path_scheme: str = "bids_like"

    # ── Compute device ───────────────────────────────────────────────
    # "cpu" | "mps" | "cuda" | "auto"  — auto picks the best available.
    # Plug-ins that can use accelerators (CNN AIF, future Tikhonov port)
    # honour this; pure-numpy plug-ins ignore it. See pbrain.core.devices.
    device: str = "cpu"

    # ── Acquisition (paper §4.2 defaults) ────────────────────────────
    tr_s: float = 0.01118        # TR = 11.18 ms (3D T1 reference)
    te_s: float = 0.00193        # TE = 1.93 ms (DCE)
    flip_angle_deg: float = 30.0     # DCE readout flip angle (paper protocol)
    r1_per_s_mM: float = 4.0     # relaxivity at 3 T
    prepulse_to_readout_s: float = 0.120
    # Optional AIF flow correction (OFF by default = pure voxelwise pipeline).
    # Flowing blood breaks the static saturation-recovery T1 fit (inflow → faster
    # apparent recovery → underestimated T1), which can inflate / log-clamp the
    # AIF for high-enhancement arterial voxels. Set to a literature blood T1
    # (≈1600 ms at 3 T; Lu 2004) to convert the AIF curve with that instead of
    # the per-voxel fit. 0 = use the voxelwise fitted T1 everywhere (default).
    aif_blood_t1_ms: float = 0.0
    dt_s: float = 2.463          # DCE temporal sampling
    n_bolus: int = 2             # single- vs double-bolus

    # ── T1/M0 fit ────────────────────────────────────────────────────
    t1_min_ms: float = 100.0
    t1_max_ms: float = 6000.0
    inversion_times_ms: tuple[float, ...] = (120, 300, 600, 1000, 2000, 4000, 10_000)

    # ── Normalisation ────────────────────────────────────────────────
    baseline_frames: int = 5
    rescale_percentile: float = 95.0

    # ── Paths ────────────────────────────────────────────────────────
    data_root: Path = Path("data")
    subject_id: str = ""

    # ── Per-plug-in options (the ONE place for plug-in parameters) ────
    # Every model/method-specific knob lives here, keyed by
    # "<plug-point>.<plugin-key>" → {opt_name: value}. Set from the CLI
    # via ``--opt models.tikhonov.lambda_selection=gcv`` or from a config
    # file. Research groups add their own keys with zero core changes —
    # there are intentionally NO per-model fields baked into Config.
    plugin_options: dict[str, dict[str, Any]] = field(default_factory=dict)

    def options_for(self, plug_point: str, plugin_key: str) -> dict[str, Any]:
        """Return per-plug-in opts merged with empty defaults."""
        return dict(self.plugin_options.get(f"{plug_point}.{plugin_key}", {}))

    def subject_dir(self) -> Path:
        """Path to this subject's directory (``data_root / subject_id``)."""
        return self.data_root / self.subject_id
