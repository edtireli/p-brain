"""Stage classes that wrap each plug-point into the :class:`Stage` Protocol.

Each Stage:

* declares the upstream manifest names it requires;
* picks the plug-in implementation from the appropriate REGISTRY using
  ``ctx.config``;
* delegates the actual work to that plug-in;
* writes outputs via ``ctx.path_scheme`` and returns a :class:`StageOutput`.

The default 9-stage pipeline is assembled by ``default_pipeline()`` in
``pbrain.cli.run``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pbrain.aggregation import REGISTRY as AGGREGATORS
from pbrain.aif import REGISTRY as AIF_EXTRACTORS
from pbrain.core import Config, Stage, StageContext, StageOutput, get_logger

_log = get_logger("stages")
from pbrain.diffusion import REGISTRY as DIFFUSION_MODELS, DWIInputs
from pbrain.io.loaders import load_4d
from pbrain.io.loaders.base import Series4D
from pbrain.io.loaders.dwi import load_dwi
from pbrain.models import REGISTRY as MODELS, CurveInputs
from pbrain.normalisation import REGISTRY as NORMALISERS
from pbrain.signal_to_conc import REGISTRY as CONVERTERS
from pbrain.t1_m0 import REGISTRY as T1M0_FITTERS
from pbrain.tissue_roi import REGISTRY as TISSUE_ROIS


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────


def _save_nifti(arr: np.ndarray, affine: np.ndarray, path: Path) -> Path:
    import nibabel as nib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), str(path))
    return path


def _save_npy(arr: np.ndarray, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return path


def _save_json(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def _resolve_input(
    ctx: StageContext, opt_key: str, default: str | None = None,
    required: bool = True,
) -> Path | None:
    """Resolve a per-stage input path from config.plugin_options.

    Returns ``Path(value)`` if found, ``None`` if absent and not required,
    or raises ``FileNotFoundError`` if required and missing.
    """
    opts = ctx.config.options_for("inputs", "paths")
    val = opts.get(opt_key, default)
    if val is None:
        if required:
            raise FileNotFoundError(
                f"{opt_key} not provided. Set config.plugin_options['inputs.paths']['{opt_key}']."
            )
        return None
    return Path(val)


def _fit_parcel_painted(model, C: np.ndarray, ca: np.ndarray, t_s: np.ndarray,
                        parc: np.ndarray, opts: dict,
                        cbf_map: np.ndarray | None = None):
    """Average-then-fit a model per parcel and paint each parcel's result back.

    For models that declare ``supports_voxelwise = False`` (e.g. gamma, which
    is ~1800× slower than Tikhonov per curve). Fits each parcel's median curve
    once and assigns that value to every voxel of the parcel — a parcel-
    resolution map that is fast to compute and is exactly the level the model
    is meaningful at. Returns a ``ModelResult`` of (X, Y, Z) painted maps.

    ``cbf_map`` (voxelwise Tikhonov CBF, mL/100g/min): when given and the
    model accepts ``f_cbf``, each parcel's median CBF is passed so gamma pins
    its flow F to the *pipeline's* Tikhonov CBF (the two-stage design).
    """
    from pbrain.models import CurveInputs, ModelResult

    X, Y, Z, _ = C.shape
    outputs = tuple(getattr(model, "outputs", ()) or ())
    accepts_fcbf = "f_cbf" in (getattr(model, "accepts", {}) or {}) or model.key == "gamma"
    painted = {nm: np.full((X, Y, Z), np.nan, dtype=float) for nm in outputs}
    for L in (int(v) for v in np.unique(parc) if int(v) > 0):
        m = parc == L
        ct = np.nanmedian(C[m], axis=0).astype(float)      # parcel median curve (T,)
        if not np.any(np.isfinite(ct) & (ct != 0)):
            continue
        call_opts = dict(opts)
        if cbf_map is not None and accepts_fcbf and "f_cbf" not in call_opts:
            parc_cbf = float(np.nanmedian(cbf_map[m]))
            if np.isfinite(parc_cbf) and parc_cbf > 0:
                call_opts["f_cbf"] = parc_cbf / 6000.0     # mL/100g/min → 1/s
        try:
            r = model.fit(CurveInputs(c_tissue=ct, c_input=ca, t_s=t_s), **call_opts)
        except Exception:                                   # noqa: BLE001
            continue
        for nm in outputs:
            if nm in r.maps:
                painted[nm][m] = float(np.asarray(r.maps[nm]).reshape(-1)[0])
    return ModelResult(maps=painted, units=dict(getattr(model, "units", {})))


# ────────────────────────────────────────────────────────────────────────
# Stages
# ────────────────────────────────────────────────────────────────────────


@dataclass
class LoadStage:
    name: str = "load"
    requires: tuple[str, ...] = ()
    plugin_key: str = "loader"

    def run(self, ctx: StageContext) -> StageOutput:
        dce_path = _resolve_input(ctx, "dce", required=True)
        dce = load_4d(dce_path)

        t1_path = _resolve_input(ctx, "t1_anatomical", required=False)
        ir_path = _resolve_input(ctx, "ir", required=False)

        artefacts: dict[str, Path] = {}

        dce_out = ctx.path_scheme.output_path(
            ctx.subject_dir, self.name, None, None, "dce", "nii.gz"
        )
        _save_nifti(dce.data[..., 0] if dce.data.shape[-1] == 1 else dce.data, dce.affine, dce_out)
        artefacts["dce"] = dce_out
        _save_npy(dce.axis4_values,
                  ctx.path_scheme.output_path(ctx.subject_dir, self.name, None, None,
                                              "dce_time_s", "npy"))
        artefacts["dce_time_s"] = ctx.path_scheme.output_path(
            ctx.subject_dir, self.name, None, None, "dce_time_s", "npy"
        )
        _save_json(
            {"voxel_size": list(dce.voxel_size),
             "axis4_kind": dce.axis4_kind,
             "meta": dce.meta},
            ctx.path_scheme.output_path(ctx.subject_dir, self.name, None, None,
                                        "dce_meta", "json"),
        )
        artefacts["dce_meta"] = ctx.path_scheme.output_path(
            ctx.subject_dir, self.name, None, None, "dce_meta", "json"
        )

        if t1_path is not None and Path(t1_path).exists():
            t1 = load_4d(t1_path)
            t1_out = ctx.path_scheme.output_path(
                ctx.subject_dir, self.name, None, None, "t1_anatomical", "nii.gz"
            )
            _save_nifti(t1.data[..., 0], t1.affine, t1_out)
            artefacts["t1_anatomical"] = t1_out

        if ir_path is not None and Path(ir_path).exists():
            ir = load_4d(ir_path)
            ir_out = ctx.path_scheme.output_path(
                ctx.subject_dir, self.name, None, None, "ir", "nii.gz"
            )
            _save_nifti(ir.data, ir.affine, ir_out)
            artefacts["ir"] = ir_out
            _save_npy(ir.axis4_values,
                      ctx.path_scheme.output_path(ctx.subject_dir, self.name, None, None,
                                                  "ir_ti_s", "npy"))
            artefacts["ir_ti_s"] = ctx.path_scheme.output_path(
                ctx.subject_dir, self.name, None, None, "ir_ti_s", "npy"
            )

        return StageOutput(artefacts=artefacts,
                           metadata={"dce_shape": list(dce.data.shape),
                                     "dce_path": str(dce_path)})


@dataclass
class T1M0Stage:
    name: str = "t1_m0"
    requires: tuple[str, ...] = ("load",)
    plugin_key: str = ""

    def run(self, ctx: StageContext) -> StageOutput:
        import nibabel as nib

        plugin_key = ctx.config.t1_m0_fitter
        self.plugin_key = plugin_key
        fitter = T1M0_FITTERS[plugin_key]

        load_mf = ctx.upstream_manifests["load"]
        ir_path_s = load_mf.outputs.get("ir", "")
        ti_path_s = load_mf.outputs.get("ir_ti_s", "")

        ir_data: np.ndarray | None = None
        ti_s: np.ndarray | None = None
        ref_affine: np.ndarray | None = None
        if ir_path_s and Path(ir_path_s).exists():
            ir_img = nib.load(str(ir_path_s))
            ir_data = np.asarray(ir_img.dataobj, dtype=np.float32)
            ref_affine = np.asarray(ir_img.affine, dtype=float)
            if ti_path_s and Path(ti_path_s).exists():
                ti_s = np.load(ti_path_s)
            # An IR series assembled from separate per-TI files carries no
            # per-frame TI in the NIfTI — fall back to the configured
            # inversion_times_ms when they match the frame count.
            if (ti_s is None or np.unique(np.asarray(ti_s)).size < 2) and ir_data.ndim == 4:
                cfg_ti = np.asarray(ctx.config.inversion_times_ms, dtype=float) / 1000.0
                if cfg_ti.size == ir_data.shape[-1]:
                    ti_s = cfg_ti

        opts = dict(ctx.config.options_for("t1_m0", plugin_key))
        result = fitter.fit(ir_data, ti_s, **opts)

        if ref_affine is None:
            # Plugin (e.g. preloaded) supplied its own maps — derive affine from the DCE.
            dce_img = nib.load(str(load_mf.outputs["dce"]))
            ref_affine = np.asarray(dce_img.affine, dtype=float)

        t1_out = ctx.path_scheme.output_path(
            ctx.subject_dir, self.name, plugin_key, None, "t1_map_ms", "nii.gz"
        )
        m0_out = ctx.path_scheme.output_path(
            ctx.subject_dir, self.name, plugin_key, None, "m0_map", "nii.gz"
        )
        _save_nifti(result.t1_map_ms, ref_affine, t1_out)
        _save_nifti(result.m0_map, ref_affine, m0_out)

        return StageOutput(
            artefacts={"t1_map_ms": t1_out, "m0_map": m0_out},
            metadata=result.meta,
        )


@dataclass
class AIFStage:
    """Extracts the input function curve from the **concentration volume**,
    not the raw signal. The DCE→mM conversion happens once upstream in
    SignalToConcStage so every voxel uses its own T1/M0; the AIF curve is
    then just a curve-extraction from that mM volume (max-voxel, median,
    mean, etc.). No further conversion downstream.
    """
    name: str = "aif"
    requires: tuple[str, ...] = ("load", "t1_m0", "signal_to_conc")
    plugin_key: str = ""

    def run(self, ctx: StageContext) -> StageOutput:
        import nibabel as nib

        plugin_key = ctx.config.aif_extractor
        self.plugin_key = plugin_key
        extractor = AIF_EXTRACTORS[plugin_key]

        load_mf = ctx.upstream_manifests["load"]
        t1m0_mf = ctx.upstream_manifests["t1_m0"]
        stc_mf = ctx.upstream_manifests["signal_to_conc"]

        # CNN-based AIF extractors expect the *signal* volume (their
        # training distribution). The *concentration* volume goes in
        # alongside as `concentration_data` — the plug-in pulls the
        # curve from there so each voxel's own T1/M0 propagates into
        # the AIF curve units (mM).
        dce_img = nib.load(str(load_mf.outputs["dce"]))
        dce_signal = np.asarray(dce_img.dataobj, dtype=np.float32)
        if dce_signal.ndim == 3:
            dce_signal = dce_signal[..., None]
        dce_affine = np.asarray(dce_img.affine, dtype=float)

        ct_img = nib.load(str(stc_mf.outputs["concentration"]))
        ct_data = np.asarray(ct_img.dataobj, dtype=np.float32)
        if ct_data.ndim == 3:
            ct_data = ct_data[..., None]

        t_s = np.load(load_mf.outputs["dce_time_s"])

        t1 = np.asarray(nib.load(str(t1m0_mf.outputs["t1_map_ms"])).dataobj, dtype=float)
        m0 = np.asarray(nib.load(str(t1m0_mf.outputs["m0_map"])).dataobj, dtype=float)

        opts = dict(ctx.config.options_for("aif", plugin_key))
        opts.setdefault("baseline_frames", ctx.config.baseline_frames)

        ifn = extractor.extract(
            dce_signal, t_s, dce_affine,
            t1_map=t1, m0_map=m0,
            concentration_data=ct_data,
            **opts,
        )

        # Persist
        ca_out = ctx.path_scheme.output_path(
            ctx.subject_dir, self.name, plugin_key, None, "aif_signal", "npy"
        )
        _save_npy(ifn.c_a, ca_out)
        mask_out = ctx.path_scheme.output_path(
            ctx.subject_dir, self.name, plugin_key, None, "aif_mask", "nii.gz"
        )
        _save_nifti(ifn.mask.astype(np.uint8), dce_affine, mask_out)
        json_out = ctx.path_scheme.output_path(
            ctx.subject_dir, self.name, plugin_key, None, "aif", "json"
        )
        _save_json({"source": ifn.source, "n_voxels": int(ifn.mask.sum()),
                    "meta": ifn.meta}, json_out)

        return StageOutput(
            artefacts={"aif_signal": ca_out, "aif_mask": mask_out, "aif_meta": json_out},
            metadata={"source": ifn.source, "n_voxels": int(ifn.mask.sum())},
        )


@dataclass
class TissueROIStage:
    name: str = "tissue_roi"
    requires: tuple[str, ...] = ("load",)
    plugin_key: str = ""

    def run(self, ctx: StageContext) -> StageOutput:
        import nibabel as nib

        plugin_key = ctx.config.tissue_roi_provider
        self.plugin_key = plugin_key
        provider = TISSUE_ROIS[plugin_key]

        load_mf = ctx.upstream_manifests["load"]
        t1_path = load_mf.outputs.get("t1_anatomical")
        if t1_path:
            t1_img = nib.load(str(t1_path))
            t1_vol = np.asarray(t1_img.dataobj, dtype=np.float32)
            t1_aff = np.asarray(t1_img.affine, dtype=float)
        else:
            # Providers that supply a parcellation directly (preloaded) or
            # work in DCE space (voxelwise/manual) don't need a T1 — common
            # for control datasets with no 3D-T1. Segmentation providers
            # (synthseg/fastsurfer/command) that *do* need it will say so.
            t1_vol = np.zeros((1, 1, 1), dtype=np.float32)
            t1_aff = np.eye(4)

        out_dir = ctx.path_scheme.stage_dir(ctx.subject_dir, self.name, plugin_key)
        out_dir.mkdir(parents=True, exist_ok=True)

        opts = dict(ctx.config.options_for("tissue_roi", plugin_key))
        roi = provider.extract(t1_vol, t1_aff, out_dir=out_dir, **opts)

        # The parcellation must live on the DCE grid — that's where the kinetic
        # maps are aggregated. A provider that segments a high-res 3-D T1 (e.g.
        # SynthSeg) returns its own grid, so resample to the DCE grid with
        # nearest-neighbour (labels are integers). Providers already in DCE
        # space (preloaded *_in_DCE) pass through unchanged.
        parc = roi.parcellation.astype(np.int16)
        parc_affine = roi.affine
        dce_img = nib.load(str(load_mf.outputs["dce"]))
        dce_shape = tuple(dce_img.shape[:3])
        if parc.shape != dce_shape:
            from nibabel.processing import resample_from_to
            res = resample_from_to(
                nib.Nifti1Image(parc, np.asarray(parc_affine, dtype=float)),
                (dce_shape, np.asarray(dce_img.affine, dtype=float)), order=0)
            parc = np.asarray(res.dataobj).astype(np.int16)
            parc_affine = np.asarray(dce_img.affine, dtype=float)
            _log.info("tissue_roi: resampled %s parcellation %s → DCE grid %s",
                      plugin_key, roi.parcellation.shape, dce_shape)

        parc_out = ctx.path_scheme.output_path(
            ctx.subject_dir, self.name, plugin_key, None, "parcellation", "nii.gz"
        )
        _save_nifti(parc, parc_affine, parc_out)
        labels_out = ctx.path_scheme.output_path(
            ctx.subject_dir, self.name, plugin_key, None, "labels", "json"
        )
        _save_json(
            {"labels": {str(k): v for k, v in roi.labels.items()},
             "region_map": roi.region_map or {},
             "meta": roi.meta},
            labels_out,
        )

        return StageOutput(
            artefacts={"parcellation": parc_out, "labels": labels_out},
            metadata=roi.meta,
        )


@dataclass
class SignalToConcStage:
    """Convert the **entire** 4-D DCE signal volume to concentration (mM) once,
    voxel-wise, using each voxel's own T1/M0. After this stage runs, every
    downstream stage operates on concentrations — there is **no further
    signal→conc conversion** anywhere. SSS voxels keep their own T1/M0
    because the conversion is per-voxel; the AIF stage just extracts a
    curve from this mM volume.
    """
    name: str = "signal_to_conc"
    requires: tuple[str, ...] = ("load", "t1_m0")
    plugin_key: str = ""

    def run(self, ctx: StageContext) -> StageOutput:
        import nibabel as nib

        plugin_key = ctx.config.signal_to_conc
        self.plugin_key = plugin_key
        conv = CONVERTERS[plugin_key]

        load_mf = ctx.upstream_manifests["load"]
        t1m0_mf = ctx.upstream_manifests["t1_m0"]

        dce_img = nib.load(str(load_mf.outputs["dce"]))
        S = np.asarray(dce_img.dataobj, dtype=np.float32)
        if S.ndim == 3:
            S = S[..., None]
        T1 = np.asarray(nib.load(str(t1m0_mf.outputs["t1_map_ms"])).dataobj, dtype=float)
        M0 = np.asarray(nib.load(str(t1m0_mf.outputs["m0_map"])).dataobj, dtype=float)

        # Pull DCE acquisition metadata from the BIDS sidecar (load stage).
        # Critical: the DCE flip angle (Philips default 30°) differs from
        # the T1-anatomical (paper §4.2 = 8°). Wrong flip → S/(M0·sin α)
        # exceeds 1 → log clamp → nonsense concentrations.
        dce_meta_path = load_mf.outputs.get("dce_meta")
        dce_meta: dict[str, Any] = {}
        if dce_meta_path:
            try:
                with open(dce_meta_path) as f:
                    dce_meta = json.load(f).get("meta", {}) or {}
            except Exception:
                dce_meta = {}

        opts = dict(ctx.config.options_for("signal_to_conc", plugin_key))
        flip_meta = dce_meta.get("FlipAngle") or dce_meta.get("flip_angle_deg")
        tr_meta = dce_meta.get("RepetitionTimeExcitation") or dce_meta.get("RepetitionTime")
        opts.setdefault("flip_angle_deg", float(flip_meta) if flip_meta is not None else ctx.config.flip_angle_deg)
        opts.setdefault("tr_s", float(tr_meta) if tr_meta is not None else ctx.config.tr_s)
        opts.setdefault("r1_per_s_mM", ctx.config.r1_per_s_mM)
        opts.setdefault("prepulse_to_readout_s", ctx.config.prepulse_to_readout_s)

        # Per-voxel signal → mM. Each voxel uses its own (T1, M0); SSS / rICA
        # voxels keep their own relaxation parameters through this single
        # conversion. There is no per-region averaging of T1/M0 anywhere.
        C = conv.convert(S, T1, M0, **opts)
        ct_out = ctx.path_scheme.output_path(
            ctx.subject_dir, self.name, plugin_key, None, "concentration", "nii.gz"
        )
        _save_nifti(C, np.asarray(dce_img.affine, dtype=float), ct_out)

        return StageOutput(
            artefacts={"concentration": ct_out},
            metadata={
                "converter": plugin_key,
                "flip_angle_deg": opts["flip_angle_deg"],
                "tr_s": opts["tr_s"],
                "ct_peak_mM": float(np.nanmax(C)),
            },
        )


@dataclass
class NormalisationStage:
    name: str = "normalisation"
    requires: tuple[str, ...] = ("load", "signal_to_conc", "aif")
    plugin_key: str = ""

    def run(self, ctx: StageContext) -> StageOutput:
        import nibabel as nib

        plugin_key = ctx.config.normaliser
        self.plugin_key = plugin_key
        norm = NORMALISERS[plugin_key]

        stc_mf = ctx.upstream_manifests["signal_to_conc"]
        aif_mf = ctx.upstream_manifests["aif"]
        load_mf = ctx.upstream_manifests["load"]

        C_img = nib.load(str(stc_mf.outputs["concentration"]))
        C = np.asarray(C_img.dataobj, dtype=np.float32)
        # AIF curve is already in mM (the AIF stage now reads from the
        # mM volume directly, not from the raw signal).
        ca = np.load(aif_mf.outputs["aif_signal"])
        t_s = np.load(load_mf.outputs["dce_time_s"])

        opts = dict(ctx.config.options_for("normalisation", plugin_key))
        opts.setdefault("baseline_frames", ctx.config.baseline_frames)
        opts.setdefault("percentile", ctx.config.rescale_percentile)

        # Normalise the AIF on its own; then normalise tissue using the
        # AIF as a shared-scale reference. This keeps Patlak's slope
        # physically meaningful (Ct and Ca share the same scale factor,
        # so Ki = Δy/Δx is in proper mL/100g/min instead of being scaled
        # by p95(Ca)/p95(Ct)).
        ca_norm = norm.normalise(ca, t_s, **opts)
        X, Y, Z, T = C.shape
        C_flat = C.reshape(-1, T).T   # (T, V)
        C_norm_flat = norm.normalise(C_flat, t_s, reference_curve=ca, **opts)
        C_norm = C_norm_flat.T.reshape(X, Y, Z, T)

        ca_out = ctx.path_scheme.output_path(
            ctx.subject_dir, self.name, plugin_key, None, "aif_normalised", "npy"
        )
        _save_npy(ca_norm, ca_out)
        ct_out = ctx.path_scheme.output_path(
            ctx.subject_dir, self.name, plugin_key, None, "ct_normalised", "nii.gz"
        )
        _save_nifti(C_norm, np.asarray(C_img.affine, dtype=float), ct_out)

        return StageOutput(
            artefacts={"aif_normalised": ca_out, "ct_normalised": ct_out},
            metadata={"normaliser": plugin_key},
        )


@dataclass
class KineticStage:
    """Runs every configured model. Each model writes its own raw maps
    under ``<kinetic>/<model_key>/voxelwise/`` for downstream aggregation.

    Optionally consumes the tissue-ROI mask to restrict fits to brain
    voxels — this matches legacy behaviour (Ki maps in
    ``Analysis/Ki_per_voxel.nii.gz`` are brain-masked) and dramatically
    tightens output dynamic range by skipping non-brain noise.
    """
    name: str = "kinetic"
    requires: tuple[str, ...] = ("load", "normalisation", "tissue_roi")
    plugin_key: str = "multi"

    def run(self, ctx: StageContext) -> StageOutput:
        import nibabel as nib

        norm_mf = ctx.upstream_manifests["normalisation"]
        load_mf = ctx.upstream_manifests["load"]
        roi_mf = ctx.upstream_manifests.get("tissue_roi")

        C_img = nib.load(str(norm_mf.outputs["ct_normalised"]))
        C = np.asarray(C_img.dataobj, dtype=np.float32)
        ca = np.load(norm_mf.outputs["aif_normalised"])
        t_s = np.load(load_mf.outputs["dce_time_s"])
        affine = np.asarray(C_img.affine, dtype=float)

        X, Y, Z, T = C.shape
        c_t = C.reshape(-1, T).T   # (T, V)

        # Brain mask from tissue_roi.parcellation > 0 — restricts Patlak/
        # Tikhonov to brain voxels only. Non-brain voxels get NaN outputs.
        voxel_mask: np.ndarray | None = None
        parc_arr: np.ndarray | None = None
        region_map: dict[str, list[int]] = {}
        if roi_mf is not None:
            parc_arr = np.asarray(
                nib.load(str(roi_mf.outputs["parcellation"])).dataobj, dtype=np.int32
            )
            if parc_arr.shape == (X, Y, Z):
                voxel_mask = (parc_arr > 0).ravel()
            labels_path = roi_mf.outputs.get("labels")
            if labels_path:
                blob = json.loads(Path(labels_path).read_text())
                region_map = blob.get("region_map") or {}

        artefacts: dict[str, Path] = {}
        per_model_meta: dict[str, Any] = {}
        per_model_qc: dict[str, Any] = {}
        pin_cbf_map: np.ndarray | None = None      # Tikhonov CBF to pin gamma's F

        # Run voxelwise models first so a CBF map (Tikhonov) is available to
        # pin parcel-level models (gamma) to the pipeline's own CBF.
        ordered_models = sorted(
            ctx.config.kinetic_models,
            key=lambda k: not getattr(MODELS.get(k), "supports_voxelwise", True),
        )
        for model_key in ordered_models:
            if model_key not in MODELS:
                raise RuntimeError(
                    f"Kinetic model {model_key!r} not in registry. "
                    f"Available: {sorted(MODELS.keys())}"
                )
            model = MODELS[model_key]
            opts = dict(ctx.config.options_for("models", model_key))

            if getattr(model, "supports_voxelwise", True):
                inputs = CurveInputs(
                    c_tissue=c_t, c_input=ca, t_s=t_s, mask=voxel_mask,
                )
                result = model.fit(inputs, **opts)
            else:
                # Average-then-fit per parcel, painted back to a map. For
                # models too costly for voxelwise (e.g. gamma): fit each
                # parcel's median curve once and paint every voxel of that
                # parcel with the result. Fast (~one fit per parcel).
                if parc_arr is None:
                    _log.warning("kinetic: %s needs a parcellation for its "
                                 "tissue/parcel fit — skipping", model_key)
                    continue
                result = _fit_parcel_painted(model, C, ca, t_s, parc_arr, opts,
                                             cbf_map=pin_cbf_map)
                _log.info("kinetic: %s fitted at parcel level (average-then-fit)%s",
                          model_key,
                          " [F pinned to Tikhonov CBF]" if pin_cbf_map is not None else "")

            # Validate the model honoured its declared output contract.
            # Require declared ⊆ produced (every promised map exists); extra
            # maps are allowed (e.g. optional uncertainty maps like mtt_sd /
            # cth_sd that only appear when sampling is enabled) and flow
            # through aggregation + diagnostics like any other map.
            declared = tuple(getattr(model, "outputs", ()) or ())
            produced = tuple(result.maps.keys())
            if declared:
                missing = [k for k in declared if k not in produced]
                if missing:
                    raise RuntimeError(
                        f"Model {model_key!r} contract violation: declared "
                        f"outputs={declared} but fit() produced {produced}. "
                        f"Missing={missing}. Update the model's `outputs` "
                        f"classvar or make `fit()` return all declared maps."
                    )

            # Reshape each (V,) map back to (X, Y, Z)
            shaped: dict[str, np.ndarray] = {}
            for k, v in result.maps.items():
                arr = np.asarray(v)
                if arr.ndim == 0:
                    shaped[k] = np.full((X, Y, Z), float(arr))
                elif arr.ndim == 1 and arr.size == X * Y * Z:
                    shaped[k] = arr.reshape(X, Y, Z)
                else:
                    shaped[k] = arr

            # Remember a Tikhonov CBF map to pin gamma's flow F to (first
            # voxelwise tikhonov-family model wins).
            if (pin_cbf_map is None and model_key.startswith("tikhonov")
                    and "cbf" in shaped and shaped["cbf"].ndim == 3):
                pin_cbf_map = shaped["cbf"]

            # Level reflects how the map was produced: a true per-voxel fit
            # ("voxelwise") vs an average-then-fit painted back per parcel
            # ("parcel") — so a parcel-painted map is never mislabelled as
            # voxelwise.
            level = "voxelwise" if getattr(model, "supports_voxelwise", True) else "parcel"
            for name, arr in shaped.items():
                p = ctx.path_scheme.output_path(
                    ctx.subject_dir, self.name, model_key, level, name, "nii.gz"
                )
                _save_nifti(arr, affine, p)
                artefacts[f"{model_key}/{name}"] = p

            # Physiological-range QC on the brain-median of each scalar map.
            from pbrain.core.qc import check_maps
            qc_ranges = dict(ctx.config.options_for("qc", "ranges"))
            qc_ranges = {k: tuple(float(x) for x in str(v).split(","))
                         for k, v in qc_ranges.items() if "," in str(v)}
            qc_mask = (voxel_mask.reshape(X, Y, Z) if voxel_mask is not None else None)
            scalar_maps = {k: v for k, v in shaped.items() if v.ndim == 3}
            per_model_qc[model_key] = check_maps(scalar_maps, mask=qc_mask,
                                                 ranges=qc_ranges or None)
            if per_model_qc[model_key]["status"] == "warn":
                flagged = [m for m, c in per_model_qc[model_key]["maps"].items()
                           if c["status"] == "warn"]
                _log.warning("QC: %s out-of-range maps: %s", model_key, flagged)

            per_model_meta[model_key] = {
                "outputs": list(result.maps.keys()),
                "units": result.units,
                "level": level,            # "voxelwise" or "parcel"
            }

        overall = ("warn" if any(q["status"] == "warn" for q in per_model_qc.values())
                   else "pass")
        return StageOutput(
            artefacts=artefacts,
            metadata={"models": list(ctx.config.kinetic_models), "per_model": per_model_meta},
            qc={"status": overall, "per_model": per_model_qc},
        )


@dataclass
class DiffusionStage:
    """Run every configured diffusion model on a DWI series.

    Independent of the kinetic track: a subject without DWI simply omits
    ``--diffusion`` (or passes no models). The stage writes each model's
    output maps under ``<diffusion>/<model_key>/native/`` (always) and,
    if the DWI affine differs from the parcellation affine, *also* under
    ``<diffusion>/<model_key>/voxelwise/`` resampled to the parcellation
    grid for downstream aggregation.
    """
    name: str = "diffusion"
    requires: tuple[str, ...] = ("load", "tissue_roi")
    plugin_key: str = "multi"

    def run(self, ctx: StageContext) -> StageOutput:
        import nibabel as nib

        if not ctx.config.diffusion_models:
            return StageOutput(artefacts={}, metadata={"models": []})

        dwi_path = _resolve_input(ctx, "dwi", required=False)
        if dwi_path is None or not Path(dwi_path).exists():
            _log.info("diffusion: no DWI input (inputs.paths.dwi=%s) — skipping", dwi_path)
            return StageOutput(artefacts={}, metadata={"models": []})

        bval_opt = _resolve_input(ctx, "bvals", required=False)
        bvec_opt = _resolve_input(ctx, "bvecs", required=False)
        dwi = load_dwi(dwi_path, bval_path=bval_opt, bvec_path=bvec_opt)
        _log.info("diffusion: DWI %s  shells=%s  n_b0=%d",
                  dwi.data.shape, dwi.shells, dwi.n_b0)

        # Brain mask: parc>0 if grids match, else a simple b=0 percentile mask.
        roi_mf = ctx.upstream_manifests["tissue_roi"]
        parc_img = nib.load(str(roi_mf.outputs["parcellation"]))
        parc = np.asarray(parc_img.dataobj, dtype=np.int32)
        parc_affine = np.asarray(parc_img.affine, dtype=float)
        same_grid = (
            parc.shape == dwi.data.shape[:3]
            and np.allclose(parc_affine, dwi.affine, atol=1e-3)
        )
        if same_grid:
            brain_mask = (parc > 0)
        else:
            b0 = dwi.data[..., dwi.bvals <= 50].mean(axis=-1)
            thr = float(np.percentile(b0[b0 > 0], 25)) if (b0 > 0).any() else 0.0
            brain_mask = b0 > thr
        _log.info("diffusion: brain_mask n_voxels=%d  same_grid_as_dce=%s",
                  int(brain_mask.sum()), same_grid)

        diff_inputs = DWIInputs(
            signal=dwi.data, bvals=dwi.bvals, bvecs=dwi.bvecs,
            affine=dwi.affine, mask=brain_mask,
        )

        artefacts: dict[str, Path] = {}
        per_model_meta: dict[str, Any] = {}

        for model_key in ctx.config.diffusion_models:
            if model_key not in DIFFUSION_MODELS:
                raise RuntimeError(
                    f"Diffusion model {model_key!r} not in registry. "
                    f"Available: {sorted(DIFFUSION_MODELS.keys())}"
                )
            model = DIFFUSION_MODELS[model_key]
            opts = dict(ctx.config.options_for("diffusion", model_key))
            _log.info("diffusion: fitting %s…", model_key)
            try:
                result = model.fit(diff_inputs, **opts)
            except Exception as exc:
                _log.warning("diffusion: %s failed → %s", model_key, exc)
                continue

            declared = tuple(getattr(model, "outputs", ()) or ())
            produced = tuple(result.maps.keys())
            if declared:
                missing = [k for k in declared if k not in produced]
                extra = [k for k in produced if k not in declared]
                if missing or extra:
                    raise RuntimeError(
                        f"Diffusion model {model_key!r} contract violation: "
                        f"declared {declared}, produced {produced} "
                        f"(missing={missing}, extra={extra})."
                    )

            # ---- native space (always) ----
            for nm, arr in result.maps.items():
                p = ctx.path_scheme.output_path(
                    ctx.subject_dir, self.name, model_key, "native", nm, "nii.gz"
                )
                _save_nifti(arr, dwi.affine, p)
                artefacts[f"{model_key}/native/{nm}"] = p

            # ---- DCE-grid (only if affines differ) ----
            voxelwise_paths: dict[str, Path] = {}
            if not same_grid:
                from nibabel.processing import resample_from_to
                target_shape = parc.shape
                for nm, arr in result.maps.items():
                    a = np.asarray(arr)
                    # vector/RGB maps (last dim 3 with same spatial shape) → resample channel-wise
                    if a.ndim == 4 and a.shape[-1] == 3:
                        chans = []
                        for c in range(3):
                            chan_img = nib.Nifti1Image(a[..., c].astype(np.float32), dwi.affine)
                            chans.append(np.asarray(
                                resample_from_to(chan_img,
                                                 (target_shape, parc_affine),
                                                 order=1).dataobj))
                        out_arr = np.stack(chans, axis=-1)
                    else:
                        src = nib.Nifti1Image(a.astype(np.float32), dwi.affine)
                        out_arr = np.asarray(
                            resample_from_to(src, (target_shape, parc_affine),
                                             order=1).dataobj)
                    p = ctx.path_scheme.output_path(
                        ctx.subject_dir, self.name, model_key, "voxelwise", nm, "nii.gz"
                    )
                    _save_nifti(out_arr, parc_affine, p)
                    artefacts[f"{model_key}/voxelwise/{nm}"] = p
                    voxelwise_paths[nm] = p
            else:
                # Identical grids → "voxelwise" is the same as "native"; reuse paths.
                for nm in result.maps.keys():
                    voxelwise_paths[nm] = artefacts[f"{model_key}/native/{nm}"]
                    artefacts[f"{model_key}/voxelwise/{nm}"] = voxelwise_paths[nm]

            per_model_meta[model_key] = {
                "outputs": list(result.maps.keys()),
                "units": result.units,
                "native_affine_matches_dce": bool(same_grid),
            }

        return StageOutput(
            artefacts=artefacts,
            metadata={
                "models": list(ctx.config.diffusion_models),
                "per_model": per_model_meta,
                "n_b0": dwi.n_b0,
                "shells": list(dwi.shells),
            },
        )


@dataclass
class AggregationStage:
    """For every (model × aggregator) combination, run the aggregator over
    the model's voxelwise maps and write the bundle to the configured
    aggregation directory. Handles BOTH kinetic and diffusion outputs in
    a single pass — same code path, different source manifests.
    """
    name: str = "summary"
    requires: tuple[str, ...] = ("kinetic", "tissue_roi")
    plugin_key: str = "multi"

    def run(self, ctx: StageContext) -> StageOutput:
        import nibabel as nib

        kinetic_mf = ctx.upstream_manifests["kinetic"]
        diffusion_mf = ctx.upstream_manifests.get("diffusion")
        roi_mf = ctx.upstream_manifests.get("tissue_roi")
        parc = None
        labels: dict[int, str] | None = None
        region_map: dict[str, list[int]] | None = None
        ref_affine: np.ndarray | None = None
        if roi_mf is not None:
            parc_img = nib.load(str(roi_mf.outputs["parcellation"]))
            parc = np.asarray(parc_img.dataobj, dtype=np.int32)
            ref_affine = np.asarray(parc_img.affine, dtype=float)
            with open(roi_mf.outputs["labels"]) as f:
                ldata = json.load(f)
            labels = {int(k): v for k, v in ldata.get("labels", {}).items()}
            region_map = ldata.get("region_map") or None

        artefacts: dict[str, Path] = {}

        # Sources: (namespace_for_path_scheme, source_manifest, model_keys, key_prefix_in_outputs)
        sources: list[tuple[str, Any, tuple[str, ...], str]] = [
            ("kinetic", kinetic_mf, tuple(ctx.config.kinetic_models), ""),
        ]
        if diffusion_mf is not None and ctx.config.diffusion_models:
            sources.append(
                ("diffusion", diffusion_mf, tuple(ctx.config.diffusion_models), "voxelwise/"),
            )

        for namespace, mf, model_keys, key_prefix in sources:
            for model_key in model_keys:
                model_meta = mf.metadata.get("per_model", {}).get(model_key, {})
                output_names = model_meta.get("outputs", [])
                units = model_meta.get("units", {})

                maps: dict[str, np.ndarray] = {}
                for nm in output_names:
                    p = mf.outputs.get(f"{model_key}/{key_prefix}{nm}") \
                        or mf.outputs.get(f"{model_key}/{nm}")
                    if not p:
                        continue
                    arr = np.asarray(nib.load(str(p)).dataobj, dtype=float)
                    # vector / RGB maps don't aggregate scalar-wise — skip
                    if arr.ndim != 3:
                        continue
                    maps[nm] = arr
                if not maps:
                    continue

                # Parcel-painted models (gamma) have no true voxelwise map, so
                # skip the voxelwise aggregator for them — never write a
                # <model>/voxelwise/ that is really parcel data.
                model_level = model_meta.get("level", "voxelwise")
                for agg_key in ctx.config.analysis_levels:
                    if agg_key == "voxelwise" and model_level != "voxelwise":
                        continue
                    if agg_key not in AGGREGATORS:
                        raise RuntimeError(
                            f"Aggregator {agg_key!r} not in registry. "
                            f"Available: {sorted(AGGREGATORS.keys())}"
                        )
                    agg = AGGREGATORS[agg_key]
                    out_dir = ctx.path_scheme.aggregation_dir(
                        ctx.subject_dir, namespace, model_key, agg_key
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)
                    opts = dict(ctx.config.options_for("aggregation", agg_key))
                    res = agg.aggregate(
                        maps, units,
                        reference_affine=ref_affine,
                        parcellation=parc,
                        labels=labels,
                        region_map=region_map,
                        out_dir=out_dir,
                        **opts,
                    )
                    for nm, p in res.files.items():
                        artefacts[f"{namespace}/{model_key}/{agg_key}/{nm}"] = p

        return StageOutput(artefacts=artefacts,
                           metadata={"kinetic_models": list(ctx.config.kinetic_models),
                                     "diffusion_models": list(ctx.config.diffusion_models),
                                     "aggregators": list(ctx.config.analysis_levels)})


def default_stages() -> list[Stage]:
    """Default pipeline ordering. The key change from the paper's diagram:
    ``signal_to_conc`` runs BEFORE ``aif`` so all curves throughout the
    pipeline are in mM (per-voxel conversion uses each voxel's own T1/M0,
    including SSS/rICA voxels through the AIF curve extraction).
    """
    from ._diagnostics import DiagnosticsStage
    return [
        LoadStage(),
        T1M0Stage(),
        SignalToConcStage(),     # mM volume FIRST
        AIFStage(),              # extracts AIF curve from mM volume
        TissueROIStage(),
        NormalisationStage(),
        KineticStage(),
        DiffusionStage(),        # no-op if --diffusion is empty
        AggregationStage(),
        DiagnosticsStage(),
    ]
