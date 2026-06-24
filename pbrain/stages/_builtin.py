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

        # The 3-D T1 is optional: if present but unusable (empty/corrupt), proceed
        # without it rather than crash — tissue ROI then falls back to a whole-brain
        # mask. A bad *required* input (DCE/IR) still fails loudly via the loader.
        if t1_path is not None and Path(t1_path).exists():
            try:
                t1 = load_4d(t1_path)
            except Exception as exc:              # noqa: BLE001 — optional input
                _log.warning("load: T1 anatomical unusable (%s) — proceeding "
                             "without it; tissue ROI falls back to whole-brain",
                             exc)
                t1 = None
            if t1 is not None:
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
            # An IR series assembled from separate per-TI files carries only
            # frame INDICES (0,1,2,…) in its 4-th-axis values, not real
            # inversion times — using them as TIs rails the fit to the T1
            # ceiling. Fall back to the configured inversion_times_ms when they
            # match the frame count and the loaded values are missing,
            # degenerate, OR look like consecutive frame indices.
            if ir_data.ndim == 4:
                cfg_ti = np.asarray(ctx.config.inversion_times_ms, dtype=float) / 1000.0
                n = ir_data.shape[-1]
                ta = None if ti_s is None else np.ravel(np.asarray(ti_s, dtype=float))
                degenerate = ta is None or np.unique(ta).size < 2
                frame_indices = (ta is not None and ta.size == n
                                 and np.allclose(np.sort(ta), np.arange(n), atol=0.01))
                if cfg_ti.size == n and (degenerate or frame_indices):
                    if frame_indices:
                        _log.info("t1_m0: ir_ti_s looked like frame indices — "
                                  "using configured inversion_times_ms")
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

        # Flow correction for the AIF: blood voxels violate the static
        # saturation-recovery T1 model (inflow makes recovery look faster), so
        # their voxelwise T1 is underestimated (~800 ms vs ~1600 ms blood),
        # which inflates S/K and log-clamps the AIF curve to tens of mM. Re-derive
        # the concentration the AIF reads from using a fixed literature blood T1.
        # Tissue conc (signal_to_conc) is unaffected — only the blood pool needs
        # this. Disabled when aif_blood_t1_ms <= 0.
        blood_t1_ms = float(getattr(ctx.config, "aif_blood_t1_ms", 0.0) or 0.0)
        if blood_t1_ms > 0:
            conv = CONVERTERS[ctx.config.signal_to_conc]
            conv_opts = dict(ctx.config.options_for("signal_to_conc", ctx.config.signal_to_conc))
            conv_opts.setdefault("flip_angle_deg", ctx.config.flip_angle_deg)
            conv_opts.setdefault("tr_s", ctx.config.tr_s)
            conv_opts.setdefault("r1_per_s_mM", ctx.config.r1_per_s_mM)
            conv_opts.setdefault("prepulse_to_readout_s", ctx.config.prepulse_to_readout_s)
            conv_opts.setdefault("baseline_frames", ctx.config.baseline_frames)
            t1_blood = np.full(t1.shape, blood_t1_ms, dtype=float)
            ct_blood = np.asarray(conv.convert(dce_signal, t1_blood, m0, **conv_opts),
                                  dtype=np.float32)
            if ct_blood.ndim == 3:
                ct_blood = ct_blood[..., None]
            if ct_blood.shape == ct_data.shape:
                ct_data = ct_blood
                _log.info("aif: converted AIF curve with fixed blood T1 = %.0f ms "
                          "(flow correction)", blood_t1_ms)

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
        wmh_flair = opts.pop("wmh_flair", None)   # optional: add a WMH region from FLAIR
        try:
            roi = provider.extract(t1_vol, t1_aff, out_dir=out_dir, **opts)
        except Exception as exc:                  # noqa: BLE001 — segmentation unavailable
            # The segmentation provider (e.g. SynthSeg) could not run — typically a
            # subject with no structural 3-D T1 (only IR+DCE), so there is no
            # anatomy to parcellate. Degrade gracefully to a whole-brain ROI on
            # the DCE grid: the subject still yields voxelwise + whole-brain-region
            # perfusion instead of failing outright. The mask is the thresholded,
            # largest-connected, hole-filled DCE baseline — no anatomy, no
            # interpolation. (Detailed parcellation is impossible without a
            # structural T1; whole-brain is the honest fallback.)
            from pbrain.tissue_roi.base import TissueROI
            _log.warning("tissue_roi: provider '%s' failed (%s: %s); falling back "
                         "to a whole-brain DCE-space ROI (no structural T1)",
                         plugin_key, type(exc).__name__,
                         str(exc).splitlines()[0][:120] if str(exc) else "")
            dce_img = nib.load(str(load_mf.outputs["dce"]))
            dce = np.asarray(dce_img.dataobj, dtype=np.float32)
            base = dce[..., :min(10, dce.shape[-1])].mean(-1) if dce.ndim == 4 else dce
            p99 = float(np.nanpercentile(base, 99)) if np.isfinite(base).any() else 0.0
            mask = (base > 0.1 * p99).astype(np.int16)
            try:
                from scipy import ndimage
                lab, n = ndimage.label(mask)
                if n > 1:                          # keep the largest blob (the head)
                    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
                    mask = (lab == int(np.argmax(sizes)) + 1).astype(np.int16)
                mask = ndimage.binary_fill_holes(mask).astype(np.int16)
            except Exception:                      # noqa: BLE001 — mask cleanup optional
                pass
            roi = TissueROI(parcellation=mask,
                            affine=np.asarray(dce_img.affine, dtype=float),
                            labels={1: "brain"}, region_map={"brain": [1]},
                            meta={"fallback": "whole_brain_dce",
                                  "reason": type(exc).__name__})

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

        # Optional WMH region: threshold the FLAIR within white matter and give
        # those lesion voxels their own label (9000). They are thereby removed
        # from WM (so the WM region becomes normal-appearing WM) and become a
        # region "WMH" that the kinetic fit + aggregation treat like any other —
        # enabling a WMH-vs-other-regions cohort comparison.
        region_map = dict(roi.region_map or {})
        if wmh_flair and Path(wmh_flair).exists():
            from nibabel.processing import resample_from_to
            from scipy import ndimage
            wm_labels = region_map.get("WM", [2, 41, 77, 251, 252, 253, 254, 255])
            # Detect WMH at the FLAIR's *native* resolution (lesions are
            # partial-volumed away at the 9.5 mm DCE slice thickness), then map
            # the lesion fraction down onto the DCE grid. The WM mask at FLAIR
            # resolution comes from resampling the parcellation up.
            flimg = nib.load(str(wmh_flair))
            fl = np.asarray(flimg.dataobj, dtype=float)
            if fl.ndim > 3:
                fl = fl[..., 0]
            wm_hr = np.isin(np.asarray(resample_from_to(
                nib.Nifti1Image(parc, np.asarray(parc_affine, dtype=float)),
                (fl.shape, flimg.affine), order=0).dataobj, dtype=np.int32), wm_labels)
            wm_dce = np.isin(parc, wm_labels)
            if wm_hr.sum() > 2000 and wm_dce.sum() > 500:
                wv = fl[wm_hr]; med = float(np.median(wv))
                mad = 1.4826 * float(np.median(np.abs(wv - med))) + 1e-6
                wmh_hr = wm_hr & ((fl - med) / mad > 2.0) & ((fl - med) / mad < 12.0)
                wmh_hr = ndimage.binary_opening(wmh_hr, iterations=1).astype(np.float32)
                frac = np.asarray(resample_from_to(
                    nib.Nifti1Image(wmh_hr, flimg.affine),
                    (parc.shape, parc_affine), order=1).dataobj, dtype=float)
                wmh = wm_dce & (frac > 0.2)        # ≥20 % of the DCE voxel is lesion
                parc[wmh] = 9000
                region_map["WMH"] = [9000]
                roi.labels[9000] = "WM-hyperintensity"
                _log.info("tissue_roi: added WMH region — %d DCE voxels "
                          "(%d at FLAIR res)", int(wmh.sum()), int(wmh_hr.sum()))

        parc_out = ctx.path_scheme.output_path(
            ctx.subject_dir, self.name, plugin_key, None, "parcellation", "nii.gz"
        )
        _save_nifti(parc, parc_affine, parc_out)
        labels_out = ctx.path_scheme.output_path(
            ctx.subject_dir, self.name, plugin_key, None, "labels", "json"
        )
        _save_json(
            {"labels": {str(k): v for k, v in roi.labels.items()},
             "region_map": region_map,
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
        opts.setdefault("baseline_frames", ctx.config.baseline_frames)

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


def _trailing_drop_frames(curves: list[np.ndarray], n: int, *,
                          max_drop: int = 3, k_mad: float = 4.0,
                          frac: float = 0.5) -> int:
    """Count trailing frames to drop because the curve **crashes** at its end.

    A final-frame motion / coil-removal artefact makes the last point(s) of a
    DCE curve fall abruptly toward zero, far below the washout tail (the user's
    "the single last point gets very, very low"). This generalises the legacy
    ``identify_drop_points`` (fit the tail, flag samples that fall below it) but
    is restricted to the **contiguous trailing run** and is deliberately
    conservative — a frame is a drop only if, for the AIF or the mean tissue
    curve, it lies both > ``k_mad`` robust-SDs *and* below ``frac``× the tail
    median. Normal tail noise is never trimmed; interior frames are untouched.
    Returns 0 when nothing crashes.
    """
    if n < 20:
        return 0
    tail_lo = max(int(n * 0.5), 3)
    ndrop = 0
    for k in range(1, int(max_drop) + 1):
        idx = n - k                                  # candidate trailing frame
        is_drop = False
        for s in curves:
            s = np.asarray(s, dtype=float)
            tail = s[tail_lo:n - int(max_drop)]      # stable tail (excl. candidates)
            tail = tail[np.isfinite(tail)]
            if tail.size < 5:
                continue
            med = float(np.median(tail))
            mad = float(np.median(np.abs(tail - med))) * 1.4826 + 1e-9
            v = s[idx]
            if np.isfinite(v) and v < med - k_mad * mad and v < frac * med:
                is_drop = True
                break
        if is_drop:
            ndrop = k
        else:
            break
    return ndrop


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

        # Trailing-drop guard: a final-frame motion / coil-removal artefact makes
        # the last point(s) of the curves crash toward zero, biasing every fit.
        # Detect it on the AIF and the mean tissue curve and DROP those trailing
        # frames globally (t, Cₜ and Cₐ together stay aligned) so all models
        # simply fit one frame less — never interior frames, never interpolation.
        n_common = int(min(C.shape[-1], ca.shape[0], t_s.shape[0]))
        ct_mean = np.nanmean(C.reshape(-1, C.shape[-1])[:, :n_common], axis=0)
        n_drop = _trailing_drop_frames([ca[:n_common], ct_mean], n_common)
        if n_drop > 0:
            keep = n_common - n_drop
            _log.info("kinetic: dropping %d trailing artefact frame(s) "
                      "(abrupt final-point crash) before fitting", n_drop)
            C = C[..., :keep]
            ca = ca[:keep]
            t_s = t_s[:keep]

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
            # `--opt models.<key>.level=parcel` forces average-then-fit per
            # parcel even for a voxelwise-capable model (faster, higher-SNR
            # per-region curves — used for the region/WMH cohort comparison).
            force_parcel = str(opts.pop("level", "")).lower() == "parcel"
            # `also_parcel=true` keeps the full voxelwise maps AND additionally
            # paints the parcel-level fit-of-average (higher-SNR per-region
            # curves) — so you get spatial maps *and* the robust region/WMH
            # comparison from one run.
            also_parcel = str(opts.pop("also_parcel", "")).lower() in ("true", "1", "yes")
            use_voxelwise = getattr(model, "supports_voxelwise", True) and not force_parcel

            if use_voxelwise:
                inputs = CurveInputs(
                    c_tissue=c_t, c_input=ca, t_s=t_s, mask=voxel_mask,
                )
                call_opts = dict(opts)
                # Pin gamma's flow F to the pipeline's Tikhonov CBF (the two-stage
                # design) by passing it as a per-voxel f_cbf (1/s), so gamma's
                # VARPRO uses the pipeline CBF instead of recomputing its own.
                if (pin_cbf_map is not None and "f_cbf" not in call_opts
                        and (model_key == "gamma"
                             or "f_cbf" in (getattr(model, "accepts", {}) or {}))):
                    call_opts["f_cbf"] = np.asarray(pin_cbf_map, dtype=float).ravel() / 6000.0
                result = model.fit(inputs, **call_opts)
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
            level = "voxelwise" if use_voxelwise else "parcel"
            for name, arr in shaped.items():
                p = ctx.path_scheme.output_path(
                    ctx.subject_dir, self.name, model_key, level, name, "nii.gz"
                )
                _save_nifti(arr, affine, p)
                artefacts[f"{model_key}/{name}"] = p

            # Additionally paint the parcel-level fit-of-average (region/WMH
            # comparison), keeping the voxelwise maps above for spatial work.
            if use_voxelwise and also_parcel and parc_arr is not None:
                pr = _fit_parcel_painted(model, C, ca, t_s, parc_arr, opts, cbf_map=pin_cbf_map)
                for name, arr in pr.maps.items():
                    pp = ctx.path_scheme.output_path(
                        ctx.subject_dir, self.name, model_key, "parcel", name, "nii.gz")
                    _save_nifti(np.asarray(arr, dtype=float), affine, pp)
                    artefacts[f"{model_key}/parcel/{name}"] = pp
                _log.info("kinetic: %s also painted at parcel level (region comparison)", model_key)

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
    # signal_to_conc/aif/load are needed only by the median_curve aggregator
    # (it pools raw concentration curves + the AIF instead of reducing the
    # voxelwise parameter maps); they are already upstream so requiring them
    # is free.
    requires: tuple[str, ...] = ("kinetic", "tissue_roi",
                                 "load", "signal_to_conc", "aif")
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

        # The median_curve aggregator fits pooled ROI curves rather than
        # reducing voxelwise maps, so it needs the raw concentration 4-D, the
        # (un-normalised) AIF curve and the time axis. Load them once, lazily.
        curve_inputs: dict[str, Any] = {}
        if "median_curve" in ctx.config.analysis_levels:
            stc_mf = ctx.upstream_manifests["signal_to_conc"]
            aif_mf = ctx.upstream_manifests["aif"]
            load_mf = ctx.upstream_manifests["load"]
            curve_inputs = {
                "concentration_4d": np.asarray(
                    nib.load(str(stc_mf.outputs["concentration"])).dataobj, dtype=float),
                "c_a": np.load(aif_mf.outputs["aif_signal"]),
                "t_s": np.load(load_mf.outputs["dce_time_s"]),
            }

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
                    # median_curve fits a *kinetic* model on pooled curves;
                    # it has no meaning for diffusion outputs.
                    if agg_key == "median_curve" and namespace != "kinetic":
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
                    if agg_key == "median_curve":
                        opts.update(curve_inputs)
                        opts["model"] = MODELS.get(model_key)
                        opts["model_opts"] = dict(
                            ctx.config.options_for("models", model_key))
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
