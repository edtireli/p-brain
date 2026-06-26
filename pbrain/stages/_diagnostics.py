"""Stage that renders model-specific diagnostic plots after the kinetic stage.

For every (configured kinetic model) × (curve set) pair:

* ``voxel``  – pick one representative voxel per model (median of the model's
               primary map; the model declares this via the optional
               ``primary_map: ClassVar[str]`` attribute, defaulting to
               ``model.outputs[0]``) and render its diagnostic.
* ``tissue`` – for each tissue class in the parcellation's region_map,
               average voxel curves inside that class to one ROI-mean curve
               and render its diagnostic.
* ``parcel`` – same but per individual parcel label (only parcels with
               more than ``min_parcel_voxels`` covered).

Per-curve diagnostic resolution (most-specific first):

  1. ``model.diagnose(ctx)`` if the kinetic model defines that method
     (single-file models keep all their plotting logic alongside the math).
  2. ``DIAGNOSTICS[model.key].plot(ctx)`` if a separate diagnostic plug-in
     is registered.
  3. ``pbrain.diagnostics._generic.generic_diagnostic(ctx, model)`` —
     model-agnostic fallback: re-fits the model on the single curve and
     renders AIF + Cₜ + a parameter table.

This three-tier chain makes the pipeline fully blind to model identity —
a new model dropped into ``pbrain/models/`` runs through the entire
voxel/tissue/parcel diagnostic loop with zero extra glue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pbrain.core import Stage, StageContext, StageOutput, get_logger
from pbrain.diagnostics import REGISTRY as DIAGNOSTICS, DiagnosticContext
from pbrain.diagnostics._generic import generic_diagnostic
from pbrain.models import REGISTRY as MODELS

_log = get_logger("diagnostics")


def _resolve_diagnostic(model: Any) -> tuple[Callable[[DiagnosticContext], None], str]:
    """Pick the right callable to render a diagnostic for ``model``.

    Returns ``(callable, source)`` where ``source`` is one of
    ``"model.diagnose"``, ``"diagnostics.<key>"`` or ``"generic"`` (for
    logging and metadata).
    """
    key = getattr(model, "key", "?")
    own = getattr(model, "diagnose", None)
    if callable(own):
        return own, "model.diagnose"
    plug = DIAGNOSTICS.get(key)
    if plug is not None and callable(getattr(plug, "plot", None)):
        return plug.plot, f"diagnostics.{key}"
    return (lambda ctx: generic_diagnostic(ctx, model)), "generic"


def _pick_primary_map(model: Any, output_names: list[str]) -> str | None:
    """Choose which output map represents the model for voxel-rep picking.

    Precedence: explicit ``model.primary_map`` attr → first output the model
    declared (``model.outputs[0]``) → first map present in the kinetic
    manifest → ``None`` (skip voxel diagnostic).
    """
    explicit = getattr(model, "primary_map", None)
    if isinstance(explicit, str) and explicit in output_names:
        return explicit
    declared = getattr(model, "outputs", ()) or ()
    for nm in declared:
        if nm in output_names:
            return nm
    return output_names[0] if output_names else None


@dataclass
class DiagnosticsStage:
    """Final stage — render per-model diagnostic plots.

    For each fitted model, renders the model's diagnostic (single-file
    ``diagnose``, a ``pbrain/diagnostics/<key>.py`` plug-in, or the
    generic fallback) at the representative voxel, each tissue class, and
    each parcel. Writes PNG montages alongside the maps.
    """

    name: str = "diagnostics"
    requires: tuple[str, ...] = (
        "load", "t1_m0", "signal_to_conc", "normalisation", "aif", "kinetic",
        "tissue_roi", "diffusion",
    )
    plugin_key: str = "multi"

    def run(self, ctx: StageContext) -> StageOutput:
        """Render and write the diagnostic plots for every model/level."""
        import nibabel as nib

        norm_mf = ctx.upstream_manifests["normalisation"]
        load_mf = ctx.upstream_manifests["load"]
        kinetic_mf = ctx.upstream_manifests["kinetic"]
        roi_mf = ctx.upstream_manifests["tissue_roi"]

        C_img = nib.load(str(norm_mf.outputs["ct_normalised"]))
        C = np.asarray(C_img.dataobj, dtype=np.float32)
        ca = np.load(norm_mf.outputs["aif_normalised"])
        t_s = np.load(load_mf.outputs["dce_time_s"])

        parc = np.asarray(
            nib.load(str(roi_mf.outputs["parcellation"])).dataobj, dtype=np.int32
        )
        with open(roi_mf.outputs["labels"], encoding="utf-8") as f:
            labels_blob = json.load(f)
        labels = {int(k): v for k, v in labels_blob.get("labels", {}).items()}
        region_map = labels_blob.get("region_map") or {}

        opts = dict(ctx.config.options_for("diagnostics", self.plugin_key))
        min_parcel_voxels = int(opts.get("min_parcel_voxels", 50))

        artefacts: dict[str, Path] = {}
        per_model_source: dict[str, str] = {}

        # ── conversion QC: signal→concentration sanity figure, every run ──
        # Renders the *raw* concentration volume (what the user opens), so the
        # out-of-brain saturated voxels and the low tissue scale are visible at
        # a glance — not the normalised C used for fitting.
        try:
            from pbrain.diagnostics.conversion import render_conversion_qc
            stc_mf = ctx.upstream_manifests["signal_to_conc"]
            conc_raw = np.asarray(
                nib.load(str(stc_mf.outputs["concentration"])).dataobj, dtype=np.float32)
            dce_sig = np.asarray(
                nib.load(str(load_mf.outputs["dce"])).dataobj, dtype=np.float32)
            if dce_sig.ndim == 3:
                dce_sig = dce_sig[..., None]
            t1_mf = ctx.upstream_manifests.get("t1_m0")
            t1_map = None
            if t1_mf is not None and "t1_map_ms" in t1_mf.outputs:
                t1_map = np.asarray(
                    nib.load(str(t1_mf.outputs["t1_map_ms"])).dataobj, dtype=float)
            qc_out = ctx.path_scheme.aggregation_dir(
                ctx.subject_dir, "signal_to_conc", ctx.config.signal_to_conc, "diagnostics"
            ) / "conversion_qc.png"
            render_conversion_qc(conc_raw, dce_sig, ca, t_s, parc > 0, qc_out,
                                 t1_map=t1_map, baseline_frames=ctx.config.baseline_frames)
            artefacts["conversion_qc"] = qc_out
            _log.info("conversion QC -> %s", qc_out)
        except Exception as exc:  # noqa: BLE001
            _log.warning("conversion QC failed: %s", exc)

        for model_key in ctx.config.kinetic_models:
            model = MODELS.get(model_key)
            if model is None:
                # Should never happen — KineticStage would have failed first.
                continue
            diag_fn, source = _resolve_diagnostic(model)
            per_model_source[model_key] = source
            model_opts = dict(ctx.config.options_for("models", model_key))

            model_meta = kinetic_mf.metadata.get("per_model", {}).get(model_key, {})
            output_names = list(model_meta.get("outputs", []))
            primary_name = _pick_primary_map(model, output_names)

            # ── sample voxel: median of the primary map within the parcellation
            voxel_dir = ctx.path_scheme.aggregation_dir(
                ctx.subject_dir, "kinetic", model_key, "diagnostics/voxel"
            )
            if primary_name is not None:
                primary_path = kinetic_mf.outputs.get(f"{model_key}/{primary_name}")
                if primary_path is not None:
                    primary_map = np.asarray(
                        nib.load(str(primary_path)).dataobj, dtype=float
                    )
                    finite = np.isfinite(primary_map) & (parc > 0)
                    if finite.any():
                        vals = primary_map[finite]
                        idx = int(np.argmin(np.abs(vals - np.median(vals))))
                        coords = np.argwhere(finite)[idx]
                        vx, vy, vz = int(coords[0]), int(coords[1]), int(coords[2])
                        out = voxel_dir / f"voxel_{vx}_{vy}_{vz}.png"
                        ctx_d = DiagnosticContext(
                            label=(f"voxel ({vx},{vy},{vz})  "
                                   f"{primary_name}={float(primary_map[vx,vy,vz]):.4f}"),
                            c_tissue=C[vx, vy, vz, :],
                            c_input=ca, t_s=t_s,
                            out_path=out, model_opts=model_opts,
                            extras={"primary_map": primary_name, "source": source},
                        )
                        try:
                            diag_fn(ctx_d)
                            artefacts[f"{model_key}/voxel"] = out
                        except Exception as exc:  # noqa: BLE001
                            _log.warning("diagnostic voxel (%s, %s) failed: %s", model_key, source, exc)

            # ── tissue-level diagnostics (one per region in region_map)
            tissue_dir = ctx.path_scheme.aggregation_dir(
                ctx.subject_dir, "kinetic", model_key, "diagnostics/tissue"
            )
            for region_name, region_label_ids in region_map.items():
                ids = list(region_label_ids)
                if not ids:
                    continue
                mask = np.isin(parc, ids)
                if int(mask.sum()) < min_parcel_voxels:
                    continue
                roi_curve = np.nanmedian(C[mask], axis=0)
                out = tissue_dir / f"{region_name}.png"
                ctx_d = DiagnosticContext(
                    label=f"tissue {region_name}  (n={int(mask.sum())} voxels, median curve)",
                    c_tissue=roi_curve, c_input=ca, t_s=t_s,
                    out_path=out, model_opts=model_opts,
                    extras={"source": source},
                )
                try:
                    diag_fn(ctx_d)
                    artefacts[f"{model_key}/tissue/{region_name}"] = out
                except Exception as exc:  # noqa: BLE001
                    _log.warning("diagnostic tissue %s (%s, %s) failed: %s", region_name, model_key, source, exc)

            # ── parcel-level diagnostics (one per individual label)
            parcel_dir = ctx.path_scheme.aggregation_dir(
                ctx.subject_dir, "kinetic", model_key, "diagnostics/parcel"
            )
            unique_labels = [int(L) for L in np.unique(parc) if int(L) > 0]
            for L in unique_labels:
                mask = (parc == L)
                if int(mask.sum()) < min_parcel_voxels:
                    continue
                roi_curve = np.nanmedian(C[mask], axis=0)
                name = labels.get(L, f"label_{L}")
                safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
                out = parcel_dir / f"{L:04d}_{safe}.png"
                ctx_d = DiagnosticContext(
                    label=f"parcel {L} {name}  (n={int(mask.sum())})",
                    c_tissue=roi_curve, c_input=ca, t_s=t_s,
                    out_path=out, model_opts=model_opts,
                    extras={"source": source},
                )
                try:
                    diag_fn(ctx_d)
                    artefacts[f"{model_key}/parcel/{L}"] = out
                except Exception as exc:  # noqa: BLE001
                    _log.warning("diagnostic parcel %s (%s, %s) failed: %s", L, model_key, source, exc)

        # ── parameter-map montages (voxel / tissue / parcel per map) ──
        # The independent, model-agnostic generator runs on every map a model
        # produced — kinetic and diffusion — in the article style. These are
        # the figures shown in the README.
        from pbrain.diagnostics.montage import render_levels
        max_slices = int(opts.get("montage_max_slices", 16))
        units_by_map: dict[str, str] = {}
        for mk in ctx.config.kinetic_models:
            units_by_map.update(
                (kinetic_mf.metadata.get("per_model", {}).get(mk, {}) or {}).get("units", {}) or {})
        montage_sources = [("kinetic", kinetic_mf)]
        diff_mf = ctx.upstream_manifests.get("diffusion")
        if diff_mf is not None:
            montage_sources.append(("diffusion", diff_mf))
            for mk in (diff_mf.metadata.get("models") or []):
                units_by_map.update(
                    (diff_mf.metadata.get("per_model", {}).get(mk, {}) or {}).get("units", {}) or {})

        n_montage = 0
        for ns, mf in montage_sources:
            for out_key, path in mf.outputs.items():
                if not str(path).endswith(".nii.gz"):
                    continue
                try:
                    vol = np.asarray(nib.load(str(path)).dataobj, dtype=float)
                    if vol.ndim != 3:
                        continue
                    model_key, map_name = out_key.split("/")[0], out_key.split("/")[-1]
                    out_dir = ctx.path_scheme.aggregation_dir(
                        ctx.subject_dir, "kinetic", model_key, "diagnostics/montage")
                    # A parcel-painted model (average-then-fit) has no true
                    # per-voxel map — render only tissue/parcel montages so
                    # nothing is mislabelled "voxel".
                    mlevel = (kinetic_mf.metadata.get("per_model", {})
                              .get(model_key, {}).get("level", "voxelwise"))
                    levels = (("voxel", "tissue", "parcel") if mlevel == "voxelwise"
                              else ("tissue", "parcel"))
                    written = render_levels(
                        vol, out_dir, map_name,
                        parc=parc if parc.shape == vol.shape else None,
                        region_map=region_map,
                        title="",                       # clean, article style
                        units=units_by_map.get(map_name, ""),
                        max_slices=max_slices, levels=levels)
                    for level, p in written.items():
                        artefacts[f"montage/{ns}/{model_key}/{map_name}/{level}"] = p
                    n_montage += len(written)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("montage %s failed: %s", out_key, exc)

        return StageOutput(
            artefacts=artefacts,
            metadata={
                "n_plots": len(artefacts),
                "n_montages": n_montage,
                "min_parcel_voxels": min_parcel_voxels,
                "diagnostic_source_per_model": per_model_source,
            },
        )
