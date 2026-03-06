import os
import shutil
import time
from pathlib import Path

import utils.settings as settings


PBRAIN_WAITING_FOR_ROI_EXIT_CODE = 42


class InputFunctionUserInteractionRequired(RuntimeError):
    """Raised when input-function extraction requires user ROI input."""

    def __init__(self, message: str, *, missing: list[str] | None = None):
        super().__init__(message)
        self.missing = list(missing or [])


def _ai_missing_fallback_mode() -> str:
    """Return configured fallback when AI cannot find AIF/VIF.

    - deterministic (default): best-effort deterministic extraction.
    - roi: stop and request user-provided ROI masks/curves.
    """

    raw = (os.getenv("P_BRAIN_AI_INPUT_MISSING_FALLBACK") or "deterministic").strip().lower()
    if raw in {"roi", "user_roi", "user", "manual"}:
        return "roi"
    return "deterministic"


def _getenv_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    val = str(raw).strip().lower()
    if val in {"0", "false", "no", "off"}:
        return False
    if val in {"1", "true", "yes", "on"}:
        return True
    return default


def _cleanup_input_function_artifacts(analysis_directory: str, *, preserve_roi_data: bool = False) -> None:
    """Remove prior AIF/VIF + TSCC artifacts so current run selects only new outputs.

    When *preserve_roi_data* is ``True`` the ``ROI Data`` directory is kept intact.
    This is required for file-ROI mode where user-saved voxels live there.
    """

    root = Path(analysis_directory)
    targets = [
        Path("CTC Data") / "Artery",
        Path("CTC Data") / "Vein",
        Path("ITC Data") / "Artery",
        Path("ITC Data") / "Vein",
        Path("TSCC Data"),
        Path("Frame Data"),
        Path("ROI NIfTI"),
    ]
    if not preserve_roi_data:
        targets.append(Path("ROI Data"))
    for rel in targets:
        p = root / rel
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
        except Exception:
            # Best-effort cleanup; never block the pipeline on stale artifact removal.
            pass


def run_input_function(
    analysis_directory,
    nifti_directory,
    image_directory,
    filenames,
    parameters,
):
    """Dispatch input-function extraction based on ROI method.

    - ROI_METHOD=manual -> interactive user-drawn ROIs (calls opt02 input_function).
    - ROI_METHOD=file -> load user-provided ROI voxel lists from files.
    - ROI_METHOD=deterministic -> deterministic vessel ROI extraction (no AI model import).
    - ROI_METHOD=ai -> AI model-based input function extraction.
    """

    roi_method = (getattr(settings, "ROI_METHOD", "ai") or "ai").strip().lower()

    # Manual ROI mode: skip all automated logic and run the interactive ROI selector.
    if roi_method == "manual":
        from modules.opt02_input_functions import input_function as _input_function_interactive

        return _input_function_interactive(
            analysis_directory,
            nifti_directory,
            image_directory,
            filenames,
            parameters,
        )

    if _getenv_bool("P_BRAIN_INPUT_FUNCTIONS_PREFER_NEW", default=True):
        # In file-ROI mode the user has already saved ROI voxels to ROI Data;
        # we must preserve that directory.  Clean up derived artifacts only.
        _cleanup_input_function_artifacts(
            str(analysis_directory),
            preserve_roi_data=(roi_method == "file"),
        )

    def _has_any_npy(path: Path) -> bool:
        try:
            if not path.exists():
                return False
            for _ in path.rglob("*.npy"):
                return True
            return False
        except Exception:
            return False

    def _ai_has_artery_outputs(analysis_dir: Path) -> bool:
        return _has_any_npy(analysis_dir / "CTC Data" / "Artery")

    def _ai_has_sss_outputs(analysis_dir: Path) -> bool:
        return _has_any_npy(analysis_dir / "CTC Data" / "Vein" / "Sinus Sagittalis")

    def _rotate_sss_artifacts_cw(tmp_root: Path) -> None:
        """Rotate deterministic SSS artifacts into the AI frame.

        AI pipeline rotates DCE data with np.rot90(k=-1, axes=(0,1)) before ROI selection.
        If we only copy SSS from deterministic into an AI run, the ROI coordinates/mask
        otherwise end up in a different in-memory frame than the arterial ROI.
        """

        try:
            import numpy as np  # type: ignore
            import nibabel as nib  # type: ignore
        except Exception:
            return

        # 1) Rotate ROI voxel lists (ROI Data) if present.
        roi_dir = tmp_root / "ROI Data" / "Vein" / "Sinus Sagittalis"
        if roi_dir.is_dir():
            for p in roi_dir.glob("ROI_voxels_slice_*.npy"):
                try:
                    vox = np.load(str(p))
                    if not hasattr(vox, "shape") or vox.ndim != 2 or vox.shape[1] != 2:
                        continue
                    vox = np.asarray(vox, dtype=np.int64)

                    # Use any available mask (preferred) to infer (H,W); otherwise infer from vox.
                    h = w = None
                    mask_dir = tmp_root / "ROI NIfTI"
                    if mask_dir.is_dir():
                        for mp in mask_dir.glob("Vein__*Sagittalis*__mask.nii*"):
                            try:
                                img = nib.load(str(mp))
                                shape = img.shape
                                if len(shape) >= 2:
                                    h = int(shape[0])
                                    w = int(shape[1])
                                    break
                            except Exception:
                                continue
                    if h is None or w is None:
                        h = int(vox[:, 0].max()) + 1 if vox.size else 0
                        w = int(vox[:, 1].max()) + 1 if vox.size else 0
                    if h <= 0 or w <= 0:
                        continue

                    # Clockwise rotation mapping for np.rot90(k=-1): (r,c) -> (c, h-1-r)
                    r = vox[:, 0]
                    c = vox[:, 1]
                    rr = c
                    cc = (h - 1) - r
                    rot = np.stack([rr, cc], axis=1)
                    np.save(str(p), rot.astype(np.int16, copy=False))
                except Exception:
                    continue

        # 2) Rotate ROI masks (ROI NIfTI) if present.
        mask_dir = tmp_root / "ROI NIfTI"
        if mask_dir.is_dir():
            for mp in mask_dir.glob("Vein__*Sagittalis*__mask.nii*"):
                try:
                    img = nib.load(str(mp))
                    data = np.asanyarray(img.dataobj)
                    if data.ndim < 3:
                        continue
                    # Rotate in-plane axes (0,1) for each slice.
                    rot = np.rot90(data, k=-1, axes=(0, 1))
                    out = nib.Nifti1Image(rot.astype(np.uint8, copy=False), affine=img.affine, header=img.header)
                    nib.save(out, str(mp))
                except Exception:
                    continue

    def _copy_selected_outputs(
        tmp_root: Path,
        dst_root: Path,
        *,
        copy_artery: bool,
        copy_sss: bool,
        rotate_sss_to_ai_frame: bool,
    ) -> None:
        # If we are mixing AI artery + deterministic SSS, rotate the deterministic SSS artifacts first.
        if copy_sss and rotate_sss_to_ai_frame:
            _rotate_sss_artifacts_cw(tmp_root)

        # Copy only what we need so we don't override AI outputs unnecessarily.
        mappings: list[tuple[Path, Path]] = []
        if copy_artery:
            mappings.extend(
                [
                    (tmp_root / "CTC Data" / "Artery", dst_root / "CTC Data" / "Artery"),
                    (tmp_root / "ROI Data" / "Artery", dst_root / "ROI Data" / "Artery"),
                    (tmp_root / "Frame Data" / "Artery", dst_root / "Frame Data" / "Artery"),
                ]
            )
        if copy_sss:
            mappings.extend(
                [
                    (
                        tmp_root / "CTC Data" / "Vein" / "Sinus Sagittalis",
                        dst_root / "CTC Data" / "Vein" / "Sinus Sagittalis",
                    ),
                    (
                        tmp_root / "ROI Data" / "Vein" / "Sinus Sagittalis",
                        dst_root / "ROI Data" / "Vein" / "Sinus Sagittalis",
                    ),
                    (
                        tmp_root / "Frame Data" / "Vein" / "Sinus Sagittalis",
                        dst_root / "Frame Data" / "Vein" / "Sinus Sagittalis",
                    ),
                ]
            )

        for src, dst in mappings:
            try:
                if not src.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dst, dirs_exist_ok=True)
            except Exception:
                # Best-effort; a partial copy is still better than a hard failure.
                pass

        # Masks are helpful for UI inspection, but optional; copy only if present.
        try:
            src_masks = tmp_root / "ROI NIfTI"
            dst_masks = dst_root / "ROI NIfTI"
            if src_masks.is_dir():
                dst_masks.mkdir(parents=True, exist_ok=True)
                for p in src_masks.glob("*.nii*"):
                    name = p.name
                    # Filter to artery/SSS masks only.
                    if copy_artery and ("Artery__" in name):
                        shutil.copy2(p, dst_masks / name)
                    if copy_sss and ("Vein__Sinus_Sagittalis" in name or "Vein__Sinus Sagittalis" in name):
                        shutil.copy2(p, dst_masks / name)
        except Exception:
            pass

    if roi_method == "file":
        from modules.roi_file_input_functions import input_function_file

        return input_function_file(
            analysis_directory,
            nifti_directory,
            image_directory,
            filenames,
            parameters,
        )

    if roi_method in {"deterministic", "geometry"}:
        from modules.deterministic_input_functions import input_function_geometry

        return input_function_geometry(
            analysis_directory,
            nifti_directory,
            image_directory,
            filenames,
            parameters,
        )

    # ROI_METHOD=ai: attempt AI, then selectively fall back to deterministic for missing structures.
    # IMPORTANT: packaged environments may not include TensorFlow/Keras; treat missing AI deps as
    # a normal AI failure and proceed to configured fallback instead of hard-crashing.
    from modules.deterministic_input_functions import input_function_geometry

    analysis_root = Path(analysis_directory)
    ai_exc: Exception | None = None
    input_function_AI = None
    try:
        from modules.AI_input_functions import input_function_AI as _input_function_AI

        input_function_AI = _input_function_AI
    except Exception as e:
        ai_exc = e

    if input_function_AI is not None:
        try:
            input_function_AI(
                analysis_directory,
                nifti_directory,
                image_directory,
                filenames,
                parameters,
            )
        except Exception as e:
            ai_exc = e

    if ai_exc is not None and input_function_AI is None:
        try:
            print(f"[input_functions] AI unavailable; falling back ({type(ai_exc).__name__}: {ai_exc})")
        except Exception:
            pass
    elif ai_exc is not None:
        try:
            print(f"[input_functions] AI failed; falling back ({type(ai_exc).__name__}: {ai_exc})")
        except Exception:
            pass

    if ai_exc is not None:
        fallback_mode = _ai_missing_fallback_mode()
        if fallback_mode == "roi":
            try:
                print("PBRAIN_WAITING_STAGE=input_functions missing=AIF,VIF")
            except Exception:
                pass
            raise InputFunctionUserInteractionRequired(
                "AI input function extraction failed or is unavailable; waiting for user ROI input.",
                missing=["AIF", "VIF"],
            )

        # Full deterministic fallback.
        if _getenv_bool("P_BRAIN_INPUT_FUNCTIONS_PREFER_NEW", default=True):
            _cleanup_input_function_artifacts(str(analysis_directory))
        return input_function_geometry(
            analysis_directory,
            nifti_directory,
            image_directory,
            filenames,
            parameters,
        )

    has_artery = _ai_has_artery_outputs(analysis_root)
    has_sss = _ai_has_sss_outputs(analysis_root)

    fallback_mode = _ai_missing_fallback_mode()
    if fallback_mode == "roi" and (not has_artery or not has_sss):
        missing: list[str] = []
        if not has_artery:
            missing.append("AIF")
        if not has_sss:
            missing.append("VIF")
        # Emit a stable marker for p-brain-web log parsers.
        try:
            print(f"PBRAIN_WAITING_STAGE=input_functions missing={','.join(missing)}")
        except Exception:
            pass
        raise InputFunctionUserInteractionRequired(
            "AI could not find required AIF/VIF; waiting for user ROI input.",
            missing=missing,
        )

    # Rules:
    # - If AI misses arterial input -> deterministic should provide arterial output.
    # - If AI misses SSS -> deterministic should provide SSS only (do not override AI arterial).
    # - If AI misses both -> default to full deterministic.
    if has_artery and has_sss:
        return None

    if not has_artery and not has_sss:
        # Full deterministic fallback.
        if _getenv_bool("P_BRAIN_INPUT_FUNCTIONS_PREFER_NEW", default=True):
            _cleanup_input_function_artifacts(str(analysis_directory))
        return input_function_geometry(
            analysis_directory,
            nifti_directory,
            image_directory,
            filenames,
            parameters,
        )

    # Partial fallback: run deterministic in an isolated tmp dir and copy only the missing structure.
    tmp_root = analysis_root / ".pbrain_tmp_deterministic_roi" / str(int(time.time() * 1000))
    try:
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_analysis = str(tmp_root)
        tmp_images = str(tmp_root / "Images")
        os.makedirs(tmp_images, exist_ok=True)

        # Run deterministic once; it will compute both artery + SSS, but we only copy what we need.
        input_function_geometry(
            tmp_analysis,
            nifti_directory,
            tmp_images,
            filenames,
            parameters,
        )

        _copy_selected_outputs(
            tmp_root,
            analysis_root,
            copy_artery=(not has_artery),
            copy_sss=(not has_sss),
            rotate_sss_to_ai_frame=bool(has_artery) and (not bool(has_sss)),
        )
    finally:
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
            parent = tmp_root.parent
            # prune parent if empty
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent, ignore_errors=True)
        except Exception:
            pass

    return None
