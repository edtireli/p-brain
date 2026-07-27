"""Bruker ParaVision layout adapter — withheld from the released package.

Lifted verbatim out of ``pbrain/io/layout.py``. Re-insert it there and add
``BrukerLayout()`` back to ``ADAPTERS`` to restore preclinical auto-detection.
"""

# ---------------------------------------------------------------- Bruker
class BrukerLayout:
    """A Bruker ParaVision study: numbered scan dirs each holding ``pdata/<n>/2dseq``.

    Roles are classified from the acquisition headers (see
    :mod:`pbrain.io.bruker_study`), so preclinical studies need no hand-written
    config. The DCE and the anatomical are handed over as *scan directories* —
    the Bruker loader reads them natively — while the flip-angle set is stacked
    into one NIfTI (+ ``FlipAngles`` sidecar) under the study's ``pbrain/`` folder,
    because a relaxometry series has to arrive as a single 4-D input."""
    name = "bruker"

    @staticmethod
    def _scans(path: Path) -> list[Path]:
        if not path.is_dir():
            return []
        return [d for d in path.iterdir()
                if d.is_dir() and d.name.isdigit() and (d / "pdata").is_dir()]

    def is_subject(self, path: Path) -> bool:
        for scan in self._scans(path):
            for reco in (scan / "pdata").iterdir():
                if (reco / "2dseq").exists():
                    return True
        return False

    def find_subjects(self, root: Path) -> list[Path]:
        return _child_subjects(root, self.is_subject)

    def resolve_inputs(self, subject: Path) -> dict:
        from pbrain.io.bruker_study import assemble_vfa, classify, index_study
        roles = classify(index_study(subject))
        ir = None
        if len(roles.vfa) >= 2:
            try:
                ir = assemble_vfa(roles, subject / "pbrain" / "vfa.nii.gz")
            except Exception:                    # noqa: BLE001 — a bad VFA must not
                ir = None                        # block the run; t1_m0 will say so
        return {
            "dce": roles.dce.path if roles.dce else None,
            "ir": ir,
            "t1": roles.anat.path if roles.anat else None,
        }


# ---------------------------------------------------------------- flat NIfTI
