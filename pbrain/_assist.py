"""Optional LLM assist for p-Brain — the human-interface edges only.

Hard rule: **the model never touches the numbers.** It reads metadata and text
(scan headers, run provenance, QC stats, error strings) to *suggest*, *summarise*,
or *explain* — never to compute a result. Anything it suggests that could change a
run (which series to use) is confirmed by the user and recorded in the manifest,
so a re-run is byte-identical.

Local-first and dependency-free: talks to Ollama over stdlib ``urllib`` (no new
dependency). If no model is reachable, every helper returns ``None`` and the
pipeline behaves exactly as before — the assist is purely additive and opt-in.
"""
from __future__ import annotations

import json
import os
import urllib.request

_URL = os.environ.get("PBRAIN_OLLAMA", "http://127.0.0.1:11434").rstrip("/")
# small, capable local models preferred; env override wins
_PREF = ("qwen3.6:latest", "qwen3:8b", "qwen2.5:14b", "llama3.2", "llama3.1", "mistral")


def _get(path: str, timeout: float = 4.0):
    with urllib.request.urlopen(_URL + path, timeout=timeout) as r:
        return json.loads(r.read())


def _post(path: str, payload: dict, timeout: float = 90.0):
    req = urllib.request.Request(_URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _tags() -> list[str]:
    try:
        return [m["name"] for m in _get("/api/tags").get("models", [])]
    except Exception:
        return []


def model() -> str | None:
    """The model to use: ``PBRAIN_ASSIST_MODEL`` if present, else the first
    preferred model installed, else any installed model, else None."""
    tags = _tags()
    if not tags:
        return None
    env = os.environ.get("PBRAIN_ASSIST_MODEL", "").strip()
    if env and any(t == env or t.split(":")[0] == env for t in tags):
        return next(t for t in tags if t == env or t.split(":")[0] == env)
    for pref in _PREF:                     # exact preferred tag first
        if pref in tags:
            return pref
    for pref in _PREF:                     # then any tag of a preferred family
        base = pref.split(":")[0]
        for t in tags:
            if t.split(":")[0] == base:
                return t
    return tags[0]


def available() -> bool:
    return model() is not None


def chat(system: str, user: str, *, want_json: bool = False, timeout: float = 90.0):
    """One call. Returns text, or a parsed dict when ``want_json``, or None on any
    failure (unreachable, timeout, bad JSON) — callers treat None as 'no assist'."""
    m = model()
    if not m:
        return None
    payload = {
        "model": m, "stream": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"temperature": 0.1},
    }
    if want_json:
        payload["format"] = "json"
    try:
        data = _post("/api/chat", payload, timeout)
        txt = (data.get("message") or {}).get("content", "").strip()
        if want_json:
            a, b = txt.find("{"), txt.rfind("}")
            return json.loads(txt[a:b + 1] if a >= 0 else txt)
        return txt or None
    except Exception:
        return None


# ---------------------------------------------------------------- features
_SERIES_SYS = (
    "You are a neuroimaging technologist mapping raw MRI series to their role in a "
    "DCE-MRI perfusion pipeline. You are given a table of scan series with their "
    "protocol name and acquisition parameters. Assign each pipeline ROLE to the most "
    "appropriate series id, reasoning ONLY from the parameters:\n"
    "- dce: the DYNAMIC contrast series — the SAME slab acquired many times "
    "(highest 'dynamics'), a fast T1-weighted gradient echo (T1TFE/T1FFE).\n"
    "- t1_anatomical: a single-dynamic high-resolution 3D T1-weighted volume "
    "(most slices, TFE/3D), pre-contrast (not 'Gd').\n"
    "- ir: the inversion-recovery T1-mapping series — one, or a set at several "
    "inversion times (protocol like 'TI_00120'..'TI_10000' or 'IR_LL'). If several, "
    "list all their ids.\n"
    "Ignore survey, angiography, flow (QFLOW), ASL/pcasl, FLAIR, T2, DWI/DTI, SWI, "
    "spectroscopy. Return ONLY JSON: "
    '{"dce": "<id>", "t1_anatomical": "<id or null>", "ir": ["<id>", ...], '
    '"why": "<one short sentence>"}.'
)


def _series_table_text(rows: list[dict]) -> str:
    return "\n".join(
        f"- {r['id']} (protocol {r.get('protocol', '')!r}): technique={r.get('technique', '')} "
        f"mode={r.get('mode', '')} dynamics={r.get('dynamics', '')} slices={r.get('slices', '')} "
        f"TR={r.get('tr_s', '')}" for r in rows)


def identify_series(rows: list[dict]) -> dict | None:
    """rows: [{id, protocol, technique, mode, dynamics, slices, tr_s, ti}]. Returns
    a proposed {dce, t1_anatomical, ir[], why} mapping — a SUGGESTION to confirm."""
    if not rows:
        return None
    return chat(_SERIES_SYS, "SERIES:\n" + _series_table_text(rows), want_json=True, timeout=180.0)


def revise_series(rows: list[dict], proposal: dict, feedback: str) -> dict | None:
    """Re-map after the user corrects the proposal in their own words (e.g. 'the T1
    should be the sagittal one — axt1w is a reformat, not the true T1'). Returns a
    revised mapping in the same JSON shape, or None on failure."""
    if not rows or not feedback.strip():
        return None
    user = ("SERIES:\n" + _series_table_text(rows) +
            f"\n\nYOUR PREVIOUS MAPPING: {json.dumps(proposal)}"
            f"\nUSER CORRECTION: {feedback.strip()}"
            "\n\nApply the correction and return the full corrected mapping in the same JSON format. "
            "The 'why' should note what changed.")
    return chat(_SERIES_SYS, user, want_json=True, timeout=180.0)


_LAYOUT_SYS = (
    "You map a folder of neuroimaging data to p-Brain's inputs from a FILE TREE (names "
    "only — never any pixel data). Decide whether it is ONE subject or a COHORT of "
    "subjects, and for each subject identify, from the file names:\n"
    "- dce: the dynamic-contrast / perfusion series (4D, many timepoints — 'dce', "
    "'perf', 'dsc', 'dynamic').\n"
    "- t1: a single 3D T1-weighted anatomical ('t1', 't1w', 'mprage', 'tfe'), not a reformat.\n"
    "- ir: an inversion-recovery / relaxometry / T1-mapping series if present ('ir', 'ti', "
    "'irt1', 'relax'); else null.\n"
    "Prefer NIfTI (.nii/.nii.gz). A subject may instead hold Philips PAR/REC files or a "
    "DICOM folder — p-Brain converts those, so point at the .par or the folder. Paths must "
    "be RELATIVE to the given root; for a single subject use dir '.'.\n"
    'Return ONLY JSON: {"kind":"subject|cohort","subjects":[{"dir":"<rel>","dce":"<rel>",'
    '"t1":"<rel|null>","ir":"<rel|null>"}], "why":"<one short sentence>"}'
)


def propose_layout(tree_text: str) -> dict | None:
    """From a name-only file tree, propose whether it's a subject or cohort and each
    subject's {dce, t1, ir}. A SUGGESTION to confirm and freeze — never computation."""
    if not tree_text.strip():
        return None
    return chat(_LAYOUT_SYS, "ROOT FILE TREE:\n" + tree_text, want_json=True, timeout=180.0)


def revise_layout(tree_text: str, proposal: dict, feedback: str) -> dict | None:
    """Re-map the folder after the user corrects the layout proposal in their words."""
    if not tree_text.strip() or not feedback.strip():
        return None
    user = ("ROOT FILE TREE:\n" + tree_text +
            f"\n\nYOUR PREVIOUS LAYOUT: {json.dumps(proposal)}"
            f"\nUSER CORRECTION: {feedback.strip()}"
            "\n\nApply the correction and return the full corrected layout in the same JSON format.")
    return chat(_LAYOUT_SYS, user, want_json=True, timeout=180.0)


_QC_SYS = (
    "You are a DCE-MRI quality-control assistant. Given per-stage QC facts from one "
    "completed run, write a SHORT plain-English summary (3-5 sentences) a researcher "
    "can skim: what looks healthy, and any warnings worth checking. Base every "
    "statement on the numbers given — do not invent values. This is ADVISORY quality "
    "feedback, never a clinical diagnosis. No preamble."
)


def summarize_qc(facts: str) -> str | None:
    if not facts.strip():
        return None
    return chat(_QC_SYS, "QC FACTS:\n" + facts)


_METHODS_SYS = (
    "You are drafting the Methods sentence(s) for a paper from a DCE-MRI analysis "
    "run's provenance (the exact plug-ins and parameters used). Write 2-4 precise, "
    "publication-style sentences in past tense, naming the pipeline (p-Brain), the "
    "AIF method, the kinetic model(s), and key parameters. Only state what the "
    "provenance contains. No citations placeholder, no bullet points."
)


def methods_text(provenance: str) -> str | None:
    if not provenance.strip():
        return None
    return chat(_METHODS_SYS, "PROVENANCE:\n" + provenance)


_ERROR_SYS = (
    "You are p-Brain's troubleshooting assistant. A pipeline stage failed. From the "
    "stage name, the error, and context, explain in 2-3 sentences what most likely "
    "went wrong and the concrete fix (a flag, an install, an input). Be specific and "
    "practical; if the cause is genuinely unclear, say so. No preamble."
)


def explain_error(stage: str, error: str, context: str = "") -> str | None:
    return chat(_ERROR_SYS, f"STAGE: {stage}\nERROR: {error}\nCONTEXT: {context}")
