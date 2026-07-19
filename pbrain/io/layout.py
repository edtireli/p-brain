"""Layout resolution — turn *a path* into subjects and their inputs, across formats.

`pbrain run <path>` should work whether ``<path>`` is one subject or a folder of
them, and whatever the on-disk convention (raw Philips PAR/REC, flat NIfTI, BIDS).
A :class:`LayoutAdapter` recognises one convention; :func:`resolve` picks the
adapter and decides **single subject vs cohort** by probing:

1. does ``<path>`` *itself* hold a subject's scans?  → single subject
2. else, do its sub-directories?                      → cohort
3. else                                               → unknown (assist may help)

Deterministic-first: adapters are tried before any model. The optional ``--assist``
layout help is only a fallback for conventions no adapter recognises, and its
proposal is meant to be frozen to a config so re-runs never re-ask the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

_INPUT_KEYS = ("dce", "ir", "t1")
_SKIP_DIRS = ("pbrain", "derivatives", "analysis")   # a subject's own outputs — never a sub-subject


@dataclass
class Resolution:
    """What :func:`resolve` decided about a path."""
    kind: str                       # "subject" | "cohort" | "unknown"
    adapter: str = ""               # adapter that matched, or ""
    subjects: list[Path] = field(default_factory=list)   # subject dirs (1 when single)

    @property
    def n(self) -> int:
        return len(self.subjects)


@runtime_checkable
class LayoutAdapter(Protocol):
    name: str

    def is_subject(self, path: Path) -> bool:
        """Does this directory *directly* hold one subject's scans?"""

    def find_subjects(self, root: Path) -> list[Path]:
        """Subject sub-directories under a cohort root (deterministic order)."""

    def resolve_inputs(self, subject: Path) -> dict:
        """``{dce, ir, t1, …}`` input paths for a subject (values may be None)."""


def _child_subjects(root: Path, is_subject) -> list[Path]:
    """Immediate sub-dirs that look like subjects — the shared cohort-scan for
    adapters whose subjects are plain child directories. Skips hidden/underscore
    dirs and a subject's own output folders."""
    if not root.is_dir():
        return []
    return [d for d in sorted(root.iterdir())
            if d.is_dir() and not d.name.startswith((".", "_"))
            and d.name.lower() not in _SKIP_DIRS and is_subject(d)]


# ---------------------------------------------------------------- PAR/REC (Philips)
class ParRecLayout:
    """Raw Philips PAR/REC export: a subject dir has ``*.PAR`` files and a
    discoverable DCE (``hperf*``). This is p-Brain's native raw-scanner format."""
    name = "parrec"

    def is_subject(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        if not (list(path.glob("*.PAR")) or list(path.glob("*.par"))):
            return False
        from pbrain.io.subject_discovery import find_dce
        return find_dce(path) is not None

    def find_subjects(self, root: Path) -> list[Path]:
        return _child_subjects(root, self.is_subject)

    def resolve_inputs(self, subject: Path) -> dict:
        from pbrain.io.subject_discovery import discover_subject_inputs
        return discover_subject_inputs(subject, assemble=False)


# ---------------------------------------------------------------- flat NIfTI
def _first(subject: Path, *patterns: str) -> Path | None:
    """First file matching any glob (case-insensitive), skipping output folders."""
    for pat in patterns:
        for hit in sorted(subject.glob(pat)) + sorted(subject.glob(pat.upper())):
            if hit.is_file() and not any(s in hit.parts for s in _SKIP_DIRS):
                return hit
    return None


class FlatNiftiLayout:
    """A subject dir holding pre-converted NIfTIs named ``dce`` / ``t1`` / ``ir``
    (case-insensitive, ``*dce*`` etc. accepted). For users who convert upstream."""
    name = "flat-nifti"

    def _dce(self, s: Path) -> Path | None:
        return _first(s, "dce.nii.gz", "dce.nii", "*dce*.nii.gz", "*perf*.nii.gz")

    def is_subject(self, path: Path) -> bool:
        return path.is_dir() and self._dce(path) is not None

    def find_subjects(self, root: Path) -> list[Path]:
        return _child_subjects(root, self.is_subject)

    def resolve_inputs(self, subject: Path) -> dict:
        return {
            "dce": self._dce(subject),
            "t1": _first(subject, "t1.nii.gz", "t1_anatomical.nii.gz", "*t1w*.nii.gz", "*t1*.nii.gz"),
            "ir": _first(subject, "ir.nii.gz", "ir_assembled.nii.gz", "*ir*.nii.gz"),
        }


# ---------------------------------------------------------------- BIDS
class BidsLayout:
    """A BIDS dataset: ``dataset_description.json`` at the root, ``sub-<label>``
    subject dirs, scans under modality folders (possibly nested under ``ses-*``).
    ``anat/*_T1w`` is standard; DCE has no standard BIDS suffix yet, so the DCE is
    matched with flexible ``*dce*`` / ``perf/`` patterns."""
    name = "bids"

    def _dce(self, s: Path) -> Path | None:
        return _first(s, "perf/*dce*.nii.gz", "**/*dce*.nii.gz", "**/perf/*.nii.gz")

    def is_subject(self, path: Path) -> bool:
        if not (path.is_dir() and path.name.startswith("sub-")):
            return False
        return ((path / "anat").is_dir() or (path / "perf").is_dir()
                or any(path.glob("ses-*")) or self._dce(path) is not None)

    def find_subjects(self, root: Path) -> list[Path]:
        if not root.is_dir():
            return []
        if not ((root / "dataset_description.json").is_file() or any(root.glob("sub-*"))):
            return []
        return [d for d in sorted(root.glob("sub-*")) if self.is_subject(d)]

    def resolve_inputs(self, subject: Path) -> dict:
        return {
            "dce": self._dce(subject),
            "t1": _first(subject, "anat/*_T1w.nii.gz", "**/anat/*_T1w.nii.gz", "**/*_T1w.nii.gz"),
            "ir": _first(subject, "**/*_IRT1*.nii.gz", "**/*ir*.nii.gz"),
        }


# ---------------------------------------------------------------- frozen (assist)
_FROZEN = "pbrain.layout.toml"


class FrozenLayout:
    """A layout previously resolved by ``--assist`` and **frozen** to
    ``pbrain.layout.toml``. Read deterministically — the model is never re-asked, so
    a published analysis re-runs byte-for-byte. Checked first, so a frozen file wins
    over any heuristic adapter."""
    name = "frozen"

    def _load(self, toml_path: Path) -> dict:
        try:
            import tomllib                       # py3.11+ stdlib
        except ModuleNotFoundError:              # py3.10 — same API
            import tomli as tomllib
        with open(toml_path, "rb") as f:
            return tomllib.load(f)

    def is_subject(self, path: Path) -> bool:
        f = path / _FROZEN
        return f.is_file() and self._load(f).get("kind") == "subject"

    def find_subjects(self, root: Path) -> list[Path]:
        f = root / _FROZEN
        if not f.is_file() or self._load(f).get("kind") != "cohort":
            return []
        return [root / s["dir"] for s in self._load(f).get("subjects", []) if s.get("dir")]

    def resolve_inputs(self, subject: Path) -> dict:
        subject = Path(subject)
        for p in [subject, *subject.parents]:      # the toml is at the subject (single) or root (cohort)
            f = p / _FROZEN
            if f.is_file():
                data = self._load(f)
                for s in data.get("subjects", []):
                    if (p / s.get("dir", ".")).resolve() == subject.resolve():
                        return {k: (p / s[k]) if s.get(k) else None for k in _INPUT_KEYS}
        return {}


def gather_tree(path: str | Path, max_depth: int = 3, max_entries: int = 300) -> str:
    """A compact, **name-only** listing of a folder for the assist layout proposer —
    directory and file names/extensions only, never any file contents (metadata-only,
    per the assist hard rule)."""
    path = Path(path)
    lines: list[str] = []

    def walk(p: Path, depth: int, prefix: str) -> None:
        if depth > max_depth or len(lines) >= max_entries:
            return
        try:
            entries = sorted(p.iterdir())
        except Exception:
            return
        for e in entries:
            if len(lines) >= max_entries:
                lines.append(f"{prefix}… (truncated)")
                break
            if e.name.startswith(".") or e.name.lower() in _SKIP_DIRS:
                continue
            if e.is_dir():
                lines.append(f"{prefix}{e.name}/")
                walk(e, depth + 1, prefix + "  ")
            else:
                lines.append(f"{prefix}{e.name}")

    walk(path, 0, "")
    return "\n".join(lines)


def write_frozen(root: str | Path, proposal: dict) -> Path:
    """Freeze a confirmed layout proposal to ``<root>/pbrain.layout.toml``. Paths are
    kept relative to ``root`` so the file travels with the data."""
    root = Path(root)
    kind = proposal.get("kind", "subject")
    out = ["# pbrain.layout.toml — layout frozen by `pbrain --assist`.",
           "# Read deterministically on later runs; the model is never re-asked.",
           f'kind = "{kind}"', 'adapter = "assist"', ""]
    for s in proposal.get("subjects", []):
        out.append("[[subjects]]")
        out.append(f'dir = "{s.get("dir", ".")}"')
        for k in _INPUT_KEYS:
            v = s.get(k)
            if v:
                out.append(f'{k} = "{v}"')
        out.append("")
    dest = root / _FROZEN
    dest.write_text("\n".join(out))
    return dest


# ---------------------------------------------------------------- registry + resolve
ADAPTERS: list[LayoutAdapter] = [FrozenLayout(), ParRecLayout(), FlatNiftiLayout(), BidsLayout()]


def _adapter(name: str) -> LayoutAdapter | None:
    return next((a for a in ADAPTERS if a.name == name), None)


def resolve(path: str | Path) -> Resolution:
    """Auto-detect: is ``path`` a single subject or a cohort of subjects, and by
    which layout? Checks single-subject first (so a subject with a ``pbrain/``
    output folder is never mistaken for a one-subject cohort)."""
    path = Path(path)
    if not path.exists():
        return Resolution("unknown")
    for ad in ADAPTERS:
        if ad.is_subject(path):
            return Resolution("subject", ad.name, [path])
    for ad in ADAPTERS:
        subs = ad.find_subjects(path)
        if subs:
            return Resolution("cohort", ad.name, subs)
    return Resolution("unknown")


def inputs_for(subject: Path, adapter_name: str) -> dict:
    """Resolve one subject's inputs with the named adapter (empty if unknown)."""
    ad = _adapter(adapter_name)
    return ad.resolve_inputs(Path(subject)) if ad else {}
