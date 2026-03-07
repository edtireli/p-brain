from .patlak import MODEL as PATLAK_MODEL
from .tikhonov import MODEL as TIKHONOV_MODEL
from .extended_tofts import MODEL as EXTENDED_TOFTS_MODEL

PK_MODELS = {
    PATLAK_MODEL["key"]: PATLAK_MODEL,
    TIKHONOV_MODEL["key"]: TIKHONOV_MODEL,
    EXTENDED_TOFTS_MODEL["key"]: EXTENDED_TOFTS_MODEL,
}

PK_MODEL_ALIASES: dict[str, dict] = {}
for model in PK_MODELS.values():
    for alias in model.get("aliases", []):
        PK_MODEL_ALIASES[alias] = model


def normalise_pk_model(raw: str) -> str:
    """Return a canonical kinetic-model key.

    Accepts ``+``-delimited combos, e.g. ``"patlak+etofts"`` or
    ``"etofts+patlak+tikhonov"``.  Single keywords like ``"both"``
    and ``"all"`` are still supported.

    The canonical output is one of:

    * ``"patlak"`` / ``"tikhonov"`` / ``"extended_tofts"``  – single model
    * ``"both"``  – patlak + tikhonov
    * ``"all"``   – patlak + tikhonov + extended_tofts
    * A ``+``-joined sorted string, e.g. ``"extended_tofts+patlak"``
    """
    key = str(raw or "").strip().lower()

    # ── handle "+" combos first ──────────────────────────────────────
    if "+" in key:
        _SINGLE_ALIASES = {
            "patlak": "patlak",
            "tikhonov": "tikhonov", "tik": "tikhonov",
            "tik_fast": "tikhonov", "tik-fast": "tikhonov",
            "tikfast": "tikhonov", "tikhonov_fast": "tikhonov",
            "two_compartment": "tikhonov", "2comp": "tikhonov",
            "etofts": "extended_tofts", "etm": "extended_tofts",
            "tofts": "extended_tofts", "extended_tofts": "extended_tofts",
            "extended-tofts": "extended_tofts",
        }
        parts: set[str] = set()
        for token in key.split("+"):
            token = token.strip()
            if not token:
                continue
            canonical = _SINGLE_ALIASES.get(token)
            if canonical:
                parts.add(canonical)
            else:
                # Try alias table from MODEL dicts
                model = PK_MODEL_ALIASES.get(token)
                if model:
                    parts.add(model["env"])
        if not parts:
            return "both"
        # Collapse to legacy keywords when possible
        if parts == {"patlak", "tikhonov"}:
            return "both"
        if parts == {"patlak", "tikhonov", "extended_tofts"}:
            return "all"
        if len(parts) == 1:
            return parts.pop()
        return "+".join(sorted(parts))

    # ── legacy single keywords ───────────────────────────────────────
    if key == "all":
        return "all"
    if key in {"both", "patlak_then_tikhonov", "patlak-then-tikhonov",
               "patlak_tikhonov", "patlak_tikhonov_fast"}:
        return "both"
    model = PK_MODEL_ALIASES.get(key)
    if model:
        return model["env"]
    # Default to combo (Patlak then Tikhonov)
    return "both"


__all__ = [
    "PATLAK_MODEL",
    "TIKHONOV_MODEL",
    "EXTENDED_TOFTS_MODEL",
    "PK_MODELS",
    "PK_MODEL_ALIASES",
    "normalise_pk_model",
]
