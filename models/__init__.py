from .patlak import MODEL as PATLAK_MODEL
from .tikhonov import MODEL as TIKHONOV_MODEL

PK_MODELS = {
    PATLAK_MODEL["key"]: PATLAK_MODEL,
    TIKHONOV_MODEL["key"]: TIKHONOV_MODEL,
}

PK_MODEL_ALIASES: dict[str, dict] = {}
for model in PK_MODELS.values():
    for alias in model.get("aliases", []):
        PK_MODEL_ALIASES[alias] = model


def normalise_pk_model(raw: str) -> str:
    key = str(raw or "").strip().lower()
    model = PK_MODEL_ALIASES.get(key)
    if model:
        return model["env"]
    # Default to combo (Patlak then Tikhonov)
    return "both"


__all__ = [
    "PATLAK_MODEL",
    "TIKHONOV_MODEL",
    "PK_MODELS",
    "PK_MODEL_ALIASES",
    "normalise_pk_model",
]
