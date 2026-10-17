from typing import Dict
from config import LABEL_SUPPORTED, LABEL_REFUTED, LABEL_UNCERTAIN

# Canonical labels for the 3-class fact verification taxonomy
CANONICAL_LABELS = [LABEL_SUPPORTED, LABEL_REFUTED, LABEL_UNCERTAIN]

# Common variations from NLI and FEVER benchmarks
NLI_LABEL_MAP: Dict[str, str] = {
    # NLI standard terms
    "entailment": LABEL_SUPPORTED,
    "contradiction": LABEL_REFUTED,
    "neutral": LABEL_UNCERTAIN,

    # FEVER official benchmark terms
    "supports": LABEL_SUPPORTED,
    "refutes": LABEL_REFUTED,
    "not enough info": LABEL_UNCERTAIN,
    "not_enough_info": LABEL_UNCERTAIN,

    # Binary/Tertiary variants
    "true": LABEL_SUPPORTED,
    "false": LABEL_REFUTED,
    "unverified": LABEL_UNCERTAIN
}

def normalize_label(raw_label: str) -> str:
    """Normalizes any model output label into one of the three canonical labels:
    SUPPORTED, REFUTED, or UNCERTAIN.
    """
    cleaned = str(raw_label).strip().lower().replace("-", "_").replace(" ", "_")
    for key, canonical in NLI_LABEL_MAP.items():
        if key.replace(" ", "_") == cleaned:
            return canonical

    # If completely unrecognized, default conservatively to UNCERTAIN
    return LABEL_UNCERTAIN
