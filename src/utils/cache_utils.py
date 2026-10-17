import hashlib
from typing import Any, Dict

def hash_text(text: str) -> str:
    """Generates an MD5 hex digest of the normalized text string."""
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()
