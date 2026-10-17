import re
from typing import List

class TextPreprocessor:
    """Preprocesses raw text input for claim extraction and verification."""

    def __init__(self):
        # Common abbreviation patterns to prevent incorrect sentence splits
        self._abbr_pattern = re.compile(
            r'\b(Mr|Mrs|Ms|Dr|Prof|Gen|Rep|Sen|St|Inc|Corp|Ltd|Co|Jan|Feb|Mar|Apr|Aug|Sept|Oct|Nov|Dec|U\.S|U\.K|e\.g|i\.e)\.\s+',
            re.IGNORECASE
        )

    def normalize_whitespace(self, text: str) -> str:
        """Normalizes unicode whitespace, replaces tabs and consecutive spaces with single spaces."""
        if not text:
            return ""
        # Replace non-breaking spaces and irregular whitespace
        text = text.replace('\xa0', ' ').replace('\u200b', '')
        # Collapse multiple spaces and newlines
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def segment_sentences(self, text: str) -> List[str]:
        """Segments raw article text into individual sentences reliably.
        Uses a resilient regex-based tokenizer that protects common abbreviations.
        """
        cleaned = self.normalize_whitespace(text)
        if not cleaned:
            return []

        # Temporarily protect period in common abbreviations with placeholder
        protected_text = self._abbr_pattern.sub(lambda m: m.group(0).replace('.', '@@DOT@@'), cleaned)

        # Split on sentence boundaries: period, exclamation mark, question mark followed by space or newline
        raw_sentences = re.split(r'(?<=[.!?])\s+', protected_text)

        sentences = []
        for s in raw_sentences:
            s_restored = s.replace('@@DOT@@', '.').strip()
            # Clean up residual quotes or trailing punctuation anomalies
            if s_restored:
                sentences.append(s_restored)

        return sentences

    def clean_claim_text(self, text: str) -> str:
        """Lightly cleans individual claim text while strictly preserving its original semantic content."""
        if not text:
            return ""
        cleaned = self.normalize_whitespace(text)
        # Strip outer quotation marks if wrapped completely in quotes
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()
        return cleaned
