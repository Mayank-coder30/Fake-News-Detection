import re
from dataclasses import dataclass, asdict
from typing import List, Optional
from src.preprocessing.text_preprocessor import TextPreprocessor

@dataclass
class ExtractedClaim:
    claim_id: str
    claim_text: str
    position: int
    extraction_method: str
    confidence: float
    is_verifiable: bool
    filter_reason: str = ""

    def to_dict(self):
        return asdict(self)

class ClaimExtractor:
    """Identifies factual and potentially verifiable claims from news articles."""

    # Pure opinion or rhetorical indicators
    NON_CHECKABLE_PATTERNS = [
        r'^(I think|I believe|In my opinion|I feel|Personally|We hope|Let us pray|Congratulations|Happy birthday)\b',
        r'^(Hello|Hi|Welcome|Good morning|Good evening|Thank you|Thanks for reading)\b',
        r'^(Click here|Subscribe|Sign up|Follow us|Share this)\b',
    ]

    # Factual signal patterns (numbers, percentages, dates, entities, currency)
    FACTUAL_SIGNALS = [
        r'\b\d{4}\b',  # Years (e.g. 2020)
        r'\b\d+(\.\d+)?%\b',  # Percentages
        r'\$\d+',  # Currency
        r'\b(president|minister|ceo|founder|director|government|court|police|university|company|country)\b',
        r'\b(announced|stated|reported|won|defeated|signed|passed|increased|decreased|killed|discovered)\b',
        r'\b(was born|graduated|invented|founded|published|directed|starred in)\b'
    ]

    def __init__(self, min_words: int = 4, max_claims: int = 15):
        self.preprocessor = TextPreprocessor()
        self.min_words = min_words
        self.max_claims = max_claims
        self.non_checkable_re = [re.compile(p, re.IGNORECASE) for p in self.NON_CHECKABLE_PATTERNS]
        self.factual_re = [re.compile(p, re.IGNORECASE) for p in self.FACTUAL_SIGNALS]

    def is_candidate_claim(self, sentence: str) -> (bool, str, float):
        """Evaluates whether a candidate sentence is an independently verifiable factual claim.
        Returns (is_verifiable, reason, confidence).
        """
        words = sentence.split()
        if len(words) < self.min_words:
            return False, "Sentence is too short to constitute a complete verifiable claim", 0.1

        # Questions are inquiries, not assertions of fact
        if sentence.endswith("?"):
            return False, "Sentence is a question rather than a factual assertion", 0.0

        # Exclamations that are short or conversational
        if sentence.endswith("!") and len(words) < 8:
            return False, "Sentence is an exclamation or emotional utterance", 0.2

        # Check for non-checkable subjective openings
        for pattern in self.non_checkable_re:
            if pattern.search(sentence):
                return False, "Sentence expresses personal opinion, greeting, or editorial remark", 0.2

        # Compute factual confidence heuristic based on presence of factual signals
        factual_matches = sum(1 for p in self.factual_re if p.search(sentence))
        
        # Declarative check: Has a noun phrase / subject and a predicate verb
        has_verb = bool(re.search(r'\b(is|was|are|were|has|have|had|will|did|became|scored|created|led)\b', sentence, re.IGNORECASE))
        
        base_confidence = 0.6 if has_verb else 0.4
        confidence = min(0.95, base_confidence + (factual_matches * 0.1))

        return True, "Declarative factual assertion", round(confidence, 2)

    def extract_claims(self, article_text: str) -> List[ExtractedClaim]:
        """Extracts candidate claims from article text, preserving sentence order and text integrity."""
        if not article_text or not article_text.strip():
            return []

        sentences = self.preprocessor.segment_sentences(article_text)
        claims: List[ExtractedClaim] = []
        seen_texts = set()

        for idx, sentence in enumerate(sentences, start=1):
            cleaned = self.preprocessor.clean_claim_text(sentence)
            if not cleaned:
                continue

            # Deduplication
            norm_key = cleaned.lower()
            if norm_key in seen_texts:
                continue
            seen_texts.add(norm_key)

            is_verifiable, reason, conf = self.is_candidate_claim(cleaned)
            if is_verifiable:
                claim = ExtractedClaim(
                    claim_id=f"claim_{len(claims) + 1:03d}",
                    claim_text=cleaned,
                    position=idx,
                    extraction_method="heuristic_linguistic_segmentation",
                    confidence=conf,
                    is_verifiable=True,
                    filter_reason=""
                )
                claims.append(claim)
                if len(claims) >= self.max_claims:
                    break

        return claims
