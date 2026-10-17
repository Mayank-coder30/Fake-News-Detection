from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F

from src.evidence_retrieval.evidence_retriever import RetrievedPassage
from src.verification.model_loader import ModelLoader
from src.verification.label_mapping import normalize_label
from config import (
    DEFAULT_VERIFIER_MODEL,
    LABEL_SUPPORTED,
    LABEL_REFUTED,
    LABEL_UNCERTAIN
)
from src.utils.logging_utils import get_logger

logger = get_logger("ClaimVerifier")

@dataclass
class PassageVerdict:
    passage_id: str
    label: str
    confidence: float
    probabilities: Dict[str, float]
    passage_text: str

    def to_dict(self):
        return asdict(self)

@dataclass
class VerificationResult:
    claim_id: str
    claim_text: str
    label: str
    confidence: float
    retrieval_status: str
    evidence: List[Dict[str, Any]]
    explanation: str
    verification_metadata: Dict[str, Any]

    def to_dict(self):
        return asdict(self)

class ClaimVerifier:
    """Evaluates the relationship between candidate claims and retrieved evidence passages using a Transformer NLI model."""

    def __init__(
        self,
        model_name: str = DEFAULT_VERIFIER_MODEL,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.tokenizer, self.model, self.device = ModelLoader.load_nli_model(model_name, device)
        self._build_label_index_map()

    def _build_label_index_map(self):
        """Inspects model configuration to accurately map output logit indices to canonical labels."""
        id2label = getattr(self.model.config, "id2label", None)
        self.index_to_canonical: Dict[int, str] = {}

        if id2label and isinstance(id2label, dict):
            for idx, raw_lbl in id2label.items():
                self.index_to_canonical[int(idx)] = normalize_label(str(raw_lbl))
        else:
            # Standard NLI default: 0: contradiction, 1: entailment, 2: neutral (or similar)
            # Defaulting to standard distilroberta cross-encoder ordering:
            # 0: contradiction (REFUTED), 1: entailment (SUPPORTED), 2: neutral (UNCERTAIN)
            self.index_to_canonical = {
                0: LABEL_REFUTED,
                1: LABEL_SUPPORTED,
                2: LABEL_UNCERTAIN
            }
        logger.info(f"Verified label mapping from model config: {self.index_to_canonical}")

    def verify_pair(self, claim: str, evidence_text: str) -> Tuple[str, float, Dict[str, float]]:
        """Runs transformer inference on a single (evidence, claim) pair."""
        inputs = self.tokenizer(
            evidence_text,
            claim,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

        prob_dict = {
            LABEL_SUPPORTED: 0.0,
            LABEL_REFUTED: 0.0,
            LABEL_UNCERTAIN: 0.0
        }

        for idx, prob in enumerate(probs):
            canonical = self.index_to_canonical.get(idx, LABEL_UNCERTAIN)
            prob_dict[canonical] = max(prob_dict[canonical], float(prob))

        # Best predicted label
        best_label = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[best_label]

        return best_label, round(confidence, 4), {k: round(v, 4) for k, v in prob_dict.items()}

    def verify(
        self,
        claim_id: str,
        claim_text: str,
        passages: List[RetrievedPassage]
    ) -> VerificationResult:
        """Verifies a claim against all retrieved evidence passages and aggregates verdicts."""
        if not passages:
            return VerificationResult(
                claim_id=claim_id,
                claim_text=claim_text,
                label=LABEL_UNCERTAIN,
                confidence=0.0,
                retrieval_status="insufficient_evidence",
                evidence=[],
                explanation="No relevant evidence was retrieved from the knowledge source to substantiate or refute this claim.",
                verification_metadata={
                    "model_name": self.model_name,
                    "device": self.device,
                    "reason": "missing_evidence"
                }
            )

        passage_verdicts: List[PassageVerdict] = []
        for p in passages:
            lbl, conf, probs = self.verify_pair(claim_text, p.text)
            verdict = PassageVerdict(
                passage_id=p.passage_id,
                label=lbl,
                confidence=conf,
                probabilities=probs,
                passage_text=p.text
            )
            passage_verdicts.append(verdict)

        # Multi-evidence aggregation with relevance stratification
        # If the top-ranked passage is a strong semantic match (relevance >= 0.60),
        # only passages of comparable relevance (>= 0.60) are evaluated for decisive support or refutation.
        top_passage = passages[0]
        top_verdict = passage_verdicts[0]

        high_rel_verdicts = [
            v for p, v in zip(passages, passage_verdicts) if p.relevance_score >= 0.60
        ]

        if high_rel_verdicts:
            has_sup = any(v.label == LABEL_SUPPORTED for v in high_rel_verdicts)
            has_ref = any(v.label == LABEL_REFUTED for v in high_rel_verdicts)

            if has_sup and has_ref:
                final_label = LABEL_UNCERTAIN
                final_conf = 0.50
                conflict_detected = True
            else:
                # Top matching passage determines the verdict
                final_label = top_verdict.label
                final_conf = top_verdict.confidence
                conflict_detected = False
        else:
            # For lower-relevance passages, aggregate weighted probabilities
            weighted_scores = {LABEL_SUPPORTED: 0.0, LABEL_REFUTED: 0.0, LABEL_UNCERTAIN: 0.0}
            total_weight = 0.0
            for p, v in zip(passages, passage_verdicts):
                w = max(0.1, float(p.relevance_score))
                total_weight += w
                for lbl in [LABEL_SUPPORTED, LABEL_REFUTED, LABEL_UNCERTAIN]:
                    weighted_scores[lbl] += v.probabilities.get(lbl, 0.0) * w

            avg_probs = {lbl: weighted_scores[lbl] / total_weight for lbl in weighted_scores}
            final_label = max(avg_probs, key=avg_probs.get)
            final_conf = avg_probs[final_label]
            conflict_detected = False

        return VerificationResult(
            claim_id=claim_id,
            claim_text=claim_text,
            label=final_label,
            confidence=round(final_conf, 4),
            retrieval_status="evidence_found",
            evidence=[p.to_dict() for p in passages],
            explanation="",  # Will be populated by ExplanationGenerator
            verification_metadata={
                "model_name": self.model_name,
                "device": self.device,
                "conflict_detected": conflict_detected,
                "per_passage_verdicts": [v.to_dict() for v in passage_verdicts]
            }
        )
