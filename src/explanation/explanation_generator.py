import os
import re
from typing import List, Dict, Any, Optional
import urllib.request
import json

from config import (
    LABEL_SUPPORTED,
    LABEL_REFUTED,
    LABEL_UNCERTAIN,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    GROQ_API_KEY,
    GEMINI_API_KEY
)
from src.utils.logging_utils import get_logger

logger = get_logger("ExplanationGenerator")

class ExplanationGenerator:
    """Generates transparent, evidence-grounded natural language explanations for verification results."""

    def __init__(self):
        self.openai_key = OPENAI_API_KEY
        self.openai_model = OPENAI_MODEL
        self.groq_key = GROQ_API_KEY
        self.gemini_key = GEMINI_API_KEY

    def generate_explanation(
        self,
        claim_text: str,
        label: str,
        evidence_passages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Generates explanation using configured LLM API if available, else falls back to deterministic template.
        Returns a dict: {"explanation": str, "provider": str}
        """
        # If API key is available, attempt LLM generation with strict safeguards
        if self.openai_key:
            try:
                llm_exp = self._generate_openai_explanation(claim_text, label, evidence_passages)
                if llm_exp:
                    return {"explanation": llm_exp, "provider": f"OpenAI ({self.openai_model})"}
            except Exception as e:
                logger.warning(f"OpenAI explanation failed: {e}. Falling back to deterministic generator.")

        # Deterministic evidence-grounded fallback
        fallback_exp = self._generate_deterministic_explanation(claim_text, label, evidence_passages, metadata)
        return {"explanation": fallback_exp, "provider": "Deterministic Grounded Engine (Rule-based)"}

    def _generate_deterministic_explanation(
        self,
        claim_text: str,
        label: str,
        evidence_passages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Constructs a deterministic, strictly grounded explanation using factual excerpts from retrieved evidence."""
        if not evidence_passages or label == LABEL_UNCERTAIN:
            if metadata and metadata.get("conflict_detected"):
                return (
                    "Available evidence presents conflicting assertions across multiple retrieved sources. "
                    "Because verified sources disagree or remain inconclusive, this claim is classified as UNCERTAIN."
                )
            return (
                "Insufficient evidence was found in the indexed knowledge corpus to substantiate or disprove this claim. "
                "The lack of documented evidence does not prove the claim is false, but prevents verification."
            )

        top_passage = evidence_passages[0]
        title = top_passage.get("title", "Reference Source")
        text = top_passage.get("text", "").strip()
        score = top_passage.get("relevance_score", 0.0)

        # Extract most informative excerpt (up to 200 characters)
        excerpt = text[:220].rstrip()
        if len(text) > 220:
            excerpt += "..."

        if label == LABEL_SUPPORTED:
            return (
                f"The claim is SUPPORTED by verified knowledge. According to '{title}' (Relevance: {score:.2f}): "
                f"\"{excerpt}\", which directly confirms the factual premise of the claim."
            )
        elif label == LABEL_REFUTED:
            return (
                f"The claim is REFUTED by verified knowledge. According to '{title}' (Relevance: {score:.2f}): "
                f"\"{excerpt}\", which directly contradicts the claim's factual assertions."
            )
        else:
            return (
                f"The claim status is UNCERTAIN. While related context was located in '{title}', "
                f"the passage does not provide conclusive verification: \"{excerpt}\"."
            )

    def _generate_openai_explanation(
        self,
        claim_text: str,
        label: str,
        evidence_passages: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Calls OpenAI Chat Completion endpoint with strict anti-hallucination prompt constraints."""
        evidence_context = "\n".join([
            f"- [{p.get('title', 'Unknown')}]: {p.get('text', '')}"
            for p in evidence_passages[:3]
        ])

        system_prompt = (
            "You are an objective, evidence-grounded fact verification assistant. "
            "Your task is to write a concise 2-sentence explanation of why a factual claim was classified as "
            f"{label} based EXCLUSIVELY on the provided evidence. "
            "Rules:\n"
            "1. Ground every statement strictly in the provided evidence.\n"
            "2. Do NOT add outside facts or extrapolate.\n"
            "3. Do NOT alter the given verification label.\n"
            "4. Keep the explanation under 60 words."
        )

        user_content = (
            f"Claim: \"{claim_text}\"\n"
            f"Verification Verdict: {label}\n"
            f"Retrieved Evidence:\n{evidence_context}"
        )

        payload = {
            "model": self.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.2,
            "max_tokens": 120
        }

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_key}"
            }
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"].strip()
