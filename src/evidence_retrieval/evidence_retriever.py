from dataclasses import dataclass, asdict
from typing import List, Optional
import numpy as np

from src.evidence_retrieval.embedding_model import EmbeddingModel
from src.evidence_retrieval.vector_index import VectorIndex
from src.utils.logging_utils import get_logger

logger = get_logger("EvidenceRetriever")

@dataclass
class RetrievedPassage:
    passage_id: str
    title: str
    text: str
    source_id: str
    source_url: Optional[str]
    relevance_score: float
    rank: int

    def to_dict(self):
        return asdict(self)

class EvidenceRetriever:
    """Orchestrates semantic search and ranking of evidence passages for given claims."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_index: VectorIndex,
        relevance_threshold: float = 0.35,
        default_top_k: int = 3
    ):
        self.embedding_model = embedding_model
        self.vector_index = vector_index
        self.relevance_threshold = relevance_threshold
        self.default_top_k = default_top_k

    def retrieve_for_claim(
        self,
        claim_text: str,
        top_k: Optional[int] = None
    ) -> List[RetrievedPassage]:
        """Retrieves and ranks evidence passages relevant to the claim.
        Filters out passages below relevance_threshold and removes duplicates.
        """
        k = top_k if top_k is not None else self.default_top_k
        if not claim_text or not claim_text.strip():
            return []

        # Encode claim text to unit vector
        query_vec = self.embedding_model.encode(claim_text, normalize_embeddings=True)
        raw_results = self.vector_index.search(query_vec, top_k=k * 2)

        passages: List[RetrievedPassage] = []
        seen_ids = set()

        rank = 1
        for score, meta in raw_results:
            pid = meta.get("passage_id")
            if pid in seen_ids:
                continue
            seen_ids.add(pid)

            # Apply relevance threshold
            if score < self.relevance_threshold:
                continue

            passage = RetrievedPassage(
                passage_id=pid,
                title=meta.get("title", "Untitled"),
                text=meta.get("text", ""),
                source_id=meta.get("source_id", "corpus"),
                source_url=meta.get("source_url"),
                relevance_score=round(float(score), 4),
                rank=rank
            )
            passages.append(passage)
            rank += 1
            if len(passages) >= k:
                break

        return passages
