from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import time

from config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VERIFIER_MODEL,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    DEFAULT_EVIDENCE_CORPUS,
    TOP_K_EVIDENCE,
    RELEVANCE_THRESHOLD,
    MAX_CLAIMS_PER_ARTICLE,
    LABEL_SUPPORTED,
    LABEL_REFUTED,
    LABEL_UNCERTAIN
)
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.claim_extraction.claim_extractor import ClaimExtractor
from src.evidence_retrieval.embedding_model import EmbeddingModel
from src.evidence_retrieval.vector_index import VectorIndex
from src.evidence_retrieval.document_loader import DocumentLoader
from src.evidence_retrieval.evidence_retriever import EvidenceRetriever
from src.verification.verifier import ClaimVerifier, VerificationResult
from src.explanation.explanation_generator import ExplanationGenerator
from src.utils.logging_utils import get_logger

logger = get_logger("FactCheckPipeline")

@dataclass
class ArticleSummary:
    total_claims: int
    supported_count: int
    refuted_count: int
    uncertain_count: int
    error_count: int
    elapsed_seconds: float
    disclaimer: str

    def to_dict(self):
        return asdict(self)

@dataclass
class PipelineOutput:
    article_text: str
    claims: List[Dict[str, Any]]
    summary: ArticleSummary
    metadata: Dict[str, Any]

    def to_dict(self):
        return {
            "article_text": self.article_text,
            "claims": self.claims,
            "summary": self.summary.to_dict(),
            "metadata": self.metadata
        }

class FactCheckPipeline:
    """Central orchestration engine executing the 12-step evidence-grounded fact checking workflow."""

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        verifier_model_name: str = DEFAULT_VERIFIER_MODEL,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
        top_k: int = TOP_K_EVIDENCE
    ):
        self.relevance_threshold = relevance_threshold
        self.top_k = top_k

        logger.info("Initializing FactCheckPipeline components...")
        self.preprocessor = TextPreprocessor()
        self.extractor = ClaimExtractor(min_words=4, max_claims=MAX_CLAIMS_PER_ARTICLE)
        self.embedding_model = EmbeddingModel(model_name=embedding_model_name)

        # Load or initialize FAISS vector index
        if FAISS_INDEX_PATH.exists() and FAISS_METADATA_PATH.exists():
            logger.info("Loading existing FAISS index from disk...")
            self.vector_index = VectorIndex.load(FAISS_INDEX_PATH, FAISS_METADATA_PATH)
        else:
            logger.warning("FAISS index files not found on disk. Building dynamic index from default corpus...")
            docs = DocumentLoader.load_jsonl(DEFAULT_EVIDENCE_CORPUS)
            self.vector_index = VectorIndex(dimension=self.embedding_model.embedding_dim)
            self.vector_index.build_from_documents(docs, self.embedding_model)
            self.vector_index.save(FAISS_INDEX_PATH, FAISS_METADATA_PATH)

        self.retriever = EvidenceRetriever(
            embedding_model=self.embedding_model,
            vector_index=self.vector_index,
            relevance_threshold=self.relevance_threshold,
            default_top_k=self.top_k
        )

        self.verifier = ClaimVerifier(model_name=verifier_model_name)
        self.explainer = ExplanationGenerator()
        logger.info("FactCheckPipeline initialization complete.")

    def analyze_article(self, article_text: str) -> PipelineOutput:
        """Executes full claim-by-claim fact verification on a news article."""
        start_time = time.time()

        # Step 1: Input validation
        if not article_text or not article_text.strip():
            return PipelineOutput(
                article_text="",
                claims=[],
                summary=ArticleSummary(
                    total_claims=0,
                    supported_count=0,
                    refuted_count=0,
                    uncertain_count=0,
                    error_count=0,
                    elapsed_seconds=0.0,
                    disclaimer="Input text is empty or contains only whitespace."
                ),
                metadata={"error": "Empty input text provided"}
            )

        # Step 2: Normalize text
        cleaned_text = self.preprocessor.normalize_whitespace(article_text)

        # Step 3: Extract candidate claims
        extracted_claims = self.extractor.extract_claims(cleaned_text)

        claim_results: List[Dict[str, Any]] = []
        supported_cnt = 0
        refuted_cnt = 0
        uncertain_cnt = 0
        error_cnt = 0

        for claim in extracted_claims:
            try:
                # Step 4 & 5: Retrieve and rank relevant evidence
                passages = self.retriever.retrieve_for_claim(claim.claim_text, top_k=self.top_k)

                # Step 6, 7 & 8: Verify against retrieved evidence
                v_res: VerificationResult = self.verifier.verify(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    passages=passages
                )

                # Step 9: Generate grounded explanation
                exp_info = self.explainer.generate_explanation(
                    claim_text=claim.claim_text,
                    label=v_res.label,
                    evidence_passages=v_res.evidence,
                    metadata=v_res.verification_metadata
                )
                v_res.explanation = exp_info["explanation"]

                # Step 10 & 11: Assemble result
                res_dict = v_res.to_dict()
                res_dict["explanation_provider"] = exp_info["provider"]
                res_dict["position"] = claim.position
                res_dict["extraction_confidence"] = claim.confidence

                claim_results.append(res_dict)

                if v_res.label == LABEL_SUPPORTED:
                    supported_cnt += 1
                elif v_res.label == LABEL_REFUTED:
                    refuted_cnt += 1
                else:
                    uncertain_cnt += 1

            except Exception as e:
                logger.error(f"Error processing claim '{claim.claim_id}': {e}", exc_info=True)
                error_cnt += 1
                claim_results.append({
                    "claim_id": claim.claim_id,
                    "claim_text": claim.claim_text,
                    "label": "PROCESSING_ERROR",
                    "confidence": 0.0,
                    "retrieval_status": "error",
                    "evidence": [],
                    "explanation": f"An unexpected technical error occurred while verifying this claim: {str(e)}",
                    "verification_metadata": {"error": str(e)}
                })

        elapsed = round(time.time() - start_time, 2)

        summary = ArticleSummary(
            total_claims=len(extracted_claims),
            supported_count=supported_cnt,
            refuted_count=refuted_cnt,
            uncertain_count=uncertain_cnt,
            error_count=error_cnt,
            elapsed_seconds=elapsed,
            disclaimer=(
                "EvidenceLens evaluates individual verifiable claims against indexed reference evidence. "
                "These claim-level outcomes do not constitute a single black-box judgment on the entire article."
            )
        )

        return PipelineOutput(
            article_text=cleaned_text,
            claims=claim_results,
            summary=summary,
            metadata={
                "embedding_model": self.embedding_model.model_name,
                "verifier_model": self.verifier.model_name,
                "total_indexed_passages": self.vector_index.index.ntotal,
                "relevance_threshold": self.relevance_threshold,
                "top_k": self.top_k
            }
        )
