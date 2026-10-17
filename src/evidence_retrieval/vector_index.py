import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np
from src.evidence_retrieval.document_loader import EvidenceDocument
from src.evidence_retrieval.embedding_model import EmbeddingModel
from src.utils.logging_utils import get_logger

logger = get_logger("VectorIndex")

class VectorIndex:
    """FAISS-based vector index for fast semantic similarity search over evidence passages."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict[str, Any]] = []

    def build_from_documents(
        self,
        documents: List[EvidenceDocument],
        embedding_model: EmbeddingModel,
        batch_size: int = 32
    ):
        """Encodes documents and builds the FAISS inner-product index."""
        if not documents:
            logger.warning("No documents provided to build vector index.")
            return

        logger.info(f"Encoding {len(documents)} evidence documents...")
        texts = [f"{doc.title}: {doc.text}" if doc.title != "Untitled" else doc.text for doc in documents]
        embeddings = embedding_model.encode(texts, batch_size=batch_size, normalize_embeddings=True)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.metadata = [doc.to_dict() for doc in documents]
        logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors.")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Searches index with a normalized query vector.
        Returns list of (similarity_score, document_metadata).
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        # FAISS search returns distances (inner products) and indices
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector.astype(np.float32), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append((float(score), self.metadata[idx]))

        return results

    def save(self, index_path: Path, metadata_path: Path):
        """Persists the FAISS index and corresponding metadata to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            for item in self.metadata:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Persisted index to {index_path} and metadata to {metadata_path}")

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> "VectorIndex":
        """Loads index and metadata from disk."""
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Index or metadata file missing: {index_path}, {metadata_path}")

        index = faiss.read_index(str(index_path))
        metadata = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    metadata.append(json.loads(line))

        inst = cls(dimension=index.d)
        inst.index = index
        inst.metadata = metadata
        logger.info(f"Loaded FAISS index with {inst.index.ntotal} vectors from {index_path}")
        return inst
