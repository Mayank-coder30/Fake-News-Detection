"""Script to build and persist FAISS vector index from evidence corpus.
"""

import sys
from pathlib import Path
import argparse

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DEFAULT_EVIDENCE_CORPUS,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIM
)
from src.evidence_retrieval.document_loader import DocumentLoader
from src.evidence_retrieval.embedding_model import EmbeddingModel
from src.evidence_retrieval.vector_index import VectorIndex
from src.utils.logging_utils import get_logger

logger = get_logger("build_evidence_index")

def build_index(
    corpus_path: Path = DEFAULT_EVIDENCE_CORPUS,
    index_out: Path = FAISS_INDEX_PATH,
    meta_out: Path = FAISS_METADATA_PATH,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    dimension: int = EMBEDDING_DIM
):
    """Loads evidence documents, generates embeddings, builds and saves FAISS index."""
    logger.info(f"Loading corpus from: {corpus_path}")
    docs = DocumentLoader.load_jsonl(corpus_path)
    logger.info(f"Loaded {len(docs)} passages.")

    logger.info(f"Initializing embedding model: {model_name}")
    embedding_model = EmbeddingModel(model_name=model_name)

    vector_index = VectorIndex(dimension=dimension)
    vector_index.build_from_documents(docs, embedding_model)

    vector_index.save(index_out, meta_out)
    logger.info(f"FAISS index successfully saved to {index_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS evidence index")
    parser.add_argument("--corpus", type=str, default=str(DEFAULT_EVIDENCE_CORPUS),
                        help="Path to evidence corpus JSONL")
    parser.add_argument("--index_out", type=str, default=str(FAISS_INDEX_PATH),
                        help="Path to save FAISS index")
    parser.add_argument("--meta_out", type=str, default=str(FAISS_METADATA_PATH),
                        help="Path to save metadata JSONL")
    args = parser.parse_args()

    build_index(Path(args.corpus), Path(args.index_out), Path(args.meta_out))
