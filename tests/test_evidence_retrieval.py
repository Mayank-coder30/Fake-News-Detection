import pytest
import numpy as np
from pathlib import Path

from src.evidence_retrieval.document_loader import DocumentLoader, EvidenceDocument
from src.evidence_retrieval.embedding_model import EmbeddingModel
from src.evidence_retrieval.vector_index import VectorIndex
from src.evidence_retrieval.evidence_retriever import EvidenceRetriever
from config import DEFAULT_EVIDENCE_CORPUS, DEFAULT_EMBEDDING_MODEL

@pytest.fixture(scope="module")
def embedding_model():
    return EmbeddingModel(model_name=DEFAULT_EMBEDDING_MODEL, device="cpu")

def test_document_loader():
    docs = DocumentLoader.load_jsonl(DEFAULT_EVIDENCE_CORPUS)
    assert len(docs) > 0
    first_doc = docs[0]
    assert first_doc.passage_id is not None
    assert len(first_doc.text) > 20
    assert first_doc.source_id is not None

def test_embedding_model_encode(embedding_model):
    texts = ["Neil Armstrong walked on the Moon.", "Python is a programming language."]
    embeddings = embedding_model.encode(texts, normalize_embeddings=True)
    assert embeddings.shape == (2, embedding_model.embedding_dim)
    # Check unit normalization (norm close to 1.0)
    norm0 = np.linalg.norm(embeddings[0])
    assert abs(norm0 - 1.0) < 1e-4

def test_vector_index_build_and_search(embedding_model, tmp_path):
    docs = [
        EvidenceDocument("d1", "Apollo 11", "Neil Armstrong walked on the moon in 1969.", "source_a"),
        EvidenceDocument("d2", "Eiffel Tower", "The Eiffel Tower is located in Paris, France.", "source_b")
    ]
    index = VectorIndex(dimension=embedding_model.embedding_dim)
    index.build_from_documents(docs, embedding_model)
    assert index.index.ntotal == 2

    # Query for lunar landing
    q_vec = embedding_model.encode("Who was the first person on the moon?")
    results = index.search(q_vec, top_k=1)
    assert len(results) == 1
    score, meta = results[0]
    assert meta["passage_id"] == "d1"
    assert score > 0.4

    # Save and reload
    idx_file = tmp_path / "test.faiss"
    meta_file = tmp_path / "test.jsonl"
    index.save(idx_file, meta_file)
    assert idx_file.exists()
    assert meta_file.exists()

    reloaded = VectorIndex.load(idx_file, meta_file)
    assert reloaded.index.ntotal == 2

def test_evidence_retriever_filtering(embedding_model):
    docs = [
        EvidenceDocument("d1", "Apollo 11", "Neil Armstrong walked on the moon.", "source_a")
    ]
    index = VectorIndex(dimension=embedding_model.embedding_dim)
    index.build_from_documents(docs, embedding_model)

    retriever = EvidenceRetriever(embedding_model, index, relevance_threshold=0.85)
    # With a high threshold of 0.85, an unrelated query should return empty list
    results = retriever.retrieve_for_claim("Recipe for chocolate cake")
    assert len(results) == 0
