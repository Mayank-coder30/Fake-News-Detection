import pytest
from src.verification.label_mapping import normalize_label
from src.verification.verifier import ClaimVerifier, VerificationResult
from src.evidence_retrieval.evidence_retriever import RetrievedPassage
from config import LABEL_SUPPORTED, LABEL_REFUTED, LABEL_UNCERTAIN

def test_label_mapping():
    assert normalize_label("entailment") == LABEL_SUPPORTED
    assert normalize_label("SUPPORTS") == LABEL_SUPPORTED
    assert normalize_label("contradiction") == LABEL_REFUTED
    assert normalize_label("REFUTES") == LABEL_REFUTED
    assert normalize_label("neutral") == LABEL_UNCERTAIN
    assert normalize_label("NOT ENOUGH INFO") == LABEL_UNCERTAIN
    assert normalize_label("not_enough_info") == LABEL_UNCERTAIN
    assert normalize_label("completely_unknown_token") == LABEL_UNCERTAIN

def test_missing_evidence_yields_uncertain():
    # If no evidence passages are provided, verifier should safely return UNCERTAIN
    verifier = ClaimVerifier(device="cpu")
    result: VerificationResult = verifier.verify("claim_01", "The moon is made of cheddar cheese", [])
    assert result.label == LABEL_UNCERTAIN
    assert result.retrieval_status == "insufficient_evidence"
    assert len(result.evidence) == 0
    assert "No relevant evidence" in result.explanation

def test_pairwise_verification():
    verifier = ClaimVerifier(device="cpu")
    passage = RetrievedPassage(
        passage_id="doc_test",
        title="Apollo 11",
        text="Apollo 11 landed humans on the Moon on July 20, 1969. Commander Neil Armstrong was the first person to walk on the Moon.",
        source_id="wiki_apollo",
        source_url="https://en.wikipedia.org/wiki/Apollo_11",
        relevance_score=0.88,
        rank=1
    )

    # Test supporting claim
    res_sup = verifier.verify("c_sup", "Neil Armstrong was the first person to walk on the Moon.", [passage])
    assert res_sup.label == LABEL_SUPPORTED
    assert res_sup.confidence > 0.5

    # Test refuting claim
    res_ref = verifier.verify("c_ref", "Neil Armstrong never walked on the Moon.", [passage])
    assert res_ref.label == LABEL_REFUTED
    assert res_ref.confidence > 0.5
