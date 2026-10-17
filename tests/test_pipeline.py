import pytest
from src.pipeline.fact_check_pipeline import FactCheckPipeline
from config import LABEL_SUPPORTED, LABEL_REFUTED, LABEL_UNCERTAIN

@pytest.fixture(scope="module")
def pipeline():
    return FactCheckPipeline()

def test_pipeline_empty_article(pipeline):
    res = pipeline.analyze_article("")
    assert res.summary.total_claims == 0
    assert len(res.claims) == 0

def test_pipeline_supported_claim(pipeline):
    article = "NASA confirmed that Neil Armstrong was the first person to step onto the lunar surface in 1969."
    res = pipeline.analyze_article(article)
    assert res.summary.total_claims >= 1
    assert any(c["label"] == LABEL_SUPPORTED for c in res.claims)
    first_claim = res.claims[0]
    assert first_claim["retrieval_status"] == "evidence_found"
    assert len(first_claim["evidence"]) > 0
    assert "explanation" in first_claim
    assert len(first_claim["explanation"]) > 10

def test_pipeline_refuted_claim(pipeline):
    article = "Viral posts claim that the Eiffel Tower was built entirely out of pure solid gold."
    res = pipeline.analyze_article(article)
    assert res.summary.total_claims >= 1
    refuted_claims = [c for c in res.claims if c["label"] == LABEL_REFUTED]
    assert len(refuted_claims) >= 1
    assert "REFUTED" in refuted_claims[0]["explanation"]

def test_pipeline_uncertain_missing_evidence(pipeline):
    article = "Alien beings constructed underground pyramids on Jupiter in 5000 BC."
    res = pipeline.analyze_article(article)
    assert res.summary.total_claims >= 1
    assert any(c["label"] == LABEL_UNCERTAIN for c in res.claims)

def test_pipeline_article_summary_counts(pipeline):
    article = (
        "Albert Einstein developed the theory of relativity. "
        "The Eiffel Tower was built entirely out of pure solid gold. "
        "Some ancient philosopher owned a bright orange flying bicycle in Rome."
    )
    res = pipeline.analyze_article(article)
    assert res.summary.total_claims == 3
    # Check that sum of status counts equals total claims
    total_accounted = (
        res.summary.supported_count +
        res.summary.refuted_count +
        res.summary.uncertain_count +
        res.summary.error_count
    )
    assert total_accounted == res.summary.total_claims
    assert res.summary.error_count == 0
