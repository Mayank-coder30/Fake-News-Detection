import pytest
from src.explanation.explanation_generator import ExplanationGenerator
from config import LABEL_SUPPORTED, LABEL_REFUTED, LABEL_UNCERTAIN

@pytest.fixture
def generator():
    return ExplanationGenerator()

def test_fallback_explanation_supported(generator):
    evidence = [{
        "passage_id": "p1",
        "title": "NASA JWST",
        "text": "The James Webb Space Telescope operates in infrared wavelengths to observe early galaxies.",
        "relevance_score": 0.85
    }]
    res = generator.generate_explanation("The JWST observes early galaxies.", LABEL_SUPPORTED, evidence)
    assert res["provider"] == "Deterministic Grounded Engine (Rule-based)"
    assert "SUPPORTED" in res["explanation"]
    assert "NASA JWST" in res["explanation"]

def test_fallback_explanation_refuted(generator):
    evidence = [{
        "passage_id": "p2",
        "title": "Apollo 11",
        "text": "Neil Armstrong walked on the Moon on July 20, 1969.",
        "relevance_score": 0.90
    }]
    res = generator.generate_explanation("Neil Armstrong walked on Mars in 1969.", LABEL_REFUTED, evidence)
    assert "REFUTED" in res["explanation"]
    assert "Apollo 11" in res["explanation"]

def test_fallback_explanation_missing_evidence(generator):
    res = generator.generate_explanation("Aliens built the pyramids in 10000 BC.", LABEL_UNCERTAIN, [])
    assert "Insufficient evidence" in res["explanation"]
    assert "does not prove the claim is false" in res["explanation"]

def test_fallback_explanation_conflicting_evidence(generator):
    res = generator.generate_explanation(
        "Economic policy caused immediate growth.",
        LABEL_UNCERTAIN,
        [],
        metadata={"conflict_detected": True}
    )
    assert "conflicting assertions" in res["explanation"]
