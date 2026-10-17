import pytest
from src.preprocessing.text_preprocessor import TextPreprocessor

@pytest.fixture
def preprocessor():
    return TextPreprocessor()

def test_normalize_whitespace(preprocessor):
    raw = "  This   has  irregular \t whitespace \n\n\n and newlines.  "
    normalized = preprocessor.normalize_whitespace(raw)
    assert normalized == "This has irregular whitespace \n\n and newlines."

def test_segment_sentences_basic(preprocessor):
    text = "The quick brown fox jumps over the lazy dog. Dr. Smith was surprised! What happened next?"
    sentences = preprocessor.segment_sentences(text)
    assert len(sentences) == 3
    # Check that abbreviation Dr. did not cause a sentence split
    assert sentences[1] == "Dr. Smith was surprised!"
    assert sentences[2] == "What happened next?"

def test_segment_empty_input(preprocessor):
    assert preprocessor.segment_sentences("") == []
    assert preprocessor.segment_sentences("   \n\t  ") == []
    assert preprocessor.segment_sentences(None) == []

def test_clean_claim_text(preprocessor):
    assert preprocessor.clean_claim_text('  "The Earth orbits the Sun."  ') == "The Earth orbits the Sun."
    assert preprocessor.clean_claim_text("'Vaccines undergo clinical trials.'") == "Vaccines undergo clinical trials."
