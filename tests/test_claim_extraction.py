import pytest
from src.claim_extraction.claim_extractor import ClaimExtractor

@pytest.fixture
def extractor():
    return ClaimExtractor(min_words=4, max_claims=10)

def test_extract_from_normal_article(extractor):
    article = (
        "NASA announced that the James Webb Space Telescope discovered an exoplanet in 2023. "
        "The telescope uses infrared instruments to capture distant galaxies. "
        "Scientists confirmed the findings in a peer-reviewed publication."
    )
    claims = extractor.extract_claims(article)
    assert len(claims) == 3
    assert claims[0].claim_id == "claim_001"
    assert "James Webb Space Telescope" in claims[0].claim_text
    assert claims[0].is_verifiable is True

def test_extract_empty_and_short_articles(extractor):
    assert extractor.extract_claims("") == []
    assert extractor.extract_claims("   ") == []
    # Very short sentence (< 4 words)
    assert extractor.extract_claims("It happened.") == []

def test_filter_opinions_and_questions(extractor):
    text = (
        "In my opinion, this is the best phone ever made. "
        "Did Apple release the iPhone 15 in September 2023? "
        "Apple released the iPhone 15 in September 2023. "
        "Hello and welcome to our technology blog!"
    )
    claims = extractor.extract_claims(text)
    # Only the factual declarative sentence should be extracted as candidate claim
    assert len(claims) == 1
    assert "Apple released the iPhone 15" in claims[0].claim_text

def test_filter_repeated_sentences(extractor):
    text = (
        "The Eiffel Tower is located in Paris, France. "
        "The Eiffel Tower is located in Paris, France. "
        "The Eiffel Tower is located in Paris, France."
    )
    claims = extractor.extract_claims(text)
    # Deduplication should preserve only 1 claim
    assert len(claims) == 1

def test_extract_with_quotations(extractor):
    text = 'The government announced "unemployment dropped to 3.5% in the third quarter."'
    claims = extractor.extract_claims(text)
    assert len(claims) == 1
    assert "unemployment dropped to 3.5%" in claims[0].claim_text

def test_malformed_input(extractor):
    text = "....   ???? !!!!   "
    claims = extractor.extract_claims(text)
    assert claims == []
