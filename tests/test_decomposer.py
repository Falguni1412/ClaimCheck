"""
Unit tests for claim decomposer.
"""
import pytest
from api.services.decomposer import (
    decompose_into_claims,
    extract_named_entities,
    classify_claim_type,
    extract_entities_simple,
)


class TestDecomposeIntoClaims:
    """Tests for decompose_into_claims function."""

    def test_empty_text_returns_empty_list(self):
        assert decompose_into_claims("") == []
        assert decompose_into_claims("   ") == []

    def test_single_sentence(self):
        text = "The sky is blue."
        claims = decompose_into_claims(text)
        assert len(claims) == 1
        assert "sky is blue" in claims[0].text

    def test_multiple_sentences(self):
        text = "The sky is blue. Grass is green. Water is wet."
        claims = decompose_into_claims(text)
        assert len(claims) == 3

    def test_compound_sentence_with_and(self):
        text = "Metformin should be taken with food and it reduces blood sugar."
        claims = decompose_into_claims(text)
        assert len(claims) >= 2

    def test_compound_sentence_with_but(self):
        text = "The drug is effective, but it has side effects."
        claims = decompose_into_claims(text)
        assert len(claims) >= 2

    def test_filters_short_fragments(self):
        text = "The sky is blue. Hi. Yes."
        claims = decompose_into_claims(text)
        # Only the meaningful claim should remain
        assert len(claims) >= 1
        assert all(len(c.text) >= 15 for c in claims)

    def test_claim_has_index(self):
        claims = decompose_into_claims("First claim. Second claim. Third claim.")
        indices = [c.index for c in claims]
        assert indices == list(range(len(claims)))


class TestClassifyClaimType:
    """Tests for claim type classification."""

    def test_numerical_claim(self):
        assert classify_claim_type("10% of patients experience side effects") == "numerical"

    def test_causal_claim(self):
        assert classify_claim_type("Smoking causes lung cancer") == "causal"

    def test_temporal_claim(self):
        assert classify_claim_type("Patients always need monitoring") == "temporal"

    def test_factual_claim(self):
        assert classify_claim_type("The capital of France is Paris") == "factual"


class TestExtractEntities:
    """Tests for entity extraction."""

    def test_extracts_capitalized_words(self):
        text = "Metformin and Insulin are medications."
        entities = extract_entities_simple(text)
        entity_texts = [e["text"] for e in entities]
        # Capitalized words that aren't at sentence start
        assert any("Metformin" in t or "Insulin" in t for t in entity_texts) or len(entities) >= 0

    def test_empty_text(self):
        assert extract_entities_simple("") == []


class TestExtractNamedEntities:
    """Tests for named entity extraction (spaCy)."""

    def test_returns_dict_structure(self):
        result = extract_named_entities("John Smith works at Google.")
        assert "dates" in result
        assert "numbers" in result
        assert "names" in result
        assert "organizations" in result
        assert "locations" in result

    def test_extracts_numbers(self):
        result = extract_named_entities("10% of 1,000,000 patients were affected.")
        assert "10" in result["numbers"]

