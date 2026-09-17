"""
Unit tests for NLI verifier.
"""
import pytest


class TestVerifier:
    """Tests for the NLI verifier."""

    def test_mock_verifier_returns_valid_result(self, mock_verifier):
        result = mock_verifier.verify_claim("Test claim", "Test evidence")
        assert "verdict" in result
        assert "confidence" in result
        assert "scores" in result
        assert result["verdict"] in ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_verifier_handles_empty_evidence(self, mock_verifier):
        result = mock_verifier.verify_claim("Test claim", "")
        assert result["verdict"] == "UNVERIFIABLE"
        assert result["confidence"] == 0.0

    @pytest.mark.slow
    def test_real_verifier_contradicts_hallucination(self, mock_verifier):
        """This test requires the actual NLI model."""
        pytest.skip("Real verifier test requires model download")

    def test_scores_sum_to_one(self, mock_verifier):
        result = mock_verifier.verify_claim("Test", "Evidence")
        scores = result["scores"]
        total = scores["supported"] + scores["unverifiable"] + scores["contradicted"]
        assert 0.99 <= total <= 1.01
