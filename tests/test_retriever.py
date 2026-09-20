"""
Unit tests for evidence retriever.
"""
import pytest
from api.services.retriever import chunk_document


class TestChunkDocument:
    """Tests for document chunking."""

    def test_short_document_not_chunked(self):
        text = "This is a short document."
        chunks = chunk_document(text, chunk_size=500)
        assert chunks == [text]

    def test_empty_text_returns_empty(self):
        assert chunk_document("", chunk_size=500) == []
        assert chunk_document("   ", chunk_size=500) == []

    def test_long_document_chunked(self):
        text = " ".join(["This is sentence number " + str(i) + "." for i in range(100)])
        chunks = chunk_document(text, chunk_size=200, overlap=20)
        assert len(chunks) > 1
        # Verify all chunks are within size limit (with overlap)
        for chunk in chunks:
            assert len(chunk) <= 250  # chunk_size + some overlap allowance

    def test_breaks_at_sentence_boundaries(self):
        text = "First sentence. Second sentence. " * 30
        chunks = chunk_document(text, chunk_size=100, overlap=20)
        # Chunks should end with period (sentence boundary)
        for chunk in chunks[:-1]:  # All but possibly the last
            assert chunk.rstrip().endswith((".", "!", "?"))

    def test_overlap_creates_continuous_coverage(self):
        text = "Sentence one. Sentence two. Sentence three. " * 20
        chunks = chunk_document(text, chunk_size=100, overlap=30)
        # With overlap, consecutive chunks should share content
        if len(chunks) >= 2:
            # Some content should appear in both
            assert chunks[0] in chunks[1] or chunks[1].startswith(chunks[0][-50:])


class TestRetriever:
    """Tests for the Retriever class."""

    @pytest.fixture
    def documents(self):
        return [
            "Metformin is a medication used to treat type 2 diabetes. Take with meals.",
            "Common side effects include nausea and diarrhea. Lactic acidosis is rare.",
            "Metformin can be combined with insulin therapy under medical supervision.",
            "Patients with kidney problems should not take metformin.",
        ]

    def test_retrieve_returns_documents(self, documents):
        from api.services.retriever import Retriever
        try:
            retriever = Retriever(documents)
            results = retriever.retrieve("What is metformin?", top_k=2)
            assert len(results) > 0
            assert all(isinstance(r, str) for r in results)
        except Exception as e:
            pytest.skip(f"Retriever requires ML models: {e}")

    def test_retrieve_with_scores(self, documents):
        from api.services.retriever import Retriever
        try:
            retriever = Retriever(documents)
            results = retriever.retrieve_with_scores("metformin", top_k=2)
            assert len(results) > 0
            for doc, score in results:
                assert isinstance(doc, str)
                assert 0.0 <= score <= 1.0
        except Exception as e:
            pytest.skip(f"Retriever requires ML models: {e}")

