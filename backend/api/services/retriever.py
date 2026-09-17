"""
Enhanced evidence retriever.
Hybrid BM25 + semantic search with Redis-cached embeddings and cross-encoder reranking.
"""
import logging
from typing import List, Tuple, Optional, Dict, Any
import hashlib
import numpy as np

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class Retriever:
    """
    Hybrid retriever combining:
    - BM25 (lexical match, fast, exact keyword coverage)
    - Semantic embeddings (paraphrase & conceptual similarity)
    - Optional cross-encoder reranking (slower but more precise)

    Embeddings are cached in Redis to avoid recomputation.
    """

    def __init__(
        self,
        source_documents: List[str],
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        cross_encoder_model: Optional[str] = None,
        cache_service: Optional[Any] = None,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
    ):
        self.documents = source_documents

        if not source_documents:
            raise ValueError("source_documents cannot be empty")

        # BM25 setup
        if BM25_AVAILABLE:
            tokenized_docs = [doc.lower().split() for doc in source_documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            self.bm25 = None
            logger.warning("rank_bm25 not available; using semantic-only retrieval")

        # Embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedder = SentenceTransformer(embedding_model)
            self._compute_doc_embeddings(cache_service)
        else:
            self.embedder = None
            self.doc_embeddings = None
            logger.warning("sentence-transformers not available; using BM25-only retrieval")

        # Optional cross-encoder reranker
        self.cross_encoder = None
        if cross_encoder_model and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_model)
                logger.info(f"Loaded cross-encoder: {cross_encoder_model}")
            except Exception as e:
                logger.warning(f"Could not load cross-encoder {cross_encoder_model}: {e}")

        # Weights for hybrid scoring
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

    def _compute_doc_embeddings(self, cache_service: Optional[Any]) -> None:
        """Compute (or fetch from cache) embeddings for all documents."""
        if not self.embedder:
            self.doc_embeddings = None
            return

        # Hash the corpus to use as a cache key
        corpus_hash = hashlib.sha256(
            "||".join(self.documents).encode()
        ).hexdigest()[:16]
        cache_key = f"embeddings:{corpus_hash}"

        if cache_service:
            import asyncio
            try:
                cached = asyncio.get_event_loop().run_until_complete(
                    cache_service.get(cache_key)
                )
                if cached is not None:
                    logger.info(f"Loaded {len(cached)} embeddings from cache")
                    self.doc_embeddings = np.array(cached)
                    return
            except RuntimeError:
                # No event loop in sync context; skip cache
                pass

        # Compute fresh
        self.doc_embeddings = self.embedder.encode(
            self.documents,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )

        if cache_service:
            import asyncio
            try:
                asyncio.get_event_loop().run_until_complete(
                    cache_service.set(cache_key, self.doc_embeddings.tolist(), ttl_seconds=86400)
                )
            except RuntimeError:
                pass

    def retrieve(
        self,
        claim: str,
        top_k: int = 3,
        rerank: bool = False,
    ) -> List[str]:
        """Retrieve top-k most relevant passages for a claim."""
        results = self.retrieve_with_scores(claim, top_k=top_k * 2 if rerank else top_k, rerank=rerank)
        return [doc for doc, _ in results]

    def retrieve_with_scores(
        self,
        claim: str,
        top_k: int = 3,
        rerank: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve passages with relevance scores.

        If rerank=True, fetches more candidates and reranks with cross-encoder.
        """
        if not self.documents:
            return []

        scores: Optional[np.ndarray] = None

        # BM25
        if self.bm25:
            bm25_scores = self.bm25.get_scores(claim.lower().split())
            bm25_norm = self._normalize(bm25_scores)
            scores = bm25_norm * self.bm25_weight if scores is None else scores + bm25_norm * self.bm25_weight

        # Semantic
        if self.embedder and self.doc_embeddings is not None:
            claim_embedding = self.embedder.encode([claim], normalize_embeddings=True, show_progress_bar=False)[0]
            semantic_scores = np.dot(self.doc_embeddings, claim_embedding)
            semantic_norm = self._normalize(semantic_scores)
            scores = semantic_norm * self.semantic_weight if scores is None else scores + semantic_norm * self.semantic_weight

        if scores is None:
            # Pure lexical fallback
            scores = np.ones(len(self.documents))

        # Get top candidates
        candidate_k = min(top_k * 3, len(self.documents)) if rerank else top_k
        top_indices = np.argsort(scores)[-candidate_k:][::-1]

        candidates = [(self.documents[i], float(scores[i])) for i in top_indices]

        # Optional cross-encoder rerank
        if rerank and self.cross_encoder and candidates:
            try:
                pairs = [(claim, doc) for doc, _ in candidates]
                ce_scores = self.cross_encoder.predict(pairs)
                candidates = [(doc, float(score)) for (doc, _), score in zip(candidates, ce_scores)]
                candidates.sort(key=lambda x: x[1], reverse=True)
            except Exception as e:
                logger.warning(f"Cross-encoder rerank failed: {e}")

        return candidates[:top_k]

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1]."""
        if scores.max() == scores.min():
            return np.ones_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())


def chunk_document(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    min_chunk_size: int = 50,
) -> List[str]:
    """
    Split a long document into smaller chunks for better retrieval.

    Tries to break at sentence boundaries; falls back to character splits.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Try to break at a sentence boundary within the last 100 chars
            for i in range(end, max(start + chunk_size - 100, start), -1):
                if i < len(text) and text[i] in ".!?\n":
                    end = i + 1
                    break

        chunk = text[start:end].strip()
        if chunk and len(chunk) >= min_chunk_size:
            chunks.append(chunk)

        start = end - overlap
        if start < 0:
            start = end

    return chunks
