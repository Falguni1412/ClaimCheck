"""
Evidence retriever: hybrid BM25 + semantic search with optional cross-encoder
reranking.

Performance notes
-----------------
The previous version constructed ``SentenceTransformer(...)`` inside
``Retriever.__init__``, and ``/verify`` constructed a ``Retriever`` per request
— so every single request downloaded/deserialised an embedding model and
re-embedded the whole corpus. Embedding models are now process-level
singletons, and built retrievers are kept in a small LRU keyed by corpus hash,
so repeat requests against the same sources skip both the BM25 build and the
embedding pass entirely.
"""
import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.config import settings

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on install profile
    BM25_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on install profile
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# ---------------- process-level model singletons ----------------

_model_lock = threading.Lock()
_embedder: Optional[Any] = None
_embedder_name: Optional[str] = None
_cross_encoder: Optional[Any] = None
_cross_encoder_name: Optional[str] = None


def get_embedder(model_name: Optional[str] = None) -> Optional[Any]:
    """Load the embedding model once per process; None if unavailable/disabled."""
    global _embedder, _embedder_name

    if not SENTENCE_TRANSFORMERS_AVAILABLE or not settings.enable_semantic_retrieval:
        return None

    name = model_name or settings.embedding_model
    with _model_lock:
        if _embedder is not None and _embedder_name == name:
            return _embedder
        try:
            logger.info("Loading embedding model '%s'...", name)
            model = SentenceTransformer(name, device=settings.device or "cpu")
            model.max_seq_length = min(getattr(model, "max_seq_length", 512) or 512, 256)
            _embedder, _embedder_name = model, name
            logger.info("Embedding model ready")
        except Exception as e:
            logger.warning("Could not load embedding model '%s': %s", name, e)
            _embedder, _embedder_name = None, None
    return _embedder


def get_cross_encoder(model_name: Optional[str] = None) -> Optional[Any]:
    """Load the reranker once per process; None if unavailable/disabled."""
    global _cross_encoder, _cross_encoder_name

    if not SENTENCE_TRANSFORMERS_AVAILABLE or not settings.use_cross_encoder_rerank:
        return None

    name = model_name or settings.cross_encoder_model
    with _model_lock:
        if _cross_encoder is not None and _cross_encoder_name == name:
            return _cross_encoder
        try:
            logger.info("Loading cross-encoder '%s'...", name)
            _cross_encoder = CrossEncoder(name, device=settings.device or "cpu", max_length=256)
            _cross_encoder_name = name
        except Exception as e:
            logger.warning("Could not load cross-encoder '%s': %s", name, e)
            _cross_encoder, _cross_encoder_name = None, None
    return _cross_encoder


def retrieval_backends() -> Dict[str, bool]:
    """What retrieval is actually available right now (surfaced by /ready)."""
    return {
        "bm25": BM25_AVAILABLE,
        "semantic": bool(get_embedder()),
        "rerank": bool(get_cross_encoder()),
    }


class Retriever:
    """
    Hybrid retriever combining:
    - BM25 (lexical match, fast, exact keyword coverage)
    - Semantic embeddings (paraphrase & conceptual similarity)
    - Optional cross-encoder reranking (slower but more precise)
    """

    def __init__(
        self,
        source_documents: List[str],
        embedding_model: Optional[str] = None,
        cross_encoder_model: Optional[str] = None,
        cache_service: Optional[Any] = None,  # kept for API compatibility
        bm25_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
    ):
        if not source_documents:
            raise ValueError("source_documents cannot be empty")

        self.documents = source_documents
        self.bm25_weight = settings.bm25_weight if bm25_weight is None else bm25_weight
        self.semantic_weight = settings.semantic_weight if semantic_weight is None else semantic_weight

        # BM25 setup — cheap, always on when installed.
        if BM25_AVAILABLE:
            self.bm25 = BM25Okapi([doc.lower().split() for doc in source_documents])
        else:
            self.bm25 = None
            logger.warning("rank_bm25 not available; using semantic-only retrieval")

        # Embeddings (shared model, corpus embedded once per Retriever).
        self.embedder = get_embedder(embedding_model)
        self.doc_embeddings: Optional[np.ndarray] = None
        if self.embedder is not None:
            try:
                self.doc_embeddings = np.asarray(
                    self.embedder.encode(
                        self.documents,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        batch_size=16,
                        convert_to_numpy=True,
                    ),
                    dtype=np.float32,
                )
            except Exception as e:
                logger.warning("Embedding the corpus failed; falling back to BM25: %s", e)
                self.doc_embeddings = None

        if self.bm25 is None and self.doc_embeddings is None:
            logger.warning(
                "Neither BM25 nor embeddings are available; retrieval will return "
                "documents in their original order."
            )

        self.cross_encoder = get_cross_encoder(cross_encoder_model)

    # ---------------- retrieval ----------------

    def retrieve(self, claim: str, top_k: int = 3, rerank: bool = False) -> List[str]:
        """Retrieve top-k most relevant passages for a claim."""
        return [doc for doc, _ in self.retrieve_with_scores(claim, top_k=top_k, rerank=rerank)]

    def retrieve_with_scores(
        self,
        claim: str,
        top_k: int = 3,
        rerank: bool = False,
    ) -> List[Tuple[str, float]]:
        """Retrieve passages with relevance scores, optionally cross-encoder reranked."""
        if not self.documents:
            return []

        top_k = max(1, min(top_k, len(self.documents)))
        scores: Optional[np.ndarray] = None

        if self.bm25 is not None:
            bm25_norm = self._normalize(np.asarray(self.bm25.get_scores(claim.lower().split())))
            scores = bm25_norm * self.bm25_weight

        if self.embedder is not None and self.doc_embeddings is not None:
            try:
                claim_embedding = np.asarray(
                    self.embedder.encode(
                        [claim], normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True,
                    )[0],
                    dtype=np.float32,
                )
                semantic_norm = self._normalize(self.doc_embeddings @ claim_embedding)
                weighted = semantic_norm * self.semantic_weight
                scores = weighted if scores is None else scores + weighted
            except Exception as e:
                logger.warning("Semantic scoring failed for one claim: %s", e)

        if scores is None:
            scores = np.ones(len(self.documents), dtype=np.float32)

        candidate_k = min(top_k * 3, len(self.documents)) if (rerank and self.cross_encoder) else top_k
        top_indices = np.argsort(scores)[-candidate_k:][::-1]
        candidates = [(self.documents[i], float(scores[i])) for i in top_indices]

        if rerank and self.cross_encoder and candidates:
            try:
                ce_scores = self.cross_encoder.predict([(claim, doc) for doc, _ in candidates])
                candidates = [(doc, float(s)) for (doc, _), s in zip(candidates, ce_scores)]
                candidates.sort(key=lambda x: x[1], reverse=True)
            except Exception as e:
                logger.warning("Cross-encoder rerank failed: %s", e)

        return candidates[:top_k]

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1]."""
        scores = np.asarray(scores, dtype=np.float32)
        if scores.size == 0:
            return scores
        smin, smax = float(scores.min()), float(scores.max())
        if smax == smin:
            return np.ones_like(scores)
        return (scores - smin) / (smax - smin)


# ---------------- bounded retriever cache ----------------

_retriever_cache: "OrderedDict[str, Retriever]" = OrderedDict()
_retriever_lock = threading.Lock()


def corpus_key(documents: List[str]) -> str:
    """Stable fingerprint for a corpus."""
    digest = hashlib.sha256()
    for doc in documents:
        digest.update(doc.encode("utf-8", "ignore"))
        digest.update(b"\x00")
    return digest.hexdigest()[:32]


def get_retriever(documents: List[str], cache_service: Optional[Any] = None) -> Retriever:
    """
    Return a Retriever for this corpus, reusing a warm one when possible.

    Bounded by ``RETRIEVER_CACHE_SIZE`` so a busy instance cannot accumulate
    embedding matrices until the OOM killer intervenes.
    """
    key = corpus_key(documents)
    with _retriever_lock:
        cached = _retriever_cache.get(key)
        if cached is not None:
            _retriever_cache.move_to_end(key)
            return cached

    retriever = Retriever(documents, cache_service=cache_service)

    with _retriever_lock:
        _retriever_cache[key] = retriever
        _retriever_cache.move_to_end(key)
        while len(_retriever_cache) > max(1, settings.retriever_cache_size):
            _retriever_cache.popitem(last=False)
    return retriever


def clear_retriever_cache() -> None:
    with _retriever_lock:
        _retriever_cache.clear()


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
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Guard against a pathological overlap making the loop non-advancing.
    overlap = max(0, min(overlap, chunk_size - 1))

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

        next_start = end - overlap
        start = next_start if next_start > start else end

    return chunks
