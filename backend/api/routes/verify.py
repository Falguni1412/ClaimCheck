"""
Verification endpoints: /verify, /verify/batch, /verify/stream.
"""
import asyncio
import hashlib
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..core.metrics import (
    CLAIMS_VERIFIED,
    VERIFICATION_LATENCY,
    VERIFICATION_RISK_SCORE,
    VERIFY_REQUESTS,
)
from ..models.database import Verification, get_db_session
from ..models.schemas import (
    BatchVerifyRequest,
    BatchVerifyResponse,
    VerificationSummary,
    VerifyRequest,
    VerifyResponse,
)
from ..services.cache import CacheService, get_cache
from ..services.decomposer import decompose_into_claims
from ..services.retriever import chunk_document, get_retriever
from ..services.streaming import make_streaming_response, stream_verification
from ..services.verifier import NLIVerifier, get_verifier, verifier_is_loaded
from ..utils.exceptions import (
    DecompositionException,
    InvalidRequestException,
    ModelNotLoadedException,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix=settings.api_prefix, tags=["verification"])

# Module-level singletons, populated by init_verifier() on startup.
_verifier: Optional[NLIVerifier] = None
_cache_service: Optional[CacheService] = None

# One in-flight inference at a time by default. Two concurrent transformer
# forward passes on a 512MB instance is how you get an OOM kill instead of a
# slow response.
_inference_semaphore: Optional[asyncio.Semaphore] = None


def _semaphore() -> asyncio.Semaphore:
    global _inference_semaphore
    if _inference_semaphore is None:
        _inference_semaphore = asyncio.Semaphore(max(1, settings.max_concurrent_inferences))
    return _inference_semaphore


def get_verifier_instance() -> NLIVerifier:
    """Dependency-style accessor for the process-wide verifier."""
    return _verifier or get_verifier()


async def init_verifier() -> None:
    """Load models and connect the cache. Called once from the app lifespan."""
    global _verifier, _cache_service

    _cache_service = get_cache()
    await _cache_service.connect()

    loop = asyncio.get_running_loop()
    # Model loading is blocking and slow; keep the event loop free so the
    # platform health check can still be answered during a cold start.
    _verifier = await loop.run_in_executor(None, get_verifier)
    logger.info("Verification services initialized")


def models_ready() -> bool:
    return _verifier is not None or verifier_is_loaded()


# ---------------------------------------------------------------- helpers


def _prepare_chunks(request: VerifyRequest) -> List[str]:
    """Turn raw sources into retrievable chunks."""
    all_chunks: List[str] = []
    for source in request.sources:
        if request.use_chunking:
            all_chunks.extend(
                chunk_document(
                    source,
                    chunk_size=settings.chunk_size,
                    overlap=settings.chunk_overlap,
                )
            )
        else:
            all_chunks.append(source)
    # chunk_document drops fragments below min length; never return an empty corpus.
    return [c for c in all_chunks if c and c.strip()] or [s for s in request.sources if s.strip()]


def _sources_hash(sources: List[str]) -> str:
    return hashlib.sha256("||".join(sources).encode("utf-8", "ignore")).hexdigest()


def _empty_claim_result(claim_text: str) -> Dict[str, Any]:
    return {
        "claim": claim_text,
        "verdict": "UNVERIFIABLE",
        "confidence": 0.0,
        "evidence": "",
        "verdict_scores": {"supported": 0.0, "unverifiable": 1.0, "contradicted": 0.0},
        "numerical_check": None,
        "explanation": "No evidence found",
    }


def _summarise(results: List[Dict[str, Any]], elapsed: float) -> Tuple[VerificationSummary, float]:
    total = len(results)
    supported = sum(1 for r in results if r["verdict"] == "SUPPORTED")
    contradicted = sum(1 for r in results if r["verdict"] == "CONTRADICTED")
    unverifiable = total - supported - contradicted
    risk_score = contradicted / total if total else 0.0

    summary = VerificationSummary(
        total_claims=total,
        supported=supported,
        contradicted=contradicted,
        unverifiable=unverifiable,
        processing_time_seconds=round(elapsed, 3),
        average_confidence=(
            round(sum(r["confidence"] for r in results) / total, 3) if total else 0.0
        ),
    )
    return summary, risk_score


def _retrieve_all(chunks: List[str], claim_texts: List[str], top_k: int) -> List[str]:
    """Blocking retrieval for every claim. Runs in a worker thread."""
    retriever = get_retriever(chunks)
    evidence: List[str] = []
    for text in claim_texts:
        passages = retriever.retrieve(
            text, top_k=top_k, rerank=settings.use_cross_encoder_rerank
        )
        evidence.append(passages[0] if passages else "")
    return evidence


async def _persist_verification(
    session: Optional[AsyncSession],
    response: VerifyResponse,
    request: VerifyRequest,
    model_used: str,
) -> None:
    """
    Best-effort write to the history table.

    A failure here must never fail the verification the caller asked for — but
    unlike the previous version, it is at least attempted, so /history returns
    something other than an empty list.
    """
    if session is None or not settings.persist_verifications:
        return
    try:
        session.add(
            Verification(
                request_id=response.request_id,
                answer=request.answer[:100_000],
                sources_hash=_sources_hash(request.sources),
                sources_count=len(request.sources),
                claims=[c.model_dump() for c in response.claims],
                summary=response.summary.model_dump(),
                risk_score=response.risk_score,
                processing_time_ms=int(response.summary.processing_time_seconds * 1000),
                model_used=model_used,
                tags=request.tags,
            )
        )
        await session.flush()
    except Exception as e:
        logger.warning("Could not persist verification %s: %s", response.request_id, e)


# ---------------------------------------------------------------- core


async def run_verification(
    request: VerifyRequest,
    session: Optional[AsyncSession] = None,
) -> VerifyResponse:
    """
    Verify one LLM answer against its sources.

    1. Check cache (dedup by content hash)
    2. Decompose answer into claims
    3. Retrieve relevant evidence per claim
    4. Classify all claims in batched forward passes
    5. Summarise into verdicts + risk score
    """
    if not request.answer or not request.answer.strip():
        raise InvalidRequestException("Answer cannot be empty")
    if not request.sources:
        raise InvalidRequestException("At least one source document is required")

    verifier = _verifier or (get_verifier() if verifier_is_loaded() else None)
    if verifier is None:
        raise ModelNotLoadedException("NLI")

    cache = _cache_service or get_cache()
    cache_key = "verify:" + cache.hash_key(
        {
            "answer": request.answer,
            "sources": sorted(request.sources),
            "top_k": request.top_k,
            "use_chunking": request.use_chunking,
            "model": verifier.model_name,
        }
    )
    cached = await cache.get(cache_key)
    if cached:
        try:
            payload = dict(cached)
            payload["cached"] = True
            VERIFY_REQUESTS.labels(user_tier="free", status="cached").inc()
            return VerifyResponse(**payload)
        except Exception as e:
            logger.warning("Discarding malformed cache entry %s: %s", cache_key, e)
            await cache.delete(cache_key)

    start_time = time.perf_counter()

    claims = decompose_into_claims(request.answer)
    if not claims:
        raise DecompositionException()
    if len(claims) > settings.max_claims_per_request:
        raise InvalidRequestException(
            f"Too many claims ({len(claims)}). Maximum is {settings.max_claims_per_request}. "
            "Use /verify/batch for longer answers."
        )

    chunks = _prepare_chunks(request)
    if not chunks:
        raise InvalidRequestException("Could not process source documents")

    claim_texts = [c.text for c in claims]
    loop = asyncio.get_running_loop()

    # Retrieval and inference are both CPU-bound; hold the semaphore across
    # both so total resident model work stays bounded.
    async with _semaphore():
        evidence = await loop.run_in_executor(
            None, _retrieve_all, chunks, claim_texts, request.top_k
        )
        verifications = await verifier.verify_pairs_async(
            list(zip(claim_texts, evidence)), True
        )

    results: List[Dict[str, Any]] = []
    for claim_text, best_evidence, verification in zip(claim_texts, evidence, verifications):
        if not best_evidence:
            results.append(_empty_claim_result(claim_text))
            CLAIMS_VERIFIED.labels(verdict="UNVERIFIABLE").inc()
            continue
        results.append(
            {
                "claim": claim_text,
                "verdict": verification["verdict"],
                "confidence": verification["confidence"],
                "evidence": best_evidence,
                "verdict_scores": verification["scores"],
                "numerical_check": verification.get("numerical_check"),
                "explanation": verification.get("explanation"),
            }
        )
        CLAIMS_VERIFIED.labels(verdict=verification["verdict"]).inc()

    elapsed = time.perf_counter() - start_time
    summary, risk_score = _summarise(results, elapsed)

    response_data = VerifyResponse(
        request_id=str(uuid.uuid4()),
        claims=results,
        risk_score=round(risk_score, 3),
        summary=summary,
        metadata={
            "model": verifier.model_name,
            "sources_processed": len(chunks),
            "ensemble": verifier.use_ensemble and verifier.secondary_model is not None,
            "numerical_check_enabled": True,
            "semantic_retrieval": settings.enable_semantic_retrieval,
            "timestamp": time.time(),
        },
    )

    await cache.set(cache_key, response_data.model_dump(), ttl_seconds=settings.cache_ttl)
    await _persist_verification(session, response_data, request, verifier.model_name)

    VERIFY_REQUESTS.labels(user_tier="free", status="completed").inc()
    VERIFICATION_LATENCY.observe(elapsed)
    VERIFICATION_RISK_SCORE.observe(risk_score)

    return response_data


# ---------------------------------------------------------------- routes


@router.post(
    "/verify",
    response_model=VerifyResponse,
    summary="Verify an LLM answer against source documents",
    response_description="Per-claim verdicts, evidence, and an aggregate risk score",
)
async def verify(
    request: VerifyRequest,
    session: AsyncSession = Depends(get_db_session),
) -> VerifyResponse:
    """
    Break an LLM answer into atomic claims, retrieve the most relevant passage
    from the supplied sources for each, and classify the claim as
    **SUPPORTED**, **CONTRADICTED**, or **UNVERIFIABLE**.

    `risk_score` is the fraction of claims that were contradicted, so 0.0 means
    nothing in the answer conflicts with the sources.
    """
    return await run_verification(request, session=session)


@router.post(
    "/verify/batch",
    response_model=BatchVerifyResponse,
    summary="Verify multiple LLM answers in one request",
)
async def verify_batch(
    batch: BatchVerifyRequest,
    session: AsyncSession = Depends(get_db_session),
) -> BatchVerifyResponse:
    """
    Verify up to 50 answers sequentially.

    Items are processed in order; with `fail_fast: false` (the default) a failed
    item is counted in `failed` and the rest still run.
    """
    if not models_ready():
        raise ModelNotLoadedException("NLI")

    results: List[VerifyResponse] = []
    successful = 0
    failed = 0
    errors: List[Dict[str, Any]] = []

    for index, item in enumerate(batch.items):
        try:
            results.append(await run_verification(item, session=session))
            successful += 1
        except Exception as e:
            if batch.fail_fast:
                raise
            logger.warning("Batch item %d failed: %s", index, e)
            errors.append({"index": index, "error": str(e)})
            failed += 1

    return BatchVerifyResponse(
        results=results,
        total=len(batch.items),
        successful=successful,
        failed=failed,
        errors=errors or None,
        total_processing_time_seconds=round(
            sum(r.summary.processing_time_seconds for r in results), 3
        ),
    )


@router.post(
    "/verify/stream",
    summary="Verify with Server-Sent Events streaming",
    response_description="text/event-stream of start / claim / complete events",
)
async def verify_stream(request: VerifyRequest):
    """
    Same pipeline as `/verify`, but each claim is emitted as soon as it is
    classified. Events: `start`, `claim` (one per claim), `complete`, `error`.
    """
    verifier = _verifier or (get_verifier() if verifier_is_loaded() else None)
    if verifier is None:
        raise ModelNotLoadedException("NLI")

    if not request.answer or not request.answer.strip():
        raise InvalidRequestException("Answer cannot be empty")

    claims = decompose_into_claims(request.answer)
    if not claims:
        raise DecompositionException()

    chunks = _prepare_chunks(request)
    if not chunks:
        raise InvalidRequestException("Could not process source documents")

    claim_texts = [c.text for c in claims]
    request_id = str(uuid.uuid4())

    async def claim_generator():
        """Yield one verification result per claim, off the event loop."""
        loop = asyncio.get_running_loop()
        async with _semaphore():
            evidence = await loop.run_in_executor(
                None, _retrieve_all, chunks, claim_texts, request.top_k
            )
            for claim_text, best_evidence in zip(claim_texts, evidence):
                if not best_evidence:
                    yield _empty_claim_result(claim_text)
                    continue
                scored = await verifier.verify_pairs_async([(claim_text, best_evidence)])
                result = dict(scored[0])
                result["claim"] = claim_text
                result["evidence"] = best_evidence
                result["verdict_scores"] = result.pop("scores", None)
                yield result

    return make_streaming_response(
        stream_verification(claim_generator(), request_id=request_id, total=len(claims))
    )
