"""
Verification endpoints: /verify, /verify/batch, /verify/stream.
"""
import uuid
import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..core.metrics import (
    VERIFY_REQUESTS,
    VERIFICATION_LATENCY,
    VERIFICATION_RISK_SCORE,
)
from ..models.database import Verification, get_db_session, User
from ..models.schemas import (
    VerifyRequest,
    VerifyResponse,
    BatchVerifyRequest,
    BatchVerifyResponse,
    VerificationSummary,
)
from ..services.decomposer import decompose_into_claims, Claim
from ..services.retriever import Retriever, chunk_document
from ..services.verifier import NLIVerifier
from ..services.cache import CacheService
from ..services.streaming import stream_verification, make_streaming_response
from ..utils.exceptions import (
    InvalidRequestException,
    ModelNotLoadedException,
    DecompositionException,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix=settings.api_prefix, tags=["verification"])


def get_verifier() -> NLIVerifier:
    """Get the global NLI verifier instance."""
    return _verifier


# Module-level verifier instance
_verifier: Optional[NLIVerifier] = None
_cache_service: Optional[CacheService] = None


async def init_verifier():
    """Initialize the NLI verifier and cache on startup."""
    global _verifier, _cache_service
    _verifier = NLIVerifier()
    _cache_service = CacheService()
    await _cache_service.connect()
    logger.info("Verification services initialized")


@router.post("/verify", response_model=VerifyResponse)
async def verify(
    request: VerifyRequest,
    user: Optional[User] = Depends(lambda: None),  # Optional auth
    session: AsyncSession = Depends(get_db_session),
):
    """
    Verify an LLM answer against source documents.

    Process:
    1. Check cache (dedup by content hash)
    2. Decompose answer into claims
    3. Retrieve relevant evidence
    4. Classify each claim
    5. Return verdicts + risk score
    """
    if not request.answer or not request.answer.strip():
        raise InvalidRequestException("Answer cannot be empty")

    if not request.sources or len(request.sources) == 0:
        raise InvalidRequestException("At least one source document is required")

    if _verifier is None:
        raise ModelNotLoadedException("NLI")

    # Check cache first
    cache_key = None
    if _cache_service:
        content_hash = _cache_service.hash_key({
            "answer": request.answer,
            "sources": sorted(request.sources),
            "top_k": request.top_k,
        })
        cache_key = f"verify:{content_hash}"
        cached = await _cache_service.get(cache_key)
        if cached:
            cached["cached"] = True
            return cached

    start_time = asyncio.get_event_loop().time()

    # Step 1: Decompose
    claims = decompose_into_claims(request.answer)
    if not claims:
        raise DecompositionException()

    if len(claims) > settings.max_claims_per_request:
        raise InvalidRequestException(
            f"Too many claims ({len(claims)}). Maximum is {settings.max_claims_per_request}. "
            "Use /verify/batch for longer answers."
        )

    # Step 2: Process sources
    all_chunks = []
    for source in request.sources:
        if request.use_chunking:
            chunks = chunk_document(source, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
            all_chunks.extend(chunks)
        else:
            all_chunks.append(source)

    if not all_chunks:
        raise InvalidRequestException("Could not process source documents")

    # Step 3: Initialize retriever with cache
    retriever = Retriever(all_chunks, cache_service=_cache_service)

    # Step 4: Verify each claim
    results = []
    contradicted_count = 0
    supported_count = 0
    unverifiable_count = 0

    for claim_obj in claims:
        # Retrieve top evidence
        evidence_passages = retriever.retrieve(
            claim_obj.text, top_k=request.top_k
        )
        best_evidence = evidence_passages[0] if evidence_passages else ""

        if not best_evidence:
            results.append({
                "claim": claim_obj.text,
                "verdict": "UNVERIFIABLE",
                "confidence": 0.0,
                "evidence": "",
                "verdict_scores": {"supported": 0.0, "unverifiable": 1.0, "contradicted": 0.0},
                "numerical_check": None,
                "explanation": "No evidence found",
            })
            unverifiable_count += 1
            continue

        # Verify claim
        verification = _verifier.verify_claim(
            claim_obj.text, best_evidence, run_numerical_check=True
        )

        if verification["verdict"] == "CONTRADICTED":
            contradicted_count += 1
        elif verification["verdict"] == "SUPPORTED":
            supported_count += 1
        else:
            unverifiable_count += 1

        results.append({
            "claim": claim_obj.text,
            "verdict": verification["verdict"],
            "confidence": verification["confidence"],
            "evidence": best_evidence,
            "verdict_scores": verification["scores"],
            "numerical_check": verification.get("numerical_check"),
            "explanation": verification.get("explanation"),
        })

    # Step 5: Risk score
    total_claims = len(results)
    risk_score = contradicted_count / total_claims if total_claims > 0 else 0.0
    processing_time = asyncio.get_event_loop().time() - start_time

    # Build response
    summary = VerificationSummary(
        total_claims=total_claims,
        supported=supported_count,
        contradicted=contradicted_count,
        unverifiable=unverifiable_count,
        processing_time_seconds=round(processing_time, 2),
        average_confidence=sum(r["confidence"] for r in results) / total_claims if total_claims > 0 else 0.0,
    )

    response_data = VerifyResponse(
        request_id=str(uuid.uuid4()),
        claims=results,
        risk_score=round(risk_score, 3),
        summary=summary,
        metadata={
            "model": _verifier.model_name,
            "sources_processed": len(all_chunks),
            "ensemble": _verifier.use_ensemble,
            "numerical_check_enabled": True,
            "timestamp": asyncio.get_event_loop().time(),
        },
    )

    # Cache the result
    if _cache_service and cache_key:
        await _cache_service.set(cache_key, response_data.model_dump(), ttl_seconds=settings.cache_ttl)

    # Track metrics
    VERIFY_REQUESTS.labels(user_tier="free", status="completed").inc()
    VERIFICATION_LATENCY.observe(processing_time)
    VERIFICATION_RISK_SCORE.observe(risk_score)

    return response_data


@router.post("/verify/batch", response_model=BatchVerifyResponse)
async def verify_batch(
    batch: BatchVerifyRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """
    Verify multiple LLM answers in a single request.
    """
    if _verifier is None:
        raise ModelNotLoadedException("NLI")

    results = []
    successful = 0
    failed = 0

    for item in batch.items:
        try:
            # Reuse /verify logic
            response = await verify(item, session=session)
            results.append(response)
            successful += 1
        except Exception as e:
            if batch.fail_fast:
                raise
            logger.warning(f"Batch item failed: {e}")
            failed += 1

    total_time = sum(
        r.summary.processing_time_seconds for r in results if r.summary
    )

    return BatchVerifyResponse(
        results=results,
        total=len(batch.items),
        successful=successful,
        failed=failed,
        total_processing_time_seconds=round(total_time, 2),
    )


@router.post("/verify/stream")
async def verify_stream(
    request: VerifyRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """
    Verify with Server-Sent Events streaming.
    Returns incremental results as each claim is verified.
    """
    if _verifier is None:
        raise ModelNotLoadedException("NLI")

    # Decompose first
    claims = decompose_into_claims(request.answer)
    if not claims:
        raise DecompositionException()

    all_chunks = []
    for source in request.sources:
        if request.use_chunking:
            chunks = chunk_document(source, chunk_size=settings.chunk_size)
            all_chunks.extend(chunks)
        else:
            all_chunks.append(source)

    if not all_chunks:
        raise InvalidRequestException("Could not process source documents")

    retriever = Retriever(all_chunks)
    verifier = _verifier

    async def claim_generator():
        """Async generator that yields verification results per claim."""
        for i, claim_obj in enumerate(claims):
            evidence_passages = retriever.retrieve(claim_obj.text, top_k=request.top_k)
            best_evidence = evidence_passages[0] if evidence_passages else ""

            if not best_evidence:
                result = {
                    "claim": claim_obj.text,
                    "verdict": "UNVERIFIABLE",
                    "confidence": 0.0,
                    "evidence": "",
                    "verdict_scores": {"supported": 0.0, "unverifiable": 1.0, "contradicted": 0.0},
                    "numerical_check": None,
                    "explanation": "No evidence found",
                }
            else:
                result = verifier.verify_claim(claim_obj.text, best_evidence)
                result["claim"] = claim_obj.text
                result["evidence"] = best_evidence

            yield result
            await asyncio.sleep(0.05)  # Allow other coroutines to run

    # Return SSE streaming response
    return make_streaming_response(
        stream_verification(claim_generator(), request_id=str(uuid.uuid4()), total=len(claims))
    )
