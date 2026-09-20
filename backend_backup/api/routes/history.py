"""
Verification history endpoints.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.database import Verification, get_db_session
from ..models.schemas import (
    VerificationDetail,
    VerificationHistoryItem,
    VerificationHistoryList,
    VerificationSummary,  # was used but never imported -> NameError on every call
)
from ..utils.exceptions import NotFoundException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["history"])

_EMPTY_SUMMARY = {
    "total_claims": 0, "supported": 0, "contradicted": 0,
    "unverifiable": 0, "processing_time_seconds": 0.0,
}


def _summary_of(raw) -> VerificationSummary:
    """Tolerate rows written by older versions with a missing/odd summary blob."""
    data = dict(_EMPTY_SUMMARY)
    if isinstance(raw, dict):
        for key in _EMPTY_SUMMARY:
            if raw.get(key) is not None:
                data[key] = raw[key]
        if raw.get("average_confidence") is not None:
            data["average_confidence"] = raw["average_confidence"]
    return VerificationSummary(**data)


@router.get(
    "/",
    response_model=VerificationHistoryList,
    summary="List past verifications (newest first)",
)
async def get_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    session: AsyncSession = Depends(get_db_session),
):
    """Paginated verification history. `total` reflects the same filter as `items`."""
    query = select(Verification)
    count_query = select(func.count()).select_from(Verification)
    if user_id is not None:
        query = query.where(Verification.user_id == user_id)
        # The count must honour the filter too, or page counts lie.
        count_query = count_query.where(Verification.user_id == user_id)

    total = await session.scalar(count_query) or 0

    result = await session.execute(
        query.order_by(desc(Verification.created_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = result.scalars().all()

    return VerificationHistoryList(
        items=[
            VerificationHistoryItem(
                id=item.id,
                request_id=item.request_id,
                risk_score=item.risk_score,
                summary=_summary_of(item.summary),
                tags=item.tags,
                created_at=item.created_at,
            )
            for item in items
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{request_id}",
    response_model=VerificationDetail,
    summary="Full result of one verification",
)
async def get_verification_detail(
    request_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Retrieve the stored claims, verdicts, and evidence for a `request_id`."""
    result = await session.execute(
        select(Verification).where(Verification.request_id == request_id)
    )
    item = result.scalar_one_or_none()
    if not item:
        raise NotFoundException("Verification", request_id)

    return VerificationDetail(
        id=item.id,
        user_id=item.user_id,
        request_id=item.request_id,
        claims=item.claims or [],
        risk_score=item.risk_score,
        summary=_summary_of(item.summary),
        # NOTE: `item.metadata` is SQLAlchemy's MetaData object, not user data.
        # Reading it here (as the previous version did) produced an unserialisable
        # response. Rebuild the metadata block from stored columns instead.
        metadata={
            "model": item.model_used,
            "sources_count": item.sources_count,
            "sources_hash": item.sources_hash,
            "processing_time_ms": item.processing_time_ms,
        },
        cached=False,
        tags=item.tags,
        created_at=item.created_at,
    )
