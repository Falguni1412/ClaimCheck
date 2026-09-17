"""
Verification history endpoints.
"""
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.database import Verification, get_db_session
from ..models.schemas import (
    VerificationHistoryList,
    VerificationHistoryItem,
    VerificationDetail,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["history"])


@router.get("/", response_model=VerificationHistoryList)
async def get_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    session: AsyncSession = Depends(get_db_session),
):
    """
    Get verification history with pagination.
    """
    query = select(Verification)
    if user_id:
        query = query.where(Verification.user_id == user_id)
    query = query.order_by(desc(Verification.created_at))

    # Get total
    count_query = select(func.count()).select_from(Verification)
    total = await session.scalar(count_query)

    # Paginate
    offset = (page - 1) * page_size
    result = await session.execute(
        query.offset(offset).limit(page_size)
    )
    items = result.scalars().all()

    return VerificationHistoryList(
        items=[
            VerificationHistoryItem(
                id=item.id,
                request_id=item.request_id,
                risk_score=item.risk_score,
                summary=VerificationSummary(
                    total_claims=item.summary.get("total_claims", 0),
                    supported=item.summary.get("supported", 0),
                    contradicted=item.summary.get("contradicted", 0),
                    unverifiable=item.summary.get("unverifiable", 0),
                    processing_time_seconds=item.summary.get("processing_time_seconds", 0),
                ) if isinstance(item.summary, dict) else VerificationSummary(
                    total_claims=0, supported=0, contradicted=0,
                    unverifiable=0, processing_time_seconds=0,
                ),
                tags=item.tags,
                created_at=item.created_at,
            )
            for item in items
        ],
        total=total or 0,
        page=page,
        page_size=page_size,
    )


@router.get("/{request_id}", response_model=VerificationDetail)
async def get_verification_detail(
    request_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Get details of a specific verification request."""
    result = await session.execute(
        select(Verification).where(Verification.request_id == request_id)
    )
    item = result.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail=f"Verification {request_id} not found")

    return VerificationDetail(
        id=item.id,
        request_id=item.request_id,
        claims=item.claims,
        risk_score=item.risk_score,
        summary=item.summary,
        metadata=item.metadata if hasattr(item, "metadata") else {},
        cached=False,
        tags=item.tags,
    )
