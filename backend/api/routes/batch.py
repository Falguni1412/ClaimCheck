"""Batch pre-flight validation endpoint."""
from fastapi import APIRouter

from ..core.config import settings
from ..models.schemas import BatchVerifyRequest

router = APIRouter(tags=["verification"])

MAX_BATCH_ITEMS = 50


@router.post("/batch/validate", summary="Pre-flight check for a batch payload")
async def validate_batch(batch: BatchVerifyRequest):
    """
    Validate a batch body without running any inference.

    Useful for checking size limits and getting a rough duration estimate
    before committing to a long request.
    """
    total_claims_estimate = sum(
        max(1, len(item.answer) // 120) for item in batch.items
    )
    return {
        "valid": True,
        "total_items": len(batch.items),
        "max_supported": MAX_BATCH_ITEMS,
        "estimated_claims": total_claims_estimate,
        "estimated_time_seconds": round(total_claims_estimate * 0.35, 1),
        "max_claims_per_item": settings.max_claims_per_request,
    }
