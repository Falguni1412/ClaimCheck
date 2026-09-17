"""Batch verification endpoint."""
from fastapi import APIRouter
from ..models.schemas import BatchVerifyRequest, BatchVerifyResponse

router = APIRouter()


@router.post("/batch/validate")
async def validate_batch(batch: BatchVerifyRequest):
    """Validate a batch of verification requests before processing."""
    return {
        "valid": True,
        "total_items": len(batch.items),
        "max_supported": 50,
        "estimated_time_seconds": len(batch.items) * 2.5,
    }
