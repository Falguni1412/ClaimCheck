"""
Server-Sent Events (SSE) streaming service.
Provides real-time incremental results for the /verify/stream endpoint.
"""
import asyncio
import json
import logging
from typing import AsyncIterator, Dict, Any, List, Optional

from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


async def sse_format(data: Dict[str, Any], event: Optional[str] = None) -> str:
    """Format a dict as an SSE message."""
    lines = []
    if event:
        lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data, default=str)}")
    lines.append("")  # Empty line terminator
    lines.append("")
    return "\n".join(lines)


async def stream_verification(
    claims_with_evidence: AsyncIterator[Dict[str, Any]],
    request_id: str,
    total: Optional[int] = None,
) -> AsyncIterator[str]:
    """
    Stream verification results as Server-Sent Events.

    Yields:
    - event: start        — emitted once with request_id and total
    - event: claim        — emitted per-claim as it completes
    - event: complete     — emitted with summary at the end
    - event: error        — emitted on failure
    """
    try:
        # Start event
        yield await sse_format(
            {"request_id": request_id, "total_claims": total},
            event="start",
        )

        processed = 0
        contradicted = 0
        supported = 0
        unverifiable = 0

        async for result in claims_with_evidence:
            processed += 1

            if result["verdict"] == "CONTRADICTED":
                contradicted += 1
            elif result["verdict"] == "SUPPORTED":
                supported += 1
            else:
                unverifiable += 1

            yield await sse_format(result, event="claim")

            # Cooperative yield
            await asyncio.sleep(0)

        # Complete event
        risk_score = contradicted / processed if processed > 0 else 0.0
        yield await sse_format(
            {
                "request_id": request_id,
                "summary": {
                    "total_claims": processed,
                    "supported": supported,
                    "contradicted": contradicted,
                    "unverifiable": unverifiable,
                    "risk_score": round(risk_score, 3),
                },
            },
            event="complete",
        )

    except asyncio.CancelledError:
        logger.info(f"Stream cancelled for request {request_id}")
        raise
    except Exception as e:
        logger.exception(f"Stream error for request {request_id}: {e}")
        yield await sse_format(
            {"error": str(e), "request_id": request_id},
            event="error",
        )


def make_streaming_response(
    generator: AsyncIterator[str],
) -> StreamingResponse:
    """Build a StreamingResponse with SSE headers."""
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Connection": "keep-alive",
        },
    )
