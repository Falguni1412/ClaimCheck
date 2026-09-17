"""
Webhook management endpoints.
"""
import uuid
import logging
import hmac
import hashlib
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.database import Webhook, get_db_session
from ..models.schemas import WebhookCreateRequest, WebhookResponse, Verification

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def verify_webhook_signature(payload: str, signature: str, secret: str) -> bool:
    """Verify HMAC-SHA256 signature of a webhook payload."""
    expected = hmac.new(
        secret.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.post("/", response_model=WebhookResponse)
async def create_webhook(
    payload: WebhookCreateRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """Register a new webhook endpoint."""
    webhook = Webhook(
        url=payload.url,
        secret=hashlib.sha256(uuid.uuid4().hex.encode()).hexdigest(),  # Random secret
        events=payload.events,
    )
    session.add(webhook)
    await session.commit()
    await session.refresh(webhook)

    return WebhookResponse(
        id=webhook.id, url=webhook.url, events=webhook.events,
        is_active=webhook.is_active, created_at=webhook.created_at,
        secret=webhook.secret,
    )


@router.get("/", response_model=list[WebhookResponse])
async def list_webhooks(session: AsyncSession = Depends(get_db_session)):
    """List all webhook subscriptions."""
    result = await session.execute(select(Webhook).order_by(Webhook.created_at.desc()))
    webhooks = result.scalars().all()
    return [
        WebhookResponse(
            id=w.id, url=w.url, events=w.events, is_active=w.is_active,
            created_at=w.created_at, last_triggered=w.last_triggered,
            failure_count=w.failure_count,
        )
        for w in webhooks
    ]


@router.delete("/{webhook_id}")
async def delete_webhook(webhook_id: int, session: AsyncSession = Depends(get_db_session)):
    """Delete a webhook subscription."""
    result = await session.execute(select(Webhook).where(Webhook.id == webhook_id))
    webhook = result.scalar_one_or_none()
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    await session.delete(webhook)
    await session.commit()
    return {"status": "deleted", "id": webhook_id}


async def deliver_webhook(
    webhook_url: str,
    secret: str,
    event_type: str,
    payload: dict,
):
    """
    Deliver a webhook notification with HMAC signature.
    """
    import httpx

    body = json.dumps(payload).encode()
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    async with httpx.AsyncClient(timeout=settings.webhook_timeout_seconds) as client:
        try:
            response = await client.post(
                webhook_url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Webhook-Signature": signature,
                    "X-Webhook-Event": event_type,
                    "User-Agent": "ClaimCheck/1.0",
                },
            )
            if response.status_code >= 200 and response.status_code < 300:
                return True
        except Exception as e:
            logger.warning(f"Webhook delivery failed: {e}")
    return False
