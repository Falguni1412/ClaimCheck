"""
Admin endpoints for managing users, API keys, and system configuration.
"""
import uuid
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.database import APIKey, User, Verification, get_db_session  # Verification was missing -> NameError in /admin/stats
from ..models.schemas import APIKeyCreateRequest, APIKeyResponse, UserRegisterRequest, UserResponse
from ..core.config import settings
from ..core.security import hash_password, verify_password, create_access_token, create_refresh_token, generate_api_key, hash_api_key
from ..utils.exceptions import AuthenticationException, InvalidRequestException, NotFoundException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


@router.post("/login", response_model=LoginResponse)
async def login(payload: LoginRequest, session: AsyncSession = Depends(get_db_session)):
    """Authenticate with username/password and receive JWT tokens."""
    result = await session.execute(
        select(User).where(User.username == payload.username)
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(payload.password, user.hashed_password):
        raise AuthenticationException("Invalid username or password")

    access_token = create_access_token(user.id, {"role": user.role, "tier": user.tier})
    refresh_token = create_refresh_token(user.id)

    # Update last login
    from datetime import datetime
    user.last_login = datetime.utcnow()
    await session.commit()

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.jwt_access_token_expire_minutes * 60,
        user=UserResponse(
            id=user.id, email=user.email, username=user.username,
            role=user.role, tier=user.tier, is_active=user.is_active,
            created_at=user.created_at, last_login=user.last_login,
        ),
    )


@router.post("/register", response_model=UserResponse)
async def register(
    payload: UserRegisterRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """
    Create a new user account.

    Previously accepted a raw dict and raised a bare KeyError (500) when a field
    was missing; it is now validated, so a bad body returns 422 with the field.
    """
    existing = await session.execute(
        select(User).where(
            (User.email == payload.email) | (User.username == payload.username)
        )
    )
    if existing.scalar_one_or_none():
        raise InvalidRequestException("Email or username already registered")

    hashed_pw = hash_password(payload.password)
    new_user = User(
        email=payload.email,
        username=payload.username,
        hashed_password=hashed_pw,
        role="user",
        tier="free",
    )
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)

    return UserResponse(
        id=new_user.id, email=new_user.email, username=new_user.username,
        role=new_user.role, tier=new_user.tier, is_active=new_user.is_active,
        created_at=new_user.created_at,
    )


@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    payload: APIKeyCreateRequest,
    user: User = Depends(lambda: None),  # Would need auth in practice
    session: AsyncSession = Depends(get_db_session),
):
    """Create a new API key for programmatic access."""
    # In production, this would require authentication
    full_key = generate_api_key()
    prefix = full_key[:20]
    key_hash = hash_api_key(full_key)

    api_key = APIKey(
        user_id=user.id if user else 1,  # Default to first user for demo
        name=payload.name,
        key_hash=key_hash,
        key_prefix=prefix,
        expires_at=None,
    )
    if payload.expires_in_days:
        from datetime import datetime, timedelta
        api_key.expires_at = datetime.utcnow() + timedelta(days=payload.expires_in_days)

    session.add(api_key)
    await session.commit()
    await session.refresh(api_key)

    return APIKeyResponse(
        id=api_key.id, name=api_key.name, key_prefix=api_key.key_prefix,
        is_active=api_key.is_active, created_at=api_key.created_at, key=full_key,
    )


@router.get("/stats", summary="System-wide verification statistics")
async def get_admin_stats(session: AsyncSession = Depends(get_db_session)):
    """Aggregate counters across all stored verifications."""
    total = await session.scalar(select(func.count()).select_from(Verification)) or 0
    avg_risk = await session.scalar(select(func.avg(Verification.risk_score))) or 0.0
    avg_latency_ms = await session.scalar(
        select(func.avg(Verification.processing_time_ms))
    ) or 0.0

    # Claim-level totals come from the stored summary blobs.
    claims = supported = contradicted = unverifiable = 0
    rows = await session.execute(select(Verification.summary).limit(5000))
    for (summary,) in rows:
        if isinstance(summary, dict):
            claims += int(summary.get("total_claims") or 0)
            supported += int(summary.get("supported") or 0)
            contradicted += int(summary.get("contradicted") or 0)
            unverifiable += int(summary.get("unverifiable") or 0)

    return {
        "total_verifications": total,
        "average_risk_score": round(float(avg_risk), 3),
        "average_latency_ms": round(float(avg_latency_ms), 1),
        "total_claims": claims,
        "supported": supported,
        "contradicted": contradicted,
        "unverifiable": unverifiable,
        "environment": settings.environment,
        "app_version": settings.app_version,
    }
