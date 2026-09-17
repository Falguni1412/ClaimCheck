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

from ..models.database import User, APIKey, get_db_session
from ..models.schemas import UserResponse, APIKeyCreateRequest, APIKeyResponse
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
async def register(payload: dict, session: AsyncSession = Depends(get_db_session)):
    """Create a new user account."""
    # Check existing
    existing = await session.execute(
        select(User).where(User.email == payload.get("email"))
    )
    if existing.scalar_one_or_none():
        raise InvalidRequestException("Email already registered")

    hashed_pw = hash_password(payload["password"])
    new_user = User(
        email=payload["email"],
        username=payload["username"],
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


@router.get("/stats")
async def get_admin_stats(
    session: AsyncSession = Depends(get_db_session),
):
    """Get system-wide statistics."""
    # Total verifications
    total_query = select(func.count()).select_from(Verification)
    total = await session.scalar(total_query) or 0

    # Average risk score
    avg_query = select(func.avg(Verification.risk_score)).select_from(Verification)
    avg_risk = await session.scalar(avg_query) or 0.0

    # Count by verdict
    verdict_query = select(Verification.summary)
    # Simplified count
    return {
        "total_verifications": total,
        "average_risk_score": round(avg_risk, 3),
        "total_claims": total,  # Placeholder
        "cache_hit_rate": 0.0,  # Would come from cache metrics
        "uptime_seconds": 0,  # Would track startup time
        "environment": settings.environment,
        "app_version": settings.app_version,
    }
