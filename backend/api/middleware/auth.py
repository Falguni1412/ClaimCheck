"""
JWT / API-key authentication dependencies.

These are real FastAPI dependencies. The previous version called
``Security(http_bearer)`` inside a function body (which returns a marker object,
not credentials) and hashed the ``APIKeyHeader`` *instance* rather than the
submitted key, so any request carrying auth headers would have been rejected or
crashed. Nothing was wired to it, so it failed silently; it now works if you
attach it to a route.

Usage:
    @router.post("/verify", dependencies=[Depends(require_user)])
    async def verify(...): ...
or, for optional auth:
    async def verify(user: Optional[User] = Depends(get_current_user)): ...
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, Security
from fastapi.security import (
    APIKeyHeader,
    APIKeyQuery,
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.metrics import AUTH_FAILURES
from ..core.security import hash_api_key, verify_token
from ..models.database import APIKey, User, get_db_session
from ..utils.exceptions import AuthenticationException, InsufficientPermissionsException

logger = logging.getLogger(__name__)

http_bearer = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(http_bearer),
    header_key: Optional[str] = Security(api_key_header),
    query_key: Optional[str] = Security(api_key_query),
    session: AsyncSession = Depends(get_db_session),
) -> Optional[User]:
    """
    Resolve the caller from a bearer JWT or an API key.

    Returns None when no credentials are supplied, so the same dependency works
    for public endpoints. Supplying *invalid* credentials is an error.
    """
    now = datetime.now(timezone.utc)

    if credentials and credentials.credentials:
        payload = verify_token(credentials.credentials)
        if not payload or payload.get("type") != "access":
            AUTH_FAILURES.labels(reason="invalid_token").inc()
            raise AuthenticationException("Invalid or expired token")

        subject = payload.get("sub")
        try:
            user_id = int(subject)
        except (TypeError, ValueError):
            AUTH_FAILURES.labels(reason="invalid_token").inc()
            raise AuthenticationException("Invalid token subject")

        user = (
            await session.execute(select(User).where(User.id == user_id))
        ).scalar_one_or_none()
        if not user or not user.is_active:
            AUTH_FAILURES.labels(reason="inactive_user").inc()
            raise AuthenticationException("Account not found or disabled")
        user.last_login = now.replace(tzinfo=None)
        return user

    submitted_key = header_key or query_key
    if submitted_key:
        record = (
            await session.execute(
                select(APIKey)
                .where(APIKey.key_hash == hash_api_key(submitted_key))
                .where(APIKey.is_active.is_(True))
            )
        ).scalar_one_or_none()
        if not record:
            AUTH_FAILURES.labels(reason="unknown_api_key").inc()
            raise AuthenticationException("Invalid API key")
        if record.expires_at and record.expires_at < now.replace(tzinfo=None):
            AUTH_FAILURES.labels(reason="expired").inc()
            raise AuthenticationException("API key expired")

        record.last_used = now.replace(tzinfo=None)
        user = (
            await session.execute(select(User).where(User.id == record.user_id))
        ).scalar_one_or_none()
        if not user or not user.is_active:
            AUTH_FAILURES.labels(reason="inactive_user").inc()
            raise AuthenticationException("Account not found or disabled")
        return user

    return None


async def require_user(user: Optional[User] = Depends(get_current_user)) -> User:
    """Dependency for endpoints that must have an authenticated caller."""
    if user is None:
        AUTH_FAILURES.labels(reason="missing_credentials").inc()
        raise AuthenticationException("Authentication required")
    return user


async def require_admin(user: User = Depends(require_user)) -> User:
    """Dependency for admin-only endpoints."""
    if user.role != "admin":
        raise InsufficientPermissionsException("perform this action")
    return user
