"""
JWT authentication middleware.
Provides dependency injection for current user and API key validation.
"""
import logging
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, Security, Request
from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials,
    APIKeyHeader,
    APIKeyQuery,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..core.config import settings
from ..core.security import verify_token, verify_api_key, hash_api_key, generate_api_key
from ..core.metrics import AUTH_FAILURES
from ..models.database import User, APIKey, get_session_factory, get_db_session
from ..utils.exceptions import AuthenticationException, NotFoundException, InsufficientPermissionsException

logger = logging.getLogger(__name__)

# Security schemes
http_bearer = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


async def get_current_user(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> Optional[User]:
    """
    Extract and verify the current user from JWT or API key.
    Returns None if no auth present (for public endpoints).
    """
    # Try JWT from Authorization header
    credentials: Optional[HTTPAuthorizationCredentials] = Security(http_bearer)
    if credentials and credentials.credentials:
        token = credentials.credentials.split("Bearer ")[-1]
        payload = verify_token(token)
        if payload:
            user_id = payload.get("sub")
            if user_id:
                result = await session.execute(
                    select(User).where(User.id == int(user_id))
                )
                user = result.scalar_one_or_none()
                if user and user.is_active:
                    # Update last login
                    user.last_login = request.state.request_time if hasattr(request, 'state') else None
                    await session.commit()
                    return user
                AUTH_FAILURES.labels(reason="invalid_token").inc()
                raise AuthenticationException("Invalid or expired token")
        AUTH_FAILURES.labels(reason="invalid_token").inc()
        raise AuthenticationException("Invalid token")

    # Try API key (header or query)
    api_key_str = None
    if api_key_header:
        api_key_str = api_key_header
    elif api_key_query:
        api_key_str = api_key_query

    if api_key_str:
        api_key_hash = hash_api_key(api_key_str)
        result = await session.execute(
            select(APIKey).where(APIKey.key_hash == api_key_hash).where(APIKey.is_active == True)
        )
        api_key = result.scalar_one_or_none()
        if api_key:
            # Check expiry
            if api_key.expires_at and api_key.expires_at < request.state.request_time:
                AUTH_FAILURES.labels(reason="expired").inc()
                raise AuthenticationException("API key expired")
            # Update last used
            api_key.last_used = request.state.request_time if hasattr(request, 'state') else None
            await session.commit()
            result = await session.execute(select(User).where(User.id == api_key.user_id))
            user = result.scalar_one_or_none()
            return user
        AUTH_FAILURES.labels(reason="no_token").inc()

    return None  # Public endpoint (no auth)


async def require_auth(user: Optional[User] = Depends(get_current_user)) -> User:
    """Require that a user is authenticated."""
    if not user:
        raise AuthenticationException("Authentication required")
    return user


async def require_role(required_role: str):
    """Dependency that requires a specific role."""
    def checker(user: User = Depends(require_auth)) -> User:
        if user.role != required_role:
            raise InsufficientPermissionsException(f"requires {required_role} role")
        return user
    return checker


async def get_request_time(request: Request) -> None:
    """Middleware to set request time on request state."""
    request.state.request_time = request.state.request_time if hasattr(request.state, 'request_time') else None


# For backwards compatibility, expose generate_api_key at module level
def create_api_key(user_id: int, name: str) -> tuple[str, str, str]:
    """
    Create a new API key. Returns (full_key, key_prefix, key_hash).
    """
    full_key = generate_api_key()
    prefix = full_key[:20]
    key_hash = hash_api_key(full_key)
    return full_key, prefix, key_hash
