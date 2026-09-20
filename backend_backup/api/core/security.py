"""
Security utilities: JWT, password hashing, API key generation.
"""
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import settings


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False


def create_access_token(
    subject: str | int,
    extra_claims: Optional[Dict[str, Any]] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        subject: User ID or identifier
        extra_claims: Additional claims to include
        expires_delta: Custom expiration time
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.jwt_access_token_expire_minutes)

    expire = datetime.now(timezone.utc) + expires_delta
    to_encode: Dict[str, Any] = {
        "sub": str(subject),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }
    if extra_claims:
        to_encode.update(extra_claims)

    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_refresh_token(
    subject: str | int,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a JWT refresh token."""
    expire = datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_token_expire_days)
    to_encode: Dict[str, Any] = {
        "sub": str(subject),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "refresh",
    }
    if extra_claims:
        to_encode.update(extra_claims)

    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode a JWT token.

    Returns:
        Decoded claims if valid, None otherwise
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError:
        return None


def generate_api_key() -> str:
    """
    Generate a secure API key.

    Format: cck_<32 random hex chars>
    """
    random_part = secrets.token_hex(32)
    return f"{settings.api_key_prefix}{random_part}"


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for storage.
    Uses SHA-256 (faster than bcrypt for high-throughput key lookups).
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(plain_key: str, hashed_key: str) -> bool:
    """Verify an API key against its hash."""
    return hash_api_key(plain_key) == hashed_key
