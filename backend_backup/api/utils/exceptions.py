"""
Custom exception types for ClaimCheck.
Provides structured error handling with HTTP status codes.
"""
from fastapi import HTTPException, status
from typing import Optional, Any, Dict


class ClaimCheckException(HTTPException):
    """Base exception for ClaimCheck errors."""

    def __init__(
        self,
        detail: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.extra = extra or {}


class ModelNotLoadedException(ClaimCheckException):
    """Raised when a required model is not loaded."""

    def __init__(self, model_name: str = "NLI"):
        super().__init__(
            detail=f"{model_name} model is not loaded. Please restart the server.",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="MODEL_NOT_LOADED",
        )


class InvalidRequestException(ClaimCheckException):
    """Raised when request input is invalid."""

    def __init__(self, detail: str, error_code: str = "INVALID_REQUEST"):
        super().__init__(
            detail=detail,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=error_code,
        )


class AuthenticationException(ClaimCheckException):
    """Raised when authentication fails."""

    def __init__(self, detail: str = "Authentication required"):
        super().__init__(
            detail=detail,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="AUTH_REQUIRED",
        )


class RateLimitException(ClaimCheckException):
    """Raised when rate limit is exceeded."""

    def __init__(self, retry_after: int = 60):
        super().__init__(
            detail="Rate limit exceeded. Please try again later.",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMITED",
            extra={"retry_after_seconds": retry_after},
        )


class NotFoundException(ClaimCheckException):
    """Raised when a resource is not found."""

    def __init__(self, resource: str, identifier: Any = None):
        detail = f"{resource} not found"
        if identifier is not None:
            detail = f"{resource} '{identifier}' not found"
        super().__init__(
            detail=detail,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
        )


class InsufficientPermissionsException(ClaimCheckException):
    """Raised when user lacks required permissions."""

    def __init__(self, action: str):
        super().__init__(
            detail=f"Insufficient permissions to {action}",
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="FORBIDDEN",
        )


class DecompositionException(ClaimCheckException):
    """Raised when claim decomposition fails."""

    def __init__(self, detail: str = "Could not decompose answer into claims"):
        super().__init__(
            detail=detail,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="DECOMPOSITION_FAILED",
        )


class RetrievalException(ClaimCheckException):
    """Raised when evidence retrieval fails."""

    def __init__(self, detail: str = "Evidence retrieval failed"):
        super().__init__(
            detail=detail,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="RETRIEVAL_FAILED",
        )
