"""
Pydantic schemas for request/response validation.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# ============== Auth Schemas ==============

class UserRegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=8, max_length=128)


class UserLoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    role: str
    tier: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None


# ============== API Key Schemas ==============

class APIKeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    id: int
    name: str
    key_prefix: str
    is_active: bool
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: datetime
    # Only returned on creation
    key: Optional[str] = None


# ============== Verification Schemas ==============

class VerifyRequest(BaseModel):
    """An LLM answer plus the source documents it should be checked against."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "Metformin should be taken on an empty stomach.",
                "sources": [
                    "Take metformin with meals to reduce stomach upset. "
                    "Do not take it on an empty stomach."
                ],
                "top_k": 3,
            }
        }
    )

    answer: str = Field(
        ..., min_length=1, max_length=50_000,
        description="The LLM-generated text to verify.",
    )
    sources: List[str] = Field(
        ..., min_length=1, max_length=50,
        description="Ground-truth documents. Each is capped at 100,000 characters.",
    )
    use_chunking: bool = Field(
        default=True,
        description="Split long sources into passages before retrieval. Leave on unless sources are already short.",
    )
    top_k: int = Field(default=3, ge=1, le=10, description="Passages retrieved per claim.")
    tags: Optional[List[str]] = Field(default=None, max_length=10, description="Free-form labels stored with the result.")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Reserved for forward-compatible flags.")

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("answer cannot be blank")
        return v

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: List[str]) -> List[str]:
        cleaned: List[str] = []
        for i, s in enumerate(v):
            if not s or not s.strip():
                raise ValueError(f"Source {i} is empty")
            if len(s) > 100_000:
                raise ValueError(f"Source {i} exceeds 100,000 character limit")
            cleaned.append(s)
        return cleaned

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        tags = [t.strip() for t in v if t and t.strip()]
        for t in tags:
            if len(t) > 64:
                raise ValueError("Each tag must be 64 characters or fewer")
        return tags or None


class VerdictScores(BaseModel):
    supported: float
    unverifiable: float
    contradicted: float


class ClaimResult(BaseModel):
    claim: str
    verdict: Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: str = ""
    evidence_score: Optional[float] = None
    verdict_scores: Optional[VerdictScores] = None
    numerical_check: Optional[Dict[str, Any]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    explanation: Optional[str] = None


class VerificationSummary(BaseModel):
    total_claims: int
    supported: int
    contradicted: int
    unverifiable: int
    processing_time_seconds: float
    average_confidence: Optional[float] = None


class VerifyResponse(BaseModel):
    request_id: str
    claims: List[ClaimResult]
    risk_score: float = Field(..., ge=0.0, le=1.0)
    summary: VerificationSummary
    metadata: Dict[str, Any]
    cached: bool = False


# ============== Batch Verification ==============

class BatchVerifyRequest(BaseModel):
    items: List[VerifyRequest] = Field(..., min_length=1, max_length=50)
    parallel: bool = Field(default=True)
    fail_fast: bool = Field(default=False)


class BatchItemError(BaseModel):
    index: int
    error: str


class BatchVerifyResponse(BaseModel):
    results: List[VerifyResponse]
    total: int
    successful: int
    failed: int
    total_processing_time_seconds: float
    errors: Optional[List[BatchItemError]] = Field(
        default=None, description="Per-item failures when fail_fast is false."
    )


# ============== Webhook Schemas ==============

class WebhookCreateRequest(BaseModel):
    url: str = Field(..., pattern=r"^https?://.+")
    events: List[Literal["verification.completed", "verification.failed"]] = Field(
        default_factory=lambda: ["verification.completed"]
    )


class WebhookResponse(BaseModel):
    id: int
    url: str
    events: List[str]
    is_active: bool
    created_at: datetime
    last_triggered: Optional[datetime] = None
    failure_count: int
    # Only returned on creation
    secret: Optional[str] = None


# ============== History Schemas ==============

class VerificationHistoryItem(BaseModel):
    id: int
    request_id: str
    risk_score: float
    summary: VerificationSummary
    tags: Optional[List[str]] = None
    created_at: datetime


class VerificationHistoryList(BaseModel):
    items: List[VerificationHistoryItem]
    total: int
    page: int
    page_size: int


class VerificationDetail(VerifyResponse):
    id: int
    user_id: Optional[int] = None
    tags: Optional[List[str]] = None
    created_at: Optional[datetime] = None


# ============== Health / Stats ==============

class HealthResponse(BaseModel):
    """Liveness: the process is up and serving. Cheap, never touches the model."""
    status: str
    version: str
    timestamp: str


class ReadinessResponse(BaseModel):
    """Readiness: the process can actually serve a verification right now."""
    status: str
    ready: bool
    model_loaded: bool
    version: str
    timestamp: str
    uptime_seconds: float
    database: bool
    cache: bool
    retrieval: Dict[str, bool] = Field(default_factory=dict)


class ErrorDetail(BaseModel):
    code: str
    message: str
    extra: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Every non-2xx response from this API has this shape."""
    error: ErrorDetail
    detail: str = Field(description="Mirror of error.message, for older clients.")
    path: str
    request_id: Optional[str] = None


class StatsResponse(BaseModel):
    total_verifications: int
    average_risk_score: float
    total_claims: int
    supported: int
    contradicted: int
    unverifiable: int
    verifications_last_24h: int
    average_latency_ms: float
    cache_hit_rate: float
