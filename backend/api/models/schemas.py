"""
Pydantic schemas for request/response validation.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, EmailStr, field_validator


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
    answer: str = Field(..., min_length=1, max_length=50_000)
    sources: List[str] = Field(..., min_length=1, max_length=50)
    use_chunking: bool = Field(default=True)
    top_k: int = Field(default=3, ge=1, le=10)
    tags: Optional[List[str]] = Field(default=None, max_length=10)
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: List[str]) -> List[str]:
        for i, s in enumerate(v):
            if not s or not s.strip():
                raise ValueError(f"Source {i} is empty")
            if len(s) > 100_000:
                raise ValueError(f"Source {i} exceeds 100,000 character limit")
        return v


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


class BatchVerifyResponse(BaseModel):
    results: List[VerifyResponse]
    total: int
    successful: int
    failed: int
    total_processing_time_seconds: float


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
    user_id: Optional[int]
    tags: Optional[List[str]] = None


# ============== Health / Stats ==============

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    timestamp: str
    database: bool
    cache: bool


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
