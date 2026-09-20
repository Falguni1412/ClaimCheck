"""
Centralized configuration for ClaimCheck.
Loads from environment variables with sensible defaults.

Notes on list-valued settings
-----------------------------
pydantic-settings v2 parses ``List[str]`` env vars as JSON, so
``CORS_ORIGINS=http://a.com,http://b.com`` raises a SettingsError at import
time and the process dies before it serves a single request. List-ish settings
are therefore stored as plain strings and exposed through parsed properties.
"""
import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _split_csv(raw: str) -> List[str]:
    """Parse a comma-separated (or JSON-list) env value into a list of strings."""
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        import json
        try:
            return [str(x).strip() for x in json.loads(raw) if str(x).strip()]
        except Exception:
            pass
    return [part.strip() for part in raw.split(",") if part.strip()]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),  # allow model_* setting names without warnings
    )

    # App metadata
    app_name: str = "ClaimCheck"
    app_version: str = "2.1.0"
    environment: str = Field(default="development", description="development | staging | production | test")
    debug: bool = False

    # API settings
    api_prefix: str = ""
    # Comma-separated list. "*" allows all origins (credentials are then disabled,
    # because browsers reject Access-Control-Allow-Origin: * with credentials).
    cors_origins: str = "*"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Database (SQLite by default for local/dev; override for Postgres in production)
    database_url: str = "sqlite+aiosqlite:///./claimcheck.db"
    database_pool_size: int = 5
    database_max_overflow: int = 10
    persist_verifications: bool = True

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 512  # in-memory entries; verification payloads are large

    # JWT / Auth
    jwt_secret: str = "change-me-in-production-please-use-a-long-random-string"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    api_key_prefix: str = "cck_"

    # Rate limiting
    rate_limit_enabled: bool = False  # off by default for local demo
    rate_limit_default: str = "60/minute"
    rate_limit_free: str = "20/minute"
    rate_limit_pro: str = "200/minute"
    rate_limit_enterprise: str = "2000/minute"

    # ---------------- Models ----------------
    nli_model_name: str = "cross-encoder/nli-MiniLM2-L6-H768"
    nli_model_secondary: str = "cross-encoder/nli-MiniLM2-L6-H768"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_cross_encoder_rerank: bool = False  # extra model = extra RAM; opt in
    device: Optional[str] = None  # auto-detect cuda/cpu
    model_quantization: bool = True  # INT8 dynamic quantization for CPU inference
    use_ensemble: bool = False  # single model by default for faster local runs

    # Explicit label order override for NLI heads whose config only reports
    # LABEL_0/1/2. Comma-separated, e.g. "contradiction,entailment,neutral".
    nli_label_order: str = ""

    # Inference tuning
    nli_max_length: int = 256          # 512 doubles latency for little accuracy gain
    nli_batch_size: int = 8            # claims per forward pass
    torch_num_threads: int = 1         # >1 thrashes on a 0.1-0.5 vCPU free tier
    model_warmup: bool = True          # run one dummy forward pass at startup
    max_concurrent_inferences: int = 1 # hard cap on simultaneous model work

    # Retrieval
    enable_semantic_retrieval: bool = True   # set false on <512MB instances
    retriever_cache_size: int = 8            # corpora kept warm (BM25 + embeddings)
    bm25_weight: float = 0.5
    semantic_weight: float = 0.5

    # Verification defaults
    default_top_k: int = 3
    max_source_length: int = 100_000  # Max chars per source
    max_sources: int = 50
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_claims_per_request: int = 100
    # Risk weights
    risk_weight_contradicted: float = 1.0
    risk_weight_unverifiable: float = 0.35
    risk_boost_numerical: float = 1.5
    risk_boost_medical: float = 1.35

    # Streaming
    enable_streaming: bool = True
    stream_chunk_size: int = 1  # Emit per-claim

    # Webhooks
    webhook_timeout_seconds: int = 30
    webhook_max_retries: int = 3

    # Observability
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_path: str = "/metrics"

    # Storage (S3 / file)
    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 50

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        return (v or "INFO").upper()

    @field_validator("api_prefix")
    @classmethod
    def _normalise_prefix(cls, v: str) -> str:
        """FastAPI rejects a prefix of "/" and requires no trailing slash."""
        v = (v or "").strip()
        if v in ("", "/"):
            return ""
        if not v.startswith("/"):
            v = "/" + v
        return v.rstrip("/")

    # ---------------- Derived properties ----------------

    @property
    def cors_origin_list(self) -> List[str]:
        origins = _split_csv(self.cors_origins)
        return origins or ["*"]

    @property
    def cors_allow_all(self) -> bool:
        return "*" in self.cors_origin_list

    @property
    def nli_label_order_list(self) -> List[str]:
        return [x.lower() for x in _split_csv(self.nli_label_order)]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def is_testing(self) -> bool:
        return self.environment == "test" or bool(os.getenv("PYTEST_CURRENT_TEST"))

    def startup_warnings(self) -> List[str]:
        """Config problems worth shouting about at boot, not at 3am."""
        warnings: List[str] = []
        if self.is_production:
            if self.jwt_secret.startswith("change-me"):
                warnings.append("JWT_SECRET is still the default value — set a real secret.")
            if self.cors_allow_all:
                warnings.append("CORS_ORIGINS is '*' — set explicit origins in production.")
            if self.database_url.startswith("sqlite"):
                warnings.append(
                    "DATABASE_URL is SQLite — history will be lost on redeploy "
                    "(free tier filesystems are ephemeral)."
                )
        return warnings


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Module-level singleton
settings = get_settings()
