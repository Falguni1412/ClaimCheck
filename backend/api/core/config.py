"""
Centralized configuration for ClaimCheck.
Loads from environment variables with sensible defaults.
"""
from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App metadata
    app_name: str = "ClaimCheck"
    app_version: str = "2.0.0"
    environment: str = Field(default="development", description="development | staging | production")
    debug: bool = False

    # API settings
    api_prefix: str = ""
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Database (SQLite by default for local/dev; override for Postgres in production)
    database_url: str = "sqlite+aiosqlite:///./claimcheck.db"
    database_pool_size: int = 5
    database_max_overflow: int = 10

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 10000

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

    # Models
    nli_model_name: str = "cross-encoder/nli-MiniLM2-L6-H768"
    nli_model_secondary: str = "cross-encoder/nli-MiniLM2-L6-H768"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_cross_encoder_rerank: bool = True
    device: Optional[str] = None  # auto-detect cuda/cpu
    model_quantization: bool = True  # Use INT8 quantization for CPU inference
    use_ensemble: bool = False  # single model by default for faster local runs

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

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Module-level singleton
settings = get_settings()
