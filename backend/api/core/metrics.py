"""
Prometheus metrics for ClaimCheck.
Exposes counters, histograms, and gauges for monitoring.
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


# Application info
APP_INFO = Info(
    "claimcheck_app",
    "ClaimCheck application metadata",
)

APP_INFO.info({
    "version": "1.0.0",
    "component": "backend",
})


# Request metrics
REQUEST_COUNT = Counter(
    "claimcheck_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "claimcheck_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=(
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
    ),
)

ACTIVE_REQUESTS = Gauge(
    "claimcheck_active_requests",
    "Number of in-flight requests",
    ["endpoint"],
)


# Verification-specific metrics
VERIFY_REQUESTS = Counter(
    "claimcheck_verify_requests_total",
    "Total verification requests",
    ["user_tier", "status"],
)

CLAIMS_VERIFIED = Counter(
    "claimcheck_claims_verified_total",
    "Total claims verified",
    ["verdict"],
)

VERIFICATION_LATENCY = Histogram(
    "claimcheck_verification_duration_seconds",
    "End-to-end verification latency",
    buckets=(
        0.1,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        30.0,
        60.0,
    ),
)

VERIFICATION_RISK_SCORE = Histogram(
    "claimcheck_risk_score",
    "Distribution of risk scores",
    buckets=(
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ),
)


# Cache metrics
CACHE_HITS = Counter(
    "claimcheck_cache_hits_total",
    "Cache hits",
    ["cache_type"],
)

CACHE_MISSES = Counter(
    "claimcheck_cache_misses_total",
    "Cache misses",
    ["cache_type"],
)

CACHE_SIZE = Gauge(
    "claimcheck_cache_size",
    "Number of items in cache",
    ["cache_type"],
)


# Model metrics
MODEL_INFERENCE_LATENCY = Histogram(
    "claimcheck_model_inference_seconds",
    "Model inference time",
    ["model", "operation"],
)

MODEL_LOADED = Gauge(
    "claimcheck_model_loaded",
    "Whether a model is loaded (1) or not (0)",
    ["model"],
)


# Rate limit metrics
RATE_LIMIT_REJECTED = Counter(
    "claimcheck_rate_limit_rejected_total",
    "Number of requests rejected by rate limiter",
    ["user_tier", "endpoint"],
)


# Auth metrics
AUTH_FAILURES = Counter(
    "claimcheck_auth_failures_total",
    "Authentication failures",
    ["reason"],
)


# Webhook metrics
WEBHOOK_DELIVERIES = Counter(
    "claimcheck_webhook_deliveries_total",
    "Webhook delivery attempts",
    ["event_type", "status"],
)


def render_metrics() -> tuple[bytes, str]:
    """Render all metrics in Prometheus text format."""
    return generate_latest(), CONTENT_TYPE_LATEST