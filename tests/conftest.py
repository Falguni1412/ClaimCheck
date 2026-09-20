"""
Shared test fixtures for ClaimCheck tests.
"""
import asyncio
import os
import sys
import tempfile
import pytest
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Configure environment variables for tests."""
    os.environ.setdefault("ENVIRONMENT", "test")
    os.environ.setdefault("JWT_SECRET", "test-secret-key-for-testing")
    # NOT ":memory:" — each aiosqlite connection would get its own blank DB,
    # so tables created by init_db() disappear before the first query.
    _db = Path(tempfile.gettempdir()) / "claimcheck_test.db"
    _db.unlink(missing_ok=True)
    os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{_db}")
    os.environ.setdefault("REDIS_URL", "memory://")
    os.environ.setdefault("ENABLE_SEMANTIC_RETRIEVAL", "false")
    os.environ.setdefault("MODEL_WARMUP", "false")
    os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    yield


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_medical_answer():
    return (
        "Metformin should be taken on an empty stomach. "
        "It can be combined with insulin therapy. "
        "Lactic acidosis occurs in 10% of patients. "
        "Nausea is a common side effect."
    )


@pytest.fixture
def sample_medical_sources():
    return [
        "Metformin is a medication for type 2 diabetes. "
        "Take metformin with meals to reduce stomach upset. "
        "Do not take on an empty stomach.",
        "Common side effects of metformin include nausea, diarrhea, and stomach pain. "
        "Lactic acidosis is rare, occurring in approximately 1 in 30,000 patient-years.",
        "Metformin can be safely combined with insulin therapy under medical supervision. "
        "This combination is commonly prescribed for patients with poorly controlled diabetes.",
    ]


@pytest.fixture
def sample_company_answer():
    return (
        "Refunds are processed within 24 hours. "
        "You will receive a confirmation email within 30 minutes. "
        "Your account will be automatically credited. "
        "You can track the refund status online."
    )


@pytest.fixture
def sample_company_sources():
    return [
        "Our refund policy states that refunds are processed within 5-7 business days. "
        "A confirmation email is sent immediately after the refund is initiated. "
        "The refunded amount will be credited to your original payment method within 10 business days.",
        "You can track your refund status by logging into your account and visiting the Orders section.",
    ]


@pytest.fixture
def mock_verifier(monkeypatch):
    """Mock NLI verifier for tests that don't need real model."""
    class MockVerifier:
        MODEL_NAME = "mock-model"

        def verify_claim(self, claim, evidence):
            return {
                "verdict": "SUPPORTED",
                "confidence": 0.85,
                "scores": {"supported": 0.85, "unverifiable": 0.1, "contradicted": 0.05},
                "numerical_check": None,
                "explanation": "Mock verification",
            }

    return MockVerifier()
