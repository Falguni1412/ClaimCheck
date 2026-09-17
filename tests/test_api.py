"""
Integration tests for the API.
"""
import pytest
import httpx
from fastapi.testclient import TestClient


class TestAPI:
    """Tests for the FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        from backend.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data

    def test_docs_endpoint(self, client):
        response = client.get("/api/docs")
        assert response.status_code == 200
        data = response.json()
        assert "endpoints" in data
        assert "features" in data

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        # Prometheus text format
        assert "claimcheck" in response.text

    def test_verify_empty_answer_returns_400(self, client):
        response = client.post(
            "/verify",
            json={"answer": "", "sources": ["some source"]},
        )
        assert response.status_code in (400, 503)  # 400 for bad input, 503 if model not loaded

    def test_verify_no_sources_returns_400(self, client):
        response = client.post(
            "/verify",
            json={"answer": "test answer", "sources": []},
        )
        assert response.status_code in (400, 503)

    def test_history_endpoint(self, client):
        response = client.get("/history/")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data


class TestAuth:
    """Tests for authentication endpoints."""

    @pytest.fixture
    def client(self):
        from backend.main import app
        return TestClient(app)

    def test_login_endpoint_exists(self, client):
        response = client.post(
            "/admin/login",
            json={"username": "test", "password": "test"},
        )
        # Either succeeds (200) or fails with auth error (401)
        assert response.status_code in (200, 401, 422)


class TestStreaming:
    """Tests for SSE streaming."""

    @pytest.fixture
    def client(self):
        from backend.main import app
        return TestClient(app)

    def test_stream_endpoint_accepts_post(self, client):
        """Streaming endpoint should accept POST and return text/event-stream."""
        response = client.post(
            "/verify/stream",
            json={
                "answer": "Test claim.",
                "sources": ["Test evidence supporting the claim."],
            },
        )
        # Either 200 (success) or 503 (model not loaded)
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            assert "text/event-stream" in response.headers.get("content-type", "")
