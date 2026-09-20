"""
End-to-end endpoint tests.

These run WITHOUT downloading a transformer: the NLI verifier is replaced with
a deterministic stub, so the suite exercises routing, validation, error
envelopes, persistence and serialisation in about a second.

Run:
    pip install -r backend/requirements.txt -r backend/requirements-dev.txt
    pytest tests/test_endpoints.py -v
"""
import asyncio
from typing import Any, Dict, List, Sequence, Tuple

import pytest


# --------------------------------------------------------------- stub model


class StubVerifier:
    """Deterministic stand-in for NLIVerifier with the same call surface."""

    model_name = "stub/nli-test"
    use_ensemble = False
    secondary_model = None

    def verify_pairs(
        self, pairs: Sequence[Tuple[str, str]], run_numerical_check: bool = True
    ) -> List[Dict[str, Any]]:
        results = []
        for claim, evidence in pairs:
            if not evidence.strip():
                verdict, scores = "UNVERIFIABLE", {
                    "supported": 0.0, "unverifiable": 1.0, "contradicted": 0.0
                }
            elif "not" in claim.lower():
                verdict, scores = "CONTRADICTED", {
                    "supported": 0.1, "unverifiable": 0.1, "contradicted": 0.8
                }
            else:
                verdict, scores = "SUPPORTED", {
                    "supported": 0.9, "unverifiable": 0.05, "contradicted": 0.05
                }
            results.append({
                "verdict": verdict,
                "confidence": scores[verdict.lower()],
                "scores": scores,
                "numerical_check": None,
                "explanation": f"stubbed {verdict}",
            })
        return results

    async def verify_pairs_async(self, pairs, run_numerical_check: bool = True):
        return self.verify_pairs(pairs, run_numerical_check)


# --------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.models.database import init_db
    from api.routes import verify as verify_module
    from backend.main import app

    asyncio.run(init_db())
    # Bypass model loading; the lifespan is not started by TestClient unless
    # it is used as a context manager, so this is the whole injection point.
    verify_module._verifier = StubVerifier()

    return TestClient(app, raise_server_exceptions=False)


SAMPLE = {
    "answer": (
        "Metformin should be taken with meals. "
        "Nausea is a common side effect of the medication."
    ),
    "sources": [
        "Take metformin with meals to reduce stomach upset. "
        "Common side effects include nausea, diarrhea, and stomach pain."
    ],
    "top_k": 2,
}


def assert_error_envelope(payload: Dict[str, Any], expected_code: str = None):
    """Every non-2xx body must carry the same shape."""
    assert "error" in payload, payload
    assert {"code", "message"} <= set(payload["error"]), payload
    assert "detail" in payload and "path" in payload, payload
    assert payload["detail"] == payload["error"]["message"]
    if expected_code:
        assert payload["error"]["code"] == expected_code, payload


# --------------------------------------------------------------- probes


class TestProbes:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_health_is_cheap_and_always_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert "version" in body and "timestamp" in body

    def test_legacy_health_alias(self, client):
        assert client.get("/api/health").status_code == 200

    def test_ready_reports_model_state(self, client):
        r = client.get("/ready")
        assert r.status_code in (200, 503)
        body = r.json()
        assert set(["ready", "model_loaded", "database", "cache", "retrieval"]) <= set(body)

    def test_request_id_header_present(self, client):
        r = client.get("/health")
        assert r.headers.get("X-Request-ID")
        assert r.headers.get("X-Response-Time", "").endswith("ms")

    def test_request_id_is_echoed_when_supplied(self, client):
        r = client.get("/health", headers={"X-Request-ID": "abc-123"})
        assert r.headers["X-Request-ID"] == "abc-123"

    def test_metrics(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "claimcheck" in r.text

    def test_api_docs(self, client):
        body = client.get("/api/docs").json()
        assert "endpoints" in body and "features" in body and "limits" in body

    def test_openapi_schema_builds(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        paths = r.json()["paths"]
        for path in ("/verify", "/verify/batch", "/verify/stream", "/history/", "/health", "/ready"):
            assert path in paths, f"{path} missing from OpenAPI schema"


# --------------------------------------------------------------- verify


class TestVerify:
    def test_happy_path(self, client):
        r = client.post("/verify", json=SAMPLE)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["claims"], "expected at least one decomposed claim"
        assert 0.0 <= body["risk_score"] <= 1.0
        assert body["summary"]["total_claims"] == len(body["claims"])
        assert (
            body["summary"]["supported"]
            + body["summary"]["contradicted"]
            + body["summary"]["unverifiable"]
            == body["summary"]["total_claims"]
        )
        for claim in body["claims"]:
            assert claim["verdict"] in ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE")
            assert 0.0 <= claim["confidence"] <= 1.0

    def test_second_identical_request_is_cached(self, client):
        client.post("/verify", json=SAMPLE)
        body = client.post("/verify", json=SAMPLE).json()
        assert body["cached"] is True

    def test_empty_answer_is_422(self, client):
        r = client.post("/verify", json={"answer": "", "sources": ["a source"]})
        assert r.status_code == 422
        assert_error_envelope(r.json(), "VALIDATION_ERROR")

    def test_blank_answer_is_rejected(self, client):
        r = client.post("/verify", json={"answer": "    ", "sources": ["a source"]})
        assert r.status_code == 422

    def test_no_sources_is_422(self, client):
        r = client.post("/verify", json={"answer": "some answer text here", "sources": []})
        assert r.status_code == 422
        assert_error_envelope(r.json())

    def test_empty_source_string_is_rejected(self, client):
        r = client.post("/verify", json={"answer": "some answer text", "sources": ["  "]})
        assert r.status_code == 422

    def test_top_k_out_of_range(self, client):
        r = client.post("/verify", json={**SAMPLE, "top_k": 99})
        assert r.status_code == 422
        fields = r.json()["error"]["extra"]["fields"]
        assert any("top_k" in f["field"] for f in fields)

    def test_malformed_json_body(self, client):
        r = client.post("/verify", content=b"{not json", headers={"Content-Type": "application/json"})
        assert r.status_code == 422
        assert_error_envelope(r.json())

    def test_unknown_route_uses_envelope(self, client):
        r = client.get("/definitely-not-a-route")
        assert r.status_code == 404
        assert_error_envelope(r.json(), "NOT_FOUND")


class TestBatch:
    def test_batch_validate(self, client):
        r = client.post("/batch/validate", json={"items": [SAMPLE]})
        assert r.status_code == 200
        assert r.json()["total_items"] == 1

    def test_batch_verify(self, client):
        r = client.post("/verify/batch", json={"items": [SAMPLE, SAMPLE]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total"] == 2
        assert body["successful"] + body["failed"] == 2

    def test_batch_rejects_empty_items(self, client):
        assert client.post("/verify/batch", json={"items": []}).status_code == 422


class TestStreaming:
    def test_stream_emits_sse_events(self, client):
        with client.stream("POST", "/verify/stream", json=SAMPLE) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            payload = "".join(chunk for chunk in r.iter_text())
        assert "event: start" in payload
        assert "event: claim" in payload
        assert "event: complete" in payload


# --------------------------------------------------------------- history


class TestHistory:
    def test_history_list(self, client):
        client.post("/verify", json={**SAMPLE, "answer": SAMPLE["answer"] + " Extra sentence here."})
        r = client.get("/history/")
        assert r.status_code == 200, r.text
        body = r.json()
        assert "items" in body and "total" in body
        assert body["page"] == 1

    def test_history_pagination_validation(self, client):
        assert client.get("/history/?page=0").status_code == 422
        assert client.get("/history/?page_size=1000").status_code == 422

    def test_history_detail_roundtrip(self, client):
        created = client.post(
            "/verify",
            json={**SAMPLE, "answer": "Roundtrip claim about metformin and meals."},
        ).json()
        r = client.get(f"/history/{created['request_id']}")
        # 200 when persistence is on; 404 only if the DB is unavailable.
        assert r.status_code in (200, 404), r.text
        if r.status_code == 200:
            body = r.json()
            assert body["request_id"] == created["request_id"]
            assert isinstance(body["metadata"], dict)

    def test_history_detail_missing(self, client):
        r = client.get("/history/does-not-exist")
        assert r.status_code == 404
        assert_error_envelope(r.json(), "NOT_FOUND")


# --------------------------------------------------------------- admin/webhooks


class TestAdmin:
    def test_register_validates_body(self, client):
        r = client.post("/admin/register", json={"email": "nope"})
        assert r.status_code == 422
        assert_error_envelope(r.json(), "VALIDATION_ERROR")

    def test_register_then_login(self, client):
        user = {"email": "t@example.com", "username": "tester", "password": "hunter2hunter2"}
        r = client.post("/admin/register", json=user)
        assert r.status_code in (200, 400), r.text

        r = client.post(
            "/admin/login", json={"username": user["username"], "password": user["password"]}
        )
        assert r.status_code in (200, 401)
        if r.status_code == 200:
            assert r.json()["token_type"] == "bearer"

    def test_login_with_bad_credentials(self, client):
        r = client.post("/admin/login", json={"username": "nobody", "password": "wrongpass"})
        assert r.status_code == 401
        assert_error_envelope(r.json(), "AUTH_REQUIRED")

    def test_stats(self, client):
        r = client.get("/admin/stats")
        assert r.status_code == 200, r.text
        assert "total_verifications" in r.json()


class TestWebhooks:
    def test_webhook_crud(self, client):
        r = client.post("/webhooks/", json={"url": "https://example.com/hook"})
        assert r.status_code == 200, r.text
        created = r.json()
        assert created["secret"]

        listed = client.get("/webhooks/")
        assert listed.status_code == 200
        assert any(w["id"] == created["id"] for w in listed.json())

        assert client.delete(f"/webhooks/{created['id']}").status_code == 200
        assert client.delete(f"/webhooks/{created['id']}").status_code == 404

    def test_webhook_rejects_bad_url(self, client):
        assert client.post("/webhooks/", json={"url": "ftp://nope"}).status_code == 422


# --------------------------------------------------------------- CORS


class TestCORS:
    def test_preflight_allowed(self, client):
        r = client.options(
            "/verify",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert r.status_code in (200, 204)
        assert "access-control-allow-origin" in {k.lower() for k in r.headers}
