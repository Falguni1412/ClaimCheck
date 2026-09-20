# ClaimCheck Backend API

Interactive docs are generated from the code and are always current:

| URL | What it is |
| --- | --- |
| `/docs` | Swagger UI (try requests in the browser) |
| `/redoc` | ReDoc reference |
| `/openapi.json` | Raw OpenAPI 3.1 schema |
| `/api/docs` | Compact JSON summary (endpoints, features, limits) |

---

## Running it

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # CPU-only torch
cp ../.env.example .env                    # then edit
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

First boot downloads the NLI model (~90MB) into the HuggingFace cache. Until it
finishes, `/health` returns 200 and `/ready` returns 503.

Tests:

```bash
pip install -r requirements-dev.txt
pytest ../tests/test_endpoints.py -v       # no model download, ~1s
../scripts/smoke_test.sh http://localhost:8000
```

---

## Health & readiness

Two separate probes, on purpose.

| Endpoint | Meaning | Use it for |
| --- | --- | --- |
| `GET /health` | The process is serving. Never touches the model. | Platform health check (Render `healthCheckPath`, Docker `HEALTHCHECK`, k8s liveness) |
| `GET /ready` | The model is loaded and this instance can verify. 503 while loading. | k8s readiness, load-balancer gating, deploy verification |

`GET /ready`:

```json
{
  "status": "ready",
  "ready": true,
  "model_loaded": true,
  "version": "2.1.0",
  "timestamp": "2026-01-01T12:00:00+00:00",
  "uptime_seconds": 143.2,
  "database": true,
  "cache": true,
  "retrieval": { "bm25": true, "semantic": false, "rerank": false }
}
```

Pointing a platform health check at `/ready` on a free tier will restart-loop
the service during every cold start. Point it at `/health`.

---

## Errors

Every non-2xx response has the same body:

```json
{
  "error": { "code": "VALIDATION_ERROR", "message": "answer cannot be blank", "extra": {} },
  "detail": "answer cannot be blank",
  "path": "/verify",
  "request_id": "0f0c1e9c-..."
}
```

`detail` mirrors `error.message` so older clients reading `detail` keep working.
`request_id` is also returned as the `X-Request-ID` header and appears in the
server logs — quote it in bug reports.

| Code | Status | Cause |
| --- | --- | --- |
| `VALIDATION_ERROR` | 422 | Body failed schema validation. `extra.fields` lists each field. |
| `INVALID_REQUEST` | 400 | Semantically invalid (e.g. too many claims). |
| `DECOMPOSITION_FAILED` | 422 | No verifiable claims could be extracted from `answer`. |
| `AUTH_REQUIRED` | 401 | Missing or bad credentials. |
| `FORBIDDEN` | 403 | Authenticated but not permitted. |
| `NOT_FOUND` | 404 | Unknown route or resource. |
| `RATE_LIMITED` | 429 | Rate limit hit; see the `Retry-After` header. |
| `MODEL_NOT_LOADED` | 503 | Still starting up, or the model failed to load. Retry after `/ready` is 200. |
| `INTERNAL_ERROR` | 500 | Unhandled. The traceback is in the logs under this `request_id`. |

---

## `POST /verify`

Decompose an answer into atomic claims, retrieve the best supporting passage
for each from your sources, and classify it.

Request:

```json
{
  "answer": "Metformin should be taken on an empty stomach.",
  "sources": ["Take metformin with meals to reduce stomach upset."],
  "use_chunking": true,
  "top_k": 3,
  "tags": ["medical"]
}
```

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `answer` | string | required | 1–50,000 chars, not blank |
| `sources` | string[] | required | 1–50 items, each ≤100,000 chars, none blank |
| `use_chunking` | bool | `true` | Split long sources into passages first |
| `top_k` | int | `3` | 1–10 passages retrieved per claim |
| `tags` | string[] | `null` | ≤10 tags, ≤64 chars each; stored with the result |

Response (200):

```json
{
  "request_id": "b2b0...",
  "claims": [
    {
      "claim": "Metformin should be taken on an empty stomach.",
      "verdict": "CONTRADICTED",
      "confidence": 0.83,
      "evidence": "Take metformin with meals to reduce stomach upset.",
      "verdict_scores": { "supported": 0.06, "unverifiable": 0.11, "contradicted": 0.83 },
      "numerical_check": null,
      "explanation": "Model classified as CONTRADICTED (confidence 83.0%)."
    }
  ],
  "risk_score": 1.0,
  "summary": {
    "total_claims": 1, "supported": 0, "contradicted": 1, "unverifiable": 0,
    "processing_time_seconds": 0.42, "average_confidence": 0.83
  },
  "metadata": { "model": "cross-encoder/nli-MiniLM2-L6-H768", "sources_processed": 1,
                "ensemble": false, "semantic_retrieval": false, "timestamp": 1767225600.0 },
  "cached": false
}
```

`risk_score` is the fraction of claims that were contradicted. `cached: true`
means the result was served from cache (identical answer + sources + `top_k`).

---

## `POST /verify/batch`

`{ "items": [<VerifyRequest>, ...], "fail_fast": false }` — up to 50 items,
processed sequentially. With `fail_fast: false` a failed item is recorded in
`failed` and `errors[]` and the rest still run.

## `POST /verify/stream`

Same pipeline, emitted as Server-Sent Events:

```
event: start     data: {"request_id": "...", "total_claims": 4}
event: claim     data: {"claim": "...", "verdict": "SUPPORTED", ...}
event: complete  data: {"request_id": "...", "summary": {...}}
event: error     data: {"error": "...", "request_id": "..."}
```

```bash
curl -N -X POST http://localhost:8000/verify/stream \
  -H 'Content-Type: application/json' -d @request.json
```

## Other endpoints

| Method & path | Purpose |
| --- | --- |
| `POST /batch/validate` | Size check + duration estimate, no inference |
| `GET /history/` | Paginated past verifications (`page`, `page_size`, `user_id`) |
| `GET /history/{request_id}` | Full stored result |
| `POST /admin/register` | Create an account |
| `POST /admin/login` | JWT access + refresh tokens |
| `POST /admin/api-keys` | Issue an API key (full key returned once) |
| `GET /admin/stats` | Aggregate counters |
| `POST /webhooks/`, `GET /webhooks/`, `DELETE /webhooks/{id}` | Webhook subscriptions |
| `GET /metrics` | Prometheus exposition |

Webhook payloads are signed: `X-Webhook-Signature` is
`HMAC-SHA256(secret, raw_body)` in hex. Compare it with a constant-time compare.

---

## Configuration

Full list with comments in `.env.example`. The ones that matter most on a small
instance:

| Variable | Default | Effect |
| --- | --- | --- |
| `TORCH_NUM_THREADS` | `1` | More threads on a fractional vCPU cause contention, not speed |
| `NLI_MAX_LENGTH` | `256` | 512 roughly doubles latency for little accuracy gain |
| `NLI_BATCH_SIZE` | `8` | Claims per forward pass |
| `MAX_CONCURRENT_INFERENCES` | `1` | Ceiling on simultaneous model work; raise only with RAM to spare |
| `ENABLE_SEMANTIC_RETRIEVAL` | `true` | `false` drops the embedding model (~130MB) and uses BM25 only |
| `MODEL_QUANTIZATION` | `true` | INT8 dynamic quantization of Linear layers |
| `USE_ENSEMBLE` | `false` | Loads a second NLI model — doubles RAM |
| `CORS_ORIGINS` | `*` | Comma-separated. With `*`, credentialed CORS is disabled (browsers reject the combination) |
| `NLI_LABEL_ORDER` | unset | Override when a checkpoint reports only `LABEL_0/1/2` |

**If every verdict comes back `UNVERIFIABLE`**, the checkpoint is reporting
generic labels and the logit order is being guessed. Set
`NLI_LABEL_ORDER=contradiction,entailment,neutral` (or whatever the model card
says) and restart. The resolved mapping is logged at startup.

---

## Free-tier notes

- **One worker.** Each uvicorn worker loads its own copy of the model. Two
  workers on 512MB is an OOM kill.
- **Cold starts are slow.** Free instances sleep; the first request after a
  sleep pays model load (~10–30s). `/health` answers during that window.
- **Disk is ephemeral.** The default SQLite file lives in `/tmp` and is wiped
  on every deploy, taking `/history` with it. Use managed Postgres to keep it.
- **Redis is optional.** If `REDIS_URL` is unreachable the app logs a warning
  and uses a bounded in-process LRU cache instead.
