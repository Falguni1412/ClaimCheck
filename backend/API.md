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

---

## Verdict policy (accuracy)

A claim is **SUPPORTED** only when the evidence *asserts* it. Similarity is not
support, and mentioning the same entity is not support.

Three rules run in order, in `api/services/decision.py` and
`NLIVerifier._assemble`:

**1. Multi-premise scoring.** Each claim is scored against every retrieved
passage *and* each sentence inside it, not just the single best-ranked chunk.
A contradiction in any premise is decisive and outranks support elsewhere —
the conservative direction for a hallucination detector.

**2. Margin rule (abstention).** A class must clear an absolute floor *and*
lead the others by a margin:

| | Requirement |
| --- | --- |
| SUPPORTED | `P(entail) ≥ MIN_ENTAILMENT` and leads neutral and contradiction by `DECISION_MARGIN` |
| CONTRADICTED | `P(contradict) ≥ MIN_CONTRADICTION` and leads neutral and entailment by `DECISION_MARGIN` |
| UNVERIFIABLE | everything else |

Defaults: `MIN_ENTAILMENT=0.55`, `MIN_CONTRADICTION=0.50`, `DECISION_MARGIN=0.10`.
This replaces a bare `argmax`, which reported `entail 0.38 / neutral 0.36` as
SUPPORTED — a model expressing no opinion, recorded as an assertion.

**3. Grounding gate.** Retrieval always returns its best match, even when the
corpus contains nothing relevant, and min-max score normalisation gave that
best match a relevance of 1.0 by construction. So the evidence must mention the
claim's *entities* — proper-noun-like tokens and numeric literals — before it is
allowed to support the claim. Ordinary content words are deliberately not
required, since demanding lexical overlap on those would reject valid
paraphrases. A claim that fails this gate is downgraded to UNVERIFIABLE, never
upgraded to CONTRADICTED. Disable with `REQUIRE_ENTITY_GROUNDING=false`.

**Numerical override.** Only a genuine value-vs-value *mismatch* can override
the model. Comparisons are unit-scoped: percentages and ratios share a scale
("1 in 30,000" is normalised to 0.0033%), currency compares with currency, and
category labels like "type 2" are not quantities at all. A claim asserting a
number the evidence never discusses is `not_comparable` → UNVERIFIABLE, **not**
a contradiction.

**Categorical override (rule 4).** The NLI checkpoint (`cross-encoder/nli-
MiniLM2-L6-H768`, a 6-layer model chosen for CPU/RAM budget) is confirmed, by
direct logit inspection, to sometimes score plain category errors as confident
entailment — e.g. `P(entailment)=0.80` for *"Metformin is a person"* against
*"Metformin is a medication..."*. That clears the margin rule on its own; no
threshold adjustment fixes a model that is wrong with high confidence about a
specific input.

The fix is symbolic and syntax-driven, in `api/services/decision.py`: for a
bare copular claim (`SUBJECT is/are/was/were [a/an/the] NOUN PHRASE`), extract
the subject and predicate noun phrase using only closed-class function words
(prepositions, relative pronouns, a short list of passive/linking verbs) as
parse boundaries — never a content word or entity name. Do the same for each
evidence sentence. If both assert a category for the same subject and the
categories are incompatible, that is a **structural** contradiction,
independent of the model's own opinion — injected as an extra premise into the
same contradiction-beats-support aggregation every other premise goes through,
not a bolted-on special case.

This only ever adds a contradiction signal; it never manufactures support. A
single confirming sentence anywhere in the evidence stands down the check even
if another sentence conflicts. Claims outside the bare-copula pattern — "X is
*used to* treat Y", "X is *called* Y", "X *should be* taken with Y" — don't
match and fall through to NLI unchanged; the check narrows its own scope by
grammar, not by a list of approved sentences, which is what lets it generalize
to any subject without special-casing. Toggle with
`REQUIRE_CATEGORY_CONSISTENCY=false`; reported confidence for a caught
mismatch is `CATEGORY_MISMATCH_CONFIDENCE` (default `0.90`).

Known scope limits, stated rather than hidden: predicate comparison folds
"medication"/"medications" but does no real synonym matching, so `"Metformin
is a drug"` against `"Metformin is a medication"` would currently be flagged
as a mismatch (a false positive) rather than recognised as synonymous — this
trades a rare false contradiction for reliably catching the class of errors in
the bug report, and is a candidate for a small synonym table if it turns out
to matter in practice. Negated copulas ("is not a fruit") and sentences with
no closed-class boundary word after 6+ tokens are treated as unparseable and
skipped entirely, on the same "don't guess" principle as the length cap in the
margin rule.

### Documented behaviour: category mismatch

For claims like *"Metformin is a person"* against a source describing a
medication, ClaimCheck returns **CONTRADICTED when the NLI model asserts
contradiction with margin, and UNVERIFIABLE otherwise.** It will not return
SUPPORTED.

ClaimCheck does not implement type or ontology reasoning, so it cannot on its
own conclude that "medication" and "person" are disjoint categories. That
judgement comes entirely from the NLI checkpoint. If the checkpoint scores such
a pair as entailment with high confidence, the correct remedy is a stronger
checkpoint — not a post-processing rule, which could only be a hand-written
category list that fails on the next claim.

Run `python scripts/diagnose_nli.py --raw` to see the actual probabilities and
decide which situation you are in. If the model is at fault, set a better
checkpoint (RAM cost in brackets):

```bash
# ~370MB — noticeably better NLI, still CPU-viable on a paid instance
NLI_MODEL_NAME=MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

`cross-encoder/nli-MiniLM2-L6-H768` is a 6-layer model chosen for the free
tier's memory budget. It is fast and small, and it is the weakest link in
accuracy.

### Regression suites

```bash
pytest tests/test_decision.py -v              # margin rule, grounding, numerical scoping
pytest tests/test_category_consistency.py -v  # rule 4: the exact reported bug + generalizations
python scripts/diagnose_nli.py                # raw probabilities + 10-case + category tables
python scripts/diagnose_nli.py --categories   # category-mismatch regression only
```

`test_category_consistency.py` reproduces the exact reported NLI scores
(`supported=0.937`) as a `PremiseVerdict` and asserts aggregation still lands
on CONTRADICTED once the categorical signal is added — a regression test of
the specific numbers in the bug report, not just of the general mechanism.
