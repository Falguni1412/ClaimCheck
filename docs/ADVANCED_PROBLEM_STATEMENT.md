# ClaimCheck Advanced — Problem Statement

## The real failure mode

LLMs are not only wrong in obvious ways. The dangerous case is a **mostly correct, fully confident answer** where **one false claim** causes harm:

- A medical reply that correctly describes Metformin side effects but invents a **10% lactic acidosis rate**
- A support bot that correctly explains tracking while inventing a **24-hour refund SLA**
- A coding assistant that correctly uses `requests.get` while inventing **`fetch_url()`**

Users trust the whole answer because it *looks* coherent. Whole-answer scores and “be accurate” prompts do not show **which** sentence failed or **what evidence** contradicts it.

## Why current solutions still fail

| Approach | What it misses |
|----------|----------------|
| Manual review | Not scalable for every LLM turn |
| Prompt engineering | Reduces volume, does not guarantee claim-level truth |
| RAG | Retrieval helps; the model can still mis-synthesize retrieved text |
| Enterprise SaaS (Patronus, Arize, etc.) | Closed, costly, often privacy-sensitive |
| Research tools (G-Eval, FACTScore) | Hard to deploy as local infrastructure |
| Single-score detectors (e.g. HHEM) | One number; no per-claim evidence trail |

**The gap:** there is still no widely deployed **open-source, evidence-backed, claim-level** verification engine that returns attributed SUPPORTED / CONTRADICTED / UNVERIFIABLE verdicts with explanations.

## Round-2 problem (advanced)

Round 1 asked: *“Was the AI right?”*

Round 2 asks: *“Which atomic claim is wrong, by how much, against which source span, and why should a human trust that verdict?”*

Success criteria:

1. **Atomicity** — Complex answers become independently checkable claims.
2. **Attribution** — Every verdict cites a source index and evidence span.
3. **Numerical honesty** — Magnitude and unit mismatches (10% vs 1 in 30,000) force contradiction even when prose NLI is soft.
4. **Explainability** — Short, human-readable reasons (NLI + numerical + evidence).
5. **Risk that reflects harm** — One contradicted medical/numerical claim should dominate risk, not average away.

## Who is hurt without this

- Patients acting on confident but false clinical statistics
- Customers promised SLAs that do not exist
- Developers shipping invented APIs
- Enterprises that cannot send proprietary docs to closed SaaS checkers

## ClaimCheck’s answer

Decompose → retrieve (hybrid + rerank) → NLI classify → numerical override → explain → weighted risk.

Local-first, MIT-licensed, evidence-first. Infrastructure for any LLM app—not a one-off demo chat.
