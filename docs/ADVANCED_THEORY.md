# ClaimCheck Advanced — Theory

## Thesis

**Better atoms + better evidence ranking + numerical overrides + short explanations beat more infrastructure** for catching hidden LLM hallucinations.

## Pipeline theory

```
LLM answer + sources
        │
        ▼
 Claim decomposition (atomic propositions)
        │
        ▼
 Hybrid retrieval (BM25 + dense) + cross-encoder rerank
        │
        ▼
 NLI (DeBERTa MNLI ± ensemble)
        │
        ▼
 Numerical / unit consistency plugin
        │
        ▼
 Explanation + attribution spans
        │
        ▼
 Weighted risk score
```

### 1. Claim atomicity

Verification quality is bounded by claim quality. Compound sentences hide mixed truth (“A is true and B is false”). The decomposer:

- Splits independent coordinated propositions
- Preserves conditionals (`if … then …`) when splitting would destroy meaning
- Strips hedges that dilute entailment signals
- Applies a **complex-claim heuristic rewrite** (subject–predicate atoms) as a local substitute for a small LLM fallback

### 2. Hybrid retrieval + reranking

Lexical BM25 catches exact entities and numbers; dense embeddings catch paraphrase. A cross-encoder reranker re-scores top candidates with full claim–passage interaction, reducing “retrieved but irrelevant” evidence that poisons NLI.

Attribution is first-class: `source_index`, `chunk_id`, and character offsets so the UI can say *Source 2, passage …*.

### 3. Natural Language Inference

Premise = evidence, hypothesis = claim. Labels map to:

| NLI | Verdict |
|-----|---------|
| Entailment | SUPPORTED |
| Contradiction | CONTRADICTED |
| Neutral | UNVERIFIABLE |

Optional ensemble averages complementary MNLI checkpoints to reduce single-model bias.

### 4. Numerical override

NLI alone underweights magnitude clashes. A dedicated numerical plugin:

- Extracts claim/evidence numbers and ratio forms (`1 in 30,000`)
- Flags unit mismatches (percent vs ratio)
- Overrides toward CONTRADICTED when values disagree beyond tolerance

This is the mechanism behind the demo “wow”: catching **10% lactic acidosis** against **~1/30,000**.

### 5. Explanation and calibrated confidence

Each verdict carries a short explanation combining label, confidence, numerical note, and evidence cue. Soft calibration reduces overconfident raw max-probabilities.

### 6. Weighted risk

Risk is not merely `contradicted / total`. CONTRADICTED weighs more than UNVERIFIABLE; `numerical` / medical-tagged claims get a severity boost so one dangerous false statistic dominates the score.

## Relation to Round 1

Round 1 proved the product shape: claim-level, evidence-backed, open-source. Round 2 hardens the **accuracy theory**—the layers that decide whether the demo catches the hidden hallucination every time.

## Non-goals (this version)

ONNX export, browser extension, paid LLM decomposition, and full Next.js productization are deferred. Accuracy of the verification core comes first.
