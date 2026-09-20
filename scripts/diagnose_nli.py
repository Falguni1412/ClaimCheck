#!/usr/bin/env python3
"""
NLI diagnostic + accuracy regression harness.

Run from the `backend` directory (needs torch/transformers installed):

    python ../scripts/diagnose_nli.py --raw     # raw logits/probabilities only
    python ../scripts/diagnose_nli.py           # raw dump + the 10-case table
    python ../scripts/diagnose_nli.py --cases   # the 10-case table only

`--raw` answers the question "is the model wrong, or is our post-processing
wrong?" — it prints the label mapping and the untouched softmax output, before
any ClaimCheck logic runs. If P(entailment) is high for a hypothesis the
premise plainly does not entail, the model is the problem and no amount of
post-processing will fix it; see the notes printed at the end.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

SOURCE_MED = "Metformin is a medication for type 2 diabetes."
SOURCE_FULL = (
    "Metformin is a medication for type 2 diabetes. "
    "Take metformin with meals to reduce stomach upset. "
    "Do not take on an empty stomach."
)

# (name, claim, source, expected)
CASES = [
    ("0  reported bug", "Metformin is person.", SOURCE_FULL, "NOT SUPPORTED"),
    ("1  category mismatch", "Metformin is a person.", SOURCE_MED, "CONTRADICTED or UNVERIFIABLE"),
    ("2  direct support", "Metformin is a medication.", SOURCE_MED, "SUPPORTED"),
    ("3  paraphrase support", "Metformin is used to treat type 2 diabetes.", SOURCE_MED, "SUPPORTED"),
    ("4  category mismatch", "Metformin is a fruit.", SOURCE_MED, "CONTRADICTED or UNVERIFIABLE"),
    ("5  unstated quantity", "Metformin costs $500.", SOURCE_MED, "UNVERIFIABLE"),
    ("6  wrong entity", "Metformin is a medication.", "Aspirin is a medication.", "UNVERIFIABLE"),
    ("7  reworded support", "The medication is called Metformin.", SOURCE_MED, "SUPPORTED"),
    ("8  explicit contradiction", "Metformin should be taken on an empty stomach.",
     "Take metformin with meals to reduce stomach upset. Do not take on an empty stomach.",
     "CONTRADICTED"),
    ("9  numeric contradiction", "Lactic acidosis occurs in 10% of patients.",
     "Lactic acidosis is rare, occurring in approximately 1 in 30,000 patient-years.",
     "CONTRADICTED"),
    ("10 list support", "Nausea is a common side effect.",
     "Common side effects include nausea, diarrhea, and stomach pain.", "SUPPORTED"),
]


CATEGORY_CASES = [
    # (name, claim, evidence, expected)
    ("cat-1 metformin/person (reported bug)", "Metformin is a person.", SOURCE_MED, "NOT SUPPORTED"),
    ("cat-2 metformin/fruit", "Metformin is a fruit.", SOURCE_MED, "NOT SUPPORTED"),
    ("cat-3 aspirin/fruit", "Aspirin is a fruit.",
     "Aspirin is a medication used to relieve pain and reduce fever.", "NOT SUPPORTED"),
    ("cat-4 python/person", "Python is a person.",
     "Python is a programming language created by Guido van Rossum.", "NOT SUPPORTED"),
    ("cat-5 paris/prog-lang", "Paris is a programming language.",
     "Paris is the capital of France.", "NOT SUPPORTED"),
    ("cat-6 dog/element", "A dog is a chemical element.",
     "A dog is a domesticated mammal often kept as a pet.", "NOT SUPPORTED"),
    ("cat-7 metformin/medication (control)", "Metformin is a medication.", SOURCE_MED, "SUPPORTED"),
    ("cat-8 aspirin/medication (control)", "Aspirin is a medication.",
     "Aspirin is a medication used to relieve pain and reduce fever.", "SUPPORTED"),
    ("cat-9 python/prog-lang (control)", "Python is a programming language.",
     "Python is a programming language created by Guido van Rossum.", "SUPPORTED"),
]


def run_category_diagnostics(repeats: int = 3):
    """
    Regression run for the categorical consistency check (decision.py rule 4).

    Unlike run_cases(), the "NOT SUPPORTED" rows here are expected to be
    CONTRADICTED specifically via the symbolic override — if a row instead
    comes back UNVERIFIABLE, the categorical check silently failed to fire
    (e.g. because the sentence didn't parse as a bare copula) and NLI's own
    weak margin abstained instead. Both outcomes pass "NOT SUPPORTED", but
    that distinction is printed so a silent parser miss doesn't hide.
    """
    from api.core.config import settings
    from api.services.decision import category_mismatch_verdict
    from api.services.retriever import chunk_document, get_retriever
    from api.services.verifier import get_verifier

    verifier = get_verifier()
    header = f"{'TEST':<34}{'EXPECTED':<16}{'ACTUAL':<15}{'CONF':<8}{'VIA':<12}{'PASS':<6}"
    print(header)
    print("-" * len(header))

    rows = []
    for name, claim, source, expected in CATEGORY_CASES:
        chunks = chunk_document(source, chunk_size=settings.chunk_size,
                                overlap=settings.chunk_overlap) or [source]
        retriever = get_retriever(chunks)
        passages = retriever.retrieve(claim, top_k=settings.default_top_k)

        seen = set()
        result = None
        for _ in range(repeats):
            result = verifier.verify_claims([(claim, passages)])[0]
            seen.add((result["verdict"], result["confidence"]))
        nondeterministic = len(seen) > 1

        via = "symbolic" if category_mismatch_verdict(claim, passages, verifier.policy) else "NLI"
        actual = result["verdict"]
        passed = (actual != "SUPPORTED") if expected == "NOT SUPPORTED" else (actual == expected)
        print(f"{name:<34}{expected:<16}{actual:<15}{result['confidence']:<8.2f}{via:<12}"
              f"{'PASS' if passed else 'FAIL':<6}"
              + ("  [NONDETERMINISTIC]" if nondeterministic else ""))
        rows.append((name, expected, actual, passed, via))

    print()
    for name, expected, actual, passed, via in rows:
        if expected == "NOT SUPPORTED" and passed and via == "NLI":
            print(f"NOTE: {name} avoided SUPPORTED via NLI's own margin, not the symbolic "
                  f"override — the categorical parser did not recognise this sentence pattern. "
                  f"Fine for this run, but means it depends on the model being right unaided.")

    print(f"\npassed {sum(1 for r in rows if r[3])}/{len(rows)}")
    return all(r[3] for r in rows)


def dump_raw(pairs):
    """Print the untouched model output for (premise, hypothesis) pairs."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from api.core.config import settings
    from api.services.verifier import _resolve_label_map

    name = settings.nli_model_name
    print(f"model: {name}")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()

    print(f"config.id2label: {dict(model.config.id2label)}")
    resolved = _resolve_label_map(model.config, settings.nli_label_order_list)
    print(f"resolved label map: {resolved}")
    if settings.nli_label_order_list:
        print(f"(overridden by NLI_LABEL_ORDER={settings.nli_label_order})")
    print()

    for premise, hypothesis in pairs:
        inputs = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256)
        with torch.inference_mode():
            logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()

        print(f"PREMISE:    {premise}")
        print(f"HYPOTHESIS: {hypothesis}")
        print(f"  logits: {[round(float(x), 4) for x in logits.tolist()]}")
        for idx, prob in enumerate(probs):
            print(f"  index {idx} ({resolved.get(idx, '?'):<14}) P = {prob:.4f}")
        entail = sum(p for i, p in enumerate(probs) if "entail" in resolved.get(i, ""))
        contra = sum(p for i, p in enumerate(probs) if "contradict" in resolved.get(i, ""))
        neutral = max(0.0, 1.0 - entail - contra)
        print(f"  => P(entailment)={entail:.4f}  P(neutral)={neutral:.4f}  P(contradiction)={contra:.4f}")
        print(f"  => raw argmax would say: {resolved.get(int(max(range(len(probs)), key=probs.__getitem__)), '?')}")
        print()


def run_cases(repeats: int = 3):
    """Run every case through the real pipeline: chunk -> retrieve -> NLI -> decide."""
    from api.core.config import settings
    from api.services.retriever import chunk_document, get_retriever
    from api.services.verifier import get_verifier

    verifier = get_verifier()
    print(f"model: {verifier.model_name}   policy: {verifier.policy}\n")

    header = f"{'TEST':<26}{'EXPECTED':<32}{'ACTUAL':<15}{'CONF':<8}{'PASS':<6}EVIDENCE"
    print(header)
    print("-" * len(header))

    rows = []
    nondeterministic = []

    for name, claim, source, expected in CASES:
        chunks = chunk_document(source, chunk_size=settings.chunk_size,
                                overlap=settings.chunk_overlap) or [source]
        retriever = get_retriever(chunks)
        passages = retriever.retrieve(claim, top_k=settings.default_top_k)

        seen = set()
        result = None
        for _ in range(repeats):
            result = verifier.verify_claims([(claim, passages)])[0]
            seen.add((result["verdict"], result["confidence"]))
        if len(seen) > 1:
            nondeterministic.append((name, seen))

        actual = result["verdict"]
        passed = actual in expected or (expected == "NOT SUPPORTED" and actual != "SUPPORTED")
        evidence = (result.get("evidence") or "")[:46]
        print(f"{name:<26}{expected:<32}{actual:<15}{result['confidence']:<8.2f}"
              f"{'PASS' if passed else 'FAIL':<6}{evidence}")
        rows.append((name, expected, actual, result, passed))

    print()
    for name, expected, actual, result, passed in rows:
        if not passed:
            print(f"FAILED {name}")
            print(f"   scores: {result['scores']}")
            print(f"   reason: {result.get('decision_reason')}")
            print(f"   numerical: {result.get('numerical_check')}")

    print()
    print(f"determinism: {repeats} identical runs per case -> "
          + ("IDENTICAL for all cases" if not nondeterministic
             else f"VARIED for {[n for n, _ in nondeterministic]}"))
    print(f"passed {sum(1 for r in rows if r[4])}/{len(rows)}")
    return all(r[4] for r in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="store_true", help="raw probabilities only")
    parser.add_argument("--cases", action="store_true", help="10-case table only")
    parser.add_argument("--categories", action="store_true",
                         help="category-mismatch regression only (Aspirin/fruit, Python/person, etc.)")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    ok = True
    if not args.cases:
        print("=" * 78)
        print("RAW MODEL OUTPUT (no ClaimCheck post-processing)")
        print("=" * 78)
        dump_raw([
            (SOURCE_MED, "Metformin is person."),
            (SOURCE_FULL, "Metformin is person."),
            (SOURCE_MED, "Metformin is a person."),
            (SOURCE_MED, "Metformin is a medication."),
            (SOURCE_MED, "Metformin is a fruit."),
            ("Aspirin is a medication.", "Metformin is a medication."),
        ])
        print("How to read this:")
        print("  * P(entailment) high for 'Metformin is person' => the MODEL is wrong.")
        print("    Post-processing cannot repair it; change NLI_MODEL_NAME (see backend/API.md).")
        print("  * P(entailment) low but ClaimCheck still said SUPPORTED => POST-PROCESSING was")
        print("    wrong; that is what the decision policy in api/services/decision.py fixes.")
        print()

    if args.categories:
        print("=" * 78)
        print("CATEGORY-MISMATCH REGRESSION (decision.py rule 4)")
        print("=" * 78)
        return 0 if run_category_diagnostics(args.repeats) else 1

    if not args.raw:
        print("=" * 78)
        print("END-TO-END CASES (chunk -> retrieve -> NLI -> decide)")
        print("=" * 78)
        ok = run_cases(args.repeats) and ok

        print()
        print("=" * 78)
        print("CATEGORY-MISMATCH REGRESSION (decision.py rule 4)")
        print("=" * 78)
        ok = run_category_diagnostics(args.repeats) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
