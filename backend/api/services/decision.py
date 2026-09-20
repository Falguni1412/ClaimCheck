"""
Verdict decision policy.

This module holds the rules that turn raw NLI probabilities into a verdict. It
exists because the original pipeline used a bare `argmax` over three classes and
called the winner the answer, which conflates two very different situations:

    P(entail)=0.91, P(neutral)=0.05   -> the model is asserting entailment
    P(entail)=0.38, P(neutral)=0.36   -> the model is not asserting anything

Both used to produce SUPPORTED. Only the first should.

Three rules are applied, in this order:

1. **Grounding gate** — evidence that does not mention the claim's entities is
   not evidence *about* the claim. An ungrounded claim can never be SUPPORTED.
   This is the "same-entity != support" rule: retrieval returns the most
   *similar* passage, always, even when nothing relevant exists.

2. **Margin rule** — a class must win by a margin and clear an absolute floor.
   Otherwise the verdict is UNVERIFIABLE (abstain).

3. **Multi-premise aggregation** — a claim is checked against each retrieved
   passage and each sentence within it. A contradiction anywhere is decisive;
   support requires at least one premise that actually entails.

4. **Categorical consistency check** — a symbolic, model-independent check for
   one specific and important failure mode: simple "X is a/an NOUN" claims,
   where the NLI model itself is the weak point. A 6-layer cross-encoder can
   score P(entailment) > 0.8 for "Metformin is a person" against "Metformin is
   a medication..." — a category error the model does not reliably catch,
   confirmed by direct inspection of its logits (see scripts/diagnose_nli.py).
   No amount of threshold tuning fixes a model that is confidently wrong.

   The fix is architectural, not statistical: extract the copula subject and
   predicate noun phrase from the claim and from each evidence sentence using
   only closed-class function words (prepositions, relative pronouns, a short
   list of passive/linking verbs) as parse boundaries — never any content word,
   entity name, or domain vocabulary. If both sides assert a category for the
   same subject and the categories are incompatible, that is a **structural**
   contradiction, independent of what the neural network thinks. It is treated
   exactly like the existing numerical-mismatch override (see numerical.py):
   a symbolic signal that can override a wrong NLI verdict, injected into the
   same aggregation the NLI premises go through.

   This deliberately covers only simple bare copular assertions. Claims like
   "Metformin is used to treat diabetes" or "The medication is called
   Metformin" do not match the pattern (the predicate is a verb phrase, not a
   noun phrase) and fall through to NLI unchanged — this check narrows its own
   scope by grammar, not by a list of approved sentences, which is what lets it
   generalize to any subject without special-casing.

None of this is claim-specific or keyword-based. There is no term list tied to
any particular subject matter — every stoplist below is a closed grammatical
class (articles, prepositions, relative pronouns, linking verbs), the same
category of list a syntax textbook would use for any language, not a domain
vocabulary tied to medicine, or to "Metformin," or to "person."
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Words that are capitalised for grammatical reasons rather than because they
# name something. Used only to avoid treating sentence-initial function words as
# entities — it is not a domain vocabulary.
_NON_ENTITY_CAPITALS = {
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "they",
    "there", "here", "he", "she", "his", "her", "we", "our", "you", "your",
    "i", "in", "on", "at", "of", "for", "to", "from", "by", "with", "and",
    "but", "or", "if", "when", "while", "after", "before", "as", "is", "are",
    "was", "were", "be", "been", "being", "do", "does", "did", "have", "has",
    "had", "can", "could", "may", "might", "will", "would", "should", "must",
    "not", "no", "all", "some", "many", "most", "each", "every", "both",
}

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-']*|\d[\d,\.]*%?")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class DecisionPolicy:
    """Thresholds for the abstention rule. All configurable; see config.py."""

    min_entailment: float = 0.55
    min_contradiction: float = 0.50
    margin: float = 0.10
    require_grounding: bool = True
    max_premise_units: int = 12
    require_category_consistency: bool = True
    category_mismatch_confidence: float = 0.90


@dataclass
class PremiseVerdict:
    """One claim scored against one premise."""

    premise: str
    verdict: str
    confidence: float
    scores: Dict[str, float]
    reason: str


# --------------------------------------------------------------- premises


def split_premises(passages: Sequence[str], max_units: int = 12) -> List[str]:
    """
    Turn retrieved passages into the premise units the NLI model will see.

    Both the whole passage and its individual sentences are included. A long
    premise dilutes the signal (the model sees mostly unrelated text and leans
    on lexical overlap), while a sentence-level premise can miss context that
    spans sentences — so both are scored and the strongest signal wins.
    """
    units: List[str] = []
    seen: Set[str] = set()

    def add(text: str) -> None:
        text = text.strip()
        if len(text) < 10:
            return
        key = text.lower()
        if key not in seen:
            seen.add(key)
            units.append(text)

    for passage in passages:
        if not passage or not passage.strip():
            continue
        add(passage)
        sentences = [s for s in _SENTENCE_SPLIT.split(passage) if s.strip()]
        if len(sentences) > 1:
            for sentence in sentences:
                add(sentence)

    return units[:max_units]


# --------------------------------------------------------------- grounding


def entity_terms(text: str) -> Set[str]:
    """
    Terms whose presence in the evidence is a precondition for support:
    proper-noun-like tokens and numeric literals.

    Heuristic, deliberately narrow: a capitalised token that is not a common
    function word, or any number. Ordinary content words ("medication",
    "called") are excluded, because requiring lexical overlap on those would
    just be keyword matching and would reject valid paraphrases.
    """
    terms: Set[str] = set()
    for sentence in _SENTENCE_SPLIT.split(text):
        tokens = _TOKEN.findall(sentence)
        for position, token in enumerate(tokens):
            lowered = token.lower().rstrip(".,")
            if not lowered:
                continue
            if token[0].isdigit():
                terms.add(_normalise_number(lowered))
                continue
            if not token[0].isupper():
                continue
            # A capitalised token in first position may just be the start of a
            # sentence, so only keep it if it is not a common function word.
            if position == 0 and lowered in _NON_ENTITY_CAPITALS:
                continue
            if lowered in _NON_ENTITY_CAPITALS:
                continue
            terms.add(lowered)
    return terms


def _normalise_number(token: str) -> str:
    return token.replace(",", "").rstrip("%")


def grounding_gap(claim: str, evidence: str) -> List[str]:
    """Entity terms present in the claim but absent from the evidence."""
    evidence_terms = entity_terms(evidence)
    evidence_lower = evidence.lower()
    missing = []
    for term in sorted(entity_terms(claim)):
        if term in evidence_terms:
            continue
        # Fall back to a substring check so morphological variants
        # ("Metformin" / "metformin's") still count as grounded.
        if term in evidence_lower:
            continue
        missing.append(term)
    return missing


# --------------------------------------------------------------- classify


def classify_scores(
    scores: Dict[str, float],
    policy: DecisionPolicy,
) -> Tuple[str, float, str]:
    """
    Apply the margin rule to one set of NLI probabilities.

    Returns (verdict, confidence, reason). Confidence is always the probability
    of the class that was actually returned, so an UNVERIFIABLE verdict reports
    how uncertain the model was rather than borrowing the losing class's score.
    """
    entail = float(scores.get("supported", 0.0))
    neutral = float(scores.get("unverifiable", 0.0))
    contra = float(scores.get("contradicted", 0.0))

    if (
        contra >= policy.min_contradiction
        and contra - neutral >= policy.margin
        and contra - entail >= policy.margin
    ):
        return "CONTRADICTED", contra, (
            f"contradiction {contra:.2f} clears floor {policy.min_contradiction:.2f} "
            f"and leads neutral/entailment by >= {policy.margin:.2f}"
        )

    if (
        entail >= policy.min_entailment
        and entail - neutral >= policy.margin
        and entail - contra >= policy.margin
    ):
        return "SUPPORTED", entail, (
            f"entailment {entail:.2f} clears floor {policy.min_entailment:.2f} "
            f"and leads neutral/contradiction by >= {policy.margin:.2f}"
        )

    top = max(("supported", entail), ("unverifiable", neutral), ("contradicted", contra),
              key=lambda kv: kv[1])
    return "UNVERIFIABLE", neutral, (
        f"no class cleared the decision margin (entail {entail:.2f}, neutral {neutral:.2f}, "
        f"contradict {contra:.2f}; highest was {top[0]})"
    )


def aggregate(
    per_premise: Sequence[PremiseVerdict],
    policy: DecisionPolicy,
) -> Optional[PremiseVerdict]:
    """
    Combine per-premise verdicts into one.

    Contradiction wins over support: if any single passage contradicts the
    claim, the claim is contradicted, even if another passage looks supportive.
    That is the conservative direction for a hallucination detector.
    """
    if not per_premise:
        return None

    contradictions = [p for p in per_premise if p.verdict == "CONTRADICTED"]
    if contradictions:
        return max(contradictions, key=lambda p: p.confidence)

    supports = [p for p in per_premise if p.verdict == "SUPPORTED"]
    if supports:
        return max(supports, key=lambda p: p.confidence)

    # Nothing decisive: report the premise the model was least unsure about.
    return min(per_premise, key=lambda p: p.scores.get("unverifiable", 1.0))


# --------------------------------------------------------------- categorical


# Every set below is a closed class of English function/linking words — the
# same kind of list a parser uses for any sentence in the language. None of
# them names an entity, a domain, or a specific claim.

_COPULA_VERBS = {"is", "are", "was", "were"}
_ARTICLES = {"a", "an", "the"}

# Words that end a bare noun-phrase predicate. If the FIRST predicate token is
# one of these, the predicate is not a noun phrase at all (it's a passive verb,
# a negation, or a subordinate clause), and the sentence is not a category
# assertion this check can parse — e.g. "is called X", "is not a fruit", "is
# used to treat Y". If one appears LATER, it marks where a trailing
# prepositional/relative/verbal clause begins, and everything from there is
# dropped — e.g. "a medication for type 2 diabetes" -> "a medication".
_PREDICATE_BOUNDARY_WORDS = {
    # prepositions
    "for", "of", "in", "on", "at", "with", "without", "by", "from", "to",
    "about", "under", "over", "into", "onto", "upon", "than", "as", "among",
    "between", "during",
    # relative pronouns / subordinators / coordinators
    "that", "which", "who", "whom", "whose", "when", "where", "because",
    "since", "although", "though", "while", "and", "or", "but", "if",
    # passive / linking-verb continuations: these mean the predicate is a verb
    # phrase, not a noun phrase (e.g. "is called Metformin", "is used to...")
    "used", "called", "named", "known", "termed", "dubbed", "referred",
    "described", "classified", "considered", "regarded", "believed",
    "thought", "said", "seen", "viewed", "defined", "given", "found", "made",
    "based", "taken", "prescribed", "administered", "indicated", "associated",
    "kept", "held",
    # negation — "is not a fruit" is not an assertion that it IS anything
    "not", "never",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:'[A-Za-z]+)?|\d[\d,\.]*")

# Scope limit, stated plainly: this is crude singular/plural folding on a
# single word, not real morphology. It is applied only to a one-word
# predicate, where the failure mode ("medication" vs "medications") is common
# and the risk of a bad fold is low.
def _fold_plural(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def extract_copula(sentence: str) -> Optional[Tuple[str, str]]:
    """
    Pull (subject, predicate_noun_phrase) out of a bare copular sentence.

    Returns None — deliberately, and often — for anything that is not a
    simple "SUBJECT is/are/was/were [a/an/the] NOUN PHRASE" assertion. That
    includes verb-phrase predicates, passive constructions, negation, and
    anything long or ambiguous enough that a confident parse isn't warranted.
    A narrow, well-understood scope beats a broad, unreliable one here — the
    aggregation step never sees this signal for claims outside that scope, so
    it can't get anything else wrong.
    """
    words = _WORD_RE.findall(sentence)
    lowered = [w.lower() for w in words]

    copula_idx = next(
        (i for i, w in enumerate(lowered) if w in _COPULA_VERBS and i > 0), None
    )
    if copula_idx is None:
        return None

    subject_words = words[:copula_idx]
    if not (1 <= len(subject_words) <= 4):
        return None  # empty, or long enough that this probably isn't a simple subject

    predicate_words = words[copula_idx + 1:]
    if not predicate_words:
        return None

    boundary = next(
        (i for i, w in enumerate(predicate_words) if w.lower() in _PREDICATE_BOUNDARY_WORDS),
        None,
    )
    if boundary == 0:
        return None  # predicate isn't a noun phrase at all
    if boundary is not None:
        predicate_words = predicate_words[:boundary]
    if len(predicate_words) > 6:
        return None  # no boundary found and it's long — likely a run-on, don't guess

    if predicate_words[0].lower() in _ARTICLES:
        predicate_words = predicate_words[1:]
    if not predicate_words:
        return None
    if predicate_words[0][0].isdigit():
        return None  # a quantity, not a category — numerical.py handles this

    subject = subject_words[1:] if (
        len(subject_words) > 1 and subject_words[0].lower() in _ARTICLES
    ) else subject_words

    subject_norm = " ".join(w.lower() for w in subject)
    predicate_norm = " ".join(w.lower() for w in predicate_words)
    if len(predicate_words) == 1:
        predicate_norm = _fold_plural(predicate_norm)

    if not subject_norm or not predicate_norm:
        return None
    return subject_norm, predicate_norm


def _predicate_compatible(claim_predicate: str, evidence_predicate: str) -> bool:
    """
    Whether two predicate noun phrases could reasonably describe the same
    category. Deliberately permissive — this function's job is to avoid false
    mismatches, not to prove synonymy. A miss here (real synonym not caught)
    just means the check abstains and NLI decides, same as if there were no
    evidence-side copula at all; a false positive here would suppress a real
    contradiction, which is the direction this check must not err in.
    """
    if claim_predicate == evidence_predicate:
        return True
    if claim_predicate in evidence_predicate or evidence_predicate in claim_predicate:
        return True
    claim_head = claim_predicate.split()[-1]
    evidence_head = evidence_predicate.split()[-1]
    return claim_head == evidence_head


def category_mismatch_verdict(
    claim: str,
    premises: Sequence[str],
    policy: DecisionPolicy,
) -> Optional[PremiseVerdict]:
    """
    Check the claim's copular category assertion against every evidence
    sentence's copular assertion about the same subject.

    Returns None whenever there is nothing to compare (the claim isn't a bare
    "is-a" statement, or no evidence sentence makes one about the same
    subject) — this check only ever *adds* a contradiction signal; it never
    manufactures support, and a single confirming sentence anywhere in the
    evidence is enough to stand down even if another sentence conflicts,
    which keeps this conservative in the face of mixed or noisy sources.
    """
    if not policy.require_category_consistency:
        return None

    claim_copula = extract_copula(claim.strip())
    if claim_copula is None:
        return None
    claim_subject, claim_predicate = claim_copula

    confirmed = False
    mismatch: Optional[Tuple[str, str]] = None

    for premise in premises:
        for sentence in _SENTENCE_SPLIT.split(premise):
            sentence = sentence.strip()
            if not sentence:
                continue
            evidence_copula = extract_copula(sentence)
            if evidence_copula is None:
                continue
            evidence_subject, evidence_predicate = evidence_copula
            if evidence_subject != claim_subject:
                continue
            if _predicate_compatible(claim_predicate, evidence_predicate):
                confirmed = True
            elif mismatch is None:
                mismatch = (evidence_predicate, sentence)

    if confirmed or mismatch is None:
        return None

    evidence_predicate, evidence_sentence = mismatch
    reason = (
        f"structural category mismatch: evidence states '{claim_subject}' is "
        f"'{evidence_predicate}', but the claim asserts it is '{claim_predicate}' "
        "— an explicit is-a conflict, independent of the NLI model's own score"
    )
    return PremiseVerdict(
        premise=evidence_sentence,
        verdict="CONTRADICTED",
        confidence=policy.category_mismatch_confidence,
        scores={
            "supported": round(1.0 - policy.category_mismatch_confidence, 3),
            "unverifiable": round((1.0 - policy.category_mismatch_confidence) / 2, 3),
            "contradicted": policy.category_mismatch_confidence,
        },
        reason=reason,
    )
