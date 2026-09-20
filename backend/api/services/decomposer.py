"""
Enhanced claim decomposer.
Splits LLM outputs into atomic, verifiable claims.
Uses spaCy when available, falls back to rule-based decomposition,
plus a complex-claim heuristic rewrite (local substitute for small-LLM fallback).
"""
import re
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

try:
    import spacy
    _nlp = None

    def _get_nlp():
        global _nlp
        if _nlp is None:
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model en_core_web_sm not installed; using fallback")
                _nlp = False
        return _nlp if _nlp else None

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

    def _get_nlp():
        return None


@dataclass
class Claim:
    """A single atomic claim extracted from an LLM answer."""
    text: str
    index: int
    confidence: float = 1.0
    claim_type: str = "factual"  # factual, numerical, causal, temporal, medical
    entities: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Patterns for splitting compound sentences on coordinating conjunctions
COMPOUND_SPLIT_PATTERNS = [
    (r",\s+and\s+(?=(?:it|this|that|they|he|she|the|these|those|metformin|patients?|users?|you|we|i|also|can|may|will|should|must|is|are|was|were)\b)", re.IGNORECASE),
    (r"\s+and\s+(?=(?:it|this|that|they|he|she|the|these|those|can|may|will|should|must|is|are|was|were|has|have|had|reduces?|increases?|decreases?|causes?|leads?)\b)", re.IGNORECASE),
    (r",\s+(?:but|however|although|though|yet|whereas)\s+", re.IGNORECASE),
    (r";\s+(?=(?:it|this|that|they|he|she|the|these|those|metformin|patients?|users?|you|we|i)\b)", re.IGNORECASE),
    (r",\s+(?:additionally|moreover|furthermore|also)\s+", re.IGNORECASE),
    (r",\s+(?=(?:needs?|has|have|had)\s+to\b)", re.IGNORECASE),
    (r",\s+(?=(?:needs?|has|have|had)\s+to\b)", re.IGNORECASE),
]
HEDGE_PREFIX = re.compile(
    r"^(?:allegedly|reportedly|supposedly|apparently|arguably|possibly|perhaps|"
    r"it\s+(?:is|seems|appears)\s+(?:that|likely\s+that)|"
    r"some\s+(?:say|claim|believe)\s+(?:that\s+)?|"
    r"studies\s+suggest\s+(?:that\s+)?|"
    r"according\s+to\s+(?:some|many)\s+(?:sources?|experts?)[,\s]+)+",
    re.IGNORECASE,
)

CONDITIONAL_PATTERN = re.compile(
    r"^\s*(?:if|when|unless|provided\s+that|in\s+case)\b",
    re.IGNORECASE,
)

MEDICAL_CUES = re.compile(
    r"\b(dose|dosage|mg|ml|tablet|insulin|metformin|acidosis|side\s+effect|"
    r"contraindicat|patient|clinical|therapy|medication|drug)\b",
    re.IGNORECASE,
)

COMPLEX_MARKERS = re.compile(
    r"\b(which|who|that|although|while|whereas|because|since|due\s+to|"
    r"as\s+well\s+as|in\s+addition\s+to)\b",
    re.IGNORECASE,
)


def classify_claim_type(claim: str) -> str:
    """Classify the type of claim based on linguistic cues."""
    text = claim.lower()

    if MEDICAL_CUES.search(text) and re.search(r"\d", text):
        return "numerical"
    if MEDICAL_CUES.search(text):
        return "medical"
    if re.search(r"\d", text):
        return "numerical"
    if re.search(r"\b(causes?|leads?\s+to|results?\s+in|because|due\s+to|therefore|thus)\b", text):
        return "causal"
    if re.search(r"\b(when|after|before|during|since|until|always|never|often|rarely|sometimes)\b", text):
        return "temporal"
    return "factual"


def extract_entities_simple(text: str) -> List[Dict[str, Any]]:
    """Simple entity extraction without spaCy."""
    entities = []
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
        if m.start() == 0 or text[m.start() - 1] in ".!?\n":
            continue
        entities.append({"text": m.group(1), "label": "ENTITY", "start": m.start(), "end": m.end()})
    return entities


def strip_hedges(text: str) -> str:
    """Remove leading hedge phrases that dilute NLI signals."""
    cleaned = HEDGE_PREFIX.sub("", text).strip()
    return cleaned if cleaned else text.strip()


def is_conditional(sentence: str) -> bool:
    """True if splitting would destroy conditional meaning."""
    return bool(CONDITIONAL_PATTERN.search(sentence))


def is_complex_claim(sentence: str) -> bool:
    """Heuristic: long sentence with relative/causal markers."""
    return len(sentence) > 120 and bool(COMPLEX_MARKERS.search(sentence))


def rewrite_complex_claim(sentence: str) -> List[str]:
    """
    Local substitute for small-LLM fallback.
    Rewrites complex sentences into simpler subject–predicate atoms via rules.
    """
    text = sentence.strip()
    atoms: List[str] = []

    # Split non-restrictive relative clauses: "X, which Y, Z" → keep main + which-clause as claim
    which_split = re.split(r",\s*which\s+", text, maxsplit=1, flags=re.IGNORECASE)
    if len(which_split) == 2:
        head = which_split[0].strip()
        rest = which_split[1].strip()
        # Attach subject from head to relative clause if possible
        subj = _guess_subject(head)
        clause = rest.rstrip(".,;")
        if subj and not clause.lower().startswith(subj.lower()):
            atoms.append(f"{subj} {clause}.")
        else:
            atoms.append(clause if clause.endswith(".") else clause + ".")
        # Continue with head for further splits
        text = head if head.endswith((".", "!", "?")) else head + "."

    # Split "because / since / due to" into effect + cause when both look propositional
    cause_split = re.split(
        r"\s+(?:because|since|due\s+to\s+the\s+fact\s+that)\s+",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    if len(cause_split) == 2 and len(cause_split[0]) > 20 and len(cause_split[1]) > 15:
        effect = cause_split[0].strip(" ,;")
        cause = cause_split[1].strip(" ,;")
        if not effect.endswith((".", "!", "?")):
            effect += "."
        if not cause.endswith((".", "!", "?")):
            cause += "."
        atoms.extend([effect, cause])
        return [a for a in atoms if len(a) >= 15]

    # "as well as" coordination of predicates
    as_well = re.split(r"\s+as\s+well\s+as\s+", text, maxsplit=1, flags=re.IGNORECASE)
    if len(as_well) == 2:
        left, right = as_well[0].strip(), as_well[1].strip()
        subj = _guess_subject(left)
        if subj and right and not re.match(rf"^{re.escape(subj)}\b", right, re.IGNORECASE):
            right = f"{subj} {right}"
        for part in (left, right):
            part = part.strip(" ,;")
            if not part.endswith((".", "!", "?")):
                part += "."
            if len(part) >= 15:
                atoms.append(part)
        return atoms if atoms else [sentence]

    if atoms:
        # Also keep the simplified head
        if text and len(text) >= 15:
            atoms.insert(0, text if text.endswith((".", "!", "?")) else text + ".")
        return atoms

    return [sentence]


def _guess_subject(text: str) -> Optional[str]:
    """Very rough subject guess: first noun-ish token sequence."""
    m = re.match(
        r"^((?:The|A|An)\s+)?([A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+){0,3}|[A-Za-z]+)",
        text.strip(),
    )
    if not m:
        return None
    return (m.group(0) or "").strip()


def decompose_into_claims(text: str, min_claim_length: int = 15) -> List[Claim]:
    """
    Split an LLM answer into atomic, verifiable claims.

    Args:
        text: The LLM-generated answer
        min_claim_length: Minimum length for a claim to be considered valid

    Returns:
        List of Claim objects
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    nlp = _get_nlp() if SPACY_AVAILABLE else None
    if nlp:
        sentences = [sent.text.strip() for sent in nlp(text).sents if sent.text.strip()]
        regex_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(regex_sentences) > len(sentences):
            sentences = regex_sentences
    else:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    raw_claims: List[str] = []
    for sentence in sentences:
        sentence = strip_hedges(sentence)
        if len(sentence.strip().rstrip('.!?')) < 15:
            words = sentence.strip().rstrip('.!?').split()
            if len(words) < 3:
                continue

        # Preserve conditionals as single claims
        if is_conditional(sentence):
            pieces = [sentence]
        elif is_complex_claim(sentence):
            pieces = rewrite_complex_claim(sentence)
            # Still try compound splits on rewritten pieces
            expanded: List[str] = []
            for p in pieces:
                if is_conditional(p):
                    expanded.append(p)
                else:
                    expanded.extend(_split_compound_sentence(p))
            pieces = expanded
        else:
            pieces = _split_compound_sentence(sentence)

        for sub in pieces:
            sub = re.sub(
                r"^(and|but|however|although|moreover|furthermore|also)\s+",
                "",
                sub,
                flags=re.IGNORECASE,
            ).strip()
            sub = re.sub(r"^[,\s]+", "", sub).strip()
            sub = strip_hedges(sub)
            if len(sub.strip().rstrip('.!?')) < 15:
                words = sub.strip().rstrip('.!?').split()
                if len(words) < 3:
                    continue
            if not sub.endswith((".", "!", "?")):
                sub += "."
            raw_claims.append(sub)

    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for c in raw_claims:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)

    result = []
    for i, claim_text in enumerate(unique):
        claim_type = classify_claim_type(claim_text)
        entities = extract_entities_simple(claim_text)
        result.append(Claim(
            text=claim_text,
            index=i,
            claim_type=claim_type,
            entities=entities or None,
        ))

    return result


def _split_compound_sentence(sentence: str) -> List[str]:
    """Split a compound sentence into individual claims while preserving the subject."""
    result = [sentence]

    for pattern, flags in COMPOUND_SPLIT_PATTERNS:
        new_result = []

        for piece in result:
            parts = re.split(pattern, piece, flags=flags)

            if len(parts) == 1:
                new_result.append(piece)
                continue

            subject = _guess_subject(piece)

            for index, p in enumerate(parts):
                p = p.strip(" ,;:")

                if not p:
                    continue

                if index > 0 and subject:
                    if not re.match(
                        rf"^{re.escape(subject)}\b",
                        p,
                        re.IGNORECASE,
                    ):
                        p = f"{subject} {p}"

                new_result.append(p)

        result = new_result

    return result
def extract_named_entities(text: str) -> dict:
    """Extract named entities (kept for backward compatibility)."""
    nlp = _get_nlp() if SPACY_AVAILABLE else None
    entities = {"dates": [], "numbers": [], "names": [], "organizations": [], "locations": []}

    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "DATE":
                entities["dates"].append(ent.text)
            elif ent.label_ == "PERSON":
                entities["names"].append(ent.text)
            elif ent.label_ == "ORG":
                entities["organizations"].append(ent.text)
            elif ent.label_ == "GPE":
                entities["locations"].append(ent.text)
    numbers = re.findall(
        r"(?<!\w)(\d+(?:,\d{3})*(?:\.\d+)?)(?:%|percent|million|billion|thousand)?(?!\w)",
        text,
        re.IGNORECASE,
    )
    entities["numbers"] = numbers
    return entities
