"""
Numerical & entity verification service.
Extracts and cross-checks numbers, percentages, dates, and named entities
between claims and source documents.
"""
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


# Number extraction patterns
NUMBER_PATTERN = re.compile(
    r"""
    (?<!\w)              # negative lookbehind for word char
    (?:
        \d{1,3}(?:,\d{3})+(?:\.\d+)?   # 1,234 or 1,234.56
      | \d+(?:\.\d+)?                  # 1234 or 1234.56
    )
    (?:
        \s*(?:%|percent|per\s*cent)    # percent
      | \s*(?:million|billion|thousand|mn|bn|tn|k|m|b)
    )?
    (?!\w)               # negative lookahead
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Date patterns
DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),  # ISO
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),  # US format
    re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+\d{4})?\b", re.IGNORECASE),
]

# Unit conversion multipliers
UNIT_MULTIPLIERS = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "tn": 1e12, "t": 1e12,
}


def parse_number(text: str) -> Optional[float]:
    """
    Parse a number string into a float, handling commas, decimals, and unit suffixes.
    Returns None if not parseable.
    """
    if not text:
        return None

    raw = text.strip().lower()

    # Try plain float first
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        pass

    # Try with unit multipliers
    m = re.match(
        r"^([\d,]+(?:\.\d+)?)\s*(%?|percent|per\s*cent|million|billion|thousand|mn|bn|tn|k|m|b|t)?$",
        raw,
    )
    if m:
        number_str, unit = m.groups()
        try:
            value = float(number_str.replace(",", ""))
            if unit and unit in UNIT_MULTIPLIERS:
                value *= UNIT_MULTIPLIERS[unit]
            return value
        except ValueError:
            return None

    return None


def extract_numbers(text: str) -> List[Dict[str, Any]]:
    """
    Extract numerical values from text along with their context.

    Returns:
        List of dicts with keys: text, value, position, context
    """
    results = []
    for m in NUMBER_PATTERN.finditer(text):
        raw = m.group(0)
        value = parse_number(raw)
        if value is not None:
            # Capture context: ±40 chars around the number
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 40)
            results.append({
                "text": raw,
                "value": value,
                "start": m.start(),
                "end": m.end(),
                "context": text[start:end].strip(),
            })
    return results


def extract_dates(text: str) -> List[Dict[str, Any]]:
    """Extract date mentions from text."""
    results = []
    for pattern in DATE_PATTERNS:
        for m in pattern.finditer(text):
            results.append({
                "text": m.group(0),
                "start": m.start(),
                "end": m.end(),
            })
    return results


# Numbers that label a category rather than measure a quantity. "type 2
# diabetes" contains no quantity, but the old nearest-value matcher happily
# compared it against "$500" in a claim and declared a contradiction.
_ORDINAL_CONTEXT = re.compile(
    r"\b(?:type|stage|phase|grade|class|level|version|step|part|chapter|"
    r"figure|table|section|group|tier)\s*$",
    re.IGNORECASE,
)

_RATIO_PATTERN = re.compile(
    r"(\d[\d,\.]*)\s*(?:in|per|out\s+of)\s*(\d[\d,\.]*)", re.IGNORECASE
)

_CURRENCY_BEFORE = re.compile(r"[$£€¥]\s*$")
_CURRENCY_AFTER = re.compile(r"^\s*(?:dollars?|usd|eur|gbp|euros?|pounds?)\b", re.IGNORECASE)


def _unit_class(text: str, raw: str, start: int, end: int) -> str:
    """
    Classify what kind of quantity a number is, so unrelated numbers are never
    compared. Returns one of: percent, ratio, currency, plain.
    """
    token = raw.lower()
    if "%" in token or "percent" in token or "per cent" in token:
        return "percent"
    trailing = text[end:end + 12]
    if re.match(r"^\s*(?:%|percent|per\s*cent)", trailing, re.IGNORECASE):
        return "percent"
    if _CURRENCY_BEFORE.search(text[max(0, start - 3):start]) or _CURRENCY_AFTER.match(trailing):
        return "currency"
    return "plain"


@dataclass
class Quantity:
    """A number with enough context to know whether it is comparable."""
    value: float          # percent-normalised for percent/ratio classes
    unit_class: str
    text: str
    context: str


def extract_quantities(text: str) -> List[Quantity]:
    """
    Extract comparable quantities, dropping category labels and normalising
    ratios ("1 in 30,000") onto the percent scale so they can be compared with
    percentages.
    """
    quantities: List[Quantity] = []
    consumed: List[Tuple[int, int]] = []

    for m in _RATIO_PATTERN.finditer(text):
        numerator = parse_number(m.group(1))
        denominator = parse_number(m.group(2))
        if numerator is None or not denominator:
            continue
        quantities.append(
            Quantity(
                value=numerator / denominator * 100.0,
                unit_class="percent",   # ratios are compared on the percent scale
                text=m.group(0),
                context=text[max(0, m.start() - 40):m.end() + 40].strip(),
            )
        )
        consumed.append((m.start(), m.end()))

    for m in NUMBER_PATTERN.finditer(text):
        if any(start <= m.start() < end for start, end in consumed):
            continue  # already captured as part of a ratio
        if _ORDINAL_CONTEXT.search(text[max(0, m.start() - 16):m.start()]):
            continue  # "type 2", "stage 3" — a label, not a measurement
        value = parse_number(m.group(0))
        if value is None:
            continue
        quantities.append(
            Quantity(
                value=value,
                unit_class=_unit_class(text, m.group(0), m.start(), m.end()),
                text=m.group(0),
                context=text[max(0, m.start() - 40):m.end() + 40].strip(),
            )
        )

    return quantities


@dataclass
class NumericalCheck:
    """Result of a numerical consistency check."""
    consistent: bool
    claim_value: Optional[float]
    evidence_value: Optional[float]
    ratio: Optional[float]
    confidence: float
    note: str = ""
    # match | mismatch | not_comparable | no_numbers
    # Only "mismatch" is allowed to override an NLI verdict.
    status: str = "no_numbers"


def check_numerical_consistency(
    claim: str,
    evidence: str,
    tolerance: float = 0.05,
) -> NumericalCheck:
    """
    Check whether numerical claims in the claim are consistent with numbers in the evidence.

    Strategy:
    1. Extract numbers from claim and evidence
    2. Match by position (first claim number vs. first evidence number, etc.)
    3. If numbers exist in claim but not in evidence, look for them in larger context
    4. If claim and evidence numbers match within tolerance, mark consistent

    Args:
        claim: The claim text
        evidence: The evidence text
        tolerance: Allowed relative difference (0.05 = 5%)

    Returns:
        NumericalCheck with consistency verdict
    """
    claim_quantities = extract_quantities(claim)
    evidence_quantities = extract_quantities(evidence)

    if not claim_quantities:
        return NumericalCheck(
            consistent=True, claim_value=None, evidence_value=None, ratio=None,
            confidence=0.5, note="No numerical values in claim", status="no_numbers",
        )

    if not evidence_quantities:
        # The claim asserts a number the evidence never mentions. That is a
        # failure to verify, NOT a contradiction — there is nothing to conflict
        # with. evidence_value stays None so no verdict override can fire.
        return NumericalCheck(
            consistent=True,
            claim_value=claim_quantities[0].value,
            evidence_value=None,
            ratio=None,
            confidence=0.4,
            note="Claim contains a number the evidence does not mention",
            status="not_comparable",
        )

    matches = 0
    mismatches = 0
    uncomparable = 0
    first_pair: Optional[Tuple[float, float]] = None
    notes: List[str] = []

    for cq in claim_quantities:
        # Only compare like with like. percent and ratio share a scale (ratios
        # are normalised to percent above); currency and plain numbers do not
        # mix with anything else.
        candidates = [eq for eq in evidence_quantities if eq.unit_class == cq.unit_class]
        if not candidates:
            uncomparable += 1
            notes.append(f"{cq.text}: no comparable {cq.unit_class} value in evidence")
            continue

        best = min(
            candidates,
            key=lambda eq: abs(cq.value - eq.value) / abs(eq.value) if eq.value else float("inf"),
        )
        if first_pair is None:
            first_pair = (cq.value, best.value)

        if not best.value:
            relative_diff = float("inf") if cq.value else 0.0
        else:
            relative_diff = abs(cq.value - best.value) / abs(best.value)

        if relative_diff <= tolerance:
            matches += 1
        else:
            mismatches += 1
            notes.append(f"{cq.text} vs {best.text} in evidence")

    claim_value = claim_quantities[0].value
    evidence_value = first_pair[1] if first_pair else None
    ratio_value = (
        first_pair[0] / first_pair[1] if first_pair and first_pair[1] else None
    )

    if mismatches:
        return NumericalCheck(
            consistent=False,
            claim_value=claim_value,
            evidence_value=evidence_value,
            ratio=ratio_value,
            confidence=0.85 if matches == 0 else 0.7,
            note="Numerical values do not match: " + "; ".join(notes[:3]),
            status="mismatch",
        )

    if matches:
        return NumericalCheck(
            consistent=True,
            claim_value=claim_value,
            evidence_value=evidence_value,
            ratio=ratio_value,
            confidence=0.95,
            note=f"All {matches} numerical value(s) match within {tolerance * 100:.0f}% tolerance",
            status="match",
        )

    return NumericalCheck(
        consistent=True,
        claim_value=claim_value,
        evidence_value=None,
        ratio=None,
        confidence=0.4,
        note="; ".join(notes[:3]) or "No comparable numerical values",
        status="not_comparable",
    )


def detect_unit_mismatch(claim: str, evidence: str) -> Optional[str]:
    """
    Detect if a claim and evidence use different units for the same quantity
    (e.g. "10%" vs "1 in 30,000").
    """
    claim_numbers = extract_numbers(claim)
    evidence_numbers = extract_numbers(evidence)

    if not claim_numbers or not evidence_numbers:
        return None

    claim_text = claim.lower()
    evidence_text = evidence.lower()

    # Look for common unit markers
    has_percent_claim = "%" in claim_text or "percent" in claim_text
    has_percent_ev = "%" in evidence_text or "percent" in evidence_text

    has_ratio_claim = re.search(r"\b\d+\s*(?:in|per|out\s+of)\s*\d+", claim_text) is not None
    has_ratio_ev = re.search(r"\b\d+\s*(?:in|per|out\s+of)\s*\d+", evidence_text) is not None

    if has_percent_claim and has_ratio_ev and not has_percent_ev:
        return "claim uses percentage, evidence uses ratio"
    if has_ratio_claim and has_percent_ev and not has_percent_claim:
        return "claim uses ratio, evidence uses percentage"

    return None
