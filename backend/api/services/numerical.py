"""
Numerical & entity verification service.
Extracts and cross-checks numbers, percentages, dates, and named entities
between claims and source documents.
"""
import re
from typing import List, Dict, Any, Optional, Tuple
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


@dataclass
class NumericalCheck:
    """Result of a numerical consistency check."""
    consistent: bool
    claim_value: Optional[float]
    evidence_value: Optional[float]
    ratio: Optional[float]
    confidence: float
    note: str = ""


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
    claim_numbers = extract_numbers(claim)
    evidence_numbers = extract_numbers(evidence)

    # No numbers to check
    if not claim_numbers:
        return NumericalCheck(
            consistent=True,
            claim_value=None,
            evidence_value=None,
            ratio=None,
            confidence=0.5,
            note="No numerical values in claim",
        )

    # If claim has numbers but evidence has none, cannot verify
    if not evidence_numbers:
        return NumericalCheck(
            consistent=False,
            claim_value=claim_numbers[0]["value"],
            evidence_value=None,
            ratio=None,
            confidence=0.6,
            note="Claim contains numbers not found in evidence",
        )

    # Compare each claim number against the closest evidence number
    inconsistencies = 0
    matches = 0
    for cn in claim_numbers:
        best_match = None
        best_diff = float("inf")
        for en in evidence_numbers:
            if en["value"] == 0:
                continue
            diff = abs(cn["value"] - en["value"]) / abs(en["value"])
            if diff < best_diff:
                best_diff = diff
                best_match = en

        if best_match and best_diff <= tolerance:
            matches += 1
        else:
            inconsistencies += 1

    if inconsistencies == 0:
        return NumericalCheck(
            consistent=True,
            claim_value=claim_numbers[0]["value"],
            evidence_value=evidence_numbers[0]["value"] if evidence_numbers else None,
            ratio=claim_numbers[0]["value"] / evidence_numbers[0]["value"] if evidence_numbers and evidence_numbers[0]["value"] != 0 else None,
            confidence=0.95,
            note=f"All {matches} numerical value(s) match within {tolerance * 100:.0f}% tolerance",
        )
    elif matches > 0:
        return NumericalCheck(
            consistent=False,
            claim_value=claim_numbers[0]["value"],
            evidence_value=evidence_numbers[0]["value"] if evidence_numbers else None,
            ratio=claim_numbers[0]["value"] / evidence_numbers[0]["value"] if evidence_numbers and evidence_numbers[0]["value"] != 0 else None,
            confidence=0.7,
            note=f"{inconsistencies} of {len(claim_numbers)} number(s) inconsistent",
        )
    else:
        return NumericalCheck(
            consistent=False,
            claim_value=claim_numbers[0]["value"],
            evidence_value=evidence_numbers[0]["value"] if evidence_numbers else None,
            ratio=claim_numbers[0]["value"] / evidence_numbers[0]["value"] if evidence_numbers and evidence_numbers[0]["value"] != 0 else None,
            confidence=0.85,
            note="Numerical values do not match",
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
