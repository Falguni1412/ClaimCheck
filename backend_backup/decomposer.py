"""
Claim Decomposer: Split LLM answer into atomic, verifiable claims
Integrates spaCy, heuristic compound sentence splitting, hedge stripping, and complex-claim rewrites.
"""
from typing import List, Dict, Any
from api.services.decomposer import decompose_into_claims as service_decompose_into_claims, extract_entities_simple, strip_hedges

def decompose_into_claims(text: str) -> List[str]:
    """
    Split LLM answer into atomic claims using advanced decomposition pipeline.

    Args:
        text: The LLM-generated answer to decompose

    Returns:
        List of atomic claim strings
    """
    if not text or not text.strip():
        return []
    claims = service_decompose_into_claims(text)
    return [c.text for c in claims]


def extract_named_entities(text: str) -> Dict[str, Any]:
    """Extract named entities from text for context."""
    entities = extract_entities_simple(text)
    return {"entities": entities}

if __name__ == "__main__":
    test_text = """
    Metformin should be taken on an empty stomach.
    It can be combined with insulin therapy.
    Lactic acidosis occurs in 10% of patients.
    Nausea is a common side effect.
    """
    claims = decompose_into_claims(test_text)
    print(f"Decomposed {len(claims)} claims:")
    for i, claim in enumerate(claims, 1):
        print(f"  {i}. {claim}")

