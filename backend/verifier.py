"""
NLI Verifier: Classify claims as SUPPORTED, CONTRADICTED, or UNVERIFIABLE
Ensemble NLI, numerical/unit mismatch override, confidence calibration, human-readable explanations.
"""
from typing import Dict, List, Optional
from api.services.verifier import NLIVerifier as EnhancedNLIVerifier, verify_claims as enhanced_verify_claims

class NLIVerifier:
    """
    Multi-model NLI verifier with numerical consistency overrides.
    """

    def __init__(self, device: Optional[str] = None):
        self._verifier = EnhancedNLIVerifier(device=device)

    def verify_claim(self, claim: str, evidence: str, run_numerical_check: bool = True) -> Dict:
        res = self._verifier.verify_claim(claim, evidence, run_numerical_check=run_numerical_check)
        res["claim"] = claim
        res["evidence"] = evidence
        return res

    def verify_batch(self, claims: List[str], evidence: str) -> List[Dict]:
        results = []
        for claim in claims:
            results.append(self.verify_claim(claim, evidence))
        return results

def verify_claims(claims: List[str], evidence_passages: List[List[str]]) -> List[Dict]:
    return enhanced_verify_claims(claims, evidence_passages)

__all__ = ["NLIVerifier", "verify_claims"]

if __name__ == "__main__":
    verifier = NLIVerifier()

    test_cases = [
        {
            "claim": "Metformin should be taken on an empty stomach",
            "evidence": "Take metformin with meals to reduce stomach upset. Do not take on an empty stomach."
        },
        {
            "claim": "Lactic acidosis occurs in 10% of patients",
            "evidence": "Lactic acidosis is rare, occurring in approximately 1 in 30,000 patient-years."
        },
        {
            "claim": "Metformin can be combined with insulin therapy",
            "evidence": "Metformin can be safely combined with insulin therapy under medical supervision."
        }
    ]

    for case in test_cases:
        result = verifier.verify_claim(case["claim"], case["evidence"])
        emoji = "✅" if result["verdict"] == "SUPPORTED" else "❌" if result["verdict"] == "CONTRADICTED" else "⚠️"
        print(f"\n{emoji} {result['verdict']} (confidence: {result['confidence']:.1%})")
        print(f"   Claim: {case['claim']}")
        print(f"   Evidence: {case['evidence'][:80]}...")

