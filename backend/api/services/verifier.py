"""
Enhanced NLI verifier with multi-model ensemble, numerical verification,
confidence calibration, and async batch support.
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..core.config import settings
from ..core.metrics import MODEL_INFERENCE_LATENCY, MODEL_LOADED
from .numerical import check_numerical_consistency, detect_unit_mismatch

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Structured verification result."""
    verdict: str  # SUPPORTED, CONTRADICTED, UNVERIFIABLE
    confidence: float
    scores: Dict[str, float]
    numerical_check: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "scores": self.scores,
            "numerical_check": self.numerical_check,
            "explanation": self.explanation,
        }


class NLIVerifier:
    """
    Multi-model NLI verifier.

    Supports:
    - Single-model inference (default)
    - Ensemble of two NLI models with weighted vote
    - Numerical consistency plugin
    - Confidence calibration
    - Async batch verification
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        secondary_model_name: Optional[str] = None,
        device: Optional[str] = None,
        use_ensemble: Optional[bool] = None,
        use_quantization: Optional[bool] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model_name = model_name or settings.nli_model_name
        self.secondary_model_name = secondary_model_name or settings.nli_model_name_secondary
        self.use_ensemble = (
            use_ensemble if use_ensemble is not None else settings.use_ensemble
        )
        self.use_quantization = (
            use_quantization if use_quantization is not None else settings.model_quantization
        )

        # Load primary model
        logger.info(f"Loading primary NLI model '{self.model_name}' on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        if self.use_quantization and device == "cpu":
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Applied INT8 dynamic quantization")
            except Exception as e:
                logger.warning(f"Quantization failed: {e}")
        self.model.to(device)
        self.model.eval()
        MODEL_LOADED.labels(model="nli_primary").set(1)
        logger.info("Primary NLI model loaded")

        # Optional secondary model for ensemble
        self.secondary_tokenizer = None
        self.secondary_model = None
        if self.use_ensemble and self.secondary_model_name != self.model_name:
            try:
                logger.info(f"Loading secondary NLI model '{self.secondary_model_name}'...")
                self.secondary_tokenizer = AutoTokenizer.from_pretrained(self.secondary_model_name)
                self.secondary_model = AutoModelForSequenceClassification.from_pretrained(self.secondary_model_name)
                self.secondary_model.to(device)
                self.secondary_model.eval()
                MODEL_LOADED.labels(model="nli_secondary").set(1)
                logger.info("Secondary NLI model loaded")
            except Exception as e:
                logger.warning(f"Could not load secondary model: {e}")
                self.secondary_model = None

        # Thread pool for batched CPU/GPU inference
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nli")

    def verify_claim(
        self,
        claim: str,
        evidence: str,
        run_numerical_check: bool = True,
    ) -> Dict[str, Any]:
        """
        Verify a single claim against evidence.

        Combines:
        1. NLI classification (primary or ensemble)
        2. Numerical consistency check
        3. Confidence calibration
        """
        if not evidence or not evidence.strip():
            return VerificationResult(
                verdict="UNVERIFIABLE",
                confidence=0.0,
                scores={"supported": 0.0, "unverifiable": 1.0, "contradicted": 0.0},
                explanation="No evidence provided",
            ).to_dict()

        # NLI classification
        primary_scores = self._nli_scores(claim, evidence, model="primary")
        scores = primary_scores
        if self.secondary_model is not None:
            secondary_scores = self._nli_scores(claim, evidence, model="secondary")
            # Weighted average: 0.6 primary, 0.4 secondary
            scores = {
                "supported": 0.6 * primary_scores["supported"] + 0.4 * secondary_scores["supported"],
                "unverifiable": 0.6 * primary_scores["unverifiable"] + 0.4 * secondary_scores["unverifiable"],
                "contradicted": 0.6 * primary_scores["contradicted"] + 0.4 * secondary_scores["contradicted"],
            }

        # Determine verdict
        verdict = max(scores, key=lambda k: {
            "supported": scores["supported"],
            "unverifiable": scores["unverifiable"],
            "contradicted": scores["contradicted"],
        }[k])
        verdict_label = {
            "supported": "SUPPORTED",
            "unverifiable": "UNVERIFIABLE",
            "contradicted": "CONTRADICTED",
        }[verdict]
        confidence = scores[verdict]

        # Numerical consistency plugin
        numerical_check = None
        if run_numerical_check:
            try:
                num_result = check_numerical_consistency(claim, evidence)
                unit_mismatch = detect_unit_mismatch(claim, evidence)
                numerical_check = {
                    "consistent": num_result.consistent,
                    "claim_value": num_result.claim_value,
                    "evidence_value": num_result.evidence_value,
                    "ratio": num_result.ratio,
                    "note": num_result.note,
                    "unit_mismatch": unit_mismatch,
                }
                # If claim and evidence numbers disagree, override toward CONTRADICTED
                if not num_result.consistent and num_result.claim_value is not None and num_result.evidence_value is not None:
                    if verdict_label != "CONTRADICTED":
                        verdict_label = "CONTRADICTED"
                        confidence = max(confidence, num_result.confidence)
                        scores = {
                            "supported": min(scores["supported"], 0.1),
                            "unverifiable": 0.1,
                            "contradicted": max(scores["contradicted"], 0.8),
                        }
            except Exception as e:
                logger.warning(f"Numerical check failed: {e}")

        # Confidence calibration
        calibrated_confidence = self._calibrate_confidence(confidence, verdict_label, numerical_check)

        explanation = self._build_explanation(verdict_label, scores, numerical_check)

        result = VerificationResult(
            verdict=verdict_label,
            confidence=round(calibrated_confidence, 3),
            scores={
                "supported": round(scores["supported"], 3),
                "unverifiable": round(scores["unverifiable"], 3),
                "contradicted": round(scores["contradicted"], 3),
            },
            numerical_check=numerical_check,
            explanation=explanation,
        )
        return result.to_dict()

    async def verify_claims_batch(
        self,
        claims: List[str],
        evidence_per_claim: List[str],
        run_numerical_check: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Verify multiple claims concurrently.

        Each claim is verified against its own evidence passage.
        """
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                self._executor,
                self.verify_claim,
                claim,
                evidence,
                run_numerical_check,
            )
            for claim, evidence in zip(claims, evidence_per_claim)
        ]
        return await asyncio.gather(*tasks)

    def _nli_scores(
        self,
        claim: str,
        evidence: str,
        model: str = "primary",
    ) -> Dict[str, float]:
        """Run NLI classification for a single claim/evidence pair."""
        if model == "primary":
            tokenizer, mdl = self.tokenizer, self.model
        else:
            tokenizer, mdl = self.secondary_tokenizer, self.secondary_model

        inputs = tokenizer(
            evidence,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            with MODEL_INFERENCE_LATENCY.labels(model=f"nli_{model}", operation="classify").time():
                outputs = mdl(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]

        # Label mapping: most DeBERTa MNLI variants use 0=entailment, 1=neutral, 2=contradiction
        # Verify with id2label if available
        if hasattr(mdl.config, "id2label"):
            label_map = {int(k): v.lower() for k, v in mdl.config.id2label.items()}
        else:
            label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

        # Build scores dict
        scores = {"supported": 0.0, "unverifiable": 0.0, "contradicted": 0.0}
        for idx, label in label_map.items():
            if "entail" in label:
                scores["supported"] = float(probs[idx])
            elif "contradict" in label:
                scores["contradicted"] = float(probs[idx])
            else:
                scores["unverifiable"] = float(probs[idx])

        return scores

    def _calibrate_confidence(
        self,
        raw_confidence: float,
        verdict: str,
        numerical_check: Optional[Dict[str, Any]],
    ) -> float:
        """
        Apply temperature scaling for confidence calibration.
        Helps avoid overconfident predictions.
        """
        # Simple Platt-style scaling: soft-cap confidence
        # The model's raw max probability tends to be overconfident
        TEMPERATURE = 1.2
        calibrated = raw_confidence ** (1.0 / TEMPERATURE)

        # Boost if numerical check agrees
        if numerical_check:
            if verdict == "CONTRADICTED" and not numerical_check.get("consistent", True):
                calibrated = min(0.99, calibrated + 0.05)
            elif verdict == "SUPPORTED" and numerical_check.get("consistent", True) and numerical_check.get("claim_value") is not None:
                calibrated = min(0.99, calibrated + 0.05)

        return calibrated

    def _build_explanation(
        self,
        verdict: str,
        scores: Dict[str, float],
        numerical_check: Optional[Dict[str, Any]],
    ) -> str:
        """Build a human-readable explanation for the verdict."""
        parts = [f"Model classified as {verdict} (confidence {scores.get(verdict.lower(), 0):.1%})"]

        if numerical_check:
            if numerical_check.get("consistent"):
                parts.append(f"numerical values consistent ({numerical_check.get('note', '')})")
            else:
                parts.append(f"numerical mismatch detected: {numerical_check.get('note', '')}")
            if numerical_check.get("unit_mismatch"):
                parts.append(f"unit mismatch: {numerical_check['unit_mismatch']}")

        return ". ".join(parts) + "."


# ============== Backward-compat top-level function ==============

def verify_claims(claims: List[str], evidence_passages: List[List[str]]) -> List[Dict]:
    """Backward-compatible verification helper."""
    verifier = NLIVerifier()
    results = []
    for claim, passages in zip(claims, evidence_passages):
        if not passages:
            results.append({
                "claim": claim,
                "verdict": "UNVERIFIABLE",
                "confidence": 0.0,
                "evidence": "",
                "reason": "No evidence provided",
            })
            continue
        result = verifier.verify_claim(claim, passages[0])
        result["claim"] = claim
        result["evidence"] = passages[0]
        results.append(result)
    return results
