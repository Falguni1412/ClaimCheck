"""
NLI verifier with batched CPU inference, multi-model ensemble, numerical
verification, confidence calibration, and async batch support.

Performance notes
-----------------
* Models are loaded once per process (see ``get_verifier``) — never per request.
* Claims are tokenised and scored in batches, so an N-claim answer costs one
  forward pass per ``nli_batch_size`` claims instead of N separate passes.
* ``torch.inference_mode`` plus a pinned thread count keeps peak RSS flat on
  small CPU instances; oversubscribing threads on a fractional vCPU makes
  inference slower, not faster.
"""
import asyncio
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..core.config import settings
from ..core.metrics import MODEL_INFERENCE_LATENCY, MODEL_LOADED
from .numerical import check_numerical_consistency, detect_unit_mismatch

logger = logging.getLogger(__name__)

_GENERIC_LABEL = re.compile(r"^label_\d+$")

# Sentence-transformers cross-encoder NLI heads ship without readable labels.
# This is their documented output order.
_DEFAULT_LABEL_ORDER = {0: "contradiction", 1: "entailment", 2: "neutral"}

_EMPTY_SCORES = {"supported": 0.0, "unverifiable": 1.0, "contradicted": 0.0}


def configure_torch_runtime() -> None:
    """Pin thread counts and disable autograd globally. Safe to call twice."""
    try:
        torch.set_num_threads(max(1, settings.torch_num_threads))
        torch.set_num_interop_threads(1)
    except Exception:  # interop threads can only be set once per process
        pass
    torch.set_grad_enabled(False)


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


def _resolve_label_map(config: Any, override: Sequence[str]) -> Dict[int, str]:
    """
    Work out which logit index means what.

    ``AutoConfig`` always exposes ``id2label``, but for many NLI checkpoints it
    is the useless default ``{0: 'LABEL_0', ...}``. Trusting it blindly funnels
    every class into the "unverifiable" bucket, which is why every verdict comes
    back UNVERIFIABLE with a model that is actually working fine.
    """
    raw = getattr(config, "id2label", None) or {}
    labels = {int(k): str(v).lower() for k, v in raw.items()}

    if override:
        if labels and len(override) != len(labels):
            logger.warning(
                "NLI_LABEL_ORDER has %d entries but the model has %d labels; ignoring override",
                len(override), len(labels),
            )
        else:
            return dict(enumerate(override))

    if labels and not all(_GENERIC_LABEL.match(v) for v in labels.values()):
        return labels

    logger.warning(
        "Model reports generic labels %s; assuming contradiction/entailment/neutral. "
        "Set NLI_LABEL_ORDER to override.", sorted(labels.values()) or "none",
    )
    return dict(_DEFAULT_LABEL_ORDER)


def _scores_from_probs(probs: Sequence[float], label_map: Dict[int, str]) -> Dict[str, float]:
    scores = {"supported": 0.0, "unverifiable": 0.0, "contradicted": 0.0}
    for idx, label in label_map.items():
        if idx >= len(probs):
            continue
        value = float(probs[idx])
        if "entail" in label:
            scores["supported"] = value
        elif "contradict" in label:
            scores["contradicted"] = value
        else:
            scores["unverifiable"] = value
    return scores


class NLIVerifier:
    """
    Multi-model NLI verifier.

    Supports:
    - Single-model inference (default)
    - Ensemble of two NLI models with weighted vote
    - Numerical consistency plugin
    - Confidence calibration
    - Batched + async verification
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        secondary_model_name: Optional[str] = None,
        device: Optional[str] = None,
        use_ensemble: Optional[bool] = None,
        use_quantization: Optional[bool] = None,
    ):
        configure_torch_runtime()

        if device is None:
            device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model_name = model_name or settings.nli_model_name
        self.secondary_model_name = secondary_model_name or settings.nli_model_secondary
        self.use_ensemble = use_ensemble if use_ensemble is not None else settings.use_ensemble
        self.use_quantization = (
            use_quantization if use_quantization is not None else settings.model_quantization
        )
        self.max_length = settings.nli_max_length
        self.batch_size = max(1, settings.nli_batch_size)

        # Load primary model
        logger.info("Loading primary NLI model '%s' on %s...", self.model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.model.to(device)
        self.model = self._maybe_quantize(self.model)
        self.label_map = _resolve_label_map(self.model.config, settings.nli_label_order_list)
        MODEL_LOADED.labels(model="nli_primary").set(1)
        logger.info("Primary NLI model loaded (labels=%s)", self.label_map)

        # Optional secondary model for ensemble
        self.secondary_tokenizer = None
        self.secondary_model = None
        self.secondary_label_map: Dict[int, str] = {}
        if self.use_ensemble and self.secondary_model_name != self.model_name:
            try:
                logger.info("Loading secondary NLI model '%s'...", self.secondary_model_name)
                self.secondary_tokenizer = AutoTokenizer.from_pretrained(self.secondary_model_name)
                self.secondary_model = AutoModelForSequenceClassification.from_pretrained(
                    self.secondary_model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True,
                )
                self.secondary_model.eval()
                self.secondary_model.to(device)
                self.secondary_model = self._maybe_quantize(self.secondary_model)
                self.secondary_label_map = _resolve_label_map(
                    self.secondary_model.config, settings.nli_label_order_list
                )
                MODEL_LOADED.labels(model="nli_secondary").set(1)
                logger.info("Secondary NLI model loaded")
            except Exception as e:
                logger.warning("Could not load secondary model: %s", e)
                self.secondary_model = None
                MODEL_LOADED.labels(model="nli_secondary").set(0)

        # Single worker thread: the model is not thread-safe under quantization
        # and parallel forward passes on one vCPU only add contention.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nli")
        self._lock = threading.Lock()

        if settings.model_warmup:
            self._warmup()

    # ---------------- lifecycle ----------------

    def _maybe_quantize(self, model: Any) -> Any:
        """Apply INT8 dynamic quantization on CPU (roughly 4x smaller Linear weights)."""
        if not (self.use_quantization and self.device == "cpu"):
            return model
        quantize = getattr(getattr(torch, "ao", None), "quantization", None)
        quantize_fn = getattr(quantize, "quantize_dynamic", None) or getattr(
            torch.quantization, "quantize_dynamic", None
        )
        if quantize_fn is None:
            return model
        try:
            quantized = quantize_fn(model, {torch.nn.Linear}, dtype=torch.qint8)
            logger.info("Applied INT8 dynamic quantization")
            return quantized
        except Exception as e:
            logger.warning("Quantization failed, continuing with fp32: %s", e)
            return model

    def _warmup(self) -> None:
        """First inference pays lazy-init costs; do it at boot, not in a user request."""
        try:
            self._nli_scores_batch([("warmup claim", "warmup evidence")], model="primary")
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning("Model warmup failed (non-fatal): %s", e)

    def close(self) -> None:
        self._executor.shutdown(wait=False)

    # ---------------- public API ----------------

    def verify_claim(
        self,
        claim: str,
        evidence: str,
        run_numerical_check: bool = True,
    ) -> Dict[str, Any]:
        """Verify a single claim against evidence. Blocking — call via a thread."""
        return self.verify_pairs([(claim, evidence)], run_numerical_check)[0]

    def verify_pairs(
        self,
        pairs: Sequence[Tuple[str, str]],
        run_numerical_check: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Verify many (claim, evidence) pairs in as few forward passes as possible.

        Blocking; run it in a worker thread from async code.
        """
        results: List[Optional[Dict[str, Any]]] = [None] * len(pairs)

        # Pairs with no evidence never reach the model.
        scorable: List[Tuple[int, Tuple[str, str]]] = []
        for i, (claim, evidence) in enumerate(pairs):
            if not evidence or not evidence.strip():
                results[i] = VerificationResult(
                    verdict="UNVERIFIABLE",
                    confidence=0.0,
                    scores=dict(_EMPTY_SCORES),
                    explanation="No evidence provided",
                ).to_dict()
            else:
                scorable.append((i, (claim, evidence)))

        if scorable:
            batch_pairs = [p for _, p in scorable]
            primary = self._nli_scores_batch(batch_pairs, model="primary")

            secondary = None
            if self.secondary_model is not None:
                try:
                    secondary = self._nli_scores_batch(batch_pairs, model="secondary")
                except Exception as e:
                    logger.warning("Secondary model failed, using primary only: %s", e)

            for slot, (idx, (claim, evidence)) in enumerate(scorable):
                scores = primary[slot]
                if secondary is not None:
                    s2 = secondary[slot]
                    scores = {k: 0.6 * scores[k] + 0.4 * s2[k] for k in scores}
                results[idx] = self._assemble(claim, evidence, scores, run_numerical_check)

        return [r for r in results if r is not None]

    async def verify_pairs_async(
        self,
        pairs: Sequence[Tuple[str, str]],
        run_numerical_check: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run ``verify_pairs`` off the event loop so the server stays responsive."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self.verify_pairs, list(pairs), run_numerical_check
        )

    async def verify_claims_batch(
        self,
        claims: List[str],
        evidence_per_claim: List[str],
        run_numerical_check: bool = True,
    ) -> List[Dict[str, Any]]:
        """Verify multiple claims, each against its own evidence passage."""
        return await self.verify_pairs_async(
            list(zip(claims, evidence_per_claim)), run_numerical_check
        )

    # ---------------- inference ----------------

    def _nli_scores_batch(
        self,
        pairs: Sequence[Tuple[str, str]],
        model: str = "primary",
    ) -> List[Dict[str, float]]:
        """Score (claim, evidence) pairs in mini-batches. Returns one dict per pair."""
        if model == "primary":
            tokenizer, mdl, label_map = self.tokenizer, self.model, self.label_map
        else:
            tokenizer, mdl, label_map = (
                self.secondary_tokenizer, self.secondary_model, self.secondary_label_map,
            )
        if tokenizer is None or mdl is None:
            raise RuntimeError(f"{model} NLI model is not loaded")

        out: List[Dict[str, float]] = []
        # Serialise forward passes: one shared model, bounded memory.
        with self._lock:
            for start in range(0, len(pairs), self.batch_size):
                chunk = pairs[start:start + self.batch_size]
                # NLI convention is (premise, hypothesis) = (evidence, claim).
                premises = [evidence for _, evidence in chunk]
                hypotheses = [claim for claim, _ in chunk]

                inputs = tokenizer(
                    premises,
                    hypotheses,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                )
                if self.device != "cpu":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.inference_mode():
                    with MODEL_INFERENCE_LATENCY.labels(
                        model=f"nli_{model}", operation="classify"
                    ).time():
                        logits = mdl(**inputs).logits
                        probs = torch.softmax(logits, dim=-1)

                for row in probs:
                    out.append(_scores_from_probs(row.tolist(), label_map))

                # Release intermediate tensors promptly on small instances.
                del inputs, logits, probs

        return out

    def _nli_scores(self, claim: str, evidence: str, model: str = "primary") -> Dict[str, float]:
        """Single-pair scoring (kept for backward compatibility)."""
        return self._nli_scores_batch([(claim, evidence)], model=model)[0]

    # ---------------- verdict assembly ----------------

    def _assemble(
        self,
        claim: str,
        evidence: str,
        scores: Dict[str, float],
        run_numerical_check: bool,
    ) -> Dict[str, Any]:
        verdict_key = max(scores, key=lambda k: scores[k])
        verdict_label = {
            "supported": "SUPPORTED",
            "unverifiable": "UNVERIFIABLE",
            "contradicted": "CONTRADICTED",
        }[verdict_key]
        confidence = scores[verdict_key]

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
                # Hard numeric disagreement overrides a soft NLI verdict.
                if (
                    not num_result.consistent
                    and num_result.claim_value is not None
                    and num_result.evidence_value is not None
                    and verdict_label != "CONTRADICTED"
                ):
                    verdict_label = "CONTRADICTED"
                    confidence = max(confidence, num_result.confidence)
                    scores = {
                        "supported": min(scores["supported"], 0.1),
                        "unverifiable": 0.1,
                        "contradicted": max(scores["contradicted"], 0.8),
                    }
            except Exception as e:
                logger.warning("Numerical check failed: %s", e)

        calibrated = self._calibrate_confidence(confidence, verdict_label, numerical_check)

        return VerificationResult(
            verdict=verdict_label,
            confidence=round(min(1.0, max(0.0, calibrated)), 3),
            scores={k: round(v, 3) for k, v in scores.items()},
            numerical_check=numerical_check,
            explanation=self._build_explanation(verdict_label, scores, numerical_check),
        ).to_dict()

    def _calibrate_confidence(
        self,
        raw_confidence: float,
        verdict: str,
        numerical_check: Optional[Dict[str, Any]],
    ) -> float:
        """Temperature scaling — raw max-probability is systematically overconfident."""
        TEMPERATURE = 1.2
        calibrated = max(0.0, raw_confidence) ** (1.0 / TEMPERATURE)

        if numerical_check:
            if verdict == "CONTRADICTED" and not numerical_check.get("consistent", True):
                calibrated = min(0.99, calibrated + 0.05)
            elif (
                verdict == "SUPPORTED"
                and numerical_check.get("consistent", True)
                and numerical_check.get("claim_value") is not None
            ):
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
                note = numerical_check.get("note") or "no conflicting values"
                parts.append(f"numerical values consistent ({note})")
            else:
                parts.append(f"numerical mismatch detected: {numerical_check.get('note', '')}")
            if numerical_check.get("unit_mismatch"):
                parts.append(f"unit mismatch: {numerical_check['unit_mismatch']}")

        return ". ".join(parts) + "."


# ============== Process-wide singleton ==============

_verifier_instance: Optional[NLIVerifier] = None
_verifier_lock = threading.Lock()


def get_verifier() -> NLIVerifier:
    """
    Return the process-wide verifier, loading it on first use.

    Constructing NLIVerifier downloads and materialises a transformer; doing it
    per request is the single most expensive mistake available here.
    """
    global _verifier_instance
    if _verifier_instance is None:
        with _verifier_lock:
            if _verifier_instance is None:
                _verifier_instance = NLIVerifier()
    return _verifier_instance


def verifier_is_loaded() -> bool:
    return _verifier_instance is not None


def reset_verifier() -> None:
    """Drop the singleton (tests, or to reclaim memory on shutdown)."""
    global _verifier_instance
    with _verifier_lock:
        if _verifier_instance is not None:
            _verifier_instance.close()
        _verifier_instance = None


# ============== Backward-compat top-level function ==============

def verify_claims(claims: List[str], evidence_passages: List[List[str]]) -> List[Dict]:
    """Backward-compatible verification helper."""
    verifier = get_verifier()
    results: List[Dict[str, Any]] = []
    indexed_pairs: List[Tuple[int, Tuple[str, str]]] = []

    for i, (claim, passages) in enumerate(zip(claims, evidence_passages)):
        if not passages:
            results.append({
                "claim": claim,
                "verdict": "UNVERIFIABLE",
                "confidence": 0.0,
                "evidence": "",
                "reason": "No evidence provided",
            })
        else:
            results.append({})  # placeholder, filled below
            indexed_pairs.append((i, (claim, passages[0])))

    if indexed_pairs:
        scored = verifier.verify_pairs([p for _, p in indexed_pairs])
        for (idx, (claim, evidence)), result in zip(indexed_pairs, scored):
            result = dict(result)
            result["claim"] = claim
            result["evidence"] = evidence
            results[idx] = result

    return results
