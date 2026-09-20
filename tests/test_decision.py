"""
Regression tests for the verdict decision layer.

These are the accuracy guardrails. They need no model and no network: they
exercise the logic that turns NLI probabilities into a verdict, which is where
the "similarity treated as entailment" bug lived.

The model-dependent half of the story (are the probabilities themselves
sensible?) is covered by scripts/diagnose_nli.py, which needs torch.
"""
import pytest

from api.services.decision import (
    DecisionPolicy,
    PremiseVerdict,
    aggregate,
    classify_scores,
    entity_terms,
    grounding_gap,
    split_premises,
)
from api.services.numerical import check_numerical_consistency, extract_quantities

POLICY = DecisionPolicy()
SOURCE_MED = "Metformin is a medication for type 2 diabetes."


def verdict(entail, neutral, contra, policy=POLICY):
    return classify_scores(
        {"supported": entail, "unverifiable": neutral, "contradicted": contra}, policy
    )[0]


class TestMarginRule:
    """argmax over three classes is not a decision procedure."""

    def test_confident_entailment_is_supported(self):
        assert verdict(0.91, 0.05, 0.04) == "SUPPORTED"

    def test_confident_contradiction(self):
        assert verdict(0.05, 0.15, 0.80) == "CONTRADICTED"

    def test_narrow_entailment_win_abstains(self):
        # argmax says SUPPORTED. The model is not asserting anything.
        assert verdict(0.38, 0.36, 0.26) == "UNVERIFIABLE"

    def test_entailment_below_floor_abstains(self):
        assert verdict(0.52, 0.30, 0.18) == "UNVERIFIABLE"

    def test_three_way_tie_abstains(self):
        assert verdict(0.34, 0.33, 0.33) == "UNVERIFIABLE"

    def test_unverifiable_confidence_is_the_neutral_probability(self):
        _, confidence, _ = classify_scores(
            {"supported": 0.38, "unverifiable": 0.36, "contradicted": 0.26}, POLICY
        )
        # Not the losing class's score â€” an abstention reports its own certainty.
        assert confidence == pytest.approx(0.36)

    def test_reason_is_reported(self):
        _, _, reason = classify_scores(
            {"supported": 0.9, "unverifiable": 0.05, "contradicted": 0.05}, POLICY
        )
        assert "margin" in reason or "leads" in reason


class TestGrounding:
    """A passage about a different entity is not evidence, however similar."""

    def test_same_entity_is_grounded(self):
        assert grounding_gap("Metformin is a person.", SOURCE_MED) == []

    def test_different_entity_is_not_grounded(self):
        assert grounding_gap("Metformin is a medication.", "Aspirin is a medication.") == [
            "metformin"
        ]

    def test_unstated_number_is_not_grounded(self):
        assert grounding_gap("Metformin costs $500.", SOURCE_MED) == ["500"]

    def test_reworded_claim_stays_grounded(self):
        # Content words must NOT be required â€” that would be keyword matching
        # and would reject valid paraphrases.
        assert grounding_gap("The medication is called Metformin.", SOURCE_MED) == []

    def test_claims_without_entities_are_grounded(self):
        assert grounding_gap(
            "Nausea is a common side effect.",
            "Common side effects include nausea, diarrhea, and stomach pain.",
        ) == []

    def test_sentence_initial_function_words_are_not_entities(self):
        assert "the" not in entity_terms("The medication is called Metformin.")
        assert "metformin" in entity_terms("The medication is called Metformin.")


class TestPremiseSplitting:
    def test_passage_and_sentences_both_scored(self):
        units = split_premises(
            ["Take metformin with meals. Do not take on an empty stomach."]
        )
        assert len(units) == 3
        assert any(u.startswith("Do not take") for u in units)

    def test_single_sentence_not_duplicated(self):
        assert len(split_premises(["One sentence only here."])) == 1

    def test_respects_cap(self):
        assert len(split_premises(["Alpha beta gamma. " * 40], max_units=5)) <= 5

    def test_empty_input(self):
        assert split_premises([]) == []
        assert split_premises(["", "   "]) == []


class TestAggregation:
    @staticmethod
    def pv(verdict_, confidence):
        return PremiseVerdict(
            "premise", verdict_, confidence,
            {"supported": 0.1, "unverifiable": 0.1, "contradicted": confidence}, "",
        )

    def test_contradiction_beats_support(self):
        result = aggregate([self.pv("SUPPORTED", 0.9), self.pv("CONTRADICTED", 0.7)], POLICY)
        assert result.verdict == "CONTRADICTED"

    def test_support_when_nothing_contradicts(self):
        result = aggregate([self.pv("SUPPORTED", 0.9), self.pv("UNVERIFIABLE", 0.4)], POLICY)
        assert result.verdict == "SUPPORTED"

    def test_all_neutral(self):
        assert aggregate([self.pv("UNVERIFIABLE", 0.4)], POLICY).verdict == "UNVERIFIABLE"

    def test_empty(self):
        assert aggregate([], POLICY) is None


class TestNumericalComparability:
    """Unrelated numbers must not manufacture contradictions."""

    def test_category_label_is_not_a_quantity(self):
        assert extract_quantities("Metformin is a medication for type 2 diabetes.") == []

    def test_unstated_price_is_not_a_contradiction(self):
        result = check_numerical_consistency("Metformin costs $500.", SOURCE_MED)
        assert result.status == "not_comparable"
        assert result.evidence_value is None  # so no verdict override can fire

    def test_percent_vs_ratio_mismatch_detected(self):
        result = check_numerical_consistency(
            "Lactic acidosis occurs in 10% of patients.",
            "Lactic acidosis is rare, occurring in approximately 1 in 30,000 patient-years.",
        )
        assert result.status == "mismatch"
        assert result.consistent is False

    def test_matching_percentages(self):
        result = check_numerical_consistency(
            "10% of patients experience nausea.", "About 10 percent of patients report nausea."
        )
        assert result.status == "match"

    def test_mismatched_doses(self):
        assert check_numerical_consistency(
            "The dose is 850 mg.", "The recommended dose is 500 mg twice daily."
        ).status == "mismatch"

    def test_currency_and_plain_numbers_do_not_mix(self):
        result = check_numerical_consistency("It costs $500.", "The dose is 500 mg.")
        assert result.status == "not_comparable"

