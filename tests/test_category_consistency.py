"""
Regression tests for the categorical consistency check (decision.py rule 4).

Context: the margin rule (tests/test_decision.py::TestMarginRule) fixed cases
where NLI produced a flat, unopinionated distribution and argmax mistook that
for an assertion. It does NOT fix a model that is confidently, incorrectly
opinionated â€” and the diagnostic run against the real checkpoint showed
exactly that:

    premise:    "Metformin is a medication for type 2 diabetes."
    hypothesis: "Metformin is a person."
    => contradiction 0.0319, entailment 0.8014, neutral 0.1667

0.8014 clears every margin-rule threshold. No threshold adjustment fixes a
model that is wrong with high confidence about this specific input; that is
what category_mismatch_verdict exists for. It is a symbolic, syntax-driven
check â€” not a lexicon of approved subjects or categories â€” so it generalizes
to any "X is a/an NOUN" claim, not just this one.

All tests here run against pure Python with no model and no network.
"""
import pytest

from api.services.decision import (
    DecisionPolicy,
    PremiseVerdict,
    aggregate,
    category_mismatch_verdict,
    extract_copula,
)

POLICY = DecisionPolicy()
SOURCE_MED = "Metformin is a medication for type 2 diabetes."
SOURCE_FULL = (
    "Metformin is a medication for type 2 diabetes. "
    "Take metformin with meals to reduce stomach upset. "
    "Do not take on an empty stomach."
)


class TestExactReportedBug:
    """Reproduces the QA report's exact numbers, not a re-description of it."""

    def test_category_check_fires_on_the_reported_pair(self):
        verdict = category_mismatch_verdict(
            "Metformin is a person.", [SOURCE_FULL], POLICY
        )
        assert verdict is not None
        assert verdict.verdict == "CONTRADICTED"

    def test_final_aggregation_overrides_the_reported_nli_output(self):
        # The exact scores from the live /verify response in the bug report:
        # verdict_scores = {supported: 0.937, unverifiable: 0.056, contradicted: 0.007}
        reported_nli_verdict = PremiseVerdict(
            premise=SOURCE_FULL,
            verdict="SUPPORTED",
            confidence=0.937,
            scores={"supported": 0.937, "unverifiable": 0.056, "contradicted": 0.007},
            reason="raw NLI (as reported)",
        )
        category_verdict = category_mismatch_verdict(
            "Metformin is a person.", [SOURCE_FULL], POLICY
        )
        assert category_verdict is not None

        final = aggregate([reported_nli_verdict, category_verdict], POLICY)
        assert final.verdict == "CONTRADICTED"

    def test_ungrammatical_variant_was_already_correct(self):
        # "Metformin is person." (no article) scored contradiction=0.681 raw â€”
        # the margin rule alone already handles this one. Included so the two
        # near-duplicate inputs from the bug report are both covered.
        verdict = category_mismatch_verdict(
            "Metformin is person.", [SOURCE_MED], POLICY
        )
        assert verdict is not None
        assert verdict.verdict == "CONTRADICTED"


class TestGenericCategoryMismatch:
    """Same mechanism, different subjects â€” proves this isn't a Metformin rule."""

    @pytest.mark.parametrize(
        "claim,evidence",
        [
            ("Aspirin is a fruit.", "Aspirin is a medication used to relieve pain."),
            ("Python is a person.", "Python is a programming language."),
            ("Paris is a programming language.", "Paris is the capital of France."),
            ("A dog is a chemical element.", "A dog is a domesticated mammal."),
            ("Metformin is a fruit.", SOURCE_MED),
        ],
    )
    def test_mismatch_detected(self, claim, evidence):
        verdict = category_mismatch_verdict(claim, [evidence], POLICY)
        assert verdict is not None, f"expected a mismatch for {claim!r} vs {evidence!r}"
        assert verdict.verdict == "CONTRADICTED"


class TestPositiveControls:
    """The check must never suppress or interfere with genuine support."""

    @pytest.mark.parametrize(
        "claim,evidence",
        [
            ("Metformin is a medication.", SOURCE_MED),
            ("Aspirin is a medication.", "Aspirin is a medication used to relieve pain."),
            ("Python is a programming language.",
             "Python is a programming language used for many purposes."),
        ],
    )
    def test_compatible_category_does_not_override(self, claim, evidence):
        assert category_mismatch_verdict(claim, [evidence], POLICY) is None


class TestUnverifiableControls:
    """No comparable evidence-side assertion => abstain, not contradict."""

    def test_no_matching_subject_in_evidence(self):
        # Evidence never mentions Metformin at all; nothing to compare against.
        # (Separately, the grounding gate in decision.py handles demoting this
        # to UNVERIFIABLE â€” this check's job is only to not manufacture a
        # contradiction it has no basis for.)
        assert category_mismatch_verdict(
            "Metformin is a medication.", ["Aspirin relieves headaches."], POLICY
        ) is None

    def test_evidence_not_copular_at_all(self):
        # "Common side effects include nausea..." asserts nothing via "is/are".
        assert category_mismatch_verdict(
            "Nausea is a common side effect.",
            ["Common side effects include nausea, diarrhea, and stomach pain."],
            POLICY,
        ) is None


class TestParaphraseSafety:
    """Claims outside the bare-copula pattern must fall through untouched."""

    @pytest.mark.parametrize(
        "claim",
        [
            "Metformin is used to treat type 2 diabetes.",   # verb-phrase predicate
            "The medication is called Metformin.",            # passive, reversed identity
            "Metformin should be taken with meals.",           # modal + passive, no "is/are"
        ],
    )
    def test_non_bare_copula_claims_are_not_parsed(self, claim):
        assert extract_copula(claim) is None
        assert category_mismatch_verdict(claim, [SOURCE_FULL], POLICY) is None


class TestExtractCopulaScope:
    """The parser's boundaries, stated as executable tests, not just prose."""

    def test_extracts_simple_assertion(self):
        assert extract_copula(SOURCE_MED) == ("metformin", "medication")

    def test_strips_trailing_prepositional_phrase(self):
        subject, predicate = extract_copula("Paris is the capital of France.")
        assert (subject, predicate) == ("paris", "capital")

    def test_negation_is_not_an_assertion(self):
        assert extract_copula("Metformin is not a fruit.") is None

    def test_numeric_predicate_is_left_to_the_numerical_checker(self):
        assert extract_copula("The dose is 500 mg.") is None

    def test_long_unbounded_predicate_is_not_guessed(self):
        # No boundary word appears, so nothing gets truncated â€” 8 predicate
        # tokens exceeds the cap, and the parser declines to guess rather
        # than accepting an unbounded, likely-mis-scoped noun phrase.
        long_claim = (
            "Metformin is a widely available generally affordable "
            "common everyday medication"
        )
        assert extract_copula(long_claim) is None

    def test_boundary_word_mid_predicate_truncates_correctly(self):
        # A boundary word partway through the predicate is exactly the
        # common case ("a medication for diabetes") â€” truncate, don't reject.
        assert extract_copula(
            "Metformin is a widely prescribed oral medication"
        ) == ("metformin", "widely")

