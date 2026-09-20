"""
Unit tests for numerical verification service.
"""
import pytest
from api.services.numerical import (
    parse_number,
    extract_numbers,
    extract_dates,
    check_numerical_consistency,
    detect_unit_mismatch,
)


class TestParseNumber:
    """Tests for number parsing."""

    def test_simple_integer(self):
        assert parse_number("42") == 42.0

    def test_decimal(self):
        assert parse_number("3.14") == 3.14

    def test_comma_separated(self):
        assert parse_number("1,234") == 1234.0
        assert parse_number("1,234,567") == 1234567.0

    def test_with_percent(self):
        # 10% parsed as plain number, % is marker
        result = parse_number("10%")
        assert result == 10.0

    def test_with_million(self):
        result = parse_number("1.5 million")
        assert result == 1_500_000.0

    def test_with_billion(self):
        result = parse_number("2.3 billion")
        assert result == 2_300_000_000.0

    def test_invalid(self):
        assert parse_number("not a number") is None
        assert parse_number("") is None


class TestExtractNumbers:
    """Tests for number extraction from text."""

    def test_extracts_multiple_numbers(self):
        text = "10% of 1000 patients had 5 events."
        numbers = extract_numbers(text)
        assert len(numbers) >= 2
        values = [n["value"] for n in numbers]
        assert 10.0 in values
        assert 1000.0 in values

    def test_returns_context(self):
        text = "The study enrolled 500 participants in 2020."
        numbers = extract_numbers(text)
        assert len(numbers) >= 1
        for n in numbers:
            assert "context" in n
            assert "value" in n


class TestExtractDates:
    """Tests for date extraction."""

    def test_extracts_iso_dates(self):
        text = "The event was on 2023-12-25 and ended 2024-01-01."
        dates = extract_dates(text)
        assert len(dates) >= 1

    def test_extracts_month_names(self):
        text = "Released on January 15, 2024 and updated February 1, 2024."
        dates = extract_dates(text)
        assert len(dates) >= 1


class TestNumericalCheck:
    """Tests for numerical consistency check."""

    def test_no_numbers_consistent(self):
        result = check_numerical_consistency(
            "The sky is blue.",
            "The sky appears blue due to Rayleigh scattering."
        )
        assert result.consistent is True

    def test_matching_numbers_consistent(self):
        result = check_numerical_consistency(
            "10% of patients experience nausea.",
            "About 10 percent of patients report nausea.",
        )
        assert result.consistent is True
        assert result.claim_value == 10.0

    def test_mismatched_numbers_inconsistent(self):
        result = check_numerical_consistency(
            "Lactic acidosis occurs in 10% of patients.",
            "Lactic acidosis is rare, occurring in 1 in 30,000 patient-years.",
        )
        # 10% vs 0.003% â€” large mismatch
        assert result.consistent is False


class TestUnitMismatch:
    """Tests for unit mismatch detection."""

    def test_detects_percentage_to_ratio(self):
        result = detect_unit_mismatch(
            "10% of patients",
            "1 in 30,000",
        )
        assert result is not None
        assert "percentage" in result.lower()

    def test_no_mismatch_for_same_units(self):
        result = detect_unit_mismatch(
            "10% of patients",
            "10 percent of patients",
        )
        assert result is None

