"""Unit tests for answer extractors."""

import pytest
import numpy as np
from callm.extractors import VerbalizedConfidenceExtractor


class TestVerbalizedConfidenceExtractor:
    """Tests for VerbalizedConfidenceExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = VerbalizedConfidenceExtractor()

    def test_perfect_format(self):
        """Test extraction with perfect format."""
        text = """Guess: Paris
Probability: 0.95"""
        answer, confidence = self.extractor.extract(text)
        assert answer == "Paris"
        assert confidence == 0.95

    def test_case_insensitive(self):
        """Test case-insensitive extraction."""
        text = """guess: London
probability: 0.8"""
        answer, confidence = self.extractor.extract(text)
        assert answer == "London"
        assert confidence == 0.8

    def test_extra_text(self):
        """Test with extra text after the answer."""
        text = """Guess: New York
Probability: 0.75
Here is my reasoning..."""
        answer, confidence = self.extractor.extract(text)
        assert answer == "New York"
        assert confidence == 0.75

    def test_missing_probability(self):
        """Test when probability is missing."""
        text = """Guess: Berlin"""
        answer, confidence = self.extractor.extract(text)
        assert answer == "Berlin"
        assert np.isnan(confidence)

    def test_missing_guess(self):
        """Test when guess is missing."""
        text = """Probability: 0.9"""
        answer, confidence = self.extractor.extract(text)
        # Should use first line as fallback
        assert "Probability" in answer
        assert confidence == 0.9

    def test_probability_out_of_range_high(self):
        """Test probability clipping when > 1.0."""
        text = """Guess: Tokyo
Probability: 1.5"""
        answer, confidence = self.extractor.extract(text)
        assert answer == "Tokyo"
        assert np.isnan(confidence)  # Invalid range -> NaN

    def test_probability_out_of_range_low(self):
        """Test probability clipping when < 0.0."""
        text = """Guess: Rome
Probability: -0.2"""
        answer, confidence = self.extractor.extract(text)
        assert answer == "Rome"
        assert np.isnan(confidence)  # Invalid range -> NaN

    def test_decimal_probability(self):
        """Test various decimal formats."""
        test_cases = [
            ("Guess: A\nProbability: 0.123", 0.123),
            ("Guess: B\nProbability: .5", 0.5),
            ("Guess: C\nProbability: 1", 1.0),
        ]
        for text, expected_conf in test_cases:
            _, confidence = self.extractor.extract(text)
            assert confidence == pytest.approx(expected_conf)

    def test_empty_string(self):
        """Test extraction from empty string."""
        answer, confidence = self.extractor.extract("")
        assert answer == ""
        assert np.isnan(confidence)  # Should be NaN

    def test_invalid_probability_format(self):
        """Test when probability is not a valid number."""
        text = """Guess: Paris
Probability: not_a_number"""
        answer, confidence = self.extractor.extract(text)
        assert answer == "Paris"
        assert np.isnan(confidence)  # Invalid format should return NaN

    def test_multiline_answer(self):
        """Test when answer might span multiple descriptions."""
        text = """Guess: The United States of America
Probability: 0.88"""
        answer, confidence = self.extractor.extract(text)
        assert answer == "The United States of America"
        assert confidence == 0.88
