"""Unit tests for answer extractors."""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from callm.extractors import (
    VerbalizedConfidenceExtractor,
    SequencePosteriorConfidenceExtractor,
)


class TestVerbalizedConfidenceExtractor:
    """Tests for VerbalizedConfidenceExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = VerbalizedConfidenceExtractor()

    def test_perfect_format(self):
        """Test extraction with perfect format."""
        text = """Guess: Paris
Probability: 0.95"""
        answer, confidence = self.extractor(text)
        assert answer == "Paris"
        assert confidence == 0.95

    def test_case_insensitive(self):
        """Test case-insensitive extraction."""
        text = """guess: London
probability: 0.8"""
        answer, confidence = self.extractor(text)
        assert answer == "London"
        assert confidence == 0.8

    def test_extra_text(self):
        """Test with extra text after the answer."""
        text = """Guess: New York
Probability: 0.75
Here is my reasoning..."""
        answer, confidence = self.extractor(text)
        assert answer == "New York"
        assert confidence == 0.75

    def test_missing_probability(self):
        """Test when probability is missing."""
        text = """Guess: Berlin"""
        answer, confidence = self.extractor(text)
        assert answer == "Berlin"
        assert np.isnan(confidence)

    def test_missing_guess(self):
        """Test when guess is missing."""
        text = """Probability: 0.9"""
        answer, confidence = self.extractor(text)
        # Should use first line as fallback
        assert "Probability" in answer
        assert confidence == 0.9

    def test_probability_out_of_range_high(self):
        """Test probability clipping when > 1.0."""
        text = """Guess: Tokyo
Probability: 1.5"""
        answer, confidence = self.extractor(text)
        assert answer == "Tokyo"
        assert np.isnan(confidence)  # Invalid range -> NaN

    def test_probability_out_of_range_low(self):
        """Test probability clipping when < 0.0."""
        text = """Guess: Rome
Probability: -0.2"""
        answer, confidence = self.extractor(text)
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
            _, confidence = self.extractor(text)
            assert confidence == pytest.approx(expected_conf)

    def test_empty_string(self):
        """Test extraction from empty string."""
        answer, confidence = self.extractor("")
        assert answer == ""
        assert np.isnan(confidence)  # Should be NaN

    def test_invalid_probability_format(self):
        """Test when probability is not a valid number."""
        text = """Guess: Paris
Probability: not_a_number"""
        answer, confidence = self.extractor(text)
        assert answer == "Paris"
        assert np.isnan(confidence)  # Invalid format should return NaN

    def test_multiline_answer(self):
        """Test when answer might span multiple descriptions."""
        text = """Guess: The United States of America
Probability: 0.88"""
        answer, confidence = self.extractor(text)
        assert answer == "The United States of America"
        assert confidence == 0.88


class TestSequencePosteriorConfidenceExtractor:
    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "</s>"
        return tokenizer

    @patch("callm.extractors.get_tokenizer_for_model")
    def test_token_extraction_offset_mapping(self, mock_get_tokenizer, mock_tokenizer):
        mock_get_tokenizer.return_value = mock_tokenizer

        # Setup extractor
        extractor = SequencePosteriorConfidenceExtractor(model_name="gpt2")
        # extractor.device is a property in LightningModule, cannot set it directly.

        # Test Case 1: "Guess: World", answer="World"
        # Layout: "Guess" (0,5), ":" (5,6), " World" (6,12)
        # Answer "World" is at index 7 in "Guess: World" (0-indexed? No wait.)
        # Text: "Guess: World"
        # 012345678901
        # G u e s s :   W o r l d
        # Answer "World" starts at 7.
        # Token " World" (including space): matches " World".
        # Offsets likely: (0,5) "Guess", (5,6) ":", (6,12) " World".
        # (Assuming space is part of World token in GPT2 style)

        text = "Guess: World"
        # output_ids/logits don't matter much for indices logic check
        output_ids = torch.tensor([1, 2, 3])
        logits = torch.randn(3, 100)

        # Mock encoding
        encoding_mock = MagicMock()
        encoding_mock.offset_mapping = [(0, 5), (5, 6), (6, 12)]
        mock_tokenizer.return_value = encoding_mock

        # Run forward
        # Only test first return value logic for finding answer tokens
        # We expect confidence to be nan because we didn't mock enough for prob calc yet
        answer, confidence = extractor(text, logits, output_ids)

        # Check that we called tokenizer with return_offsets_mapping=True
        mock_tokenizer.assert_called_with(text, return_offsets_mapping=True)

        assert answer == "World"
        assert np.isnan(confidence)

    @patch("callm.extractors.get_tokenizer_for_model")
    def test_token_extraction_no_match(self, mock_get_tokenizer, mock_tokenizer):
        mock_get_tokenizer.return_value = mock_tokenizer
        extractor = SequencePosteriorConfidenceExtractor(model_name="gpt2")

        text = "Hello Universe"
        output_ids = torch.tensor([1, 2])
        logits = torch.randn(2, 100)

        # Answer "World" not in text
        # But extractor uses self.extract_answer(text) which usually falls back to first line
        # "Hello Universe" -> "Hello Universe"
        # So it will look for "Hello Universe" tokens.

        encoding_mock = MagicMock()
        encoding_mock.offset_mapping = [(0, 5), (6, 14)]
        mock_tokenizer.return_value = encoding_mock

        answer, confidence = extractor.forward(text, logits, output_ids)
        assert answer == "Hello Universe"  # First line fallback
