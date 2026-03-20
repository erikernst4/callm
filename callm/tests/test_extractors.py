"""Unit tests for answer extractors."""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from callm.extractors import (
    VerbalizedConfidenceExtractor,
    SequencePosteriorExtractor,
    IsTruePosteriorExtractor,
    GCPSequencePosteriorExtractor,
    GCPIsTruePosteriorExtractor,
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


class TestSequencePosteriorLogic:
    """Tests for the base sequence posterior logic using IsTruePosteriorExtractor."""

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "</s>"
        return tokenizer

    @patch("callm.extractors.get_tokenizer_for_model")
    def test_token_extraction_offset_mapping(self, mock_get_tokenizer, mock_tokenizer):
        mock_get_tokenizer.return_value = mock_tokenizer

        extractor = SequencePosteriorExtractor(model_name="gpt2")

        text = "Guess: World"
        output_ids = torch.tensor([1, 2, 3])
        logits = torch.randn(3, 100)

        encoding_mock = MagicMock()
        encoding_mock.offset_mapping = [(0, 5), (5, 6), (6, 12)]
        mock_tokenizer.return_value = encoding_mock

        answer, confidence = extractor(text, logits, output_ids)

        mock_tokenizer.assert_called_with(text, return_offsets_mapping=True)
        assert answer == "World"
        assert 0 <= confidence <= 1

    @patch("callm.extractors.get_tokenizer_for_model")
    def test_token_extraction_no_match(self, mock_get_tokenizer, mock_tokenizer):
        mock_get_tokenizer.return_value = mock_tokenizer
        extractor = SequencePosteriorExtractor(model_name="gpt2")

        text = "Hello Universe"
        output_ids = torch.tensor([1, 2])
        logits = torch.randn(2, 100)

        encoding_mock = MagicMock()
        encoding_mock.offset_mapping = [(0, 5), (6, 14)]
        mock_tokenizer.return_value = encoding_mock

        answer, confidence = extractor.forward(text, logits, output_ids)
        assert answer == "Hello Universe"

    @patch("callm.extractors.get_tokenizer_for_model")
    def test_extractor_handles_1d_logits(self, mock_get_tokenizer):
        """Test that extractor handles 1D logits correctly."""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        extractor = SequencePosteriorExtractor(model_name="gpt2")

        text = "Guess: Paris"
        logits_1d = torch.tensor([-0.1, -0.2, -0.5])

        mock_encoding = MagicMock()
        mock_encoding.offset_mapping = [(0, 5), (5, 6), (7, 12)]
        mock_tokenizer.return_value = mock_encoding
        extractor.tokenizer = mock_tokenizer

        answer, confidence = extractor.forward(text, logits_1d, None)
        assert answer == "Paris"
        assert np.abs(confidence - np.exp(-0.5)) < 1e-6

    @patch("callm.extractors.get_tokenizer_for_model")
    def test_extractor_backward_compatibility(self, mock_get_tokenizer):
        """Test that Extractor still works with 2D logits."""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        extractor = SequencePosteriorExtractor(model_name="gpt2")

        logits_2d = torch.randn(3, 100)

        text = "Guess: Paris"
        mock_encoding = MagicMock()
        mock_encoding.offset_mapping = [(0, 5), (5, 6), (7, 12)]
        mock_tokenizer.return_value = mock_encoding
        extractor.tokenizer = mock_tokenizer

        answer, confidence = extractor.forward(text, logits_2d, None)
        assert answer == "Paris"
        assert 0 <= confidence <= 1.0


class TestSequencePosteriorExtractor:
    """Tests for SequencePosteriorExtractor base class."""

    @patch("callm.extractors.get_tokenizer_for_model")
    def test_compute_probability_from_logits_1d(self, mock_get_tokenizer):
        """Test probability computation with 1D logits."""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        extractor = IsTruePosteriorExtractor(model_name="gpt2")

        logits = torch.tensor([-0.5, -1.0, -0.2])
        token_indices = [0, 2]

        prob = extractor.compute_probability_from_logits(logits, token_indices)

        expected = np.exp(-0.7)
        assert np.abs(prob - expected) < 1e-6

    @patch("callm.extractors.get_tokenizer_for_model")
    def test_compute_probability_from_logits_2d(self, mock_get_tokenizer):
        """Test probability computation with 2D logits."""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        extractor = IsTruePosteriorExtractor(model_name="gpt2")

        logits = torch.zeros(3, 10)
        logits[0, 5] = 10.0
        logits[1, 3] = 8.0
        token_indices = [0, 1]

        prob = extractor.compute_probability_from_logits(logits, token_indices)

        assert 0 <= prob <= 1.0

    @patch("callm.extractors.get_tokenizer_for_model")
    def test_compute_probability_empty_indices(self, mock_get_tokenizer):
        """Test probability computation with empty indices."""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        extractor = IsTruePosteriorExtractor(model_name="gpt2")

        logits = torch.tensor([-0.5, -1.0])
        token_indices = []

        prob = extractor.compute_probability_from_logits(logits, token_indices)
        assert np.isnan(prob)


class TestIsTruePosteriorExtractor:
    """Tests for IsTruePosteriorExtractor."""

    @patch("callm.extractors.get_tokenizer_for_model")
    def test_forward_empty_text(self, mock_get_tokenizer):
        """Test forward with empty text returns nan."""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        extractor = IsTruePosteriorExtractor(model_name="gpt2")

        answer, confidence = extractor.forward("", None, None)
        assert answer == ""
        assert np.isnan(confidence)

    @patch("callm.extractors.get_tokenizer_for_model")
    def test_forward_none_logits(self, mock_get_tokenizer):
        """Test forward with None logits returns nan."""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        extractor = IsTruePosteriorExtractor(model_name="gpt2")

        # Mock tokenizer return for extract_answer and get_target_token_indices
        mock_encoding = MagicMock()
        mock_encoding.offset_mapping = [(0, 3)]
        mock_tokenizer.return_value = mock_encoding

        answer, confidence = extractor.forward("A", None, None)
        assert answer == "A"
        assert np.isnan(confidence)

    @patch("callm.extractors.get_tokenizer_for_model")
    def test_forward_choice_a_true(self, mock_get_tokenizer):
        """Test forward when choice is (A) True."""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        extractor = IsTruePosteriorExtractor(model_name="gpt2")

        text = "A True"
        logits = torch.tensor([-0.1])  # log prob for (A)

        mock_encoding = MagicMock()
        # Mocking A at 0:1
        mock_encoding.offset_mapping = [(0, 1)]
        mock_tokenizer.return_value = mock_encoding

        answer, confidence = extractor.forward(text, logits, None)

        assert answer == "A"
        assert np.abs(confidence - np.exp(-0.1)) < 1e-6

    @patch("callm.extractors.get_tokenizer_for_model")
    def test_forward_choice_b_false(self, mock_get_tokenizer):
        """Test forward when choice is (B) False - should flip confidence."""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        extractor = IsTruePosteriorExtractor(model_name="gpt2")

        text = "B False"
        logits = torch.tensor([-0.2])  # log prob for (B)

        mock_encoding = MagicMock()
        # Mocking B at 0:1
        mock_encoding.offset_mapping = [(0, 1)]
        mock_tokenizer.return_value = mock_encoding

        answer, confidence = extractor.forward(text, logits, None)

        # Confidence should NOT be flipped anymore
        expected_seq_posterior = np.exp(-0.2)
        assert answer == "B"
        assert np.abs(confidence - expected_seq_posterior) < 1e-6

    @patch("callm.extractors.get_tokenizer_for_model")
    def test_extract_answer_variants(self, mock_get_tokenizer):
        """Test extracting choice variants."""
        mock_get_tokenizer.return_value = MagicMock()
        extractor = IsTruePosteriorExtractor(model_name="gpt2")

        assert extractor.extract_answer("(A) True") == "A"
        assert extractor.extract_answer("Answer is B.") == "B"
        assert extractor.extract_answer("**A**") == "A"
        assert extractor.extract_answer("The answer is True") == "True"
        assert extractor.extract_answer("It is definitely False") == "False"

    @patch("callm.extractors.get_tokenizer_for_model")
    def test_get_target_token_indices(self, mock_get_tokenizer):
        """Test that get_target_token_indices finds the choice marker."""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        extractor = IsTruePosteriorExtractor(model_name="gpt2")

        text = "(A) True"
        mock_encoding = MagicMock()
        mock_encoding.offset_mapping = [(0, 3), (3, 4), (4, 8)]

        indices = extractor.get_target_token_indices(text, [(0, 3), (3, 4), (4, 8)])
        # Should only get the index for "(A)" which is idx 0
        assert indices == [0]

    def test_inherits_from_sequence_posterior_extractor(self):
        """Test that IsTruePosteriorExtractor inherits from SequencePosteriorExtractor."""
        assert issubclass(IsTruePosteriorExtractor, SequencePosteriorExtractor)


class TestGCPSequencePosteriorExtractor:
    """Tests for GCPSequencePosteriorExtractor."""

    def test_forward_valid(self):
        extractor = GCPSequencePosteriorExtractor()
        logits = torch.tensor([-0.5, -0.2])  # sum is -0.7
        text = "Guess: Paris"

        answer, confidence = extractor(text, logits, None)
        assert answer == "Paris"
        assert np.abs(confidence - np.exp(-0.7)) < 1e-6

    def test_forward_none_logits(self):
        extractor = GCPSequencePosteriorExtractor()
        text = "Guess: Paris"

        answer, confidence = extractor(text, None, None)
        assert answer == "Paris"
        assert np.isnan(confidence)

    def test_forward_empty_text(self):
        extractor = GCPSequencePosteriorExtractor()
        logits = torch.tensor([-0.5])

        answer, confidence = extractor("", logits, None)
        assert answer == ""
        assert np.isnan(confidence)


class TestGCPIsTruePosteriorExtractor:
    """Tests for GCPIsTruePosteriorExtractor."""

    def test_forward_choice_a(self):
        extractor = GCPIsTruePosteriorExtractor()
        logits = torch.tensor([-0.1])
        text = "(A) True"

        answer, confidence = extractor(text, logits, None)
        assert answer == "A"
        assert np.abs(confidence - np.exp(-0.1)) < 1e-6

    def test_extract_answer_variants(self):
        extractor = GCPIsTruePosteriorExtractor()

        assert extractor.extract_answer("(A) True") == "A"
        assert extractor.extract_answer("Answer is B.") == "B"
        assert extractor.extract_answer("**A**") == "A"
        assert extractor.extract_answer("The answer is True") == "True"
        assert extractor.extract_answer("It is definitely False") == "False"

    def test_inheritance(self):
        assert issubclass(GCPIsTruePosteriorExtractor, GCPSequencePosteriorExtractor)


class TestOffsetMappingFallback:
    """Tests for the fallback offset mapping when tokenizer doesn't support return_offsets_mapping."""

    @patch("callm.extractors.get_tokenizer_for_model")
    def test_fallback_triggers_on_value_error(self, mock_get_tokenizer):
        """Test that ValueError from tokenizer triggers fallback to manual offset mapping."""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        # Simulate Mistral's ValueError on return_offsets_mapping
        mock_tokenizer.side_effect = ValueError(
            "`MistralCommonBackend` does not support `return_offsets_mapping`"
        )
        # But decode should work for the fallback
        mock_tokenizer.decode = MagicMock(
            side_effect=lambda ids, **kw: {  # incremental prefix decoding
                1: "Guess",
                2: "Guess:",
                3: "Guess: Paris",
            }[len(ids)]
        )

        extractor = SequencePosteriorExtractor(model_name="mistralai/test")
        extractor.tokenizer = mock_tokenizer

        text = "Guess: Paris"
        logits = torch.tensor([-0.1, -0.2, -0.5])
        output_ids = torch.tensor([101, 102, 103])

        answer, confidence = extractor.forward(text, logits, output_ids)

        assert answer == "Paris"
        # Token index 2 maps to "Paris" (chars 7-12), so confidence = exp(-0.5)
        assert np.abs(confidence - np.exp(-0.5)) < 1e-6

    @patch("callm.extractors.get_tokenizer_for_model")
    def test_fallback_none_output_ids_returns_nan(self, mock_get_tokenizer):
        """Test that fallback with None output_ids returns nan confidence."""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        mock_tokenizer.side_effect = ValueError("no offset mapping")

        extractor = SequencePosteriorExtractor(model_name="mistralai/test")
        extractor.tokenizer = mock_tokenizer

        text = "Guess: Paris"
        logits = torch.tensor([-0.1])

        answer, confidence = extractor.forward(text, logits, None)
        assert answer == "Paris"
        assert np.isnan(confidence)

    @patch("callm.extractors.get_tokenizer_for_model")
    def test_build_offset_mapping_special_tokens(self, mock_get_tokenizer):
        """Test that special tokens producing no text get (0, 0) offsets."""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        # Token 0 is a special token that produces no text
        # Token 1 produces "Hello"
        mock_tokenizer.decode = MagicMock(
            side_effect=lambda ids, **kw: {1: "", 2: "Hello"}[len(ids)]
        )

        extractor = SequencePosteriorExtractor(model_name="gpt2")
        extractor.tokenizer = mock_tokenizer

        offsets = extractor._build_offset_mapping_from_ids(torch.tensor([999, 101]))
        assert offsets == [(0, 0), (0, 5)]

    @patch("callm.extractors.get_tokenizer_for_model")
    def test_build_offset_mapping_whitespace(self, mock_get_tokenizer):
        """Test that whitespace between tokens is handled correctly."""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        # BPE-style: token 2 includes a leading space
        mock_tokenizer.decode = MagicMock(
            side_effect=lambda ids, **kw: {
                1: "Hello",
                2: "Hello World",
            }[len(ids)]
        )

        extractor = SequencePosteriorExtractor(model_name="gpt2")
        extractor.tokenizer = mock_tokenizer

        offsets = extractor._build_offset_mapping_from_ids(torch.tensor([10, 20]))
        # "Hello" spans chars 0-5, " World" spans chars 5-11
        assert offsets == [(0, 5), (5, 11)]
