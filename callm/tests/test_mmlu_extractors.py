"""Unit tests for MMLU extractors."""

import numpy as np
import torch
from unittest.mock import patch, MagicMock
from callm.extractors.mmlu import (
    MMLUBaseExtractor,
    MMLUVerbalizedExtractor,
    MMLUSequencePosteriorExtractor,
    GCPMMLUSequencePosteriorExtractor,
)


class TestMMLUBaseExtractor:
    def setup_method(self):
        self.extractor = MMLUBaseExtractor()

    def test_extract_answer_standard(self):
        assert self.extractor.extract_answer("Answer: B") == "B"
        assert self.extractor.extract_answer("answer: c") == "C"
        assert self.extractor.extract_answer("Answer: a\nProbability: 0.9") == "A"

    def test_extract_answer_standalone_start(self):
        assert self.extractor.extract_answer("D") == "D"
        assert self.extractor.extract_answer(" c ") == "C"
        assert self.extractor.extract_answer("A. because it is...") == "A"

    def test_extract_answer_fallback(self):
        assert self.extractor.extract_answer("The answer is B") == "B"
        assert self.extractor.extract_answer("I think D is correct") == "D"

    def test_extract_answer_invalid(self):
        assert self.extractor.extract_answer("Answer: E") == ""
        assert self.extractor.extract_answer("The elephant is big") == ""


class TestMMLUVerbalizedExtractor:
    def setup_method(self):
        self.extractor = MMLUVerbalizedExtractor()

    def test_perfect_format(self):
        text = "Answer: C\nProbability: 0.85"
        answer, conf = self.extractor(text)
        assert answer == "C"
        assert conf == 0.85

    def test_case_insensitive(self):
        text = "answer: b\nprobability: .5"
        answer, conf = self.extractor(text)
        assert answer == "B"
        assert conf == 0.5

    def test_missing_probability(self):
        text = "Answer: A"
        answer, conf = self.extractor(text)
        assert answer == "A"
        assert np.isnan(conf)

    def test_out_of_range_probability(self):
        text = "Answer: D\nProbability: 1.5"
        answer, conf = self.extractor(text)
        assert answer == "D"
        assert np.isnan(conf)


class TestMMLUSequencePosteriorExtractor:
    @patch("callm.extractors.base.get_tokenizer_for_model")
    def test_extractor_handles_1d_logits(self, mock_get_tokenizer):
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        extractor = MMLUSequencePosteriorExtractor(model_name="gpt2")

        text = "Answer: B"
        logits_1d = torch.tensor([-0.1, -0.2, -0.5])  # Dummy logits

        mock_encoding = MagicMock()
        # Mocking "B" token at char index 8:9
        mock_encoding.offset_mapping = [(0, 6), (6, 7), (8, 9)]
        mock_tokenizer.return_value = mock_encoding
        extractor.tokenizer = mock_tokenizer

        answer, confidence = extractor.forward(text, logits_1d, None)
        assert answer == "B"
        # token index 2 corresponds to "B", logit is -0.5
        assert np.abs(confidence - np.exp(-0.5)) < 1e-6


class TestGCPMMLUSequencePosteriorExtractor:
    def test_forward_valid(self):
        extractor = GCPMMLUSequencePosteriorExtractor()
        logits = torch.tensor([-0.2])  # log prob for the "B" token
        text = "Answer: B"

        answer, confidence = extractor(text, logits, None)
        assert answer == "B"
        assert np.abs(confidence - np.exp(-0.2)) < 1e-6
