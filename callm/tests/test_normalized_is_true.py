"""Tests for normalized IsTrue confidences: p(answer) / (p(True) + p(False))."""

import math

import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock

from callm.extractors import (
    IsTrueExtractor,
    IsTruePosteriorExtractor,
    GCPIsTruePosteriorExtractor,
    SequencePosteriorExtractor,
    GCPSequencePosteriorExtractor,
)

LOG = math.log


class TestNormalizedAnswerPosterior:
    def test_true_answer(self):
        top = [{"True": LOG(0.6), "False": LOG(0.2), "Yes": LOG(0.1)}]
        assert IsTrueExtractor.normalized_posterior("True", top) == pytest.approx(0.75)

    def test_false_answer(self):
        top = [{"False": LOG(0.6), "True": LOG(0.2), "No": LOG(0.1)}]
        assert IsTrueExtractor.normalized_posterior("False", top) == pytest.approx(0.75)

    def test_case_and_space_variants_are_summed(self):
        top = [
            {"True": LOG(0.5), " True": LOG(0.1), "true": LOG(0.1), "False": LOG(0.3)}
        ]
        assert IsTrueExtractor.normalized_posterior("True", top) == pytest.approx(0.7)

    def test_missing_counterpart_gives_one(self):
        top = [{"True": LOG(0.9), "Yes": LOG(0.05)}]
        assert IsTrueExtractor.normalized_posterior("True", top) == pytest.approx(1.0)

    def test_ab_labels(self):
        top = [{"B": LOG(0.3), "A": LOG(0.1)}]
        assert IsTrueExtractor.normalized_posterior("B", top) == pytest.approx(0.75)

    def test_leading_whitespace_step_is_skipped(self):
        top = [{" ": LOG(0.9), "\n": LOG(0.1)}, {"False": LOG(0.4), "True": LOG(0.4)}]
        assert IsTrueExtractor.normalized_posterior("False", top) == pytest.approx(0.5)

    def test_rambling_first_token_is_nan(self):
        top = [{"Let": LOG(0.9), "True": LOG(0.05)}, {"A": LOG(0.9)}]
        assert np.isnan(IsTrueExtractor.normalized_posterior("A", top))

    def test_first_token_other_label_class_is_nan(self):
        # top-1 is "True" but the extracted answer is "False": inconsistent
        top = [{"True": LOG(0.6), "False": LOG(0.3)}]
        assert np.isnan(IsTrueExtractor.normalized_posterior("False", top))

    def test_tie_with_other_token_listed_first(self):
        # bf16 ties: greedy picked "True" but topk listed "False" first
        top = [{"False": LOG(0.4), "True": LOG(0.4), "<box>": LOG(0.19)}]
        assert IsTrueExtractor.normalized_posterior("True", top) == pytest.approx(0.5)
        top = [{"<box>": LOG(0.5), "True": LOG(0.5), "False": LOG(0.0002)}]
        assert IsTrueExtractor.normalized_posterior("True", top) == pytest.approx(
            0.5 / 0.5002
        )

    def test_unknown_answer_is_nan(self):
        assert np.isnan(IsTrueExtractor.normalized_posterior("Paris", [{"Paris": 0.0}]))

    def test_none_or_empty_is_nan(self):
        assert np.isnan(IsTrueExtractor.normalized_posterior("True", None))
        assert np.isnan(IsTrueExtractor.normalized_posterior("True", []))


@patch("callm.extractors.base.get_tokenizer_for_model")
class TestIsTruePosteriorExtractorNormalize:
    def _extractor(self, mock_get_tokenizer, normalize):
        mock_tokenizer = MagicMock()
        encoding = MagicMock()
        encoding.offset_mapping = [(0, 4)]
        mock_tokenizer.return_value = encoding
        mock_get_tokenizer.return_value = mock_tokenizer
        return IsTruePosteriorExtractor(model_name="gpt2", normalize=normalize)

    def test_default_is_not_normalized(self, mock_get_tokenizer):
        extractor = self._extractor(mock_get_tokenizer, normalize=False)
        top = [{"True": LOG(0.6), "False": LOG(0.2)}]
        answer, conf = extractor(
            "True", torch.tensor([LOG(0.6)]), None, top_logprobs=top
        )
        assert answer == "True"
        assert conf == pytest.approx(0.6)

    def test_normalized(self, mock_get_tokenizer):
        extractor = self._extractor(mock_get_tokenizer, normalize=True)
        top = [{"True": LOG(0.6), "False": LOG(0.2)}]
        answer, conf = extractor(
            "True", torch.tensor([LOG(0.6)]), None, top_logprobs=top
        )
        assert answer == "True"
        assert conf == pytest.approx(0.75)

    def test_normalized_without_top_logprobs_is_nan(self, mock_get_tokenizer):
        extractor = self._extractor(mock_get_tokenizer, normalize=True)
        answer, conf = extractor("True", torch.tensor([LOG(0.6)]), None)
        assert answer == "True"
        assert np.isnan(conf)


class TestGCPIsTruePosteriorExtractorNormalize:
    def test_default_is_not_normalized(self):
        extractor = GCPIsTruePosteriorExtractor()
        top = [{"False": LOG(0.6), "True": LOG(0.2)}]
        answer, conf = extractor(
            "False", torch.tensor([LOG(0.6)]), None, top_logprobs=top
        )
        assert answer == "False"
        assert conf == pytest.approx(0.6)

    def test_normalized(self):
        extractor = GCPIsTruePosteriorExtractor(normalize=True)
        top = [{"False": LOG(0.6), "True": LOG(0.2)}]
        answer, conf = extractor(
            "False", torch.tensor([LOG(0.6)]), None, top_logprobs=top
        )
        assert answer == "False"
        assert conf == pytest.approx(0.75)

    def test_normalized_empty_text(self):
        extractor = GCPIsTruePosteriorExtractor(normalize=True)
        answer, conf = extractor("", None, None, top_logprobs=None)
        assert answer == ""
        assert np.isnan(conf)


class TestIsTrueExtractorBase:
    def test_hierarchy(self):
        assert issubclass(IsTruePosteriorExtractor, IsTrueExtractor)
        assert issubclass(IsTruePosteriorExtractor, SequencePosteriorExtractor)
        assert issubclass(GCPIsTruePosteriorExtractor, IsTrueExtractor)
        assert issubclass(GCPIsTruePosteriorExtractor, GCPSequencePosteriorExtractor)

    def test_return_p_true_flips_false_answers(self):
        extractor = GCPIsTruePosteriorExtractor(return_p_true=True)
        _, conf = extractor("False", torch.tensor([LOG(0.6)]), None)
        assert conf == pytest.approx(0.4)
        _, conf = extractor("True", torch.tensor([LOG(0.6)]), None)
        assert conf == pytest.approx(0.6)

    def test_return_p_true_with_normalize(self):
        extractor = GCPIsTruePosteriorExtractor(normalize=True, return_p_true=True)
        top = [{"False": LOG(0.6), "True": LOG(0.2)}]
        _, conf = extractor("False", None, None, top_logprobs=top)
        assert conf == pytest.approx(0.25)

    def test_return_p_true_default_off(self):
        extractor = GCPIsTruePosteriorExtractor()
        _, conf = extractor("False", torch.tensor([LOG(0.6)]), None)
        assert conf == pytest.approx(0.6)

    @patch("callm.extractors.base.get_tokenizer_for_model")
    def test_hf_return_p_true(self, mock_get_tokenizer):
        mock_tokenizer = MagicMock()
        encoding = MagicMock()
        encoding.offset_mapping = [(0, 5)]
        mock_tokenizer.return_value = encoding
        mock_get_tokenizer.return_value = mock_tokenizer
        extractor = IsTruePosteriorExtractor(model_name="gpt2", return_p_true=True)
        _, conf = extractor("False", torch.tensor([LOG(0.7)]), None)
        assert conf == pytest.approx(0.3)
