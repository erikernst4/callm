"""Tests for the length-normalized Sequence Posterior extractors.

c_norm = exp((1/n) * sum_t log p_t) = (prod_t p_t) ** (1/n), the geometric mean of
the n scored token probabilities.
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from callm.extractors import (
    GCPIsTruePosteriorExtractor,
    GCPLengthNormalizedSequencePosteriorExtractor,
    GCPMMLULengthNormalizedSequencePosteriorExtractor,
    GCPMMLUSequencePosteriorExtractor,
    GCPSequencePosteriorExtractor,
    IsTruePosteriorExtractor,
    LengthNormalizedSequencePosteriorExtractor,
    SequencePosteriorExtractor,
)

GEOMETRIC = math.sqrt(0.5 * 0.8)  # geometric mean of p = 0.5, 0.8 (~0.632)
ARITHMETIC = (0.5 + 0.8) / 2  # 0.65, must NOT be the result


@pytest.fixture
def mock_tokenizer():
    with patch("callm.extractors.base.get_tokenizer_for_model") as get_tok:
        tokenizer = MagicMock()
        get_tok.return_value = tokenizer
        yield tokenizer


def hf(cls=LengthNormalizedSequencePosteriorExtractor):
    return cls(model_name="gpt2")


# ── Aggregation hook ────────────────────────────────────────────────────


class TestAggregationHook:
    def test_bases_sum(self, mock_tokenizer):
        lp = torch.tensor([-0.5, -1.0, -0.2])
        assert hf(SequencePosteriorExtractor).aggregate_log_probs(lp) == pytest.approx(
            -1.7
        )
        assert GCPSequencePosteriorExtractor().aggregate_log_probs(lp) == pytest.approx(
            -1.7
        )

    def test_length_normalized_mean_of_log_probs(self, mock_tokenizer):
        lp = torch.tensor([-0.5, -1.0, -0.2])
        assert hf().aggregate_log_probs(lp) == pytest.approx(-1.7 / 3)
        assert GCPLengthNormalizedSequencePosteriorExtractor().aggregate_log_probs(
            lp
        ) == pytest.approx(-1.7 / 3)
        assert GCPMMLULengthNormalizedSequencePosteriorExtractor().aggregate_log_probs(
            lp
        ) == pytest.approx(-1.7 / 3)


# ── Geometric mean ──────────────────────────────────────────────────────


class TestGeometricMean:
    def test_hf_geometric_mean_of_probabilities(self, mock_tokenizer):
        lp = torch.log(torch.tensor([0.5, 0.8]))
        conf = hf().compute_probability_from_logits(lp, [0, 1])
        assert conf == pytest.approx(GEOMETRIC)
        assert conf != pytest.approx(ARITHMETIC)

    def test_gcp_geometric_mean_of_probabilities(self):
        lp = torch.log(torch.tensor([0.5, 0.8]))
        _, conf = GCPLengthNormalizedSequencePosteriorExtractor()(
            "Guess: Paris", lp, None
        )
        assert conf == pytest.approx(GEOMETRIC)
        assert conf != pytest.approx(ARITHMETIC)


# ── HF extractor ────────────────────────────────────────────────────────


class TestLengthNormalizedSequencePosteriorExtractor:
    def test_1d_selected_indices(self, mock_tokenizer):
        logits = torch.tensor([-0.5, -1.0, -0.2])
        assert hf().compute_probability_from_logits(logits, [0, 2]) == pytest.approx(
            math.exp(-0.35)
        )
        assert hf(SequencePosteriorExtractor).compute_probability_from_logits(
            logits, [0, 2]
        ) == pytest.approx(math.exp(-0.7))

    def test_2d_logits(self, mock_tokenizer):
        logits = torch.zeros(2, 4)
        logits[0, 1] = 2.0
        logits[1, 3] = 1.0
        log_probs = torch.log_softmax(logits, dim=-1).max(dim=-1).values
        expected = math.exp(log_probs.mean().item())
        assert hf().compute_probability_from_logits(logits, [0, 1]) == pytest.approx(
            expected
        )

    def test_equals_base_confidence_to_the_one_over_n(self, mock_tokenizer):
        torch.manual_seed(0)
        logits = torch.randn(6, 50)
        indices = [1, 2, 4]
        base = hf(SequencePosteriorExtractor).compute_probability_from_logits(
            logits, indices
        )
        norm = hf().compute_probability_from_logits(logits, indices)
        assert norm == pytest.approx(base ** (1 / len(indices)), rel=1e-5)

    def test_single_token_equals_base(self, mock_tokenizer):
        logits = torch.tensor([-0.7, -0.3])
        base = hf(SequencePosteriorExtractor).compute_probability_from_logits(
            logits, [1]
        )
        assert hf().compute_probability_from_logits(logits, [1]) == pytest.approx(base)

    def test_empty_indices_nan(self, mock_tokenizer):
        assert np.isnan(hf().compute_probability_from_logits(torch.tensor([-0.5]), []))

    def test_forward_scores_answer_tokens_only(self, mock_tokenizer):
        encoding = MagicMock()
        encoding.offset_mapping = [
            (0, 5),
            (5, 6),
            (6, 9),
            (9, 12),
        ]  # "Guess", ":", " Pa", "ris"
        mock_tokenizer.return_value = encoding
        logits = torch.tensor([-3.0, -3.0, -0.4, -0.2])
        answer, conf = hf()("Guess: Paris", logits, None)
        assert answer == "Paris"
        assert conf == pytest.approx(math.exp((-0.4 - 0.2) / 2))

    def test_forward_empty_text_and_none_logits(self, mock_tokenizer):
        assert np.isnan(hf()("", torch.tensor([-0.5]), None)[1])
        answer, conf = hf()("Guess: Paris", None, None)
        assert answer == "Paris"
        assert np.isnan(conf)


# ── GCP extractors ──────────────────────────────────────────────────────


class TestGCPLengthNormalizedSequencePosteriorExtractor:
    def test_forward_mean(self):
        answer, conf = GCPLengthNormalizedSequencePosteriorExtractor()(
            "Guess: Paris", torch.tensor([-0.5, -0.2]), None
        )
        assert answer == "Paris"
        assert conf == pytest.approx(math.exp(-0.35))

    def test_none_logits_and_empty_text(self):
        ext = GCPLengthNormalizedSequencePosteriorExtractor()
        assert np.isnan(ext("Guess: Paris", None, None)[1])
        answer, conf = ext("", torch.tensor([-0.5]), None)
        assert answer == ""
        assert np.isnan(conf)


class TestGCPMMLULengthNormalizedSequencePosteriorExtractor:
    def test_letter_and_mean_over_all_generated_tokens(self):
        # "B", "\n\n", "Because ..." -> every generated token is scored
        logits = torch.tensor([-0.1, -0.5, -0.9])
        answer, conf = GCPMMLULengthNormalizedSequencePosteriorExtractor()(
            "B\n\nBecause it is", logits, None
        )
        _, base = GCPMMLUSequencePosteriorExtractor()(
            "B\n\nBecause it is", logits, None
        )
        assert answer == "B"
        assert conf == pytest.approx(math.exp(-1.5 / 3))
        assert base == pytest.approx(math.exp(-1.5))


# ── Hierarchy and isolation ─────────────────────────────────────────────


class TestHierarchy:
    @pytest.mark.parametrize(
        "cls, parent",
        [
            (LengthNormalizedSequencePosteriorExtractor, SequencePosteriorExtractor),
            (
                GCPLengthNormalizedSequencePosteriorExtractor,
                GCPSequencePosteriorExtractor,
            ),
            (
                GCPMMLULengthNormalizedSequencePosteriorExtractor,
                GCPMMLUSequencePosteriorExtractor,
            ),
        ],
    )
    def test_subclass_of_parent(self, cls, parent):
        assert issubclass(cls, parent)

    def test_is_true_extractors_still_sum(self, mock_tokenizer):
        lp = torch.tensor([-0.5, -0.2])
        assert IsTruePosteriorExtractor(model_name="gpt2").aggregate_log_probs(
            lp
        ) == pytest.approx(-0.7)
        assert GCPIsTruePosteriorExtractor().aggregate_log_probs(lp) == pytest.approx(
            -0.7
        )

    def test_instantiable_with_model_name(self, mock_tokenizer):
        for cls in (
            LengthNormalizedSequencePosteriorExtractor,
            GCPLengthNormalizedSequencePosteriorExtractor,
            GCPMMLULengthNormalizedSequencePosteriorExtractor,
        ):
            assert isinstance(cls(model_name="gpt2"), cls)
