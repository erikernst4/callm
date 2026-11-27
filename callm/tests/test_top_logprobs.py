"""Tests for storing per-step top-k log probabilities (needed for normalized IsTrue)."""

import pytest
import torch
from unittest.mock import Mock, patch

from callm.models.llm import LLM
from callm.models.gcp_llm import GCPLLM

VOCAB = ["True", "False", " True", "Yes", "No"]


def _make_llm(mock_init_model, mock_get_tokenizer, top_logprobs):
    mock_model = Mock()
    mock_model.config = Mock(spec=["pad_token_id", "eos_token_id"])
    mock_model.config.pad_token_id = 0
    mock_model.config.eos_token_id = 1
    mock_model.parameters.return_value = iter([])
    mock_init_model.return_value = mock_model

    mock_tokenizer = Mock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.decode.side_effect = lambda ids, **kw: "".join(VOCAB[i] for i in ids)
    mock_get_tokenizer.return_value = mock_tokenizer

    return LLM(
        extractor=Mock(),
        model_name="gpt2",
        return_logits=True,
        top_logprobs=top_logprobs,
    )


def _run_step(llm):
    # input length 1, generated tokens [0, 1] ("True", "False")
    scores = (
        torch.tensor([[3.0, 1.0, 0.5, 0.0, -1.0]]),
        torch.tensor([[0.0, 2.0, 0.0, 1.0, 0.0]]),
    )
    mock_output = Mock()
    mock_output.sequences = torch.tensor([[4, 0, 1]])
    mock_output.scores = scores
    llm.forward = Mock(return_value=mock_output)
    batch = {
        "input_ids": torch.tensor([[4]]),
        "attention_mask": torch.tensor([[1]]),
        "question": ["q1"],
        "label": [["a1"]],
    }
    with torch.no_grad():
        llm.validation_step(batch, 0)
    return llm.outputs[0], scores


@patch("callm.models.llm.get_tokenizer_for_model")
@patch("callm.models.llm.initialize_model")
class TestLLMTopLogprobs:
    def test_disabled_by_default(self, mock_init_model, mock_get_tokenizer):
        llm = _make_llm(mock_init_model, mock_get_tokenizer, top_logprobs=0)
        out, _ = _run_step(llm)
        assert "top_logprobs" not in out

    def test_stores_top_k_per_step(self, mock_init_model, mock_get_tokenizer):
        llm = _make_llm(mock_init_model, mock_get_tokenizer, top_logprobs=3)
        out, scores = _run_step(llm)

        top = out["top_logprobs"]
        assert len(top) == 2
        assert all(len(step) == 3 for step in top)

        log_probs = torch.log_softmax(scores[0][0], dim=-1)
        assert set(top[0]) == {"True", "False", " True"}
        assert top[0]["True"] == pytest.approx(log_probs[0].item())
        assert top[0][" True"] == pytest.approx(log_probs[2].item())
        # top-1 of each step equals the stored chosen-token logprob
        for step, chosen in zip(top, out["logits"]):
            assert max(step.values()) == pytest.approx(chosen.item())

    def test_batch_matches_per_sample_reference(
        self, mock_init_model, mock_get_tokenizer
    ):
        k = 2
        llm = _make_llm(mock_init_model, mock_get_tokenizer, top_logprobs=k)
        # Two samples whose top-k differ at every step
        scores = (
            torch.tensor([[3.0, 1.0, 0.5, 0.0, -1.0], [0.0, 0.5, 1.0, 4.0, 2.0]]),
            torch.tensor([[0.0, 2.0, 0.0, 1.0, 0.0], [1.5, 0.0, 3.0, 0.0, 0.2]]),
        )
        mock_output = Mock()
        mock_output.sequences = torch.tensor([[4, 0, 1], [4, 3, 2]])
        mock_output.scores = scores
        llm.forward = Mock(return_value=mock_output)
        batch = {
            "input_ids": torch.tensor([[4], [4]]),
            "attention_mask": torch.tensor([[1], [1]]),
            "question": ["q1", "q2"],
            "label": [["a1"], ["a2"]],
        }
        with torch.no_grad():
            llm.validation_step(batch, 0)

        assert len(llm.outputs) == 2
        for i, out in enumerate(llm.outputs):
            # Per-sample reference: callm's original step-by-step computation
            expected = []
            for step_scores in scores:
                log_probs = torch.log_softmax(step_scores[i], dim=-1)
                values, ids = torch.topk(log_probs, k)
                expected.append(
                    {
                        VOCAB[int(t)]: pytest.approx(float(v))
                        for v, t in zip(values, ids)
                    }
                )
            assert out["top_logprobs"] == expected


def _gcp_resp():
    def cand(token, lp):
        c = Mock()
        c.token = token
        c.log_probability = lp
        return c

    chosen = [cand("True", -0.1)]
    top = Mock()
    top.candidates = [cand("True", -0.1), cand("False", -2.5), cand("true", -4.0)]

    logprobs_result = Mock()
    logprobs_result.chosen_candidates = chosen
    logprobs_result.top_candidates = [top]
    candidate = Mock()
    candidate.logprobs_result = logprobs_result
    resp = Mock()
    resp.text = "True"
    resp.candidates = [candidate]
    return resp


@patch("callm.models.gcp_llm.types")
@patch("callm.models.gcp_llm.genai")
class TestGCPTopLogprobs:
    def _run(self, llm):
        llm.client.models.generate_content.return_value = _gcp_resp()
        batch = {"input": ["Instruction"], "question": ["q1"], "label": [["a1"]]}
        llm.validation_step(batch, 0)
        return llm.outputs[0]

    def test_disabled_by_default(self, mock_genai, mock_types):
        llm = GCPLLM(extractor=Mock(), model_name="m", return_logits=True)
        out = self._run(llm)
        assert "top_logprobs" not in out
        assert mock_types.GenerateContentConfig.call_args[1]["logprobs"] == 1

    def test_stores_top_candidates(self, mock_genai, mock_types):
        llm = GCPLLM(
            extractor=Mock(), model_name="m", return_logits=True, top_logprobs=20
        )
        out = self._run(llm)
        assert mock_types.GenerateContentConfig.call_args[1]["logprobs"] == 20
        assert out["top_logprobs"] == [
            {
                "True": pytest.approx(-0.1),
                "False": pytest.approx(-2.5),
                "true": pytest.approx(-4.0),
            }
        ]
