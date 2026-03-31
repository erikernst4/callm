import pytest
from unittest.mock import Mock, patch
from callm.models.gcp_llm import GCPLLM
from callm.models.gcp_evaluator import GCPEvaluatorModule


class TestGCPLLM:
    @patch("callm.models.gcp_llm.genai")
    def test_gcp_llm_initializes_with_parameters(self, mock_genai):
        extractor = Mock()
        llm = GCPLLM(
            extractor=extractor,
            model_name="my-test-model",
            location="us-central1",
        )
        assert llm.model_name == "my-test-model"
        assert llm.client is not None
        mock_genai.Client.assert_called_with(vertexai=True, location="us-central1")

    @patch("callm.models.gcp_llm.types")
    @patch("callm.models.gcp_llm.genai")
    def test_gcp_llm_validation_step(self, mock_genai, mock_types):
        extractor = Mock()
        llm = GCPLLM(extractor=extractor, model_name="test-model")

        mock_resp = Mock()
        mock_resp.text = "generated text answer"
        llm.client.models.generate_content.return_value = mock_resp

        batch = {
            "input": ["Instruction 1", "Instruction 2"],
            "question": ["q1", "q2"],
            "label": [["a1"], ["a2"]],
        }

        llm.validation_step(batch, 0)

        assert len(llm.outputs) == 2
        assert llm.outputs[0]["question"] == "q1"
        assert llm.outputs[1]["raw_output"] == "generated text answer"

    @patch("callm.models.gcp_llm.types")
    @patch("callm.models.gcp_llm.genai")
    def test_gcp_llm_extracts_logits(self, mock_genai, mock_types):
        extractor = Mock()
        llm = GCPLLM(extractor=extractor, model_name="test-model", return_logits=True)

        mock_resp = Mock()
        mock_resp.text = "generated text answer"

        # Mock logprobs sequence
        mock_candidate = Mock()
        mock_logprobs_result = Mock()

        token1 = Mock()
        token1.log_probability = -0.5
        token2 = Mock()
        token2.log_probability = -0.1

        mock_logprobs_result.chosen_candidates = [token1, token2]
        mock_candidate.logprobs_result = mock_logprobs_result
        mock_resp.candidates = [mock_candidate]

        llm.client.models.generate_content.return_value = mock_resp

        batch = {"input": ["Instruction 1"], "question": ["q1"], "label": [["a1"]]}

        llm.validation_step(batch, 0)

        assert len(llm.outputs) == 1
        logits = llm.outputs[0]["logits"]
        assert logits is not None
        assert logits.tolist() == pytest.approx([-0.5, -0.1], abs=1e-5)

        # Verify config was passed for logprobs
        config_call_args = mock_types.GenerateContentConfig.call_args
        assert config_call_args is not None
        config_kwargs = config_call_args[1]
        assert config_kwargs.get("response_logprobs") is True


class TestGCPEvaluator:
    @patch("callm.models.gcp_evaluator.types")
    @patch("callm.models.gcp_evaluator.genai")
    def test_evaluator_validation_step(self, mock_genai, mock_types):
        evaluator = GCPEvaluatorModule(model_name="test-eval-model")

        mock_resp = Mock()
        mock_resp.text = "Yes"
        evaluator.client.models.generate_content.return_value = mock_resp

        batch = {
            "exact_match": [True, False],
            "question": ["Q1", "Q2"],
            "gold_answers": ["A1", "B1"],
            "pred_answer": ["A1", "B2"],
            "confidence": ["0.9", "0.8"],
            "raw_output": ["A1", "B2"],
            "index": [0, 1],
            "input": ["", "Evaluate this: B2 vs B1"],
        }

        evaluator.validation_step(batch, 0)

        assert len(evaluator.outputs) == 2
        assert evaluator.outputs[0]["exact_match"] is True
        assert evaluator.outputs[1]["exact_match"] is False
        assert evaluator.outputs[1]["evaluator_response_raw"] == "Yes"

        # Exact Match didn't trigger api call, the second item triggered 1 call
        assert evaluator.client.models.generate_content.call_count == 1


class TestGCPLLMChatPromptInput:
    """Tests that GCPLLM handles ChatPrompt GCP-format message lists."""

    @patch("callm.models.gcp_llm.types")
    @patch("callm.models.gcp_llm.genai")
    def test_validation_step_with_chat_prompt_messages(self, mock_genai, mock_types):
        """When input is a list of dicts (ChatPrompt gcp=True), should build contents correctly."""
        extractor = Mock()
        llm = GCPLLM(extractor=extractor, model_name="test-model")

        mock_resp = Mock()
        mock_resp.text = "Paris"
        llm.client.models.generate_content.return_value = mock_resp

        # Simulate ChatPrompt(gcp=True) output
        chat_messages = [
            {"role": "system", "parts": [{"text": "Be helpful"}]},
            {
                "role": "user",
                "parts": [{"text": "Question: What is the capital of France?"}],
            },
            {"role": "model", "parts": [{"text": "Guess:"}]},
        ]

        batch = {
            "input": [chat_messages],
            "question": ["What is the capital of France?"],
            "label": [["paris"]],
        }

        llm.validation_step(batch, 0)

        assert len(llm.outputs) == 1
        assert llm.outputs[0]["raw_output"] == "Paris"

        # Verify system_instruction was passed in config
        config_call = mock_types.GenerateContentConfig.call_args
        assert config_call[1].get("system_instruction") == "Be helpful"

        # Verify contents were constructed (system should be excluded from contents)
        generate_call = llm.client.models.generate_content.call_args
        contents = generate_call[1]["contents"]
        # Should have 2 Content objects (user + model), not 3
        assert len(contents) == 2

    @patch("callm.models.gcp_llm.types")
    @patch("callm.models.gcp_llm.genai")
    def test_validation_step_with_plain_string_still_works(
        self, mock_genai, mock_types
    ):
        """Plain string input should still work as before."""
        extractor = Mock()
        llm = GCPLLM(extractor=extractor, model_name="test-model")

        mock_resp = Mock()
        mock_resp.text = "generated text"
        llm.client.models.generate_content.return_value = mock_resp

        batch = {
            "input": ["Plain prompt text"],
            "question": ["q1"],
            "label": [["a1"]],
        }

        llm.validation_step(batch, 0)

        assert len(llm.outputs) == 1
        assert llm.outputs[0]["raw_output"] == "generated text"

        # No system_instruction should be in config
        config_call = mock_types.GenerateContentConfig.call_args
        assert "system_instruction" not in config_call[1]
