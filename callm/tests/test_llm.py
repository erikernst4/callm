"""Unit tests for LLM model."""

import pytest
import torch
from unittest.mock import Mock, patch
from callm.models.llm import LLM


class TestLLMInitialization:
    """Tests for LLM initialization."""

    @patch("callm.models.llm.AutoModelForSeq2SeqLM")
    @patch("callm.models.llm.AutoTokenizer")
    @patch("callm.models.llm.CorrectnessEvaluator")
    def test_flan_t5_initialization(self, mock_evaluator, mock_tokenizer, mock_model):
        """Test initialization with flan-t5 model."""
        # Setup mock model with parameters method
        mock_model_instance = Mock()
        mock_model_instance.parameters.return_value = iter([])  # Empty iterator
        mock_model.from_pretrained.return_value = mock_model_instance

        llm = LLM(model_name="flan-t5-small", train=False)

        assert llm.model_name == "flan-t5-small"
        assert llm.is_seq2seq is True
        assert mock_model.from_pretrained.called

    @patch("callm.models.llm.AutoModelForCausalLM")
    @patch("callm.models.llm.AutoTokenizer")
    @patch("callm.models.llm.CorrectnessEvaluator")
    def test_llama_initialization(self, mock_evaluator, mock_tokenizer, mock_model):
        """Test initialization with Llama model."""
        llm = LLM(model_name="Llama-2-7b-chat-hf", hf_token="fake_token", train=False)

        assert llm.model_name == "Llama-2-7b-chat-hf"
        assert llm.is_seq2seq is False

    def test_unsupported_model(self):
        """Test that unsupported model raises error."""
        with pytest.raises(NotImplementedError):
            LLM(model_name="unsupported-model")


class TestLLMForward:
    """Tests for forward pass."""

    @patch("callm.models.llm.AutoModelForSeq2SeqLM")
    @patch("callm.models.llm.AutoTokenizer")
    @patch("callm.models.llm.CorrectnessEvaluator")
    def test_forward_generates(self, mock_evaluator, mock_tokenizer, mock_model):
        """Test that forward calls generate."""
        # Setup
        mock_model_instance = Mock()
        mock_model_instance.parameters.return_value = iter([])  # Empty iterator
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_model.from_pretrained.return_value = mock_model_instance

        llm = LLM(model_name="flan-t5-small", train=False)

        # Test
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])
        output = llm.forward(input_ids, attention_mask)

        # Verify generate was called
        assert mock_model_instance.generate.called
        assert output is not None


class TestLLMValidation:
    """Tests for validation step."""

    @patch("callm.models.llm.AutoModelForSeq2SeqLM")
    @patch("callm.models.llm.AutoTokenizer")
    @patch("callm.models.llm.CorrectnessEvaluator")
    def test_validation_step_structure(
        self, mock_evaluator, mock_tokenizer, mock_model
    ):
        """Test validation step processes batch correctly."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_model_instance.parameters.return_value = iter([])  # Empty iterator
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.batch_decode.return_value = [
            "Guess: Paris\nProbability: 0.9"
        ]

        mock_evaluator_instance = Mock()
        mock_evaluator.return_value = mock_evaluator_instance
        mock_evaluator_instance.evaluate.return_value = True

        # Create LLM
        llm = LLM(model_name="flan-t5-small", train=False)

        # Create batch (matches current datamodule structure)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "question": ["What is the capital of France?"],
            "label": [["Paris", "paris"]],
        }

        # Run validation step
        llm.validation_step(batch, 0)

        # Verify outputs stored
        assert len(llm.validation_outputs) == 1
        assert llm.validation_outputs[0]["question"] == "What is the capital of France?"
        assert "pred_answer" in llm.validation_outputs[0]
        assert "confidence" in llm.validation_outputs[0]
        assert "correct" in llm.validation_outputs[0]


class TestConfigureOptimizers:
    """Test optimizer configuration."""

    @patch("callm.models.llm.AutoModelForSeq2SeqLM")
    @patch("callm.models.llm.AutoTokenizer")
    @patch("callm.models.llm.CorrectnessEvaluator")
    def test_returns_none_for_inference(
        self, mock_evaluator, mock_tokenizer, mock_model
    ):
        """Test that configure_optimizers returns None for inference mode."""
        llm = LLM(model_name="flan-t5-small", train=False)
        assert llm.configure_optimizers() is None
