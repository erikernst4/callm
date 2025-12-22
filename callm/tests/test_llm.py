"""Unit tests for LLM model."""

import pytest
import torch
from unittest.mock import Mock, patch
from callm.models.llm import LLM


class TestLLMInitialization:
    """Tests for LLM initialization."""

    @patch("callm.models.llm.get_tokenizer_for_model")
    @patch("callm.models.llm.initialize_model")
    def test_flan_t5_initialization(self, mock_init_model, mock_get_tokenizer):
        """Test initialization with flan-t5 model."""
        # Setup mock model with required attributes
        mock_model_instance = Mock()
        mock_model_instance.parameters.return_value = iter([])
        mock_model_instance.config = Mock()
        mock_model_instance.config.pad_token_id = 0
        mock_model_instance.config.eos_token_id = 1
        mock_init_model.return_value = (mock_model_instance, True)  # is_seq2seq=True

        mock_tokenizer_instance = Mock()
        mock_get_tokenizer.return_value = mock_tokenizer_instance

        llm = LLM(model_name="google/flan-t5-small", train=False)

        assert llm.model_name == "google/flan-t5-small"
        assert llm.is_seq2seq is True
        assert mock_init_model.called

    @patch("callm.models.llm.get_tokenizer_for_model")
    @patch("callm.models.llm.initialize_model")
    def test_llama_initialization(self, mock_init_model, mock_get_tokenizer):
        """Test initialization with Llama model."""
        # Setup mock model
        mock_model_instance = Mock()
        mock_model_instance.parameters.return_value = iter([])
        mock_model_instance.config = Mock()
        mock_model_instance.config.pad_token_id = 0
        mock_model_instance.config.eos_token_id = 1
        mock_init_model.return_value = (mock_model_instance, False)  # is_seq2seq=False

        mock_tokenizer_instance = Mock()
        mock_get_tokenizer.return_value = mock_tokenizer_instance

        llm = LLM(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            hf_token="fake_token",
            train=False,
        )

        assert llm.model_name == "meta-llama/Llama-2-7b-chat-hf"
        assert llm.is_seq2seq is False

    @patch("callm.models.llm.get_tokenizer_for_model")
    @patch("callm.models.llm.initialize_model")
    def test_unsupported_model(self, mock_init_model, mock_get_tokenizer):
        """Test that unsupported model raises error."""
        # Make initialize_model raise NotImplementedError for unsupported models
        mock_init_model.side_effect = NotImplementedError(
            "Model unsupported-model not supported"
        )

        with pytest.raises(NotImplementedError):
            LLM(model_name="unsupported-model")


class TestLLMForward:
    """Tests for forward pass."""

    @patch("callm.models.llm.get_tokenizer_for_model")
    @patch("callm.models.llm.initialize_model")
    def test_forward_generates(self, mock_init_model, mock_get_tokenizer):
        """Test that forward calls generate."""
        # Setup mock model
        mock_model_instance = Mock()
        mock_model_instance.parameters.return_value = iter([])
        mock_model_instance.config = Mock()
        mock_model_instance.config.pad_token_id = 0
        mock_model_instance.config.eos_token_id = 1
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_init_model.return_value = (mock_model_instance, True)  # is_seq2seq=True

        mock_tokenizer_instance = Mock()
        mock_get_tokenizer.return_value = mock_tokenizer_instance

        llm = LLM(model_name="google/flan-t5-small", train=False)

        # Test
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])
        output = llm.forward(input_ids, attention_mask)

        # Verify generate was called
        assert mock_model_instance.generate.called
        assert output is not None


class TestLLMValidation:
    """Tests for validation step."""

    @patch("callm.models.llm.get_tokenizer_for_model")
    @patch("callm.models.llm.initialize_model")
    def test_validation_step_structure(self, mock_init_model, mock_get_tokenizer):
        """Test validation step processes batch correctly."""
        # Setup mock model
        mock_model_instance = Mock()
        mock_model_instance.parameters.return_value = iter([])
        mock_model_instance.config = Mock()
        mock_model_instance.config.pad_token_id = 0
        mock_model_instance.config.eos_token_id = 1
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_init_model.return_value = (mock_model_instance, True)  # is_seq2seq=True

        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.batch_decode.return_value = [
            "Guess: Paris\nProbability: 0.9"
        ]
        mock_get_tokenizer.return_value = mock_tokenizer_instance

        # Create LLM
        llm = LLM(model_name="google/flan-t5-small", train=False)

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
        assert "output_ids" in llm.validation_outputs[0]
        assert "input_length" in llm.validation_outputs[0]
        assert "gold_answers" in llm.validation_outputs[0]


class TestConfigureOptimizers:
    """Test optimizer configuration."""

    @patch("callm.models.llm.get_tokenizer_for_model")
    @patch("callm.models.llm.initialize_model")
    def test_returns_none_for_inference(self, mock_init_model, mock_get_tokenizer):
        """Test that configure_optimizers returns None for inference mode."""
        # Setup mock model
        mock_model_instance = Mock()
        mock_model_instance.parameters.return_value = iter([])
        mock_model_instance.config = Mock()
        mock_model_instance.config.pad_token_id = 0
        mock_model_instance.config.eos_token_id = 1
        mock_init_model.return_value = (mock_model_instance, True)

        mock_get_tokenizer.return_value = Mock()

        llm = LLM(model_name="google/flan-t5-small", train=False)
        assert llm.configure_optimizers() is None
