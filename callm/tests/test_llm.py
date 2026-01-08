"""Unit tests for LLM model."""

import pytest
import torch
import os
from unittest.mock import Mock, patch
from callm.models.llm import LLM
from callm.extractors import VerbalizedConfidenceExtractor


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

        llm = LLM(
            model_name="google/flan-t5-small",
            train=False,
            extractor=VerbalizedConfidenceExtractor(),
        )

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
            extractor=VerbalizedConfidenceExtractor(),
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
            LLM(
                model_name="unsupported-model",
                extractor=VerbalizedConfidenceExtractor(),
            )


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

        llm = LLM(
            model_name="google/flan-t5-small",
            train=False,
            extractor=VerbalizedConfidenceExtractor(),
        )

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
        llm = LLM(
            model_name="google/flan-t5-small",
            train=False,
            extractor=VerbalizedConfidenceExtractor(),
        )

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
        assert "output_ids" in llm.validation_outputs[0]
        assert "gold_answers" in llm.validation_outputs[0]

    @patch("callm.models.llm.get_tokenizer_for_model")
    @patch("callm.models.llm.initialize_model")
    def test_validation_flushing(self, mock_init_model, mock_get_tokenizer):
        """Test validation output flushing mechanism."""
        # Setup mock model
        mock_model_instance = Mock()
        mock_model_instance.parameters.return_value = iter([])
        mock_model_instance.config = Mock()
        mock_model_instance.config.pad_token_id = 0
        mock_model_instance.config.eos_token_id = 1
        # Mock generate return value for batch of 4
        mock_model_instance.generate.return_value = torch.tensor([[10, 11]] * 4)
        mock_init_model.return_value = (mock_model_instance, True)

        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.decode.return_value = "decoded answer"
        mock_get_tokenizer.return_value = mock_tokenizer_instance

        # Initialize LLM with flush every 2 steps
        llm = LLM(
            model_name="google/flan-t5-small",
            train=False,
            extractor=VerbalizedConfidenceExtractor(),
            flush_outputs_every_n_steps=2,
        )

        # Mock trainer log_dir
        llm.trainer = Mock()
        llm.trainer.log_dir = "tmp_test_dir"
        llm.trainer.global_rank = 0
        os.makedirs("tmp_test_dir", exist_ok=True)

        try:
            # Create batch with 4 items
            batch = {
                "input_ids": torch.tensor([[1, 2]] * 4),
                "attention_mask": torch.tensor([[1, 1]] * 4),
                "question": ["q1", "q2", "q3", "q4"],
                "label": [["a1"], ["a2"], ["a3"], ["a4"]],
            }

            # Run validation step
            # Batch size 4 >= flush step 2, so it should flush
            llm.validation_step(batch, 0)

            # Check flushing happened
            assert len(llm.validation_outputs) == 0
            assert len(llm.flushed_output_files) == 1
            assert os.path.exists(llm.flushed_output_files[0])

            # Run on_validation_epoch_end
            llm.on_validation_epoch_end()

            # Check if flushed files are cleaned up
            assert len(llm.flushed_output_files) == 0

            # Check if output csv was written (implies data was reloaded and processed)
            assert os.path.exists(os.path.join("tmp_test_dir", "llm_outputs.csv"))

        finally:
            import shutil

            if os.path.exists("tmp_test_dir"):
                shutil.rmtree("tmp_test_dir")


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

        llm = LLM(
            model_name="google/flan-t5-small",
            train=False,
            extractor=VerbalizedConfidenceExtractor(),
        )
        assert llm.configure_optimizers() is None

    @patch("callm.models.llm.initialize_model")
    @patch("callm.models.llm.get_tokenizer_for_model")
    def test_llm_stores_optimized_logits(self, mock_get_tokenizer, mock_init_model):
        """Test that LLM saves 1D log_softmax values instead of 2D full logits."""
        # Mock init_model return
        mock_model = Mock()
        mock_model.config.pad_token_id = 0
        mock_model.config.eos_token_id = 1
        mock_model.parameters.return_value = iter([])  # Ensure it's iterable
        mock_init_model.return_value = (mock_model, False)  # model, is_seq2seq=False

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_get_tokenizer.return_value = mock_tokenizer

        # Mock extractor
        extractor = Mock()

        # Initialize LLM with return_logits=True
        model = LLM(extractor=extractor, model_name="gpt2", return_logits=True)

        # Mock model.generate output
        # Batch size 1, seq len 5, vocab 10
        batch_size = 1
        seq_len = 5
        vocab_size = 10

        generation_scores = tuple(
            [torch.randn(batch_size, vocab_size) for _ in range(seq_len - 1)]
        )

        # generation_output.sequences
        output_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # patch forward to return a mock
        mock_output = Mock()
        mock_output.sequences = output_ids
        mock_output.scores = generation_scores

        with torch.no_grad():
            model.forward = Mock(return_value=mock_output)

            # Create a dummy batch
            batch = {
                "input_ids": torch.tensor([[1]]),
                "attention_mask": torch.tensor([[1]]),
                "question": ["q1"],
                "label": [["a1"]],
            }

            # Run validation step
            model.validation_step(batch, 0)

            # Check stored output
            assert len(model.validation_outputs) == 1
            out = model.validation_outputs[0]

            assert "logits" in out
            logits = out["logits"]

            # Verification:
            # logits should be 1D
            assert logits.ndim == 1
            # Input length was 1, total seq_len 5 => generated 4
            assert logits.shape[0] == seq_len - 1
