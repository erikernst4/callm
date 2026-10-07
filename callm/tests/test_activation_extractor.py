"""Unit tests for ActivationExtractor model."""

import os
import csv
import json
import shutil
import pytest
import functools

import torch
from unittest.mock import Mock

from callm.models.llm import LLM


def _make_mock_llm(is_seq2seq=False, model_name="Qwen/Qwen3.5-9B"):
    """Create a mock LLM with the required attributes for ActivationExtractor."""
    mock_llm = Mock()
    mock_llm.model_name = model_name
    mock_llm.is_seq2seq = is_seq2seq
    mock_llm.full_sequence = functools.partial(LLM.full_sequence, mock_llm)
    mock_llm.tokenizer = Mock()
    mock_llm.tokenizer.decode.return_value = "Guess: Paris"
    mock_llm.extractor.return_value = ("Paris", 0.9)
    mock_llm.parameters.return_value = iter([])
    mock_llm.eval.return_value = None

    # Mock forward call (generate=True): returns generated_ids
    # Mock forward call (generate=False): returns hidden_states
    hidden_dim = 64

    def mock_forward(input_ids, attention_mask, generate=True):
        batch_size = input_ids.shape[0]
        if generate:
            # Return generated_ids: [batch, prompt_len + answer_len]
            prompt_len = input_ids.shape[1]
            answer_len = 2
            return torch.randint(0, 100, (batch_size, prompt_len + answer_len))
        else:
            # Return hidden_states: tuple of (batch, seq_len, hidden_dim) per layer
            actual_seq_len = input_ids.shape[1]
            return tuple(
                torch.randn(batch_size, actual_seq_len, hidden_dim) for _ in range(4)
            )

    mock_llm.__call__ = mock_forward
    mock_llm.side_effect = mock_forward
    return mock_llm


def _make_triviaqa_batch():
    """Create a TriviaQA-shaped batch with list-of-strings gold_answers."""
    return {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "question": ["What is the capital of France?"],
        "gold_answers": [["paris", "city of paris", "Paris"]],
    }


def _make_mmlu_batch():
    """Create an MMLU-shaped batch with string labels."""
    return {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "question": ["What is 2+2?"],
        "label": ["B"],
    }


def _make_extractor(
    mock_llm, output_dir, layer_indices=None, flush_outputs_every_n_steps=100
):
    """Create an ActivationExtractor instance with a mock trainer."""
    from callm.models.activation_extractor import ActivationExtractor

    extractor = ActivationExtractor(
        llm=mock_llm,
        output_dir=output_dir,
        layer_indices=layer_indices,
        save_dtype="float32",
        flush_outputs_every_n_steps=flush_outputs_every_n_steps,
    )

    # Mock trainer
    extractor.trainer = Mock()
    extractor.trainer.log_dir = output_dir
    extractor.trainer.global_rank = 0
    return extractor


class TestActivationExtractorValidationStep:
    """Tests for ActivationExtractor.validation_step batch handling."""

    def setup_method(self):
        self.output_dir = "tmp_test_activation_extractor"
        os.makedirs(self.output_dir, exist_ok=True)

    def teardown_method(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_validation_step_triviaqa_batch(self):
        """Verify it processes a TriviaQA-shaped batch (gold_answers is list)."""
        mock_llm = _make_mock_llm()
        extractor = _make_extractor(mock_llm, self.output_dir, layer_indices=[0, 1])

        batch = _make_triviaqa_batch()
        result = extractor.validation_step(batch, 0)

        assert result["batch_size"] == 1
        assert len(extractor.outputs) == 1

        out = extractor.outputs[0]
        assert out["question"] == "What is the capital of France?"
        assert out["gold_answers"] == ["paris", "city of paris", "Paris"]
        assert "activations" in out
        assert "seq_len" in out
        assert "prompt_length" in out
        assert out["sample_idx"] == 0

    def test_validation_step_mmlu_batch(self):
        """Verify it processes an MMLU-shaped batch (label is string, fallback)."""
        mock_llm = _make_mock_llm()
        mock_llm.extractor.return_value = ("B", 0.95)
        extractor = _make_extractor(mock_llm, self.output_dir, layer_indices=[0, 1])

        batch = _make_mmlu_batch()
        result = extractor.validation_step(batch, 0)

        assert result["batch_size"] == 1
        assert len(extractor.outputs) == 1

        out = extractor.outputs[0]
        assert out["question"] == "What is 2+2?"
        # gold_answers comes from "label" fallback
        assert out["gold_answers"] == "B"

    def test_validation_step_no_gold_answers(self):
        """Verify it handles batches with neither gold_answers nor label."""
        mock_llm = _make_mock_llm()
        extractor = _make_extractor(mock_llm, self.output_dir, layer_indices=[0])

        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "question": ["Some question"],
        }
        extractor.validation_step(batch, 0)

        assert extractor.outputs[0]["gold_answers"] is None

    def test_validation_step_increments_sample_counter(self):
        """Verify sample_idx increments across batches."""
        mock_llm = _make_mock_llm()
        extractor = _make_extractor(mock_llm, self.output_dir, layer_indices=[0])

        batch1 = _make_triviaqa_batch()
        batch2 = _make_triviaqa_batch()

        extractor.validation_step(batch1, 0)
        extractor.validation_step(batch2, 1)

        assert extractor.outputs[0]["sample_idx"] == 0
        assert extractor.outputs[1]["sample_idx"] == 1

    def test_validation_step_extracts_specified_layers(self):
        """Verify only the specified layers are extracted."""
        mock_llm = _make_mock_llm()
        extractor = _make_extractor(mock_llm, self.output_dir, layer_indices=[0, 2])

        batch = _make_triviaqa_batch()
        extractor.validation_step(batch, 0)

        activations = extractor.outputs[0]["activations"]
        assert set(activations.keys()) == {0, 2}

    def test_validation_step_all_layers_when_none(self):
        """Verify all layers are extracted when layer_indices is None."""
        mock_llm = _make_mock_llm()
        extractor = _make_extractor(mock_llm, self.output_dir, layer_indices=None)

        batch = _make_triviaqa_batch()
        extractor.validation_step(batch, 0)

        activations = extractor.outputs[0]["activations"]
        # mock_llm returns 4 layers
        assert len(activations) == 4


class TestActivationExtractorSerialization:
    """Tests for gold_answers serialization in on_validation_epoch_end."""

    def setup_method(self):
        self.output_dir = "tmp_test_activation_serialization"
        os.makedirs(self.output_dir, exist_ok=True)

    def teardown_method(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_gold_answers_serialization_list(self):
        """Verify gold_answers list is joined with '|' in CSV."""
        mock_llm = _make_mock_llm()
        extractor = _make_extractor(mock_llm, self.output_dir, layer_indices=[0])

        batch = _make_triviaqa_batch()
        extractor.validation_step(batch, 0)
        extractor.on_validation_epoch_end()

        csv_path = os.path.join(self.output_dir, "llm_outputs.csv")
        assert os.path.exists(csv_path)

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert row["gold_answers"] == "paris|city of paris|Paris"

    def test_gold_answers_serialization_string(self):
        """Verify string gold_answers (MMLU-style) is serialized correctly."""
        mock_llm = _make_mock_llm()
        mock_llm.extractor.return_value = ("B", 0.95)
        extractor = _make_extractor(mock_llm, self.output_dir, layer_indices=[0])

        batch = _make_mmlu_batch()
        extractor.validation_step(batch, 0)
        extractor.on_validation_epoch_end()

        csv_path = os.path.join(self.output_dir, "llm_outputs.csv")
        assert os.path.exists(csv_path)

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            # "|".join("B") produces "B" — works by accident but should be correct
            assert row["gold_answers"] == "B"

    def test_gold_answers_serialization_none(self):
        """Verify None gold_answers produces empty string."""
        mock_llm = _make_mock_llm()
        extractor = _make_extractor(mock_llm, self.output_dir, layer_indices=[0])

        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "question": ["Some question"],
        }
        extractor.validation_step(batch, 0)
        extractor.on_validation_epoch_end()

        csv_path = os.path.join(self.output_dir, "llm_outputs.csv")
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert row["gold_answers"] == ""


class TestActivationExtractorShardSaving:
    """Tests for activation shard and metadata saving."""

    def setup_method(self):
        self.output_dir = "tmp_test_activation_shards"
        os.makedirs(self.output_dir, exist_ok=True)

    def teardown_method(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_shards_saved(self):
        """Verify activation shards are saved as .pt files."""
        mock_llm = _make_mock_llm()
        extractor = _make_extractor(mock_llm, self.output_dir, layer_indices=[0, 1])

        batch = _make_triviaqa_batch()
        extractor.validation_step(batch, 0)
        extractor.on_validation_epoch_end()

        shard_path = os.path.join(self.output_dir, "activations_0000.pt")
        assert os.path.exists(shard_path)

        shard_data = torch.load(shard_path, weights_only=False)
        assert len(shard_data) == 1
        assert "activations" in shard_data[0]
        assert "seq_len" in shard_data[0]
        assert "prompt_length" in shard_data[0]
        assert "sample_idx" in shard_data[0]

    def test_flushed_chunks_consolidated_and_cleaned_up(self):
        """Verify periodically flushed chunks are consolidated and deleted."""
        mock_llm = _make_mock_llm()
        # Set flush_outputs_every_n_steps=1 so every validation_step flushes
        extractor = _make_extractor(
            mock_llm,
            self.output_dir,
            layer_indices=[0, 1],
            flush_outputs_every_n_steps=1,
        )

        batch1 = _make_triviaqa_batch()
        batch2 = _make_triviaqa_batch()

        extractor.validation_step(batch1, 0)
        assert len(extractor.flushed_output_files) == 1
        chunk1_path = extractor.flushed_output_files[0]
        assert os.path.exists(chunk1_path)

        extractor.validation_step(batch2, 1)
        assert len(extractor.flushed_output_files) == 2
        chunk2_path = extractor.flushed_output_files[1]
        assert os.path.exists(chunk2_path)

        extractor.on_validation_epoch_end()

        # Shard should exist with 2 samples
        shard_path = os.path.join(self.output_dir, "activations_0000.pt")
        assert os.path.exists(shard_path)
        shard_data = torch.load(shard_path, weights_only=False)
        assert len(shard_data) == 2

        # Temporary chunk files must be deleted
        assert not os.path.exists(chunk1_path)
        assert not os.path.exists(chunk2_path)
        assert len(extractor.flushed_output_files) == 0

    def test_misaligned_sample_idx_raises_and_keeps_chunks(self):
        """Verify chunks whose sample_idx doesn't match row order abort consolidation."""
        mock_llm = _make_mock_llm()
        extractor = _make_extractor(
            mock_llm, self.output_dir, layer_indices=[0], flush_outputs_every_n_steps=1
        )

        batch1 = {**_make_triviaqa_batch(), "sample_idx": [0]}
        batch2 = {**_make_triviaqa_batch(), "sample_idx": [2]}  # sample 1 missing
        extractor.validation_step(batch1, 0)
        extractor.validation_step(batch2, 1)
        chunk_paths = list(extractor.flushed_output_files)

        with pytest.raises(ValueError, match="misaligned"):
            extractor.on_validation_epoch_end()

        for path in chunk_paths:
            assert os.path.exists(path)
        assert not os.path.exists(os.path.join(self.output_dir, "activations_0000.pt"))

    def test_metadata_saved(self):
        """Verify metadata.json is saved with correct fields."""
        mock_llm = _make_mock_llm()
        extractor = _make_extractor(mock_llm, self.output_dir, layer_indices=[0, 1])

        batch = _make_triviaqa_batch()
        extractor.validation_step(batch, 0)
        extractor.on_validation_epoch_end()

        metadata_path = os.path.join(self.output_dir, "metadata.json")
        assert os.path.exists(metadata_path)

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        assert metadata["model_name"] == "Qwen/Qwen3.5-9B"
        assert metadata["layer_indices"] == [0, 1]
        assert metadata["hidden_size"] == 64  # from mock
        assert metadata["num_samples"] == 1
        assert metadata["dtype"] == "float32"

    def test_llm_outputs_csv_written(self):
        """Verify llm_outputs.csv has correct header."""
        mock_llm = _make_mock_llm()
        extractor = _make_extractor(mock_llm, self.output_dir, layer_indices=[0])

        batch = _make_triviaqa_batch()
        extractor.validation_step(batch, 0)
        extractor.on_validation_epoch_end()

        csv_path = os.path.join(self.output_dir, "llm_outputs.csv")
        assert os.path.exists(csv_path)

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["question", "gold_answers", "pred_answer", "raw_output"]

    def test_counter_resets_after_epoch(self):
        """Verify sample counter resets after on_validation_epoch_end."""
        mock_llm = _make_mock_llm()
        extractor = _make_extractor(mock_llm, self.output_dir, layer_indices=[0])

        batch = _make_triviaqa_batch()
        extractor.validation_step(batch, 0)
        assert extractor.sample_counter == 1

        extractor.on_validation_epoch_end()
        assert extractor.sample_counter == 0
        assert len(extractor.outputs) == 0
