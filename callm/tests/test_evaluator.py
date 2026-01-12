"""Unit tests for Evaluator components."""

import torch
from unittest.mock import Mock, patch, mock_open
from callm.models.evaluator import EvaluatorModule
from callm.data.evaluator_data import EvaluatorDataModule


class TestEvaluatorDataModule:
    """Tests for EvaluatorDataModule."""

    @patch("callm.data.evaluator_data.get_tokenizer_for_model")
    def test_setup_reads_csv_and_tokenizes(self, mock_get_tokenizer):
        """Test that setup reads CSV and processes data correctly."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 512  # Add this attribute
        mock_tokenizer.side_effect = lambda x, return_tensors=None, **kwargs: {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_get_tokenizer.return_value = mock_tokenizer

        # Mock CSV data
        csv_content = (
            "question,gold_answers,pred_answer,confidence,raw_output\n"
            "Q1,A1|a1,A1,0.9,A1\n"  # Exact match
            "Q2,B1,B2,0.8,B2\n"  # Non-exact match
        )

        with patch("builtins.open", mock_open(read_data=csv_content)):
            dm = EvaluatorDataModule(
                llm_outputs_path="dummy.csv", model_name="dummy-model", batch_size=2
            )
            dm.setup()

        # Check dataset size
        assert len(dm.dataset) == 2

        # Check exact matches
        assert dm.dataset[0]["exact_match"]  # A1 in ['A1']|['a1']

        assert dm.dataset[0]["data"] is None

        # Check non-exact matches
        assert not dm.dataset[1]["exact_match"]
        assert dm.dataset[1]["data"] is not None  # Should be tokenized

    def test_collate_fn(self):
        """Test dataloader collation."""
        dm = EvaluatorDataModule("dummy.csv")

        # Create dummy batch items
        batch = [
            {
                "data": None,
                "exact_match": True,
                "question": "Q1",
                "gold_answers": "A1",
                "pred_answer": "A1",
                "confidence": "0.9",
                "raw_output": "A1",
                "index": 0,
            },
            {
                "data": {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                },
                "exact_match": False,
                "question": "Q2",
                "gold_answers": "B1",
                "pred_answer": "B2",
                "confidence": "0.8",
                "raw_output": "B2",
                "index": 1,
            },
        ]

        dm.dataset = batch  # Mock dataset behavior
        dataloader = dm.val_dataloader()
        collate_fn = dataloader.collate_fn

        collated = collate_fn(batch)

        # Check structure
        assert collated["input_ids"] is not None
        assert collated["input_ids"].shape[0] == 1  # Only 1 non-exact match
        assert len(collated["exact_match"]) == 2
        assert collated["exact_match"] == [True, False]


class TestEvaluatorModule:
    """Tests for EvaluatorModule."""

    @patch("callm.models.evaluator.get_tokenizer_for_model")
    @patch("callm.models.evaluator.initialize_model")
    def test_validation_step(self, mock_init_model, mock_get_tokenizer):
        """Test validation step processing."""
        # Setup mocks
        mock_model = Mock()
        mock_model.parameters.return_value = iter([])  # Fix: Make parameters iterable
        mock_model.config = Mock()
        mock_model.config.pad_token_id = 0
        mock_model.generate.return_value = torch.tensor([[1, 2]])  # Dummy output
        mock_init_model.return_value = (mock_model, True)  # is_seq2seq=True

        mock_tokenizer = Mock()
        # Decode "Yes" for correctness
        mock_tokenizer.decode.return_value = "Yes"
        mock_tokenizer.pad_token_id = 0
        mock_get_tokenizer.return_value = mock_tokenizer

        evaluator = EvaluatorModule(model_name="dummy")

        # Create batch
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "exact_match": [True, False],
            "question": ["Q1", "Q2"],
            "gold_answers": ["A1", "B1"],
            "pred_answer": ["A1", "B2"],
            "confidence": ["0.9", "0.8"],
            "raw_output": ["A1", "B2"],
            "index": [0, 1],
        }

        evaluator.validation_step(batch, 0)

        # Check results
        assert len(evaluator.evaluation_results) == 2

        # First item (Exact match) should be True without generation
        assert evaluator.evaluation_results[0]["correct"] is True

        # Second item (Non-exact) should be True because decode returned "Yes"
        assert evaluator.evaluation_results[1]["correct"] is True

        # Verify generate called only once (for the non-exact item)
        assert mock_model.generate.call_count == 1

    @patch("callm.models.evaluator.get_tokenizer_for_model")
    @patch("callm.models.evaluator.initialize_model")
    def test_on_validation_epoch_end_saves_results(
        self, mock_init_model, mock_get_tokenizer
    ):
        """Test metrics calculation and file saving."""
        # Setup basic mocks (evaluator init)
        mock_model = Mock()
        mock_model.parameters.return_value = iter([])
        mock_model.config = Mock()
        mock_model.config.pad_token_id = 0
        mock_init_model.return_value = (mock_model, True)
        mock_get_tokenizer.return_value = Mock()

        evaluator = EvaluatorModule(model_name="dummy")

        # Populate results directly
        evaluator.evaluation_results = [
            {
                "index": 0,
                "question": "Q1",
                "gold_answers": "A1",
                "pred_answer": "P1",
                "confidence": "0.9",
                "raw_output": "P1",
                "correct": True,
            },
            {
                "index": 1,
                "question": "Q2",
                "gold_answers": "A2",
                "pred_answer": "P2",
                "confidence": "0.1",  # Calibrated wrong
                "raw_output": "P2",
                "correct": False,
            },
        ]

        # Mock logger and trainer
        evaluator.log = Mock()
        evaluator.trainer = Mock()
        evaluator.trainer.log_dir = "/tmp"

        # Run epoch end
        with patch("builtins.open", mock_open()) as mock_file:
            evaluator.on_validation_epoch_end()

            # Check metrics logged
            assert evaluator.log.called
            # We expect calls for accuracy, ece, brier, etc.
            logged_metrics = [call.args[0] for call in evaluator.log.call_args_list]
            assert "val_accuracy" in logged_metrics
            assert "val_ece" in logged_metrics

            # Check file saved
            mock_file.assert_called()
