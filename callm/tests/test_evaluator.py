"""Unit tests for Evaluator components."""

import torch
import os
from unittest.mock import Mock, patch, mock_open
from callm.models.evaluator import EvaluatorModule
from callm.data.triviaqa import EvaluatorDataModule


class TestEvaluatorDataModule:
    """Tests for EvaluatorDataModule."""

    @patch("callm.data.answers_data.get_tokenizer_for_model")
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
        mock_init_model.return_value = mock_model

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
        assert len(evaluator.outputs) == 2

        # First item (Exact match)
        assert evaluator.outputs[0]["exact_match"] is True

        # Second item (Non-exact) should have output_ids stored
        assert evaluator.outputs[1]["exact_match"] is False
        assert evaluator.outputs[1]["output_ids"] is not None
        # Verify it's a tensor
        assert torch.is_tensor(evaluator.outputs[1]["output_ids"])

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
        mock_init_model.return_value = mock_model
        mock_get_tokenizer.return_value = Mock()

        evaluator = EvaluatorModule(model_name="dummy")

        # Populate results directly
        evaluator.outputs = [
            {
                "index": 0,
                "question": "Q1",
                "gold_answers": "A1",
                "pred_answer": "P1",
                "confidence": "0.9",
                "raw_output": "P1",
                "exact_match": True,
                "output_ids": None,
            },
            {
                "index": 1,
                "question": "Q2",
                "gold_answers": "A2",
                "pred_answer": "P2",
                "confidence": "0.1",
                "raw_output": "P2",
                "exact_match": False,
                "output_ids": torch.tensor([1, 2]),
            },
        ]

        # Mock logger and trainer
        evaluator.log = Mock()
        evaluator.trainer = Mock()
        evaluator.trainer.log_dir = "/tmp"

        # Setup tokenizer decode return value
        evaluator.tokenizer.decode.return_value = "No"

        # Run epoch end
        with patch("builtins.open", mock_open()) as mock_file, patch(
            "csv.writer"
        ) as mock_writer:
            mock_csv_instance = Mock()
            mock_writer.return_value = mock_csv_instance

            evaluator.on_validation_epoch_end()

            # Check metrics logged
            assert evaluator.log.called

            # Check correctness via CSV output: header + 2 rows
            assert mock_csv_instance.writerow.call_count == 1
            rows = mock_csv_instance.writerows.call_args[0][0]
            assert len(rows) == 2

            # Item 0 is an exact match: Correct=Yes, no evaluator response
            assert rows[0][4] == "Yes"  # 5th column is Correct
            assert rows[0][5] == ""

            # Item 1 -> Correct=No (since we mocked decode="No")
            assert rows[1][4] == "No"
            assert rows[1][5] == "No"

            # We expect calls for accuracy, ece, brier, etc.
            logged_metrics = [call.args[0] for call in evaluator.log.call_args_list]
            assert "val_accuracy" in logged_metrics

            assert "val_ece" in logged_metrics
            assert "val_ecuas_0" in logged_metrics

            # Check file saved
            mock_file.assert_called()

    @patch("callm.models.evaluator.get_tokenizer_for_model")
    @patch("callm.models.evaluator.initialize_model")
    def test_validation_flushing(self, mock_init_model, mock_get_tokenizer):
        """Test evaluation results flushing mechanism."""
        # Setup mock model
        mock_model = Mock()
        mock_model.parameters.return_value = iter([])
        mock_model.config = Mock()
        mock_model.config.pad_token_id = 0
        mock_model.generate.return_value = torch.tensor([[10, 11]] * 3)
        mock_init_model.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "Yes"
        mock_get_tokenizer.return_value = mock_tokenizer

        # Initialize Evaluator with flush every 2 results
        evaluator = EvaluatorModule(
            model_name="dummy",
            flush_outputs_every_n_steps=2,
        )

        # Mock trainer
        evaluator.trainer = Mock()
        evaluator.trainer.log_dir = "tmp_eval_dir"
        evaluator.trainer.global_rank = 0
        os.makedirs("tmp_eval_dir", exist_ok=True)

        try:
            # Create batch with 3 items (enough for one flush and one leftover)
            batch = {
                "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
                "exact_match": [False, False, False],
                "question": ["q1", "q2", "q3"],
                "gold_answers": ["a1", "a2", "a3"],
                "pred_answer": ["p1", "p2", "p3"],
                "confidence": ["0.9", "0.8", "0.7"],
                "raw_output": ["r1", "r2", "r3"],
                "index": [0, 1, 2],
            }

            # Run validation step
            evaluator.validation_step(batch, 0)

            # Check flushing happened (3 items >= 2 limit)
            # outputs should be empty (or contain leftovers if we flushed exactly 2)
            # Actually, the logic flushes ALL if >= limit
            assert len(evaluator.outputs) == 0
            assert len(evaluator.flushed_output_files) == 1
            assert os.path.exists(evaluator.flushed_output_files[0])

            # Run on_validation_epoch_end
            # Mock logger to avoid errors
            evaluator.log = Mock()
            evaluator.on_validation_epoch_end()

            # Check if flushed files are cleaned up
            assert len(evaluator.flushed_output_files) == 0

            # Check if output csv was written
            assert os.path.exists(
                os.path.join("tmp_eval_dir", "evaluation_results.csv")
            )

        finally:
            import shutil

            if os.path.exists("tmp_eval_dir"):
                shutil.rmtree("tmp_eval_dir")


class TestEvaluatorModuleCharacterization:
    """Pins EvaluatorModule behaviour that the shared evaluator base must keep."""

    @staticmethod
    def _make(config=None, tokenizer=None, **kwargs):
        with (
            patch("callm.models.evaluator.initialize_model") as init_model,
            patch("callm.models.evaluator.get_tokenizer_for_model") as get_tokenizer,
        ):
            model = Mock()
            model.parameters.return_value = iter([])
            model.config = config or Mock(
                spec=["pad_token_id", "eos_token_id"], pad_token_id=0, eos_token_id=1
            )
            init_model.return_value = model
            get_tokenizer.return_value = tokenizer or Mock(pad_token_id=None)
            return EvaluatorModule(model_name="judge", **kwargs)

    @staticmethod
    def _batch(**overrides):
        batch = {
            "input_ids": torch.tensor([[5, 6], [7, 8]]),
            "attention_mask": torch.ones(2, 2, dtype=torch.long),
            "exact_match": [False, True, False],
            "question": ["q1", "q2", "q3"],
            "gold_answers": [["a"], ["b"], ["c"]],
            "pred_answer": ["x", "b", "z"],
            "confidence": [0.1, 0.2, 0.3],
            "raw_output": ["r1", "r2", "r3"],
            "index": [torch.tensor(4), torch.tensor(5), torch.tensor(6)],
        }
        batch.update(overrides)
        return batch

    def test_init_sets_pad_from_eos(self):
        config = Mock(spec=["pad_token_id", "eos_token_id"])
        config.pad_token_id, config.eos_token_id = None, 3
        assert self._make(config=config).model.config.pad_token_id == 3

    def test_forward_generate_kwargs(self):
        evaluator = self._make(max_new_tokens=7)
        ids, mask = torch.tensor([[1]]), torch.tensor([[1]])
        evaluator.forward(ids, mask)
        evaluator.model.generate.assert_called_once_with(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=7,
            do_sample=False,
            pad_token_id=0,
        )

    def test_forward_pad_falls_back_to_tokenizer(self):
        config = Mock(spec=["pad_token_id", "eos_token_id"])
        config.pad_token_id, config.eos_token_id = None, None
        evaluator = self._make(config=config, tokenizer=Mock(pad_token_id=5))
        evaluator.forward(torch.tensor([[1]]), torch.tensor([[1]]))
        assert evaluator.model.generate.call_args.kwargs["pad_token_id"] == 5

    def test_validation_step_maps_generations_to_non_exact_rows(self):
        evaluator = self._make()
        evaluator.model.generate.return_value = torch.tensor([[5, 6, 7], [7, 8, 9]])

        assert evaluator.validation_step(self._batch(), 0) == {"batch_size": 3}

        assert [o["index"] for o in evaluator.outputs] == [4, 5, 6]
        assert evaluator.outputs[0]["output_ids"].tolist() == [7]
        assert evaluator.outputs[1]["output_ids"] is None
        assert evaluator.outputs[2]["output_ids"].tolist() == [9]
        assert set(evaluator.outputs[0]) == {
            "index",
            "question",
            "gold_answers",
            "pred_answer",
            "confidence",
            "raw_output",
            "exact_match",
            "cached_correct",
            "output_ids",
        }
        assert evaluator.outputs[0]["cached_correct"] is None

    def test_validation_step_without_inputs_skips_generation(self):
        evaluator = self._make()
        evaluator.validation_step(
            self._batch(input_ids=None, attention_mask=None, exact_match=[True] * 3), 0
        )
        evaluator.model.generate.assert_not_called()
        assert all(o["output_ids"] is None for o in evaluator.outputs)

    def test_process_outputs_keeps_existing_correct(self, tmp_path):
        evaluator = self._make()
        evaluator._trainer = Mock(log_dir=str(tmp_path))
        evaluator.log = Mock()
        record = {
            "question": "q",
            "gold_answers": ["a"],
            "pred_answer": "a",
            "confidence": 0.5,
            "raw_output": "r",
        }
        # Pre-labelled rows (e.g. from resumed chunks) are not re-judged
        outputs = [
            {**record, "index": 0, "exact_match": True, "correct": False},
            {
                **record,
                "index": 1,
                "exact_match": False,
                "output_ids": None,
                "correct": True,
            },
        ]
        # As in on_validation_epoch_end, outputs is self.outputs
        evaluator.outputs = outputs
        evaluator.process_outputs(evaluator.outputs)
        assert [o["correct"] for o in outputs] == [False, True]

    def test_load_csv_and_calculate_metrics(self, tmp_path):
        from callm.models.outputs import write_evaluation_results_csv

        path = str(tmp_path / "evaluation_results.csv")
        write_evaluation_results_csv(
            [
                {
                    "question": "q1",
                    "gold_answers": ["a"],
                    "pred_answer": "a",
                    "confidence": 0.9,
                    "correct": True,
                    "raw_output": "r",
                },
                {
                    "question": "q2",
                    "gold_answers": ["b"],
                    "pred_answer": "c",
                    "confidence": 0.2,
                    "correct": False,
                    "raw_output": "r",
                },
            ],
            path,
            text_limit=None,
            description="x",
        )
        evaluator = self._make()
        evaluator._trainer = Mock(log_dir=str(tmp_path / "logs"))
        evaluator.log = Mock()

        evaluator.load_evaluation_results_from_csv(path)
        evaluator.calculate_metrics()

        assert [o["correct"] for o in evaluator.outputs] == [True, False]
        logged = {c.args[0]: float(c.args[1]) for c in evaluator.log.call_args_list}
        assert logged["val_accuracy"] == 0.5
        assert "val_ecuas_0" in logged
        # calculate_metrics does not write metrics.csv
        assert not (tmp_path / "logs" / "metrics.csv").exists()
