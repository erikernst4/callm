from unittest.mock import patch
from callm.data.untokenized_triviaQA import UntokenizedTriviaQADataModule
from callm.data.untokenized_evaluator_data import UntokenizedEvaluatorDataModule


class TestUntokenizedTriviaQA:
    @patch("callm.data.untokenized_triviaQA.load_dataset")
    def test_setup_and_collate(self, mock_load):
        # Mock dataset
        mock_load.return_value = {
            "validation": {
                "question": ["Q1", "Q2"],
                "answer": [
                    {"aliases": ["A1"], "normalized_aliases": ["A1_norm"]},
                    {"aliases": ["A2"], "normalized_aliases": []},
                ],
            }
        }

        dm = UntokenizedTriviaQADataModule(batch_size=2)

        # Test setup
        dm.setup(stage="test")
        assert len(dm.triviaQA_val) == 2
        assert "input" in dm.triviaQA_val[0]
        assert "question" in dm.triviaQA_val[0]

        # Test dataloader collate
        loader = dm.val_dataloader()
        batch = next(iter(loader))

        assert "input" in batch
        assert len(batch["input"]) == 2
        assert isinstance(batch["input"][0], str)
        assert "Q1" in batch["input"][0]


class TestUntokenizedEvaluator:
    @patch(
        "callm.data.untokenized_evaluator_data.AnswersDataModule.load_llm_outputs_from_csv"
    )
    def test_setup_and_collate(self, mock_load_csv):
        # Mock LLM outputs
        mock_load_csv.return_value = [
            {
                "question": "Q1",
                "gold_answers": ["A1", "A1_alt"],
                "pred_answer": "A1_alt",
                "confidence": "0.9",
                "raw_output": "A1_alt",
                "index": 0,
            },
            {
                "question": "Q2",
                "gold_answers": ["A2"],
                "pred_answer": "Wrong_A2",
                "confidence": "0.5",
                "raw_output": "Wrong_A2 generated",
                "index": 1,
            },
        ]

        dm = UntokenizedEvaluatorDataModule(batch_size=2)

        # Test setup
        dm.setup(stage="test")

        # Q1 is an exact match, prompt should be empty.
        # Q2 is not an exact match, it should be rendered.
        assert dm.dataset[0]["exact_match"].item() is True
        assert dm.dataset[0]["input"] == ""

        assert dm.dataset[1]["exact_match"].item() is False
        assert "Q2" in dm.dataset[1]["input"]
        assert "Wrong_A2" in dm.dataset[1]["input"]

        # Test dataloader collate
        loader = dm.val_dataloader()
        batch = next(iter(loader))

        assert "input" in batch
        assert len(batch["input"]) == 2
        assert isinstance(batch["input"][1], str)
