from unittest.mock import patch
from callm.data.untokenized_is_true_data import UntokenizedIsTrueDataModule


class TestUntokenizedIsTrueData:
    @patch(
        "callm.data.untokenized_is_true_data.AnswersDataModule.load_llm_outputs_from_csv"
    )
    def test_setup_and_collate(self, mock_load_csv):
        # Mock LLM outputs
        mock_load_csv.return_value = [
            {
                "question": "Is the sky blue?",
                "gold_answers": ["yes", "true"],
                "pred_answer": "Yes",
                "confidence": "0.95",
                "raw_output": "The sky is blue.",
                "index": 0,
            },
            {
                "question": "Is the earth flat?",
                "gold_answers": ["no", "false"],
                "pred_answer": "False",
                "confidence": "0.99",
                "raw_output": "The earth is a sphere.",
                "index": 1,
            },
        ]

        dm = UntokenizedIsTrueDataModule(batch_size=2)

        # Test setup
        dm.setup(stage="val")

        assert len(dm.dataset) == 2
        assert "input" in dm.dataset.features
        assert "question" in dm.dataset.features
        assert "label" in dm.dataset.features

        # Verify prompt construction
        prompt_str = str(dm.dataset[0]["input"])
        assert "Is the sky blue?" in prompt_str
        assert "Yes" in prompt_str
        assert "Is the proposed answer correct?" in prompt_str

        # Test dataloader collate
        loader = dm.val_dataloader()
        batch = next(iter(loader))

        assert "input" in batch
        assert len(batch["input"]) == 2
        assert isinstance(batch["input"][0], list)

        assert "question" in batch
        assert batch["question"][0] == "Is the sky blue?"

        assert "label" in batch
        assert batch["label"][0] == ["yes", "true"]

        assert "pred_answer" in batch
        assert batch["pred_answer"][0] == "Yes"

        assert "confidence" in batch
        assert batch["confidence"][0] == "0.95"

        assert "index" in batch
        assert batch["index"][0] == 0

    def test_setup_tokenizer_override(self):
        dm = UntokenizedIsTrueDataModule()
        dm._setup_tokenizer()
        assert dm.tokenizer is None
