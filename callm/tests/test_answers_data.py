"""Unit tests for AnswersDataModule and IsTrueDataModule."""

import pytest
import torch
import os
import tempfile
from unittest.mock import Mock, patch, mock_open

from callm.data.answers_data import AnswersDataModule
from callm.data.triviaqa import IsTrueDataModule
from callm.prompts import CHAT_IS_TRUE_PROB_PROMPT


class TestAnswersDataModule:
    """Tests for AnswersDataModule base class."""

    def test_init_default_values(self):
        """Test default initialization values."""
        dm = AnswersDataModule()
        assert dm.llm_outputs_path is None
        assert dm.model_name == "google/flan-t5-base"
        assert dm.batch_size == 1
        assert dm.num_workers == 8
        assert dm.max_length is None
        assert dm.log_dir is None
        assert dm.resume_from is None
        assert dm.tokenizer is None
        assert dm.skip_indices == set()

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        dm = AnswersDataModule(
            llm_outputs_path="test.csv",
            model_name="custom-model",
            batch_size=16,
            num_workers=4,
            max_length=256,
            log_dir="/tmp/logs",
            resume_from="/tmp/resume",
        )
        assert dm.llm_outputs_path == "test.csv"
        assert dm.model_name == "custom-model"
        assert dm.batch_size == 16
        assert dm.num_workers == 4
        assert dm.max_length == 256
        assert dm.log_dir == "/tmp/logs"
        assert dm.resume_from == "/tmp/resume"

    @patch("callm.data.answers_data.get_tokenizer_for_model")
    def test_setup_tokenizer(self, mock_get_tokenizer):
        """Test tokenizer setup."""
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 1024
        mock_get_tokenizer.return_value = mock_tokenizer

        dm = AnswersDataModule(model_name="test-model")
        dm._setup_tokenizer()

        mock_get_tokenizer.assert_called_once_with("test-model")
        assert dm.tokenizer == mock_tokenizer
        assert dm.max_length == 1024

    @patch("callm.data.answers_data.get_tokenizer_for_model")
    def test_setup_tokenizer_uses_default_max_length(self, mock_get_tokenizer):
        """Test that default max_length is used when tokenizer has huge value."""
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 1e12  # Huge value
        mock_get_tokenizer.return_value = mock_tokenizer

        dm = AnswersDataModule()
        dm._setup_tokenizer()

        assert dm.max_length == 512  # DEFAULT_MAX_LENGTH

    @patch("callm.data.answers_data.get_tokenizer_for_model")
    def test_setup_tokenizer_preserves_explicit_max_length(self, mock_get_tokenizer):
        """Test that explicit max_length is preserved."""
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 1024
        mock_get_tokenizer.return_value = mock_tokenizer

        dm = AnswersDataModule(max_length=256)
        dm._setup_tokenizer()

        assert dm.max_length == 256  # Should keep explicit value

    def test_resolve_llm_outputs_path_with_path(self):
        """Test path resolution when path is provided."""
        dm = AnswersDataModule(llm_outputs_path="test.csv")
        result = dm._resolve_llm_outputs_path()
        assert result == "test.csv"

    @patch("callm.data.answers_data.get_last_llm_outputs_path")
    def test_resolve_llm_outputs_path_from_log_dir(self, mock_get_path):
        """Test path resolution from log_dir."""
        mock_get_path.return_value = "/tmp/logs/llm_outputs.csv"

        dm = AnswersDataModule(log_dir="/tmp/logs")
        result = dm._resolve_llm_outputs_path()

        mock_get_path.assert_called_once_with("/tmp/logs")
        assert result == "/tmp/logs/llm_outputs.csv"

    def test_resolve_llm_outputs_path_raises_when_none(self):
        """Test that ValueError is raised when path cannot be resolved."""
        dm = AnswersDataModule()
        with pytest.raises(ValueError, match="llm_outputs_path must be provided"):
            dm._resolve_llm_outputs_path()

    def test_get_skip_indices_no_resume(self):
        """Test skip indices when not resuming."""
        dm = AnswersDataModule()
        indices = dm._get_skip_indices()
        assert indices == set()

    def test_get_skip_indices_invalid_path(self):
        """Test skip indices with invalid resume path."""
        dm = AnswersDataModule(resume_from="/nonexistent/path")
        with pytest.raises(ValueError, match="does not exist"):
            dm._get_skip_indices()

    def test_get_skip_indices_with_temp_files(self):
        """Test skip indices loading from temp files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create temp file with indices
            temp_file = os.path.join(tmpdir, "temp_eval_results_rank0_0.pt")
            torch.save([{"index": 0}, {"index": 2}, {"index": 5}], temp_file)

            dm = AnswersDataModule(resume_from=tmpdir)
            indices = dm._get_skip_indices()

            assert indices == {0, 2, 5}

    def test_load_llm_outputs_from_csv(self):
        """Test CSV loading."""
        csv_content = (
            "question,gold_answers,pred_answer,confidence,raw_output\n"
            "Q1,A1|a1,A1,0.9,raw1\n"
            "Q2,B1|B2,B1,0.8,raw2\n"
        )

        with patch("builtins.open", mock_open(read_data=csv_content)):
            dm = AnswersDataModule(llm_outputs_path="dummy.csv")
            rows = dm.load_llm_outputs_from_csv()

        assert len(rows) == 2
        assert rows[0]["question"] == "Q1"
        assert rows[0]["gold_answers"] == ["a1", "a1"]  # lowercase and split
        assert rows[0]["pred_answer"] == "A1"
        assert rows[0]["confidence"] == "0.9"
        assert rows[0]["raw_output"] == "raw1"
        assert rows[0]["index"] == 0

    def test_load_llm_outputs_from_csv_with_skip(self):
        """Test CSV loading with skip indices."""
        csv_content = (
            "question,gold_answers,pred_answer,confidence,raw_output\n"
            "Q1,A1,A1,0.9,raw1\n"
            "Q2,B1,B1,0.8,raw2\n"
            "Q3,C1,C1,0.7,raw3\n"
        )

        with patch("builtins.open", mock_open(read_data=csv_content)):
            dm = AnswersDataModule(llm_outputs_path="dummy.csv")
            dm.skip_indices = {1}  # Skip second row
            # Bypass _get_skip_indices by setting directly
            with patch.object(dm, "_get_skip_indices", return_value={1}):
                rows = dm.load_llm_outputs_from_csv()

        assert len(rows) == 2
        assert rows[0]["question"] == "Q1"
        assert rows[1]["question"] == "Q3"

    def test_load_llm_outputs_from_csv_with_choices(self):
        """Test CSV loading with choices column."""
        csv_content = (
            "question,gold_answers,pred_answer,confidence,raw_output,choices\n"
            "Q1,A1,A1,0.9,raw1,Choice A|Choice B\n"
            "Q2,B1,B1,0.8,raw2,\n"
        )

        with patch("builtins.open", mock_open(read_data=csv_content)):
            dm = AnswersDataModule(llm_outputs_path="dummy.csv")
            rows = dm.load_llm_outputs_from_csv()

        assert len(rows) == 2
        assert "choices" in rows[0]
        assert rows[0]["choices"] == "Choice A|Choice B"
        assert "choices" in rows[1]
        assert rows[1]["choices"] == ""


class TestIsTrueDataModule:
    """Tests for IsTrueDataModule."""

    @patch("callm.data.answers_data.get_tokenizer_for_model")
    def test_setup_creates_is_true_prompts(self, mock_get_tokenizer):
        """Test that setup creates IS_TRUE prompts correctly."""
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 512
        mock_tokenizer.side_effect = lambda x, return_tensors=None, **kwargs: {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.apply_chat_template = Mock(
            side_effect=lambda x, return_tensors=None, **kwargs: (
                torch.tensor([[1, 2, 3]])
                if not kwargs.get("return_dict")
                else {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                }
            )
        )
        mock_get_tokenizer.return_value = mock_tokenizer

        csv_content = (
            "question,gold_answers,pred_answer,confidence,raw_output\n"
            "What is 2+2?,4,4,0.9,raw1\n"
            "Capital of France?,Paris,Paris,0.8,raw2\n"
        )

        with patch("builtins.open", mock_open(read_data=csv_content)):
            dm = IsTrueDataModule(
                llm_outputs_path="dummy.csv",
                model_name="test-model",
                batch_size=2,
            )
            dm.setup()

        assert len(dm.dataset) == 2
        assert dm.dataset[0]["question"] == "What is 2+2?"
        assert dm.dataset[0]["pred_answer"] == "4"

    @patch("callm.data.answers_data.get_tokenizer_for_model")
    def test_val_dataloader_collation(self, mock_get_tokenizer):
        """Test dataloader collation."""
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 512
        mock_tokenizer.side_effect = lambda x, return_tensors=None, **kwargs: {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.apply_chat_template = Mock(
            side_effect=lambda x, return_tensors=None, **kwargs: (
                torch.tensor([[1, 2, 3]])
                if not kwargs.get("return_dict")
                else {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                }
            )
        )
        mock_get_tokenizer.return_value = mock_tokenizer

        csv_content = (
            "question,gold_answers,pred_answer,confidence,raw_output\n"
            "Q1,A1,A1,0.9,raw1\n"
            "Q2,A2,A2,0.8,raw2\n"
        )

        with patch("builtins.open", mock_open(read_data=csv_content)):
            dm = IsTrueDataModule(
                llm_outputs_path="dummy.csv",
                model_name="test-model",
                batch_size=2,
                num_workers=0,
            )
            dm.setup()

        dataloader = dm.val_dataloader()
        batch = next(iter(dataloader))

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].shape[0] == 2
        assert len(batch["question"]) == 2
        assert len(batch["pred_answer"]) == 2

    def test_inherits_from_answers_data_module(self):
        """Test that IsTrueDataModule inherits from AnswersDataModule."""
        dm = IsTrueDataModule()
        assert isinstance(dm, AnswersDataModule)

    @patch("callm.data.answers_data.get_tokenizer_for_model")
    def test_setup_unsupported_chat_template_raises_error(self, mock_get_tokenizer):
        """Test that setup raises AttributeError when tokenizer lacks apply_chat_template but ChatPrompt is used."""
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 512
        # Mocking a tokenizer that doesn't have apply_chat_template
        del mock_tokenizer.apply_chat_template
        mock_get_tokenizer.return_value = mock_tokenizer

        csv_content = (
            "question,gold_answers,pred_answer,confidence,raw_output\n"
            "Q1,A1,A1,0.9,raw1\n"
        )

        with patch("builtins.open", mock_open(read_data=csv_content)):
            dm = IsTrueDataModule(
                llm_outputs_path="dummy.csv",
                model_name="unsupported-model",
                prompt=CHAT_IS_TRUE_PROB_PROMPT,  # Explicitly use a ChatPrompt
            )
            # Setup should fail when it tries to call apply_chat_template
            with pytest.raises(AttributeError):
                dm.setup()
