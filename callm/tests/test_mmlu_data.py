"""Unit tests for MMLU data modules."""

import pytest
from unittest.mock import patch, MagicMock
from callm.data.mmlu import MMLUDataModule, UntokenizedMMLUDataModule
from callm.prompts.mmlu import (
    CHAT_MMLU_LABEL_PROB_PROMPT,
    GCP_CHAT_MMLU_LABEL_PROB_PROMPT,
    format_choices,
    answer_index_to_letter,
)


def test_format_choices():
    choices = ["Apple", "Banana", "Cherry", "Date"]
    expected = "A. Apple\nB. Banana\nC. Cherry\nD. Date"
    assert format_choices(choices) == expected


def test_answer_index_to_letter():
    assert answer_index_to_letter(0) == "A"
    assert answer_index_to_letter(1) == "B"
    assert answer_index_to_letter(2) == "C"
    assert answer_index_to_letter(3) == "D"
    with pytest.raises(IndexError):
        answer_index_to_letter(4)


@patch("callm.data.mmlu.mmlu.load_dataset")
@patch("callm.data.mmlu.mmlu.get_tokenizer_for_model")
def test_mmlu_data_module_setup(mock_get_tokenizer, mock_load_dataset):
    """Test standard MMLUDataModule setup processing."""
    import torch

    mock_tokenizer = MagicMock()
    # Return actual tensors so dataset conversion doesn't fail on MagicMocks
    mock_tokenizer.apply_chat_template.return_value = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    mock_get_tokenizer.return_value = mock_tokenizer

    # Mock dataset using a real Dataset to avoid mock slicing issues
    from datasets import Dataset

    mock_dataset = Dataset.from_dict(
        {
            "question": ["Q1", "Q2", "Q3"],
            "choices": [
                ["a", "b", "c", "d"],
                ["1", "2", "3", "4"],
                ["w", "x", "y", "z"],
            ],
            "answer": [0, 1, 3],
        }
    )
    mock_load_dataset.return_value = {"test": mock_dataset, "validation": mock_dataset}

    # Initialize DM
    dm = MMLUDataModule(
        batch_size=2,
        model_name="dummy/model",
        prompt=CHAT_MMLU_LABEL_PROB_PROMPT,
        max_samples=2,  # Subsample to 2
        seed=42,
    )

    dm.setup("validate")

    # Assert load_dataset called correctly
    mock_load_dataset.assert_called_with("cais/mmlu", "all")

    # Assert dataset was subsampled
    assert len(dm.mmlu_val) == 2

    # Since we can't easily mock the chained dataset methods (shuffle, select, with_format)
    # perfectly, we just verify the basic setup flow completed without error.
    assert hasattr(dm, "mmlu_val")


@patch("callm.data.mmlu.untokenized_mmlu.load_dataset")
def test_untokenized_mmlu_data_module_setup(mock_load_dataset):
    """Test UntokenizedMMLUDataModule setup."""
    from datasets import Dataset

    # Creating a real mock dataset for testing the actual logic
    mock_hf_dataset = Dataset.from_dict(
        {
            "question": ["Q1", "Q2"],
            "choices": [["A", "B", "C", "D"], ["w", "x", "y", "z"]],
            "answer": [0, 2],  # A, C
        }
    )

    mock_load_dataset.return_value = {"test": mock_hf_dataset}

    # Initialize DM
    dm = UntokenizedMMLUDataModule(
        batch_size=2,
        prompt=GCP_CHAT_MMLU_LABEL_PROB_PROMPT,
        max_samples=None,
    )

    dm.setup("validate")

    assert len(dm.mmlu_val) == 2

    # First item checking
    item0 = dm.mmlu_val[0]
    assert item0["question"] == "Q1"
    assert item0["label"] == "A"

    # Check that choices are properly injected into the prompt
    input_prompt = item0["input"]
    # For GCP prompt, the prompt is a list of dicts (messages)
    assert isinstance(input_prompt, list)
    assert (
        len(input_prompt) >= 2
    )  # System + User (+ Assistant/Model context potentially)

    # Formatted choices should be in the user message
    user_msg_text = input_prompt[1]["parts"][0]["text"]
    assert "A. A" in user_msg_text
    assert "B. B" in user_msg_text
    assert "C. C" in user_msg_text
    assert "D. D" in user_msg_text

    # Second item checking
    item1 = dm.mmlu_val[1]
    assert item1["question"] == "Q2"
    assert item1["label"] == "C"
