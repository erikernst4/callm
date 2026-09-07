import csv
import pytest
from unittest.mock import MagicMock, patch

from callm.data.triviaqa.untokenized_evaluator_data import (
    UntokenizedEvaluatorDataModule,
    load_previous_eval_cache,
    lookup_cached_evaluation,
)
from callm.models.gcp_evaluator import GCPEvaluatorModule


@pytest.fixture
def sample_llm_csv(tmp_path):
    csv_path = tmp_path / "llm_outputs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["question", "gold_answers", "pred_answer", "confidence", "raw_output"]
        )
        # Row 0: Exact match with gold
        writer.writerow(
            [
                "What is the capital of France?",
                "paris|city of light",
                "Paris",
                "0.95",
                "Paris is the capital.",
            ]
        )
        # Row 1: Not exact match, but in previous eval cache
        writer.writerow(
            [
                "Who wrote 1984?",
                "george orwell|eric blair",
                "The book 1984 was written by George Orwell.",
                "0.90",
                "George Orwell wrote it.",
            ]
        )
        # Row 2: Not exact match, not in previous eval cache -> needs API call
        writer.writerow(
            [
                "What is the speed of light?",
                "299792458 m/s|300000 km/s",
                "About 300,000 km/s in vacuum.",
                "0.85",
                "Light travels fast.",
            ]
        )
    return str(csv_path)


@pytest.fixture
def sample_prev_eval_csv(tmp_path):
    csv_path = tmp_path / "prev_evaluation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Question",
                "Gold Answers",
                "Predicted Answer",
                "Confidence",
                "Correct",
                "Evaluator Response",
                "Raw Output",
            ]
        )
        writer.writerow(
            [
                "Who wrote 1984?",
                "['george orwell']",
                "The book 1984 was written by George Orwell.",
                "0.90",
                "Yes",
                "Yes, correct.",
                "George Orwell wrote it.",
            ]
        )
    return str(csv_path)


def test_load_previous_eval_cache(sample_prev_eval_csv):
    exact_lookup, prefix_lookup = load_previous_eval_cache(sample_prev_eval_csv)
    assert len(exact_lookup) == 1

    res = lookup_cached_evaluation(
        "Who wrote 1984?",
        "The book 1984 was written by George Orwell.",
        exact_lookup,
        prefix_lookup,
    )
    assert res is not None
    assert res["correct"] is True
    assert res["evaluator_response"] == "Yes, correct."

    # Non-existent sample
    res_none = lookup_cached_evaluation(
        "What is the speed of light?",
        "About 300,000 km/s in vacuum.",
        exact_lookup,
        prefix_lookup,
    )
    assert res_none is None


def test_untokenized_evaluator_incremental_setup(sample_llm_csv, sample_prev_eval_csv):
    dm = UntokenizedEvaluatorDataModule(
        llm_outputs_path=sample_llm_csv,
        previous_eval_csv=sample_prev_eval_csv,
        batch_size=3,
    )
    dm.setup()

    assert len(dm.dataset) == 3

    # Row 0: Exact match
    item0 = dm.dataset[0]
    assert bool(item0["exact_match"]) is True
    assert bool(item0["cached_correct"]) is True
    assert item0["input"] == ""  # No prompt

    # Row 1: Cached evaluation
    item1 = dm.dataset[1]
    assert bool(item1["exact_match"]) is False
    assert bool(item1["cached_correct"]) is True
    assert item1["input"] == ""  # No prompt

    # Row 2: Needs API call
    item2 = dm.dataset[2]
    assert bool(item2["exact_match"]) is False
    # None becomes False or None when formatted depending on format, check raw/bool
    assert item2["input"] != ""  # Has prompt for Gemini


@patch("callm.models.gcp_evaluator.genai.Client")
def test_gcp_evaluator_validation_step_skips_cached(
    mock_client_cls, sample_llm_csv, sample_prev_eval_csv, tmp_path
):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    # Mock response for the 1 sample that needs API call
    mock_resp = MagicMock()
    mock_resp.text = "Yes, about 300,000 km/s is correct."
    mock_client.models.generate_content.return_value = mock_resp

    dm = UntokenizedEvaluatorDataModule(
        llm_outputs_path=sample_llm_csv,
        previous_eval_csv=sample_prev_eval_csv,
        batch_size=3,
    )
    dm.setup()
    loader = dm.val_dataloader()
    batch = next(iter(loader))

    evaluator = GCPEvaluatorModule(model_name="gemini-3-flash-preview")
    evaluator.validation_step(batch, batch_idx=0)

    # Verify generate_content was called EXACTLY ONCE (only for row 2)
    assert mock_client.models.generate_content.call_count == 1

    # Verify outputs
    assert len(evaluator.outputs) == 3
    # Row 0
    assert bool(evaluator.outputs[0]["exact_match"]) is True
    # Row 1
    assert bool(evaluator.outputs[1]["cached_correct"]) is True
    # Row 2
    assert (
        evaluator.outputs[2]["evaluator_response_raw"]
        == "Yes, about 300,000 km/s is correct."
    )
