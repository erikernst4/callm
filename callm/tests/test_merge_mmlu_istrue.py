"""Tests for scripts/merge_mmlu_istrue.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.merge_mmlu_istrue import (  # noqa: E402
    backup,
    base_dir_name,
    merge_is_true,
    main,
)


def _base():
    return pd.DataFrame(
        {
            "question": ["q1", "q2", "q3"],
            "gold_answers": ["a", "b", "c"],
            "choices": ["A. x", "A. y", "A. z"],
            "pred_answer": ["A", "C", "C"],
            "confidence": [0.9, 0.8, 0.7],
            "raw_output": ["A", "C", "C"],
        }
    )


def _is_true(order=(0, 1, 2)):
    df = pd.DataFrame(
        {
            "question": ["q1", "q2", "q3"],
            "gold_answers": ["a", "b", "c"],
            "pred_answer": ["True", "False", "True"],
            "confidence": [0.6, 0.7, 0.8],
            "raw_output": ["True", "False", "True"],
        }
    )
    return df.iloc[list(order)].reset_index(drop=True)


class TestMergeIsTrue:
    def test_aligned_by_question_not_position(self):
        # IsTrue runs are shuffled (subsample_dataset shuffles with the seed)
        out = merge_is_true(_base(), _is_true(order=(2, 0, 1)))
        assert list(out["Question"]) == ["q1", "q2", "q3"]
        np.testing.assert_allclose(out["Confidence"], [0.6, 0.3, 0.8])
        assert list(out["Is True Predicted Answer"]) == ["True", "False", "True"]

    def test_correctness_from_base_answers(self):
        out = merge_is_true(_base(), _is_true())
        assert list(out["Correct"]) == ["Yes", "No", "Yes"]
        assert list(out["Predicted Answer"]) == ["A", "C", "C"]

    def test_false_answers_are_flipped(self):
        out = merge_is_true(_base(), _is_true())
        assert out["Confidence"][1] == pytest.approx(0.3)

    def test_duplicate_questions_raise(self):
        # Shuffled runs can't tell repeated questions apart: refuse to guess
        base = _base()
        base["question"] = ["q", "q", "q3"]
        is_true = _is_true()
        is_true["question"] = ["q", "q", "q3"]
        with pytest.raises(ValueError, match="duplicate"):
            merge_is_true(base, is_true)

    def test_unmatched_question_raises(self):
        is_true = _is_true()
        is_true.loc[2, "question"] = "other"
        with pytest.raises(ValueError, match="unmatched"):
            merge_is_true(_base(), is_true)

    def test_row_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="rows"):
            merge_is_true(_base(), _is_true().iloc[:2])


def test_base_dir_name():
    assert base_dir_name("is_true_Qwen3.5-4B") == "label_prob_Qwen3.5-4B"
    assert (
        base_dir_name("is_true_GLM-4.6V-Flash-no-thinking")
        == "label_prob_zero_shot_GLM-4.6V-Flash-no-thinking"
    )


def test_backup_keeps_previous_backups(tmp_path):
    f = tmp_path / "merged_results.csv"
    old_bak = tmp_path / "merged_results.csv.bak"
    old_bak.write_text("older backup")
    assert backup(f) is None  # nothing to back up

    f.write_text("current")
    bak = backup(f)
    assert bak is not None and bak.read_text() == "current"
    assert bak != old_bak and old_bak.read_text() == "older backup"


def _write_run(root, name, df):
    (root / name).mkdir(parents=True)
    df.to_csv(root / name / "llm_outputs.csv", index=False)


def test_main_with_separate_base_dir(tmp_path, monkeypatch):
    runs, bases = tmp_path / "runs", tmp_path / "bases"
    _write_run(runs, "is_true_M", _is_true(order=(1, 2, 0)))
    _write_run(bases, "label_prob_M", _base())
    (runs / "is_true_M" / "merged_results.csv").write_text("previous")

    monkeypatch.setattr(
        sys, "argv", ["merge", "--mmlu-dir", str(runs), "--base-dir", str(bases)]
    )
    main()

    out = pd.read_csv(runs / "is_true_M" / "merged_results.csv")
    np.testing.assert_allclose(out["Confidence"], [0.6, 0.3, 0.8])
    backups = list((runs / "is_true_M").glob("merged_results.csv.bak-*"))
    assert len(backups) == 1 and backups[0].read_text() == "previous"

    metrics = pd.read_csv(runs / "is_true_M" / "metrics.csv")
    # same metrics as the paper table
    for col in [
        "ER",
        "ECE",
        "AUROC",
        "CE",
        "BS",
        "AURC",
        "ECUAS_0",
        "ECUAS_1",
        "ECUAS_128",
    ]:
        assert col in metrics.columns
    assert metrics["ER"][0] == pytest.approx(1 / 3)
