"""Tests for scripts/check_istrue_confidences.py."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.check_istrue_confidences import (  # noqa: E402
    align,
    check_file,
    find_raw_outputs,
    verdict,
)


def _merged(questions, confidences, answers=None):
    df = pd.DataFrame({"Question": questions, "Confidence": confidences})
    if answers is not None:
        df["Is True Predicted Answer"] = answers
    return df


def _raw(questions, answers, confidences):
    return pd.DataFrame(
        {"question": questions, "pred_answer": answers, "confidence": confidences}
    )


class TestAlign:
    def test_positional_when_questions_match(self):
        df, how = align(
            _merged(["q1", "q2"], [0.9, 0.2]),
            _raw(["q1", "q2"], ["True", "False"], [0.9, 0.8]),
        )
        assert how == "positional"
        assert list(df["raw_confidence"]) == [0.9, 0.8]

    def test_escaped_newlines_still_positional(self):
        df, how = align(
            _merged(["a\\nb", "q2"], [0.9, 0.2]),
            _raw(["a\nb", "q2"], ["True", "False"], [0.9, 0.8]),
        )
        assert how == "positional"

    def test_shuffled_rows_are_matched_by_question(self):
        df, how = align(
            _merged(["q1", "q2", "q3"], [0.9, 0.2, 0.5]),
            _raw(["q3", "q1", "q2"], ["True", "False", "True"], [0.5, 0.9, 0.8]),
        )
        assert how == "by question"
        assert list(df["it_answer"]) == ["False", "True", "True"]
        assert list(df["raw_confidence"]) == [0.9, 0.8, 0.5]

    def test_matching_answers_do_not_force_positional(self):
        # Same answers row by row (all True), but rows are shuffled:
        # order must never be trusted when the questions differ.
        df, how = align(
            _merged(["q1", "q2"], [0.9, 0.6], answers=["True", "True"]),
            _raw(["q2", "q1"], ["True", "True"], [0.6, 0.9]),
        )
        assert how == "by question"
        assert list(df["raw_confidence"]) == [0.9, 0.6]

    def test_shared_long_prefix_matched_by_full_question(self):
        # MMLU questions often share long openings (> 150 chars)
        prefix = "This question refers to the following information. " * 5
        q1, q2 = prefix + "Which one?", prefix + "Which other?"
        df, how = align(
            _merged([q1, q2], [0.9, 0.2]),
            _raw([q2, q1], ["False", "True"], [0.8, 0.9]),
        )
        assert how == "by question"
        assert list(df["it_answer"]) == ["True", "False"]

    def test_truncated_merged_question_matched_by_prefix(self):
        long_q = "x" * 300
        df, _ = align(
            _merged(["q1", long_q[:200] + "..."], [0.9, 0.2]),
            _raw([long_q, "q1"], ["False", "True"], [0.8, 0.9]),
        )
        assert list(df["_matched"]) == [True, True]
        assert list(df["it_answer"]) == ["True", "False"]

    def test_unmatched_rows_are_flagged(self):
        df, _ = align(
            _merged(["q1", "q2"], [0.9, 0.2]),
            _raw(["q1", "other"], ["True", "False"], [0.9, 0.8]),
        )
        assert list(df["_matched"]) == [True, False]


def _write(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def test_check_file_flipped_is_ok(tmp_path):
    merged = _write(
        tmp_path / "is_true_m" / "merged_results.csv", _merged(["q1", "q2"], [0.9, 0.2])
    )
    raw = _write(
        tmp_path / "is_true_m" / "llm_outputs.csv",
        _raw(["q2", "q1"], ["False", "True"], [0.8, 0.9]),
    )
    r = check_file(merged, raw)
    assert (r["true_raw"], r["false_flipped"]) == (1, 1)
    assert verdict(r) == "OK"


def test_check_file_unflipped_is_reported(tmp_path):
    merged = _write(
        tmp_path / "is_true_m" / "merged_results.csv", _merged(["q1", "q2"], [0.9, 0.8])
    )
    raw = _write(
        tmp_path / "is_true_m" / "llm_outputs.csv",
        _raw(["q1", "q2"], ["True", "False"], [0.9, 0.8]),
    )
    assert verdict(check_file(merged, raw)) == "NOT FLIPPED"


def test_find_raw_outputs_for_triviaqa_subsets(tmp_path):
    raw = _write(
        tmp_path / "triviaqa" / "is_true_m" / "llm_outputs.csv",
        _raw(["q"], ["True"], [0.9]),
    )
    merged = _write(
        tmp_path / "triviaqa_strict" / "is_true_m" / "merged_results.csv",
        _merged(["q"], [0.9]),
    )
    assert find_raw_outputs(merged) == raw
    assert (
        find_raw_outputs(tmp_path / "other" / "is_true_m" / "merged_results.csv")
        is None
    )
