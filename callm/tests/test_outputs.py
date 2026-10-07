"""Unit tests for the shared answer-output helpers."""

import math

import pytest

from callm.models.outputs import (
    format_confidence,
    format_gold_answers,
    format_text,
    letter_match,
    read_evaluation_results,
    write_evaluation_results_csv,
)


@pytest.mark.parametrize(
    "confidence,expected",
    [
        (0.5, "0.500000"),
        ("0.25", "0.250000"),
        (float("nan"), "nan"),
        ("nan", "nan"),
        ("", "nan"),
        (None, "nan"),
    ],
)
def test_format_confidence(confidence, expected):
    assert format_confidence(confidence) == expected


@pytest.mark.parametrize(
    "gold,expected",
    [(["a", "b"], "a|b"), ([], ""), ("AB", "AB"), (None, "")],
)
def test_format_gold_answers(gold, expected):
    assert format_gold_answers(gold) == expected


def test_format_text():
    assert format_text(None) == ""
    assert format_text("a\nb") == "a\\nb"
    assert format_text("abcdef", limit=3) == "abc..."
    assert format_text("abc", limit=3) == "abc"


@pytest.mark.parametrize(
    "pred,gold,expected",
    [
        (" b ", ["B"], True),
        ("b", ["A", "b "], True),
        ("C", "c", True),
        ("C", None, False),
        (None, [], False),
        (None, "", True),
    ],
)
def test_letter_match(pred, gold, expected):
    assert letter_match(pred, gold) is expected


def test_evaluation_results_round_trip(tmp_path):
    path = str(tmp_path / "evaluation_results.csv")
    write_evaluation_results_csv(
        [
            {
                "question": "q1",
                "gold_answers": ["A"],
                "pred_answer": "A",
                "confidence": 0.8,
                "correct": True,
                "raw_output": "r",
            },
            {
                "question": "q2",
                "gold_answers": ["B"],
                "pred_answer": "c",
                "confidence": None,
                "correct": False,
                "evaluator_response": "No",
                "raw_output": "r",
            },
        ],
        path,
        text_limit=None,
        description="Evaluation results",
    )

    outputs = read_evaluation_results(path)

    assert [o["correct"] for o in outputs] == [True, False]
    assert outputs[0]["confidence"] == 0.8
    assert math.isnan(outputs[1]["confidence"])
    assert outputs[0]["gold_answers"] == "['A']"


def test_read_missing_evaluation_results(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_evaluation_results(str(tmp_path / "missing.csv"))
