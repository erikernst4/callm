"""Tests for the raw vs normalized IsTrue table cell formatting."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.compare_istrue_normalization import format_pair  # noqa: E402

GREEN = r"\cellcolor{green!25}"
RED = r"\cellcolor{red!25}"


def test_lower_is_better_bolds_smaller():
    assert format_pair(0.100, 0.098, higher_is_better=False) == (
        "0.100",
        r"\textbf{0.098}",
    )


def test_higher_is_better_bolds_larger():
    assert format_pair(0.80, 0.79, higher_is_better=True) == (
        r"\textbf{0.800}",
        "0.790",
    )


def test_tie_at_displayed_precision_bolds_both():
    raw, norm = format_pair(0.1481, 0.1484, higher_is_better=False)
    assert raw == r"\textbf{0.148}"
    assert norm == r"\textbf{0.148}"


def test_more_than_5_percent_better_is_green():
    # ECE 0.370 -> 0.104
    assert format_pair(0.370, 0.104, higher_is_better=False) == (
        "0.370",
        GREEN + r"\textbf{0.104}",
    )
    # AUC 0.435 -> 0.873
    assert (
        format_pair(0.435, 0.873, higher_is_better=True)[1] == GREEN + r"\textbf{0.873}"
    )


def test_more_than_5_percent_worse_is_red():
    # CE 0.586 -> 0.747
    assert format_pair(0.586, 0.747, higher_is_better=False) == (
        r"\textbf{0.586}",
        RED + "0.747",
    )
    assert format_pair(0.80, 0.70, higher_is_better=True)[1] == RED + "0.700"


def test_within_5_percent_is_not_colored():
    # 4% better, 4% worse
    assert format_pair(0.100, 0.096, higher_is_better=False)[1] == r"\textbf{0.096}"
    assert format_pair(0.100, 0.104, higher_is_better=False)[1] == "0.104"


def _write(path, text):
    path.write_text(text)
    return path


def test_compare_run_uses_raw_rerun_when_given(tmp_path):
    from scripts.compare_istrue_normalization import compare_run

    merged = _write(
        tmp_path / "merged.csv",
        "Question,Confidence,Correct,Is True Predicted Answer\n"
        "q1,0.9,Yes,True\nq2,0.2,No,False\n",
    )
    normalized = _write(
        tmp_path / "norm.csv",
        "question,pred_answer,confidence\nq1,True,0.95\nq2,False,0.7\n",
    )
    raw_rerun = _write(
        tmp_path / "raw.csv",
        "question,pred_answer,confidence\nq1,True,0.6\nq2,True,0.55\n",
    )

    rows = {r["variant"]: r for r in compare_run(merged, normalized)}
    assert rows["raw"]["raw_source"] == "paper"

    rows = {r["variant"]: r for r in compare_run(merged, normalized, raw_rerun)}
    assert rows["raw"]["raw_source"] == "rerun"
    # raw rerun P(True) = [0.6, 0.55] vs paper [0.9, 0.2]: Brier differs
    expected_bs = ((0.6 - 1) ** 2 + (0.55 - 0) ** 2) / 2
    assert abs(rows["raw"]["BS"] - expected_bs) < 1e-6
    # normalized P(True) = [0.95, 0.3]
    expected_bs = ((0.95 - 1) ** 2 + (0.3 - 0) ** 2) / 2
    assert abs(rows["normalized"]["BS"] - expected_bs) < 1e-6
