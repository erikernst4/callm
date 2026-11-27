"""Tests for the raw vs normalized IsTrue table cell formatting."""

import sys

import pytest
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


def test_raw_is_scored_from_llm_outputs_aligned_by_question(tmp_path):
    from scripts.compare_istrue_normalization import compare_run

    # merged Confidence was merged by position (misaligned): it must be ignored
    merged = _write(
        tmp_path / "merged.csv",
        "Question,Confidence,Correct,Is True Predicted Answer\n"
        "q1,0.11,Yes,True\nq2,0.99,No,True\n",
    )
    # IsTrue outputs are shuffled relative to the merged (base) order
    raw = _write(
        tmp_path / "raw.csv",
        "question,pred_answer,confidence\nq2,False,0.55\nq1,True,0.6\n",
    )
    normalized = _write(
        tmp_path / "norm.csv",
        "question,pred_answer,confidence\nq2,False,0.7\nq1,True,0.95\n",
    )

    rows = {r["variant"]: r for r in compare_run(merged, normalized, raw)}
    # raw P(True) = [0.6, 0.45], normalized P(True) = [0.95, 0.3]
    assert rows["raw"]["BS"] == pytest.approx(((0.6 - 1) ** 2 + 0.45**2) / 2)
    assert rows["normalized"]["BS"] == pytest.approx(((0.95 - 1) ** 2 + 0.3**2) / 2)


def _run_dir(root, *parts, text="question,pred_answer,confidence\nq1,True,0.9\n"):
    d = root.joinpath(*parts)
    d.mkdir(parents=True)
    (d / "llm_outputs.csv").write_text(text)
    return d


def test_collect_raw_from_rerun_or_paper_llm_outputs(tmp_path):
    from scripts.compare_istrue_normalization import collect

    paper, results, reruns = (
        tmp_path / "paper",
        tmp_path / "results",
        tmp_path / "reruns",
    )
    for name in ["is_true_A", "is_true_B"]:
        _run_dir(results, "mmlu", name)
        _run_dir(paper, "mmlu", name)  # paper IsTrue llm_outputs.csv
        (paper / "mmlu" / name / "merged_results.csv").write_text(
            "Question,Confidence,Correct\nq1,0.5,Yes\n"
        )
    _run_dir(reruns, "mmlu", "is_true_B")

    df = collect(results, paper, reruns)
    sources = df[df.variant == "raw"].set_index("model")["raw_source"].to_dict()
    assert sources == {"A": "paper", "B": "rerun"}
