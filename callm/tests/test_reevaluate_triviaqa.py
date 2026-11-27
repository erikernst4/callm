"""
Tests for the TriviaQA re-evaluation script.

Tests the core logic: loading labels, erasing questions, re-running exact match,
and applying human overrides.
"""

import csv
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.reevaluate_triviaqa import (
    load_labeling_csv,
    parse_gold_answers,
    reevaluate_csv,
    find_experiment_csvs,
    load_subset_questions,
    add_truncated_questions,
    load_subset_gold,
)


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def labeling_csv(tmp_path):
    """Create a sample labeling CSV."""
    csv_path = tmp_path / "labeling.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "question",
                "gold answers",
                "predicted answer",
                "correct_evaluator",
                "correct_human",
                "should_change_gold_answers",
                "erase_question",
                "system",
            ]
        )
        # Question to erase
        writer.writerow(
            [
                "Bad question?",
                "['answer']",
                "wrong",
                "No",
                "",
                "",
                "Yes",
                "['system-A']",
            ]
        )
        # Human says correct (evaluator said No)
        writer.writerow(
            [
                "Good question?",
                "['gold1', 'gold2']",
                "Correct Answer",
                "No",
                "Yes",
                "Yes",
                "",
                "['system-A']",
            ]
        )
        # Human says incorrect
        writer.writerow(
            [
                "Another question?",
                "['gold3']",
                "Wrong Answer",
                "No",
                "No",
                "",
                "",
                "['system-A']",
            ]
        )
        # No human label (only evaluator)
        writer.writerow(
            [
                "Unlabeled question?",
                "['gold4']",
                "Some Answer",
                "No",
                "",
                "",
                "",
                "['system-A']",
            ]
        )
    return csv_path


@pytest.fixture
def evaluation_csv(tmp_path):
    """Create a sample evaluation results CSV."""
    csv_path = tmp_path / "evaluation_results.csv"
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
        # Row to be erased
        writer.writerow(["Bad question?", "['answer']", "wrong", "0.9", "No", "", ""])
        # Row where human overrides to Yes
        writer.writerow(
            [
                "Good question?",
                "['gold1', 'gold2']",
                "Correct Answer",
                "0.8",
                "No",
                "",
                "",
            ]
        )
        # Row where human says No (should stay No)
        writer.writerow(
            [
                "Another question?",
                "['gold3']",
                "Wrong Answer",
                "0.7",
                "No",
                "",
                "",
            ]
        )
        # Row with no human label (should keep evaluator result)
        writer.writerow(
            [
                "Unlabeled question?",
                "['gold4']",
                "Some Answer",
                "0.6",
                "No",
                "",
                "",
            ]
        )
        # Row that already has Yes and no human label
        writer.writerow(
            [
                "Already correct?",
                "['answer5']",
                "answer5",
                "0.95",
                "Yes",
                "",
                "",
            ]
        )
    return csv_path


# ── Tests for load_labeling_csv ────────────────────────────────────────


class TestLoadLabelingCsv:
    def test_erase_questions(self, labeling_csv):
        erase_questions, _, _ = load_labeling_csv(labeling_csv)
        assert "Bad question?" in erase_questions
        assert len(erase_questions) == 1

    def test_human_corrections(self, labeling_csv):
        _, corrections, _ = load_labeling_csv(labeling_csv)
        assert ("Good question?", "Correct Answer") in corrections
        assert corrections[("Good question?", "Correct Answer")] == "Yes"
        assert ("Another question?", "Wrong Answer") in corrections
        assert corrections[("Another question?", "Wrong Answer")] == "No"

    def test_unlabeled_not_in_corrections(self, labeling_csv):
        _, corrections, _ = load_labeling_csv(labeling_csv)
        assert ("Unlabeled question?", "Some Answer") not in corrections

    def test_empty_csv(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "question",
                    "gold answers",
                    "predicted answer",
                    "correct_evaluator",
                    "correct_human",
                    "should_change_gold_answers",
                    "erase_question",
                    "system",
                ]
            )
        erase, corrections, _ = load_labeling_csv(csv_path)
        assert len(erase) == 0
        assert len(corrections) == 0


# ── Tests for parse_gold_answers ──────────────────────────────────────


class TestParseGoldAnswers:
    def test_list_string(self):
        assert parse_gold_answers("['a', 'b', 'c']") == ["a", "b", "c"]

    def test_single_element(self):
        assert parse_gold_answers("['only']") == ["only"]

    def test_fallback_string(self):
        assert parse_gold_answers("not a list") == ["not a list"]

    def test_empty_list(self):
        assert parse_gold_answers("[]") == []


# ── Tests for reevaluate_csv ──────────────────────────────────────────


class TestReevaluateCsv:
    def test_erased_rows_removed(self, evaluation_csv, labeling_csv, tmp_path):
        erase_questions, human_corrections, _ = load_labeling_csv(labeling_csv)
        out_path = tmp_path / "output.csv"
        stats = reevaluate_csv(
            evaluation_csv, out_path, erase_questions, human_corrections
        )

        assert stats["erased_rows"] == 1
        assert stats["original_rows"] == 5
        assert stats["final_rows"] == 4

        # Verify erased question is not in output
        with open(out_path) as f:
            reader = csv.DictReader(f)
            questions = [row["Question"] for row in reader]
        assert "Bad question?" not in questions

    def test_human_override_yes(self, evaluation_csv, labeling_csv, tmp_path):
        erase_questions, human_corrections, _ = load_labeling_csv(labeling_csv)
        out_path = tmp_path / "output.csv"
        stats = reevaluate_csv(
            evaluation_csv, out_path, erase_questions, human_corrections
        )

        assert stats["human_override_fixes"] >= 1

        # Verify the correction
        with open(out_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Question"] == "Good question?":
                    assert row["Correct"] == "Yes"

    def test_human_override_no_stays_no(self, evaluation_csv, labeling_csv, tmp_path):
        erase_questions, human_corrections, _ = load_labeling_csv(labeling_csv)
        out_path = tmp_path / "output.csv"
        reevaluate_csv(evaluation_csv, out_path, erase_questions, human_corrections)

        with open(out_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Question"] == "Another question?":
                    assert row["Correct"] == "No"

    def test_unlabeled_rows_unchanged(self, evaluation_csv, labeling_csv, tmp_path):
        erase_questions, human_corrections, _ = load_labeling_csv(labeling_csv)
        out_path = tmp_path / "output.csv"
        reevaluate_csv(evaluation_csv, out_path, erase_questions, human_corrections)

        with open(out_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Question"] == "Unlabeled question?":
                    assert row["Correct"] == "No"  # Unchanged

    def test_already_correct_unchanged(self, evaluation_csv, labeling_csv, tmp_path):
        erase_questions, human_corrections, _ = load_labeling_csv(labeling_csv)
        out_path = tmp_path / "output.csv"
        reevaluate_csv(evaluation_csv, out_path, erase_questions, human_corrections)

        with open(out_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Question"] == "Already correct?":
                    assert row["Correct"] == "Yes"

    def test_all_columns_preserved(self, evaluation_csv, labeling_csv, tmp_path):
        erase_questions, human_corrections, _ = load_labeling_csv(labeling_csv)
        out_path = tmp_path / "output.csv"
        reevaluate_csv(evaluation_csv, out_path, erase_questions, human_corrections)

        with open(out_path) as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == {
                "Question",
                "Gold Answers",
                "Predicted Answer",
                "Confidence",
                "Correct",
                "Evaluator Response",
                "Raw Output",
            }


class TestExactMatchReevaluation:
    """Test that improved exact match (parenthetical removal) fixes correctness."""

    def test_parenthetical_fix(self, tmp_path):
        """Parenthetical overdescription in pred should be auto-fixed."""
        csv_path = tmp_path / "eval.csv"
        with open(csv_path, "w", newline="") as f:
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
            # Giglio case: parenthetical overdescription
            writer.writerow(
                [
                    "What island?",
                    "['giglio (disambiguation)', 'giglio disambiguation', 'giglio']",
                    "Giglio (Isola del Giglio)",
                    "0.8",
                    "No",
                    "",
                    "",
                ]
            )
            # Imola case: ** + parenthetical + dot
            writer.writerow(
                [
                    "What track?",
                    "['forum cornelii', 'imola', 'rocca sforzesca']",
                    "Imola** (Autodromo Enzo e Dino Ferrari).",
                    "0.7",
                    "No",
                    "",
                    "",
                ]
            )
            # No change needed (already Yes)
            writer.writerow(
                [
                    "Normal?",
                    "['answer']",
                    "answer",
                    "0.9",
                    "Yes",
                    "",
                    "",
                ]
            )
            # Should stay No (truly wrong answer)
            writer.writerow(
                [
                    "Wrong?",
                    "['correct']",
                    "totally wrong",
                    "0.5",
                    "No",
                    "",
                    "",
                ]
            )

        out_path = tmp_path / "output.csv"
        stats = reevaluate_csv(csv_path, out_path, set(), {})

        assert stats["exact_match_fixes"] == 2

        with open(out_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Giglio → Yes
        assert rows[0]["Predicted Answer"] == "Giglio (Isola del Giglio)"
        assert rows[0]["Correct"] == "Yes"

        # Imola → Yes
        assert rows[1]["Predicted Answer"] == "Imola** (Autodromo Enzo e Dino Ferrari)."
        assert rows[1]["Correct"] == "Yes"

        # Already correct stays Yes
        assert rows[2]["Correct"] == "Yes"

        # Wrong stays No
        assert rows[3]["Correct"] == "No"

    def test_formatting_fix(self, tmp_path):
        """Test ** and quote formatting fixes via exact match."""
        csv_path = tmp_path / "eval.csv"
        with open(csv_path, "w", newline="") as f:
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
            # ** removal
            writer.writerow(
                [
                    "Q1?",
                    "['nicaragua']",
                    "Nicaragua**.",
                    "0.8",
                    "No",
                    "",
                    "",
                ]
            )
            # Quote removal
            writer.writerow(
                [
                    "Q2?",
                    "['three men and a baby']",
                    '"Three Men and a Baby"',
                    "0.7",
                    "No",
                    "",
                    "",
                ]
            )

        out_path = tmp_path / "output.csv"
        stats = reevaluate_csv(csv_path, out_path, set(), {})

        assert stats["exact_match_fixes"] == 2

        with open(out_path) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["Correct"] == "Yes"
        assert rows[1]["Correct"] == "Yes"

    def test_human_overrides_exact_match(self, tmp_path):
        """Human label should take precedence over exact match."""
        csv_path = tmp_path / "eval.csv"
        with open(csv_path, "w", newline="") as f:
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
            # Exact match would say Yes, but human says No
            writer.writerow(
                [
                    "Tricky question?",
                    "['answer']",
                    "answer",
                    "0.9",
                    "Yes",
                    "",
                    "",
                ]
            )

        human_corrections = {("Tricky question?", "answer"): "No"}
        out_path = tmp_path / "output.csv"
        stats = reevaluate_csv(csv_path, out_path, set(), human_corrections)

        assert stats["human_override_to_no"] == 1

        with open(out_path) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["Correct"] == "No"

    def test_erase_plus_corrections_combined(self, tmp_path):
        """Test erasing questions and applying corrections in the same run."""
        csv_path = tmp_path / "eval.csv"
        with open(csv_path, "w", newline="") as f:
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
            # To be erased
            writer.writerow(["Erase me?", "['x']", "y", "0.5", "No", "", ""])
            # To be corrected via human override
            writer.writerow(["Fix me?", "['x']", "almost x", "0.8", "No", "", ""])
            # To be corrected via exact match
            writer.writerow(
                [
                    "Match me?",
                    "['giglio']",
                    "Giglio (Island)",
                    "0.7",
                    "No",
                    "",
                    "",
                ]
            )
            # Normal row, no changes
            writer.writerow(["Keep me?", "['z']", "z", "0.9", "Yes", "", ""])

        erase_questions = {"Erase me?"}
        human_corrections = {("Fix me?", "almost x"): "Yes"}
        out_path = tmp_path / "output.csv"
        stats = reevaluate_csv(csv_path, out_path, erase_questions, human_corrections)

        assert stats["original_rows"] == 4
        assert stats["erased_rows"] == 1
        assert stats["exact_match_fixes"] == 1
        assert stats["human_override_fixes"] == 1
        assert stats["final_rows"] == 3

        with open(out_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert rows[0]["Question"] == "Fix me?"
        assert rows[0]["Correct"] == "Yes"
        assert rows[1]["Question"] == "Match me?"
        assert rows[1]["Correct"] == "Yes"
        assert rows[2]["Question"] == "Keep me?"
        assert rows[2]["Correct"] == "Yes"


class TestFindExperimentCsvs:
    def test_finds_evaluation_csvs(self, tmp_path):
        # Create evaluation directory structure
        eval_dir = tmp_path / "verbalized_model_evaluation" / "version_0"
        eval_dir.mkdir(parents=True)
        (eval_dir / "evaluation_results.csv").write_text("header\n")

        results = find_experiment_csvs(tmp_path)
        assert len(results) == 1
        assert "evaluation_results.csv" in str(results[0][0])

    def test_finds_is_true_csvs(self, tmp_path):
        # Create is_true directory structure
        is_true_dir = tmp_path / "is_true_model"
        is_true_dir.mkdir(parents=True)
        (is_true_dir / "merged_results.csv").write_text("header\n")

        results = find_experiment_csvs(tmp_path)
        assert len(results) == 1
        assert "merged_results.csv" in str(results[0][0])

    def test_ignores_non_matching_dirs(self, tmp_path):
        # Create a directory that doesn't match patterns
        other_dir = tmp_path / "other_stuff"
        other_dir.mkdir()
        (other_dir / "data.csv").write_text("header\n")

        results = find_experiment_csvs(tmp_path)
        assert len(results) == 0

    def test_ignores_missing_csvs(self, tmp_path):
        # Create directories without CSVs
        (tmp_path / "verbalized_model_evaluation" / "version_0").mkdir(parents=True)
        (tmp_path / "is_true_model").mkdir(parents=True)

        results = find_experiment_csvs(tmp_path)
        assert len(results) == 0


class TestReevaluateWithRealLabelingFormat:
    """Test with CSV format matching the actual equivalence_labeling.csv."""

    def test_multirow_same_question_different_predictions(self, tmp_path):
        """Same question can have different predictions across systems;
        each prediction has its own human label."""
        labeling_path = tmp_path / "labeling.csv"
        with open(labeling_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "question",
                    "gold answers",
                    "predicted answer",
                    "correct_evaluator",
                    "correct_human",
                    "should_change_gold_answers",
                    "erase_question",
                    "system",
                ]
            )
            # Same question, different predictions, different labels
            writer.writerow(
                [
                    "Capital?",
                    "['paris']",
                    "Paris!",
                    "No",
                    "Yes",
                    "",
                    "",
                    "['system-A']",
                ]
            )
            writer.writerow(
                [
                    "Capital?",
                    "['paris']",
                    "London",
                    "No",
                    "No",
                    "",
                    "",
                    "['system-B']",
                ]
            )

        erase, corrections, _ = load_labeling_csv(labeling_path)
        assert len(erase) == 0
        assert corrections[("Capital?", "Paris!")] == "Yes"
        assert corrections[("Capital?", "London")] == "No"

        # Create eval CSV for system C that happens to predict "Paris!"
        eval_path = tmp_path / "eval.csv"
        with open(eval_path, "w", newline="") as f:
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
            writer.writerow(["Capital?", "['paris']", "Paris!", "0.9", "No", "", ""])
            writer.writerow(["Capital?", "['paris']", "London", "0.8", "No", "", ""])

        out_path = tmp_path / "output.csv"
        reevaluate_csv(eval_path, out_path, erase, corrections)

        with open(out_path) as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["Predicted Answer"] == "Paris!"
        assert rows[0]["Correct"] == "Yes"
        assert rows[1]["Predicted Answer"] == "London"
        assert rows[1]["Correct"] == "No"

    def test_erase_question_removes_all_predictions(self, tmp_path):
        """When a question is erased, ALL its predictions are removed."""
        labeling_path = tmp_path / "labeling.csv"
        with open(labeling_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "question",
                    "gold answers",
                    "predicted answer",
                    "correct_evaluator",
                    "correct_human",
                    "should_change_gold_answers",
                    "erase_question",
                    "system",
                ]
            )
            writer.writerow(["Bad Q?", "['x']", "y", "No", "", "", "Yes", "['sys-A']"])

        erase, corrections, _ = load_labeling_csv(labeling_path)

        eval_path = tmp_path / "eval.csv"
        with open(eval_path, "w", newline="") as f:
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
            # Same erased question, different predictions (e.g., 2 duplicate questions)
            writer.writerow(["Bad Q?", "['x']", "y", "0.5", "No", "", ""])
            writer.writerow(["Bad Q?", "['x']", "z", "0.6", "No", "", ""])
            writer.writerow(["Good Q?", "['a']", "a", "0.9", "Yes", "", ""])

        out_path = tmp_path / "output.csv"
        stats = reevaluate_csv(eval_path, out_path, erase, corrections)

        assert stats["erased_rows"] == 2
        assert stats["final_rows"] == 1

        with open(out_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["Question"] == "Good Q?"


def test_load_subset_questions_normalizes_quotes(tmp_path):
    """Pinned subset questions are normalized like the evaluation CSV questions."""
    subset_path = tmp_path / "subset.csv"
    with open(subset_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Question ID"])
        writer.writerow(['"Quoted question?"', "q_1"])
        writer.writerow(["Plain question?", "q_2"])

    assert load_subset_questions(subset_path) == {"Quoted question?", "Plain question?"}


def test_add_truncated_questions_matches_unique_prefix(tmp_path):
    """Long subset questions stored truncated with '...' in eval CSVs are kept."""
    eval_path = tmp_path / "eval.csv"
    with open(eval_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Predicted Answer"])
        writer.writerow(["Which long question is this...", "A"])
        writer.writerow(["Unrelated question...", "B"])

    subset = {"Which long question is this one?", "Short?"}
    expanded = add_truncated_questions(subset, [(eval_path, Path("eval.csv"))])

    assert expanded == subset | {"Which long question is this..."}


def test_curated_gold_aliases_count_as_exact_match(tmp_path):
    """Aliases added to the curated dataset match, but human 'No' labels still win."""
    eval_path = tmp_path / "eval.csv"
    with open(eval_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Gold Answers", "Predicted Answer", "Correct"])
        writer.writerow(["Which line?", "['northern']", "Northern Line", "No"])
        writer.writerow(["Which colour?", "['red']", "Crimson", "No"])

    out_path = tmp_path / "out.csv"
    reevaluate_csv(
        eval_path,
        out_path,
        set(),
        {("Which colour?", "Crimson"): "No"},
        curated_gold={"Which line?": ["northern line"], "Which colour?": ["crimson"]},
    )

    with open(out_path) as f:
        rows = list(csv.DictReader(f))
    assert [r["Correct"] for r in rows] == ["Yes", "No"]


def test_load_subset_gold_maps_truncated_questions(tmp_path):
    """Truncated ('...') question spellings get the aliases of their full question."""
    subset_path = tmp_path / "subset.csv"
    with open(subset_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Question ID", "Gold Answers (aliases)"])
        writer.writerow(
            ["Which long question is this one?", "q_1", "['northern line']"]
        )
        writer.writerow(["Short?", "q_2", "['yes']"])

    gold = load_subset_gold(
        subset_path, {"Which long question is this...", "Short?", "Unknown?"}
    )

    assert gold == {
        "Which long question is this...": ["northern line"],
        "Short?": ["yes"],
    }
