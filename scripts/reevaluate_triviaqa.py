"""
Re-evaluate TriviaQA results with human labels and improved exact matching.

Produces cleaned copies of all evaluation CSVs under
`lightning_logs/paper_results/triviaqa_strict/` (or `triviaqa_cleaned/` if --no-strict).

Steps for each experiment CSV:
  1. Remove rows for questions marked erase_question=Yes in labeling CSV
  2. If --strict is set (default), keep ONLY the questions of the pinned subset (--subset-csv,
     default: the published 455-question subset); without it, keep questions that were
     human-reviewed or exact-matched across all models
  3. Re-run improved exact match against the raw and the curated (--subset-csv) gold aliases
  4. Apply human label overrides from equivalence_labeling.csv
  5. Write cleaned CSV to parallel directory structure

Usage:
    uv run python scripts/reevaluate_triviaqa.py [--strict / --no-strict] [--labeling-csv ...] [--source-dir ...] [--output-dir ...]
"""

import argparse
import ast
import csv
import shutil
import sys
from pathlib import Path

# Add project root to path so we can import callm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from callm.utils import check_exact_match


def normalize_question(q: str) -> str:
    """Normalize question text to fix Excel stripping leading quotes."""
    return q.strip().strip("'").strip('"')


def load_labeling_csv(
    csv_path: Path,
) -> tuple[set[str], dict[tuple[str, str], str], set[str]]:
    """Load equivalence_labeling.csv and return:
    - erase_questions: set of question strings to remove
    - human_corrections: dict of (question, predicted_answer) -> correct_human value
    - reviewed_questions: set of question strings that have at least one human label
    """
    erase_questions: set[str] = set()
    human_corrections: dict[tuple[str, str], str] = {}
    reviewed_questions: set[str] = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = normalize_question(row["question"])
            pred_answer = row["predicted answer"].strip()

            # Collect questions to erase
            if row.get("erase_question", "").strip() == "Yes":
                erase_questions.add(question)

            # Collect human corrections
            correct_human = row.get("correct_human", "").strip()
            if correct_human in ("Yes", "No"):
                key = (question, pred_answer)
                human_corrections[key] = correct_human
                reviewed_questions.add(question)

    return erase_questions, human_corrections, reviewed_questions


def parse_gold_answers(gold_str: str) -> list[str]:
    """Parse the gold answers string from CSV into a list."""
    try:
        gold_list = ast.literal_eval(gold_str)
        if isinstance(gold_list, list):
            return [str(g) for g in gold_list]
    except (ValueError, SyntaxError):
        pass
    return [gold_str]


def find_experiment_csvs(source_dir: Path) -> list[tuple[Path, Path]]:
    """Find all evaluation CSV files and compute their relative output paths."""
    results = []
    for d in sorted(source_dir.iterdir()):
        if not d.is_dir():
            continue

        if d.name.startswith("is_true_") and not d.name.endswith("_evaluation"):
            csv_path = d / "merged_results.csv"
            if csv_path.exists():
                rel = Path(d.name) / "merged_results.csv"
                results.append((csv_path, rel))
        elif d.name.endswith("_evaluation"):
            csv_path = d / "version_0" / "evaluation_results.csv"
            if csv_path.exists():
                rel = Path(d.name) / "version_0" / "evaluation_results.csv"
                results.append((csv_path, rel))

    return results


def find_universal_exact_matches(experiment_csvs: list[tuple[Path, Path]]) -> set[str]:
    """Find questions that have an exact match across ALL models."""
    question_exact_match_count: dict[str, int] = {}
    total_models = len(experiment_csvs)

    for csv_path, _ in experiment_csvs:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = normalize_question(row.get("Question", ""))
                if q not in question_exact_match_count:
                    question_exact_match_count[q] = 0

                pred = row.get("Predicted Answer", "").strip()
                gold = parse_gold_answers(row.get("Gold Answers", ""))

                if check_exact_match(pred, gold):
                    question_exact_match_count[q] += 1

    return {
        q for q, count in question_exact_match_count.items() if count == total_models
    }


def reevaluate_csv(
    csv_path: Path,
    output_path: Path,
    erase_questions: set[str],
    human_corrections: dict[tuple[str, str], str],
    allowed_questions: set[str] | None = None,
    curated_gold: dict[str, list[str]] | None = None,
) -> dict:
    """Re-evaluate a single evaluation CSV and write cleaned output.

    `curated_gold` maps questions to the curated dataset's gold aliases, which extend
    the raw TriviaQA aliases stored in the evaluation CSVs.
    """
    stats = {
        "original_rows": 0,
        "erased_rows": 0,
        "unreviewed_rows_dropped": 0,
        "duplicate_rows_dropped": 0,
        "exact_match_fixes": 0,
        "human_override_fixes": 0,
        "human_override_to_no": 0,
        "final_rows": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        if not fieldnames:
            print(f"  WARNING: Empty CSV at {csv_path}")
            return stats

        rows_out = []
        seen_questions = set()
        for row in reader:
            stats["original_rows"] += 1
            question = normalize_question(row.get("Question", ""))

            # Step 1: Remove erased questions
            if question in erase_questions:
                stats["erased_rows"] += 1
                continue

            # Step 2: If strict mode, filter to allowed_questions
            if allowed_questions is not None and question not in allowed_questions:
                stats["unreviewed_rows_dropped"] += 1
                continue

            # Deduplicate rows per question in strict mode
            if allowed_questions is not None:
                if question in seen_questions:
                    stats["duplicate_rows_dropped"] += 1
                    continue
                seen_questions.add(question)

            pred_answer = row.get("Predicted Answer", "").strip()
            gold_str = row.get("Gold Answers", "")
            original_correct = row.get("Correct", "").strip()

            # Step 3: Re-run improved exact match (raw + curated gold aliases)
            gold_answers = parse_gold_answers(gold_str)
            if curated_gold is not None:
                gold_answers += curated_gold.get(question, [])
            is_exact = check_exact_match(pred_answer, gold_answers)

            new_correct = original_correct

            if is_exact and original_correct != "Yes":
                new_correct = "Yes"
                stats["exact_match_fixes"] += 1

            # Step 4: Apply human label overrides
            key = (question, pred_answer)
            if key in human_corrections:
                human_label = human_corrections[key]
                if human_label != new_correct:
                    if human_label == "Yes":
                        stats["human_override_fixes"] += 1
                    else:
                        stats["human_override_to_no"] += 1
                    new_correct = human_label

            row["Correct"] = new_correct
            rows_out.append(row)

        stats["final_rows"] = len(rows_out)

    # Write output
    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    return stats


def load_subset_questions(csv_path: Path) -> set[str]:
    """Load the normalized questions of a pinned subset (e.g. the published one)."""
    with open(csv_path, "r", encoding="utf-8") as f:
        return {normalize_question(row["Question"]) for row in csv.DictReader(f)}


def load_subset_gold(csv_path: Path, questions: set[str]) -> dict[str, list[str]]:
    """Map each question (full or truncated '...' spelling) to its curated gold aliases."""
    with open(csv_path, "r", encoding="utf-8") as f:
        subset_gold = {
            normalize_question(row["Question"]): parse_gold_answers(
                row["Gold Answers (aliases)"]
            )
            for row in csv.DictReader(f)
        }
    gold = {}
    for q in questions:
        if q in subset_gold:
            gold[q] = subset_gold[q]
        elif q.endswith("..."):
            matches = [s for s in subset_gold if s.startswith(q[:-3].strip())]
            if len(matches) == 1:
                gold[q] = subset_gold[matches[0]]
    return gold


def add_truncated_questions(
    subset_questions: set[str], experiment_csvs: list[tuple[Path, Path]]
) -> set[str]:
    """Add the truncated ('...') spellings that evaluation CSVs use for long subset questions."""
    expanded = set(subset_questions)
    for csv_path, _ in experiment_csvs:
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                q = normalize_question(row.get("Question", ""))
                if q in expanded or not q.endswith("..."):
                    continue
                prefix = q[:-3].strip()
                matches = [s for s in subset_questions if s.startswith(prefix)]
                if len(matches) == 1:
                    expanded.add(q)
    return expanded


def main():
    root_dir = Path(__file__).resolve().parents[1]
    default_subset_csv = (
        root_dir
        / "lightning_logs"
        / "paper_results"
        / "triviaqa_strict"
        / "triviaqa_annotated_subset.csv"
    )

    parser = argparse.ArgumentParser(
        description="Re-evaluate TriviaQA with human labels and improved exact matching"
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to keep only the strict 455-question subset (default: True)",
    )
    parser.add_argument(
        "--labeling-csv",
        type=str,
        default=str(root_dir / "equivalence_labeling.csv"),
        help="Path to equivalence_labeling.csv",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=str(root_dir / "lightning_logs" / "paper_results" / "triviaqa"),
        help="Source TriviaQA results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for cleaned results (default: triviaqa_strict if --strict else triviaqa_cleaned)",
    )
    parser.add_argument(
        "--subset-csv",
        type=str,
        default=str(default_subset_csv) if default_subset_csv.exists() else None,
        help="CSV with a 'Question' column pinning the strict subset (default: the "
        "published triviaqa_annotated_subset.csv). Without it, the strict subset is "
        "recomputed as (universal exact matches | reviewed) - erased",
    )
    args = parser.parse_args()

    labeling_csv = Path(args.labeling_csv)
    source_dir = Path(args.source_dir)

    if args.output_dir is None:
        out_folder = "triviaqa_strict" if args.strict else "triviaqa_cleaned"
        output_dir = root_dir / "lightning_logs" / "paper_results" / out_folder
    else:
        output_dir = Path(args.output_dir)

    if not labeling_csv.exists():
        print(f"ERROR: Labeling CSV not found: {labeling_csv}")
        sys.exit(1)
    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        sys.exit(1)

    print(f"=== TriviaQA Re-evaluation (strict={args.strict}) ===")
    experiment_csvs = find_experiment_csvs(source_dir)
    print(f"Found {len(experiment_csvs)} experiment CSVs to process.")

    erase_questions, human_corrections, reviewed_questions = load_labeling_csv(
        labeling_csv
    )
    print(f"Loaded labeling CSV: {labeling_csv}")
    print(f"  Questions to erase: {len(erase_questions)}")
    print(f"  Human corrections: {len(human_corrections)}")

    allowed_questions = None
    curated_gold = None
    if args.strict and args.subset_csv:
        subset_questions = load_subset_questions(Path(args.subset_csv))
        allowed_questions = (
            add_truncated_questions(subset_questions, experiment_csvs) - erase_questions
        )
        curated_gold = load_subset_gold(Path(args.subset_csv), allowed_questions)
        print(f"  Pinned subset questions: {len(subset_questions)} ({args.subset_csv})")
        print(
            f"  Total strict allowed questions (Subset - Erased): {len(allowed_questions)}"
        )
    elif args.strict:
        universal_matches = find_universal_exact_matches(experiment_csvs)
        print(
            f"  Questions with exact match across ALL models: {len(universal_matches)}"
        )
        allowed_questions = (universal_matches | reviewed_questions) - erase_questions
        print(
            f"  Total strict allowed questions (Union - Erased): {len(allowed_questions)}"
        )
    print()

    output_dir.mkdir(parents=True, exist_ok=True)
    for f in source_dir.iterdir():
        if f.is_file() and f.suffix in (".py", ".tex"):
            dest = output_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)

    for csv_path, rel_path in experiment_csvs:
        src_parent = csv_path.parent
        dst_parent = output_dir / rel_path.parent
        dst_parent.mkdir(parents=True, exist_ok=True)
        for extra in ["config.yaml", "metrics.csv"]:
            src_extra = src_parent / extra
            dst_extra = dst_parent / extra
            if src_extra.exists() and not dst_extra.exists():
                shutil.copy2(src_extra, dst_extra)

    total_stats = {
        "original_rows": 0,
        "erased_rows": 0,
        "unreviewed_rows_dropped": 0,
        "duplicate_rows_dropped": 0,
        "exact_match_fixes": 0,
        "human_override_fixes": 0,
        "human_override_to_no": 0,
        "final_rows": 0,
    }

    for csv_path, rel_path in experiment_csvs:
        out_path = output_dir / rel_path
        print(f"Processing: {rel_path}")

        stats = reevaluate_csv(
            csv_path,
            out_path,
            erase_questions,
            human_corrections,
            allowed_questions=allowed_questions,
            curated_gold=curated_gold,
        )

        for k in total_stats:
            total_stats[k] += stats[k]

        changes = (
            stats["exact_match_fixes"]
            + stats["human_override_fixes"]
            + stats["human_override_to_no"]
        )
        print(
            f"  {stats['original_rows']} → {stats['final_rows']} rows "
            f"({stats['erased_rows']} erased, {changes} corrections)"
        )

    print()
    print("=" * 60)
    print("TOTALS across all experiments:")
    print(f"  Original rows: {total_stats['original_rows']}")
    print(f"  Erased rows: {total_stats['erased_rows']}")
    if args.strict:
        print(f"  Unreviewed rows dropped: {total_stats['unreviewed_rows_dropped']}")
        print(f"  Duplicate rows dropped: {total_stats['duplicate_rows_dropped']}")
    print(f"  Exact match fixes (No→Yes): {total_stats['exact_match_fixes']}")
    print(f"  Human override fixes (No→Yes): {total_stats['human_override_fixes']}")
    print(f"  Human override to No: {total_stats['human_override_to_no']}")
    print(f"  Final rows: {total_stats['final_rows']}")
    print(f"\nCleaned results written to: {output_dir}")


if __name__ == "__main__":
    main()
