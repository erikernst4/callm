"""
Generate a per-sample CSV comparing the LLM-as-a-Judge labels against the manual
labels of the curated TriviaQA subset.

The curated labels (`actual` in consolidated_evaluations.csv) are manual ground truth:
exact matches plus human labels for every non-exact match (equivalence_labeling.csv).
The LLM judge labels come from the raw runs in lightning_logs/paper_results/triviaqa.
Disagreements between both are realistic labeling errors that can be sampled to study
the sensitivity of each metric.

Merge with the consolidated CSV on (dataset, model, method, id).

Columns:
  - dataset, model, method, id: Merge keys (same values as consolidated_evaluations.csv)
  - predicted_answer: Model's generated answer text (for sanity checks)
  - judge_label: LLM-as-a-Judge correctness (1/0)
  - exact_match: Whether the answer exactly matches a gold answer (improved matching)
  - human_label: Human correctness label (1/0), empty if not human-reviewed
  - final_label: Manual ground-truth correctness (equals `actual`)
  - label_source: 'human', 'exact_match', or 'unreviewed' (no exact match and no
    human label for this (question, answer) pair; final_label then equals judge_label)
  - judge_error: Whether judge_label != final_label
  - judge_error_type: 'false_negative' (judge 0, manual 1), 'false_positive'
    (judge 1, manual 0), or empty

Usage:
    uv run python scripts/generate_judge_errors_csv.py
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from callm.utils import check_exact_match
from generate_consolidated_csv import normalize_question, process_triviaqa
from reevaluate_triviaqa import load_labeling_csv, parse_gold_answers

KEY_FIELDS = ("dataset", "model", "method", "id")

FIELDNAMES = [
    "dataset",
    "model",
    "method",
    "id",
    "predicted_answer",
    "judge_label",
    "exact_match",
    "human_label",
    "final_label",
    "label_source",
    "judge_error",
    "judge_error_type",
]


def load_judge_labels(triviaqa_dir: Path, annot_csv_path: Path) -> dict[tuple, dict]:
    """Map (dataset, model, method, id) to the raw LLM-judge row (first occurrence)."""
    judge = {}
    for row in process_triviaqa(triviaqa_dir, annot_csv_path):
        if not row["id"]:
            continue
        key = tuple(row[k] for k in KEY_FIELDS)
        judge.setdefault(key, row)
    return judge


def main():
    root_dir = Path(__file__).resolve().parents[1]
    paper_results = root_dir / "lightning_logs" / "paper_results"

    parser = argparse.ArgumentParser(
        description="Compare LLM-as-a-Judge labels with manual labels on curated TriviaQA"
    )
    parser.add_argument(
        "--consolidated-csv",
        type=str,
        default=str(root_dir / "consolidated_evaluations.csv"),
        help="Path to consolidated_evaluations.csv",
    )
    parser.add_argument(
        "--labeling-csv",
        type=str,
        default=str(root_dir / "equivalence_labeling.csv"),
        help="Path to equivalence_labeling.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(root_dir / "judge_errors.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    judge = load_judge_labels(
        paper_results / "triviaqa",
        paper_results / "triviaqa_strict" / "triviaqa_annotated_subset.csv",
    )
    _, human_corrections, _ = load_labeling_csv(Path(args.labeling_csv))
    # Long questions are labeled with the truncated ('...') text of the evaluation CSVs
    truncated_questions = {q for q, _ in human_corrections if q.endswith("...")}

    def labeling_questions(question: str) -> list[str]:
        full = normalize_question(question)
        return [full] + [
            q for q in truncated_questions if full.startswith(q[:-3].strip())
        ]

    rows = []
    answer_mismatches = 0
    with open(args.consolidated_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] != "triviaqa":
                continue
            key = tuple(row[k] for k in KEY_FIELDS)
            if key not in judge:
                raise KeyError(f"No LLM-judge label found for {key}")
            judge_row = judge[key]

            pred = row["predicted_answer"].strip()
            if judge_row["predicted_answer"] != pred:
                answer_mismatches += 1

            judge_label = judge_row["actual"]
            final_label = int(row["actual"])
            exact = check_exact_match(pred, parse_gold_answers(row["gold_answers"]))

            human = next(
                (
                    human_corrections[(q, pred)]
                    for q in labeling_questions(row["question"])
                    if (q, pred) in human_corrections
                ),
                None,
            )
            human_label = "" if human is None else int(human == "Yes")

            if human is not None:
                label_source = "human"
            elif exact:
                label_source = "exact_match"
            else:
                label_source = "unreviewed"

            if judge_label == final_label:
                error_type = ""
            elif final_label == 1:
                error_type = "false_negative"
            else:
                error_type = "false_positive"

            rows.append(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "method": row["method"],
                    "id": row["id"],
                    "predicted_answer": pred,
                    "judge_label": judge_label,
                    "exact_match": exact,
                    "human_label": human_label,
                    "final_label": final_label,
                    "label_source": label_source,
                    "judge_error": judge_label != final_label,
                    "judge_error_type": error_type,
                }
            )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    if answer_mismatches:
        print(
            f"WARNING: {answer_mismatches} rows with a different predicted answer in the raw judge CSV"
        )
    unreviewed = Counter(r["model"] for r in rows if r["label_source"] == "unreviewed")
    if unreviewed:
        print(
            f"WARNING: {sum(unreviewed.values())} rows are neither exact matches nor human-labeled:"
        )
        for model, n in sorted(unreviewed.items()):
            print(f"  {model}: {n}")

    print("\nRows by label source:")
    for source, n in Counter(r["label_source"] for r in rows).most_common():
        print(f"  {source}: {n}")

    errors = [r for r in rows if r["judge_error"]]
    print(f"\nJudge errors: {len(errors)}")
    for (source, kind), n in sorted(
        Counter((r["label_source"], r["judge_error_type"]) for r in errors).items()
    ):
        print(f"  {source} / {kind}: {n}")

    print("\nJudge errors per model:")
    for model, n in sorted(Counter(r["model"] for r in errors).items()):
        print(f"  {model}: {n}")


if __name__ == "__main__":
    main()
