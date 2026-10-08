"""
Evaluate exact-match correctness for curated TriviaQA outputs and export
incorrect cases for manual review.
"""

import os
import csv
import ast
import argparse
from callm.utils import check_exact_match


def parse_gold_answers(gold_str: str) -> list[str]:
    if not gold_str:
        return []
    try:
        gold_list = ast.literal_eval(gold_str)
        if isinstance(gold_list, list):
            return [str(g) for g in gold_list]
    except (ValueError, SyntaxError):
        pass
    if "|" in gold_str:
        return [g.strip() for g in gold_str.split("|")]
    return [gold_str]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate exact match and export incorrect cases."
    )
    parser.add_argument(
        "--llm_outputs_csv",
        type=str,
        required=True,
        help="Path to llm_outputs.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/curated_benchmark",
        help="Directory to save evaluation_results.csv and incorrect_cases_manual_review.csv",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    eval_rows = []
    incorrect_rows = []

    total = 0
    correct_count = 0

    with open(args.llm_outputs_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            question = row.get("question", "")
            gold_str = row.get("gold_answers", "")
            gold_answers = parse_gold_answers(gold_str)
            pred_answer = row.get("pred_answer", "")
            raw_output = row.get("raw_output", "")

            is_correct = check_exact_match(pred_answer, gold_answers)
            correct_str = "Yes" if is_correct else "No"

            eval_rows.append(
                {
                    "Question": question,
                    "Gold Answers": gold_str,
                    "Predicted Answer": pred_answer,
                    "Correct": correct_str,
                    "Raw Output": raw_output,
                }
            )

            total += 1
            if is_correct:
                correct_count += 1
            else:
                incorrect_rows.append(
                    {
                        "sample_idx": idx,
                        "question": question,
                        "gold_answers": gold_str,
                        "pred_answer": pred_answer,
                        "raw_output": raw_output,
                    }
                )

    acc = (correct_count / total * 100) if total > 0 else 0.0
    print(f"Total samples: {total}")
    print(f"Correct (Exact Match): {correct_count} ({acc:.2f}%)")
    print(f"Incorrect: {len(incorrect_rows)} ({100 - acc:.2f}%)")

    # Save evaluation_results.csv
    eval_csv_path = os.path.join(args.output_dir, "evaluation_results.csv")
    with open(eval_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Question",
            "Gold Answers",
            "Predicted Answer",
            "Correct",
            "Raw Output",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eval_rows)
    print(f"Saved evaluation results to: {eval_csv_path}")

    # Save incorrect cases for manual review
    incorrect_csv_path = os.path.join(
        args.output_dir, "incorrect_cases_manual_review.csv"
    )
    with open(incorrect_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_idx",
            "question",
            "gold_answers",
            "pred_answer",
            "raw_output",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(incorrect_rows)
    print(f"Saved incorrect cases for manual review to: {incorrect_csv_path}")


if __name__ == "__main__":
    main()
