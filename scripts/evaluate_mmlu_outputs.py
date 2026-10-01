"""
Evaluate multiple-choice correctness for MMLU LLM outputs and generate
evaluation_results.csv compatible with ActivationsDataModule.
"""

import os
import csv
import argparse


def evaluate_mmlu_csv(llm_outputs_csv: str, output_dir: str = None) -> str:
    if output_dir is None:
        output_dir = os.path.dirname(llm_outputs_csv)
    os.makedirs(output_dir, exist_ok=True)

    eval_rows = []
    total = 0
    correct_count = 0

    with open(llm_outputs_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get("question", "")
            gold_answer = row.get("gold_answers", "").strip().upper()
            pred_answer = row.get("pred_answer", "").strip().upper()
            raw_output = row.get("raw_output", "")

            # For MMLU, answer is the letter choice: A, B, C, or D
            is_correct = (pred_answer == gold_answer) and len(pred_answer) > 0
            correct_str = "Yes" if is_correct else "No"

            eval_rows.append(
                {
                    "Question": question,
                    "Gold Answers": gold_answer,
                    "Predicted Answer": pred_answer,
                    "Correct": correct_str,
                    "Raw Output": raw_output,
                }
            )
            total += 1
            if is_correct:
                correct_count += 1

    acc = (correct_count / total * 100) if total > 0 else 0.0
    print("MMLU Evaluation:")
    print(f"  Total samples: {total}")
    print(f"  Correct:       {correct_count} ({acc:.2f}%)")
    print(f"  Accuracy:      {acc:.2f}%")

    eval_csv_path = os.path.join(output_dir, "evaluation_results.csv")
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
    return eval_csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MMLU LLM outputs and produce evaluation_results.csv."
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
        default=None,
        help="Directory to save evaluation_results.csv (defaults to same dir as llm_outputs_csv)",
    )
    args = parser.parse_args()
    evaluate_mmlu_csv(args.llm_outputs_csv, args.output_dir)


if __name__ == "__main__":
    main()
