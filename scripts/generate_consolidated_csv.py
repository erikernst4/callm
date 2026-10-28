"""
Generate a consolidated CSV containing sample-level predictions, confidences,
and binary correctness across all models and methods for curated TriviaQA and MMLU.

Columns:
  - dataset: 'triviaqa' or 'mmlu'
  - model: Standardized LLM name (e.g. 'Qwen3.5 4B', 'Gemini 2.5 Flash')
  - method: Confidence method ('Sequence Posterior', 'Is True', 'Verbalized')
  - id: Unique sample ID within dataset (e.g. 'sfq_178', 'mmlu_0')
  - predicted: Confidence probability score in [0, 1]
  - actual: Ground truth binary correctness (1 for correct, 0 for incorrect)
  - predicted_answer: Model's generated answer text
  - is_true_answer: For 'Is True' rows, the True/False answer of the IsTrue
    prompt ('predicted' is then P(True), i.e. 1 - P(False) for False answers);
    empty for other methods
  - gold_answers: Ground truth target / reference answers
  - question: Cleaned question text
"""

import argparse
import csv
from pathlib import Path

# Display-name mapping for LLMs
LLM_DISPLAY_NAMES = {
    "Qwen3.5-4B": "Qwen3.5 4B",
    "Qwen3.5-9B": "Qwen3.5 9B",
    "GLM-4.6V-Flash": "GLM-4.6V-Flash",
    "Ministral-3-8B-Instruct-2512": "Ministral-3-8B-Instruct-2512",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
}

METHOD_DISPLAY_NAMES = {
    "label_prob": "Sequence Posterior",
    "is_true": "Is True",
    "verbalized": "Verbalized",
}

LLM_ORDER = [
    "Qwen3.5 4B",
    "Qwen3.5 9B",
    "GLM-4.6V-Flash",
    "Ministral-3-8B-Instruct-2512",
    "Gemini 2.5 Flash Lite",
    "Gemini 2.5 Flash",
    "Gemini 2.5 Pro",
]

METHOD_ORDER = ["Sequence Posterior", "Is True", "Verbalized"]


def normalize_question(q: str) -> str:
    """Normalize question text to fix Excel stripping leading/trailing quotes."""
    return q.strip().strip("'").strip('"')


def parse_experiment_dir(dir_name: str) -> tuple[str, str] | None:
    """Parse directory name to (method_display, llm_display)."""
    clean_name = (
        dir_name[: -len("_evaluation")]
        if dir_name.endswith("_evaluation")
        else dir_name
    )
    method_key = None
    remainder = clean_name
    for prefix in ["is_true", "label_prob", "verbalized"]:
        if clean_name.startswith(prefix + "_"):
            method_key = prefix
            remainder = clean_name[len(prefix) + 1 :]
            break
    if method_key is None:
        return None

    model_name = remainder
    model_name = (
        model_name.replace("zero_shot_", "")
        .replace("zero_shot", "")
        .replace("-no-thinking", "")
        .rstrip("_- ")
    )

    method_display = METHOD_DISPLAY_NAMES.get(method_key, method_key)
    llm_display = LLM_DISPLAY_NAMES.get(model_name, model_name)
    return method_display, llm_display


def load_triviaqa_annotated_subset(annot_csv_path: Path):
    """Load canonical question IDs and metadata for the curated 455-question TriviaQA subset."""
    metadata = []
    with open(annot_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(
                {
                    "id": row["Question ID"],
                    "question": row["Question"],
                    "gold_answers": row["Gold Answers (aliases)"],
                    "norm_q": normalize_question(row["Question"]),
                }
            )
    return metadata


def build_triviaqa_id_map(annot_metadata):
    """Map question text (including truncated variations) to canonical metadata."""

    def matcher(q_eval: str):
        q_norm = normalize_question(q_eval)
        for item in annot_metadata:
            if item["norm_q"] == q_norm:
                return item
        if q_norm.endswith("..."):
            prefix = q_norm[:-3].strip()
            matches = [m for m in annot_metadata if m["norm_q"].startswith(prefix)]
            if len(matches) == 1:
                return matches[0]
        return None

    return matcher


def process_triviaqa(triviaqa_dir: Path, annot_csv_path: Path) -> list[dict]:
    """Extract rows for curated TriviaQA."""
    annot_metadata = load_triviaqa_annotated_subset(annot_csv_path)
    id_matcher = build_triviaqa_id_map(annot_metadata)

    rows = []
    for exp_dir in sorted(triviaqa_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name in ("plots", "__pycache__"):
            continue

        parsed = parse_experiment_dir(exp_dir.name)
        if not parsed:
            continue
        method, model = parsed

        if exp_dir.name.startswith("is_true_"):
            csv_path = exp_dir / "merged_results.csv"
        else:
            csv_path = exp_dir / "version_0" / "evaluation_results.csv"

        if not csv_path.exists():
            continue

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q_text = row.get("Question", "")
                meta = id_matcher(q_text)
                qid = meta["id"] if meta else ""
                full_q = meta["question"] if meta else q_text
                gold = meta["gold_answers"] if meta else row.get("Gold Answers", "")

                # Confidence
                conf_raw = (
                    row.get("Confidence")
                    if "Confidence" in row
                    else row.get("confidence")
                )
                try:
                    conf = (
                        float(conf_raw)
                        if conf_raw is not None and conf_raw.strip() != ""
                        else float("nan")
                    )
                except ValueError:
                    conf = float("nan")

                # Correctness
                is_correct = 1 if row.get("Correct", "").strip().lower() == "yes" else 0
                pred_answer = row.get("Predicted Answer", "").strip()

                rows.append(
                    {
                        "dataset": "triviaqa",
                        "model": model,
                        "method": method,
                        "id": qid,
                        "predicted": conf,
                        "actual": is_correct,
                        "predicted_answer": pred_answer,
                        "is_true_answer": (
                            row.get("Is True Predicted Answer") or ""
                        ).strip(),
                        "gold_answers": gold,
                        "question": full_q,
                    }
                )

    return rows


def process_mmlu(mmlu_dir: Path) -> list[dict]:
    """Extract rows for MMLU 1000-sample evaluation set."""
    rows = []
    for exp_dir in sorted(mmlu_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == "plots":
            continue

        parsed = parse_experiment_dir(exp_dir.name)
        if not parsed:
            continue
        method, model = parsed

        if exp_dir.name.startswith("is_true_"):
            csv_path = exp_dir / "merged_results.csv"
        else:
            csv_path = exp_dir / "llm_outputs.csv"

        if not csv_path.exists():
            continue

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                q_text = (row.get("question") or row.get("Question") or "").replace(
                    "\\n", "\n"
                )
                qid = f"mmlu_{idx}"

                # Confidence
                conf_raw = (
                    row.get("Confidence")
                    if "Confidence" in row
                    else row.get("confidence")
                )
                try:
                    conf = (
                        float(conf_raw)
                        if conf_raw is not None and conf_raw.strip() != ""
                        else float("nan")
                    )
                except ValueError:
                    conf = float("nan")

                # Answers and Correctness
                pred_ans = (
                    row.get("pred_answer") or row.get("Predicted Answer") or ""
                ).strip()
                gold_ans = (
                    row.get("gold_answers")
                    or row.get("Gold Answers")
                    or row.get("label")
                    or row.get("answer")
                    or ""
                ).strip()

                if "Correct" in row:
                    is_correct = 1 if row["Correct"].strip().lower() == "yes" else 0
                else:
                    is_correct = 1 if pred_ans.lower() == gold_ans.lower() else 0

                rows.append(
                    {
                        "dataset": "mmlu",
                        "model": model,
                        "method": method,
                        "id": qid,
                        "predicted": conf,
                        "actual": is_correct,
                        "predicted_answer": pred_ans,
                        "is_true_answer": (
                            row.get("Is True Predicted Answer") or ""
                        ).strip(),
                        "gold_answers": gold_ans,
                        "question": q_text,
                    }
                )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate predictions and confidences for TriviaQA and MMLU"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="consolidated_evaluations.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    triviaqa_dir = project_root / "lightning_logs" / "paper_results" / "triviaqa_strict"
    annot_csv_path = triviaqa_dir / "triviaqa_annotated_subset.csv"
    mmlu_dir = project_root / "lightning_logs" / "paper_results" / "mmlu"

    print("Processing TriviaQA strict subset...")
    triviaqa_rows = process_triviaqa(triviaqa_dir, annot_csv_path)
    print(f"Extracted {len(triviaqa_rows)} TriviaQA rows.")

    print("Processing MMLU evaluation set...")
    mmlu_rows = process_mmlu(mmlu_dir)
    print(f"Extracted {len(mmlu_rows)} MMLU rows.")

    all_rows = triviaqa_rows + mmlu_rows

    fieldnames = [
        "dataset",
        "model",
        "method",
        "id",
        "predicted",
        "actual",
        "predicted_answer",
        "is_true_answer",
        "gold_answers",
        "question",
    ]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSuccessfully generated {out_path} with {len(all_rows)} total rows.")


if __name__ == "__main__":
    main()
