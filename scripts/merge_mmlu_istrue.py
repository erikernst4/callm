import sys
import pandas as pd
from pathlib import Path
import torch
import csv

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from callm.metrics import (
    ExpectedCalibrationError,
    ConfidenceBrierScore,
    ConfidenceCrossEntropy,
    ConfidenceAUCScore,
    CCAS,
)


def compute_metrics(confidences, correctness):
    conf_t = torch.tensor(confidences, dtype=torch.float32)
    corr_t = torch.tensor(correctness, dtype=torch.float32)

    accuracy = float(corr_t.mean()) if len(corr_t) > 0 else 0.0

    metrics = {
        "val_ece": ExpectedCalibrationError(n_bins=10),
        "val_brier_score": ConfidenceBrierScore(),
        "val_cross_entropy": ConfidenceCrossEntropy(),
        "val_auc": ConfidenceAUCScore(),
        "val_ccas": CCAS(),
    }

    results = {"val_accuracy": accuracy}
    for name, metric in metrics.items():
        metric.update(conf_t, corr_t)
        results[name] = metric.compute().item()

    return results


def main():
    mmlu_dir = Path("/home/eernst/callm/lightning_logs/paper_results/mmlu")

    for exp_dir in mmlu_dir.iterdir():
        if not exp_dir.is_dir() or not exp_dir.name.startswith("is_true_"):
            continue

        model_suffix = exp_dir.name[len("is_true_") :]

        # Determine the base directory
        if "GLM-4.6V-Flash-no-thinking" in model_suffix:
            base_dir_name = f"label_prob_zero_shot_{model_suffix}"
        else:
            base_dir_name = f"label_prob_{model_suffix}"

        base_dir = mmlu_dir / base_dir_name

        is_true_csv = exp_dir / "llm_outputs.csv"
        base_csv = base_dir / "llm_outputs.csv"

        if not is_true_csv.exists() or not base_csv.exists():
            print(
                f"Skipping {exp_dir.name}: missing llm_outputs.csv in the is_true or base directory {base_dir.name}"
            )
            continue

        print(f"Processing {exp_dir.name} using base {base_dir.name}...")

        df_base = pd.read_csv(base_csv)
        df_is_true = pd.read_csv(is_true_csv)

        # Standardize columns to lower case
        df_base.columns = [c.lower() for c in df_base.columns]
        df_is_true.columns = [c.lower() for c in df_is_true.columns]

        # Assume same row indices
        if len(df_base) != len(df_is_true):
            print(
                f"WARNING: row counts differ between base ({len(df_base)}) and is_true ({len(df_is_true)}). Trying to match on question..."
            )
            df_is_true = df_is_true.set_index("question")
            df_base = df_base.set_index("question")
            df_merged = df_base.join(
                df_is_true[["confidence", "raw_output"]], rsuffix="_istrue", how="inner"
            ).reset_index()
            confidences = df_merged["confidence_istrue"].values
        else:
            df_merged = df_base.copy()
            confidences = df_is_true["confidence"].values
            df_merged["confidence"] = confidences

        # compute correctness
        is_correct = (
            df_merged["pred_answer"].astype(str).str.strip().str.lower()
            == df_merged["gold_answers"].astype(str).str.strip().str.lower()
        )

        df_merged["Correct"] = is_correct.map({True: "Yes", False: "No"})

        # generate merged_results.csv mirroring evaluate_csv format
        merged_csv_path = exp_dir / "merged_results.csv"
        df_out = pd.DataFrame()
        df_out["Question"] = (
            df_merged["question"]
            .fillna("")
            .apply(lambda x: str(x).replace("\n", "\\n"))
        )
        df_out["Gold Answers"] = df_merged["gold_answers"]
        df_out["Predicted Answer"] = df_merged["pred_answer"]
        df_out["Confidence"] = confidences
        df_out["Correct"] = df_merged["Correct"]
        df_out["Evaluator Response"] = "Merged from base run"
        df_out["Raw Output"] = df_merged.get(
            "raw_output_istrue", df_merged.get("raw_output", "")
        )

        df_out.to_csv(merged_csv_path, index=False)
        print(f"  Saved {merged_csv_path}")

        # calculate metrics
        conf_array = df_out["Confidence"].values
        corr_array = is_correct.values

        res = compute_metrics(conf_array, corr_array)

        # overwrite metrics.csv
        metrics_csv_path = exp_dir / "metrics.csv"
        with open(metrics_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(res.keys())
            writer.writerow(res.values())
        print(f"  Recalculated metrics and saved to {metrics_csv_path}")


if __name__ == "__main__":
    main()
