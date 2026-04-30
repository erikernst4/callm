"""
Calculate metrics for existing MMLU experiment results.

Reads llm_outputs.csv files from lightning_logs/mmlu/ directories,
computes correctness by exact string match, and calculates calibration
metrics for experiments with valid confidence values.

Usage:
    python scripts/calculate_mmlu_metrics.py [--logs-dir lightning_logs/mmlu]
"""

import os
import csv
import argparse
import numpy as np
import torch

from callm.metrics import (
    ExpectedCalibrationError,
    ConfidenceBrierScore,
    ConfidenceCrossEntropy,
    ConfidenceAUCScore,
    CCAS,
)


def load_llm_outputs(csv_path):
    """Load llm_outputs.csv and return list of dicts."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_metrics(rows):
    """Compute correctness and calibration metrics from parsed rows.

    Returns a dict with metric name -> value.
    """
    all_correct = []
    all_confidences = []

    for row in rows:
        pred = (row.get("pred_answer") or "").strip().upper()
        gold = (row.get("gold_answers") or "").strip().upper()

        # MMLU gold_answers is a single letter (no pipe-separated list)
        correct = pred == gold
        all_correct.append(correct)

        try:
            conf = float(row.get("confidence", "nan"))
        except (ValueError, TypeError):
            conf = float("nan")
        all_confidences.append(conf)

    all_correct = np.array(all_correct)
    all_confidences = np.array(all_confidences)

    results = {}
    results["n_samples"] = len(all_correct)
    results["accuracy"] = float(all_correct.mean()) if len(all_correct) > 0 else 0.0

    # Check for NaN confidences (will be handled as 0.5 fallback by metrics)
    n_invalid = int(np.isnan(all_confidences).sum())
    results["n_nan_confidences"] = n_invalid

    if n_invalid > 0:
        print(
            f"  Warning: {n_invalid}/{len(all_confidences)} samples have NaN confidence. "
            "Using fallback confidence of 0.5 for these samples."
        )

    if len(all_correct) == 0:
        return results

    confidences = torch.tensor(all_confidences, dtype=torch.float32)
    correctness = torch.tensor(all_correct, dtype=torch.float32)

    metric_classes = {
        "ece": ExpectedCalibrationError(n_bins=10),
        "brier_score": ConfidenceBrierScore(),
        "cross_entropy": ConfidenceCrossEntropy(),
        "auc": ConfidenceAUCScore(),
        "ccas": CCAS(),
    }

    for name, metric in metric_classes.items():
        metric.update(confidences, correctness)
        results[name] = float(metric.compute())

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate metrics for MMLU experiments"
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="lightning_logs/mmlu",
        help="Directory containing MMLU experiment subdirectories",
    )
    args = parser.parse_args()

    logs_dir = args.logs_dir
    if not os.path.isdir(logs_dir):
        print(f"Error: Directory not found: {logs_dir}")
        return

    # Find all experiment directories with llm_outputs.csv
    experiments = []
    for entry in sorted(os.listdir(logs_dir)):
        exp_dir = os.path.join(logs_dir, entry)
        csv_path = os.path.join(exp_dir, "llm_outputs.csv")
        if os.path.isdir(exp_dir) and os.path.isfile(csv_path):
            experiments.append((entry, csv_path))

    if not experiments:
        print(f"No experiments with llm_outputs.csv found in {logs_dir}")
        return

    print(f"Found {len(experiments)} experiments in {logs_dir}\n")

    # Collect results for summary table
    all_results = []

    for exp_name, csv_path in experiments:
        print(f"--- {exp_name} ---")
        rows = load_llm_outputs(csv_path)
        results = compute_metrics(rows)
        results["experiment"] = exp_name
        all_results.append(results)

        print(f"  Samples:  {results['n_samples']}")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        if "ece" in results:
            print(f"  ECE:      {results['ece']:.4f}")
            print(f"  Brier:    {results['brier_score']:.4f}")
            print(f"  CE:       {results['cross_entropy']:.4f}")
            print(f"  AUC:      {results['auc']:.4f}")
            print(f"  ECUAS:    {results['ccas']:.4f}")

        # Save metrics.csv in the experiment directory
        exp_dir = os.path.dirname(csv_path)
        metrics_file = os.path.join(exp_dir, "metrics.csv")
        metrics_to_save = {"val_accuracy": results["accuracy"]}
        if "ece" in results:
            metrics_to_save.update(
                {
                    "val_ece": results["ece"],
                    "val_brier_score": results["brier_score"],
                    "val_cross_entropy": results["cross_entropy"],
                    "val_auc": results["auc"],
                    "val_ccas": results["ccas"],
                }
            )
        try:
            with open(metrics_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(list(metrics_to_save.keys()))
                writer.writerow(list(metrics_to_save.values()))
            print(f"  Saved to: {metrics_file}")
        except Exception as e:
            print(f"  Failed to save metrics: {e}")

        print()

    # Print summary table
    print("=" * 90)
    print(
        f"{'Experiment':<50} {'Acc':>7} {'ECE':>7} {'Brier':>7} {'CE':>7} {'AUC':>7} {'ECUAS':>7}"
    )
    print("-" * 90)
    for r in all_results:
        acc_str = f"{r['accuracy']:.4f}"
        ece_str = f"{r['ece']:.4f}" if "ece" in r else "N/A"
        brier_str = f"{r['brier_score']:.4f}" if "brier_score" in r else "N/A"
        ce_str = f"{r['cross_entropy']:.4f}" if "cross_entropy" in r else "N/A"
        auc_str = f"{r['auc']:.4f}" if "auc" in r else "N/A"
        ccas_str = f"{r['ccas']:.4f}" if "ccas" in r else "N/A"
        print(
            f"{r['experiment']:<50} {acc_str:>7} {ece_str:>7} {brier_str:>7} {ce_str:>7} {auc_str:>7} {ccas_str:>7}"
        )
    print("=" * 90)


if __name__ == "__main__":
    main()
