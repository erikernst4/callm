#!/usr/bin/env python3
"""
Bootstrap Evaluation Metrics for Multi-Seed Experiment Significance Testing.

Reads per-sample validation predictions (confidence, correctness) from multi-seed
re-runs and computes:
  1. Per-seed metrics (from raw predictions using ecuas)
  2. Cross-seed statistics: mean, std, 95% CI
  3. Sample-level bootstrap confidence intervals
  4. Paired bootstrap significance tests between a reference subset and all others

Expects the following directory structure:
  <exp_dir>/
    <subset_1>/
      seed_0/lightning_logs/version_*/val_predictions.csv
      seed_1/lightning_logs/version_*/val_predictions.csv
      ...
    <subset_2>/
      seed_0/...
      ...

Usage:
    python scripts/bootstrap_evaluation_metrics.py \\
        --exp_dir results/exp4_layer_pooling/layernorm \\
        --reference_subset layer_middle_30 \\
        --n_bootstraps 10000

Output files (saved to <exp_dir>/significance_results/):
    - seed_metrics.csv          — per-seed, per-subset metric values
    - bootstrap_summary.csv     — per-subset mean/std/CI for all metrics
    - pairwise_significance.csv — p-values for reference vs. each other subset
    - bootstrap_summary.png     — bar plots with error bars for key metrics
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path so we can import ecuas
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecuas import (
    ExpectedCalibrationError,
    ConfidenceCrossEntropy,
    ConfidenceAUCScore,
    ConfidenceBrierScore,
    ConfidenceECUAS,
)


# ── Metrics definitions ────────────────────────────────────────────────

METRIC_DEFINITIONS = [
    {
        "id": "val_auc",
        "display": "AUC",
        "higher_is_better": True,
    },
    {
        "id": "val_cross_entropy",
        "display": "Cross Entropy",
        "higher_is_better": False,
    },
    {
        "id": "val_ece",
        "display": "ECE",
        "higher_is_better": False,
    },
    {
        "id": "val_brier_score",
        "display": "Brier Score",
        "higher_is_better": False,
    },
    {
        "id": "val_ecuas_0",
        "display": "ECUAS (n=0)",
        "higher_is_better": False,
    },
    {
        "id": "val_ecuas_1",
        "display": "ECUAS (n=1)",
        "higher_is_better": False,
    },
    {
        "id": "val_ecuas_128",
        "display": "ECUAS (n=128)",
        "higher_is_better": False,
    },
]


def compute_metric(
    metric_id: str, confidences: torch.Tensor, correctness: torch.Tensor
) -> float:
    """Compute a single metric from raw predictions."""
    if metric_id == "val_auc":
        metric = ConfidenceAUCScore()
    elif metric_id == "val_cross_entropy":
        metric = ConfidenceCrossEntropy()
    elif metric_id == "val_ece":
        metric = ExpectedCalibrationError(n_bins=10)
    elif metric_id == "val_brier_score":
        metric = ConfidenceBrierScore()
    elif metric_id == "val_ecuas_0":
        metric = ConfidenceECUAS(n=0)
    elif metric_id == "val_ecuas_1":
        metric = ConfidenceECUAS(n=1)
    elif metric_id == "val_ecuas_128":
        metric = ConfidenceECUAS(n=128)
    else:
        raise ValueError(f"Unknown metric: {metric_id}")

    metric.update(confidences, correctness)
    return float(metric.compute())


def compute_all_metrics(
    confidences: torch.Tensor, correctness: torch.Tensor
) -> dict[str, float]:
    """Compute all metrics from raw predictions."""
    return {
        m["id"]: compute_metric(m["id"], confidences, correctness)
        for m in METRIC_DEFINITIONS
    }


# ── Data loading ────────────────────────────────────────────────────────


def load_val_predictions(csv_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load per-sample (confidence, correctness) from val_predictions.csv."""
    confidences = []
    correctness = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            confidences.append(float(row["confidence"]))
            correctness.append(float(row["correctness"]))
    return (
        torch.tensor(confidences, dtype=torch.float32),
        torch.tensor(correctness, dtype=torch.float32),
    )


def find_val_predictions(seed_dir: Path) -> Path | None:
    """Find val_predictions.csv in a seed directory's lightning_logs."""
    log_versions = sorted(
        seed_dir.glob("lightning_logs/version_*"),
        key=lambda p: int(p.name.split("_")[-1])
        if p.name.split("_")[-1].isdigit()
        else 0,
    )
    if not log_versions:
        return None
    # Use the latest version
    latest = log_versions[-1]
    pred_file = latest / "val_predictions.csv"
    return pred_file if pred_file.exists() else None


def find_metrics_csv(seed_dir: Path) -> Path | None:
    """Find metrics.csv in a seed directory's lightning_logs (fallback when no val_predictions)."""
    log_versions = sorted(
        seed_dir.glob("lightning_logs/version_*"),
        key=lambda p: int(p.name.split("_")[-1])
        if p.name.split("_")[-1].isdigit()
        else 0,
    )
    if not log_versions:
        return None
    latest = log_versions[-1]
    metrics_file = latest / "metrics.csv"
    return metrics_file if metrics_file.exists() else None


def extract_best_metrics_from_csv(metrics_csv: Path) -> dict[str, float]:
    """Extract best-epoch validation metrics from a training metrics.csv (fallback)."""
    import pandas as pd

    df = pd.read_csv(metrics_csv)
    val_rows = (
        df.dropna(subset=["val_loss_epoch"])
        if "val_loss_epoch" in df.columns
        else df.dropna(subset=["val_auc"])
    )
    if val_rows.empty:
        return {}

    best_row = val_rows.loc[val_rows["val_loss_epoch"].idxmin()]
    metrics = {}
    for m in METRIC_DEFINITIONS:
        col = m["id"]
        if col in best_row.index and not np.isnan(best_row[col]):
            metrics[col] = float(best_row[col])
    return metrics


# ── Bootstrap analysis ──────────────────────────────────────────────────


def sample_bootstrap_ci(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    metric_id: str,
    n_bootstraps: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute sample-level bootstrap confidence interval for a metric.

    Returns (point_estimate, ci_low, ci_high).
    """
    point_val = compute_metric(metric_id, confidences, correctness)
    n = len(confidences)
    if n == 0:
        return point_val, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boot_vals = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        try:
            val = compute_metric(metric_id, confidences[idx], correctness[idx])
            if not np.isnan(val):
                boot_vals.append(val)
        except Exception:
            pass

    if not boot_vals:
        return point_val, np.nan, np.nan

    alpha = (1.0 - ci_level) / 2.0
    low = float(np.percentile(boot_vals, alpha * 100))
    high = float(np.percentile(boot_vals, (1.0 - alpha) * 100))
    return point_val, low, high


def paired_bootstrap_pvalue(
    conf_a: torch.Tensor,
    corr_a: torch.Tensor,
    conf_b: torch.Tensor,
    corr_b: torch.Tensor,
    metric_id: str,
    n_bootstraps: int = 10000,
    seed: int = 42,
) -> float:
    """
    Paired bootstrap test: Is the metric for A significantly different from B?

    Uses the same validation samples (paired by index). Returns a two-sided p-value.
    """
    n = len(conf_a)
    if n == 0 or len(conf_b) != n:
        return np.nan

    point_a = compute_metric(metric_id, conf_a, corr_a)
    point_b = compute_metric(metric_id, conf_b, corr_b)
    observed_diff = point_a - point_b

    rng = np.random.default_rng(seed)
    count_extreme = 0
    valid_boots = 0

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        try:
            val_a = compute_metric(metric_id, conf_a[idx], corr_a[idx])
            val_b = compute_metric(metric_id, conf_b[idx], corr_b[idx])
            if np.isnan(val_a) or np.isnan(val_b):
                continue
            boot_diff = val_a - val_b
            valid_boots += 1
            # Two-sided: count how often bootstrap diff has opposite sign
            if observed_diff > 0 and boot_diff <= 0:
                count_extreme += 1
            elif observed_diff < 0 and boot_diff >= 0:
                count_extreme += 1
            elif observed_diff == 0:
                count_extreme += 1
        except Exception:
            pass

    if valid_boots == 0:
        return np.nan

    return min(1.0, float(count_extreme) / valid_boots)


# ── Main analysis pipeline ──────────────────────────────────────────────


def discover_subsets_and_seeds(exp_dir: Path) -> dict[str, dict[int, Path]]:
    """
    Discover all subset directories and their seed sub-runs.

    Returns: {subset_name: {seed_num: seed_dir_path}}
    """
    results = {}
    for subset_dir in sorted(exp_dir.iterdir()):
        if not subset_dir.is_dir():
            continue
        if subset_dir.name in ("significance_results", "plots", ".git"):
            continue

        seed_dirs = {}
        for seed_dir in sorted(subset_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            try:
                seed_num = int(seed_dir.name.split("_")[1])
            except (ValueError, IndexError):
                continue
            seed_dirs[seed_num] = seed_dir

        if seed_dirs:
            results[subset_dir.name] = seed_dirs

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap evaluation metrics for multi-seed experiment significance testing."
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Path to the experiment directory containing subset subdirectories with seed_N runs.",
    )
    parser.add_argument(
        "--reference_subset",
        type=str,
        default="layer_middle_30",
        help="Subset to use as reference for paired significance tests (default: layer_middle_30).",
    )
    parser.add_argument(
        "--n_bootstraps",
        type=int,
        default=10000,
        help="Number of bootstrap iterations for sample-level CIs and paired tests (default: 10000).",
    )
    parser.add_argument(
        "--ci_level",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap CIs (default: 0.95).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap sampling (default: 42).",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        sys.exit(1)

    out_dir = exp_dir / "significance_results"
    out_dir.mkdir(exist_ok=True, parents=True)

    # Discover subsets and seeds
    subset_seeds = discover_subsets_and_seeds(exp_dir)
    if not subset_seeds:
        print(f"Error: No seed_N directories found in any subset under {exp_dir}")
        sys.exit(1)

    print(f"{'=' * 70}")
    print(" Bootstrap Significance Analysis")
    print(f" Experiment: {exp_dir}")
    print(f" Reference subset: {args.reference_subset}")
    print(f" Subsets found: {len(subset_seeds)}")
    print(f"{'=' * 70}")

    for subset, seeds in subset_seeds.items():
        print(f"  {subset}: {len(seeds)} seeds ({sorted(seeds.keys())})")

    # ── Step 1: Load per-seed predictions and compute metrics ──────────

    # Structure: {subset: {seed: {metric_id: value}}}
    seed_metrics = defaultdict(lambda: defaultdict(dict))
    # Structure: {subset: {seed: (confidences, correctness)}}
    seed_predictions = defaultdict(dict)

    for subset, seeds in subset_seeds.items():
        for seed_num, seed_dir in sorted(seeds.items()):
            # Try val_predictions.csv first (sample-level data)
            pred_file = find_val_predictions(seed_dir)
            if pred_file:
                conf, corr = load_val_predictions(pred_file)
                seed_predictions[subset][seed_num] = (conf, corr)
                metrics = compute_all_metrics(conf, corr)
                seed_metrics[subset][seed_num] = metrics
                print(
                    f"  Loaded predictions: {subset}/seed_{seed_num} ({len(conf)} samples)"
                )
            else:
                # Fallback: read metrics from metrics.csv
                metrics_file = find_metrics_csv(seed_dir)
                if metrics_file:
                    metrics = extract_best_metrics_from_csv(metrics_file)
                    seed_metrics[subset][seed_num] = metrics
                    print(f"  Loaded metrics (CSV fallback): {subset}/seed_{seed_num}")
                else:
                    print(
                        f"  Warning: No predictions or metrics found for {subset}/seed_{seed_num}"
                    )

    if not seed_metrics:
        print("Error: No metrics could be loaded from any seed run.")
        sys.exit(1)

    # ── Step 2: Save per-seed metrics ──────────────────────────────────

    seed_metrics_file = out_dir / "seed_metrics.csv"
    with open(seed_metrics_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        metric_ids = [m["id"] for m in METRIC_DEFINITIONS]
        writer.writerow(["subset", "seed"] + metric_ids)
        for subset in sorted(seed_metrics.keys()):
            for seed_num in sorted(seed_metrics[subset].keys()):
                row = [subset, seed_num]
                for mid in metric_ids:
                    row.append(
                        f"{seed_metrics[subset][seed_num].get(mid, float('nan')):.6f}"
                    )
                writer.writerow(row)
    print(f"\nSaved per-seed metrics: {seed_metrics_file}")

    # ── Step 3: Compute cross-seed statistics ──────────────────────────

    print(f"\n{'=' * 70}")
    print(" Cross-Seed Summary Statistics")
    print(f"{'=' * 70}")

    summary_rows = []
    for subset in sorted(seed_metrics.keys()):
        seeds_data = seed_metrics[subset]
        n_seeds = len(seeds_data)
        row = {"subset": subset, "n_seeds": n_seeds}

        for m in METRIC_DEFINITIONS:
            mid = m["id"]
            vals = [
                seeds_data[s].get(mid, float("nan")) for s in sorted(seeds_data.keys())
            ]
            vals = [v for v in vals if not np.isnan(v)]

            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                ci_half = 1.96 * std_val / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                row[f"{mid}_mean"] = mean_val
                row[f"{mid}_std"] = std_val
                row[f"{mid}_ci_low"] = mean_val - ci_half
                row[f"{mid}_ci_high"] = mean_val + ci_half
            else:
                row[f"{mid}_mean"] = np.nan
                row[f"{mid}_std"] = np.nan
                row[f"{mid}_ci_low"] = np.nan
                row[f"{mid}_ci_high"] = np.nan

        summary_rows.append(row)

        # Print summary
        print(f"\n  {subset} (n={n_seeds}):")
        for m in METRIC_DEFINITIONS:
            mid = m["id"]
            mean_v = row[f"{mid}_mean"]
            std_v = row[f"{mid}_std"]
            if not np.isnan(mean_v):
                print(f"    {m['display']:20s}: {mean_v:.4f} ± {std_v:.4f}")

    # Save summary CSV
    summary_file = out_dir / "bootstrap_summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(summary_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                formatted = {}
                for k, v in row.items():
                    formatted[k] = f"{v:.6f}" if isinstance(v, float) else v
                writer.writerow(formatted)
        print(f"\nSaved bootstrap summary: {summary_file}")

    # ── Step 4: Sample-level bootstrap CIs ─────────────────────────────

    # Only if we have per-sample predictions for the reference subset
    bootstrap_ci_rows = []
    if seed_predictions:
        print(f"\n{'=' * 70}")
        print(" Sample-Level Bootstrap Confidence Intervals")
        print(f"{'=' * 70}")

        for subset in sorted(seed_predictions.keys()):
            seeds_preds = seed_predictions[subset]
            if not seeds_preds:
                continue

            # Use predictions from all seeds combined for a pooled bootstrap
            # (each seed has a different model, so pooling captures model variance too)
            for seed_num, (conf, corr) in sorted(seeds_preds.items()):
                row = {"subset": subset, "seed": seed_num}
                for m in METRIC_DEFINITIONS:
                    point, low, high = sample_bootstrap_ci(
                        conf,
                        corr,
                        m["id"],
                        n_bootstraps=args.n_bootstraps,
                        ci_level=args.ci_level,
                        seed=args.seed,
                    )
                    row[f"{m['id']}_point"] = point
                    row[f"{m['id']}_ci_low"] = low
                    row[f"{m['id']}_ci_high"] = high
                bootstrap_ci_rows.append(row)

        if bootstrap_ci_rows:
            ci_file = out_dir / "sample_bootstrap_ci.csv"
            fieldnames = list(bootstrap_ci_rows[0].keys())
            with open(ci_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in bootstrap_ci_rows:
                    formatted = {}
                    for k, v in row.items():
                        formatted[k] = f"{v:.6f}" if isinstance(v, float) else v
                    writer.writerow(formatted)
            print(f"Saved sample-level bootstrap CIs: {ci_file}")

    # ── Step 5: Paired significance tests ──────────────────────────────

    ref_subset = args.reference_subset
    if ref_subset not in seed_predictions:
        print(
            f"\nWarning: Reference subset '{ref_subset}' not found in predictions. "
            f"Skipping paired tests. Available: {list(seed_predictions.keys())}"
        )
    else:
        print(f"\n{'=' * 70}")
        print(f" Paired Bootstrap Significance Tests (reference: {ref_subset})")
        print(f"{'=' * 70}")

        # For paired tests, we need predictions from the same seed for both subsets.
        # We test each seed independently and report per-seed and average p-values.
        ref_seeds = seed_predictions[ref_subset]
        sig_rows = []

        for other_subset in sorted(seed_predictions.keys()):
            if other_subset == ref_subset:
                continue

            other_seeds = seed_predictions[other_subset]
            common_seeds = sorted(set(ref_seeds.keys()) & set(other_seeds.keys()))

            if not common_seeds:
                print(f"\n  {ref_subset} vs {other_subset}: No common seeds, skipping.")
                continue

            print(
                f"\n  {ref_subset} vs {other_subset} ({len(common_seeds)} common seeds):"
            )

            for m in METRIC_DEFINITIONS:
                # Average p-value across seeds (conservative approach)
                seed_pvals = []
                for seed_num in common_seeds:
                    conf_ref, corr_ref = ref_seeds[seed_num]
                    conf_other, corr_other = other_seeds[seed_num]

                    # Ensure same sample count (should be identical if seed is same)
                    min_n = min(len(conf_ref), len(conf_other))
                    pval = paired_bootstrap_pvalue(
                        conf_ref[:min_n],
                        corr_ref[:min_n],
                        conf_other[:min_n],
                        corr_other[:min_n],
                        m["id"],
                        n_bootstraps=args.n_bootstraps,
                        seed=args.seed,
                    )
                    seed_pvals.append(pval)

                # Report mean p-value and individual values
                valid_pvals = [p for p in seed_pvals if not np.isnan(p)]
                avg_pval = np.mean(valid_pvals) if valid_pvals else np.nan
                sig_marker = (
                    "***"
                    if avg_pval < 0.001
                    else "**"
                    if avg_pval < 0.01
                    else "*"
                    if avg_pval < 0.05
                    else ""
                )

                sig_rows.append(
                    {
                        "reference": ref_subset,
                        "comparison": other_subset,
                        "metric": m["id"],
                        "metric_display": m["display"],
                        "avg_pvalue": avg_pval,
                        "n_seeds": len(valid_pvals),
                        "seed_pvalues": str(valid_pvals),
                        "significant": sig_marker,
                    }
                )

                print(f"    {m['display']:20s}: p={avg_pval:.4f} {sig_marker}")

        # Save significance results
        if sig_rows:
            sig_file = out_dir / "pairwise_significance.csv"
            fieldnames = list(sig_rows[0].keys())
            with open(sig_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in sig_rows:
                    formatted = {}
                    for k, v in row.items():
                        formatted[k] = f"{v:.6f}" if isinstance(v, float) else v
                    writer.writerow(formatted)
            print(f"\nSaved pairwise significance: {sig_file}")

    # ── Step 6: Generate summary plots ─────────────────────────────────

    generate_summary_plots(summary_rows, out_dir, ref_subset)

    print(f"\n{'=' * 70}")
    print(f" All results saved to: {out_dir}")
    print(f"{'=' * 70}")


def generate_summary_plots(summary_rows: list[dict], out_dir: Path, reference: str):
    """Generate bar plots with error bars for key metrics across subsets."""
    if not summary_rows:
        return

    plot_metrics = [
        {"id": "val_auc", "display": "AUC (↑)", "higher_is_better": True},
        {
            "id": "val_cross_entropy",
            "display": "Cross Entropy (↓)",
            "higher_is_better": False,
        },
        {"id": "val_ece", "display": "ECE (↓)", "higher_is_better": False},
        {"id": "val_ecuas_0", "display": "ECUAS₀ (↓)", "higher_is_better": False},
        {"id": "val_ecuas_1", "display": "ECUAS₁ (↓)", "higher_is_better": False},
        {"id": "val_ecuas_128", "display": "ECUAS₁₂₈ (↓)", "higher_is_better": False},
    ]

    subsets = [row["subset"] for row in summary_rows]
    n_subsets = len(subsets)
    short_labels = [s.replace("layer_", "") for s in subsets]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = []
    for s in subsets:
        if s == reference:
            colors.append("#2196F3")  # Blue for reference
        else:
            colors.append("#78909C")  # Gray for others

    for idx, pm in enumerate(plot_metrics):
        ax = axes[idx]
        means = []
        stds = []
        for row in summary_rows:
            means.append(row.get(f"{pm['id']}_mean", np.nan))
            stds.append(row.get(f"{pm['id']}_std", 0.0))

        x = np.arange(n_subsets)
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=4,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_title(pm["display"], fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        # Highlight best
        valid_means = [(i, m) for i, m in enumerate(means) if not np.isnan(m)]
        if valid_means:
            if pm["higher_is_better"]:
                best_idx = max(valid_means, key=lambda t: t[1])[0]
            else:
                best_idx = min(valid_means, key=lambda t: t[1])[0]
            bars[best_idx].set_edgecolor("#FFD700")
            bars[best_idx].set_linewidth(2.5)

        # Adapt y-axis limits to zoom in on metric differences across subsets
        valid_lows = [m - s for m, s in zip(means, stds) if not np.isnan(m)]
        valid_highs = [m + s for m, s in zip(means, stds) if not np.isnan(m)]
        if valid_lows and valid_highs:
            ymin = min(valid_lows)
            ymax = max(valid_highs)
            diff = ymax - ymin
            if diff == 0:
                diff = abs(ymin) * 0.1 if ymin != 0 else 0.1
            margin = diff * 0.20
            ax.set_ylim(ymin - margin, ymax + margin)

    plt.suptitle(
        f"Cross-Seed Metric Comparison (mean ± std)\nReference: {reference}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    png_path = out_dir / "bootstrap_summary.png"
    pdf_path = out_dir / "bootstrap_summary.pdf"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"\nSaved summary plots: {png_path}")


if __name__ == "__main__":
    main()
