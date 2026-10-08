#!/usr/bin/env python3
"""
Summarize Data Scaling Experiment Results (exp9_data_scaling).

Analyzes how training set size N affects probe performance, calibration,
and overfitting when using AttentivePooling (layer 17) evaluated on a fixed
validation set of 10,000 samples.

Reads metrics from:
  <exp_dir>/n_<N>/lightning_logs/version_*/metrics.csv
And outputs:
  - <exp_dir>/experiment_summary.csv
  - <exp_dir>/experiment_summary.png
  - <exp_dir>/experiment_summary.pdf
  - <exp_dir>/plots/summary_*.png and .pdf
"""

import os
import re
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def extract_best_metrics(metrics_csv: str) -> dict:
    if not os.path.exists(metrics_csv):
        return {}

    df = pd.read_csv(metrics_csv)
    # Filter for validation epoch rows
    val_rows = (
        df.dropna(subset=["val_loss_epoch"])
        if "val_loss_epoch" in df.columns
        else df.dropna(subset=["val_auc"])
    )
    if val_rows.empty:
        return {}

    # Best epoch is the one with minimum val_loss_epoch
    best_row = val_rows.loc[val_rows["val_loss_epoch"].idxmin()]
    best_step = int(best_row.get("step", -1))
    best_epoch = int(best_row.get("epoch", -1))
    val_loss = float(best_row.get("val_loss_epoch", float("nan")))

    metrics = {
        "best_epoch": best_epoch,
        "best_step": best_step,
        "val_loss": val_loss,
        "val_cross_entropy": float(best_row.get("val_cross_entropy", val_loss)),
        "val_auc": float(best_row.get("val_auc", float("nan"))),
        "val_ece": float(best_row.get("val_ece", float("nan"))),
        "val_brier_score": float(best_row.get("val_brier_score", float("nan"))),
        "val_accuracy": float(best_row.get("val_accuracy", float("nan"))),
    }

    # Dynamically extract ECUAS metrics
    for col in best_row.index:
        if col.startswith("val_ecuas") or col == "val_ccas":
            metric_name = col.replace("val_", "")
            if metric_name == "ccas":
                metric_name = "ecuas"
            metrics[metric_name] = float(best_row[col])

    # Extract train loss at or immediately preceding best_step
    train_step_rows = df.dropna(subset=["train_loss_step"])
    if not train_step_rows.empty:
        prior_steps = train_step_rows[train_step_rows["step"] <= best_step]
        if not prior_steps.empty:
            # Use rolling average of last 5 train steps for stability
            recent_train = prior_steps.tail(min(5, len(prior_steps)))
            train_loss_at_best = float(recent_train["train_loss_step"].mean())
        else:
            train_loss_at_best = float(train_step_rows.iloc[0]["train_loss_step"])
        final_train_loss = float(train_step_rows.iloc[-1]["train_loss_step"])
    else:
        train_loss_at_best = float("nan")
        final_train_loss = float("nan")

    metrics["train_loss_at_best"] = train_loss_at_best
    metrics["generalization_gap"] = (
        val_loss - train_loss_at_best
        if not np.isnan(train_loss_at_best)
        else float("nan")
    )
    metrics["final_train_loss"] = final_train_loss

    # Epochs & steps counts
    metrics["total_steps"] = (
        int(df["step"].max())
        if "step" in df.columns and not df["step"].dropna().empty
        else -1
    )
    metrics["total_epochs"] = (
        int(df["epoch"].max())
        if "epoch" in df.columns and not df["epoch"].dropna().empty
        else -1
    )

    # Extract elapsed / training time
    total_time = None
    if "total_elapsed_time" in df.columns:
        tot = df["total_elapsed_time"].dropna()
        if not tot.empty:
            total_time = float(tot.iloc[-1])
    elif "train_elapsed_time" in df.columns:
        tot = df["train_elapsed_time"].dropna()
        if not tot.empty:
            total_time = float(tot.iloc[-1])

    json_path = os.path.join(os.path.dirname(metrics_csv), "training_time.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                t_data = json.load(f)
                if total_time is None:
                    total_time = float(
                        t_data.get("total_elapsed_time_seconds", float("nan"))
                    )
                metrics["pure_train_time_sec"] = float(
                    t_data.get("train_pure_time_seconds", float("nan"))
                )
        except Exception:
            pass

    if total_time is not None:
        metrics["elapsed_time_sec"] = total_time

    return metrics


def format_n_ticks(ax, n_values):
    """Set custom tick marks and labels for training set sizes on log scale."""
    ax.set_xscale("log")
    ax.set_xticks(n_values)
    labels = []
    for n in n_values:
        if n >= 1000:
            labels.append(f"{n // 1000}k" if n % 1000 == 0 else f"{n / 1000:.1f}k")
        else:
            labels.append(str(n))
    ax.set_xticklabels(labels, fontsize=11)
    ax.minorticks_off()


def summarize_data_scaling(exp_dir: str, output_dir: str = None):
    exp_path = Path(exp_dir)
    if not exp_path.exists():
        print(f"Error: Experiment directory '{exp_dir}' does not exist.")
        return

    out_path = Path(output_dir) if output_dir else exp_path
    out_path.mkdir(parents=True, exist_ok=True)
    plots_dir = out_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Find all n_* subdirectories
    n_dirs = []
    for d in exp_path.iterdir():
        if d.is_dir() and d.name.startswith("n_"):
            match = re.match(r"^n_(\d+)$", d.name)
            if match:
                n_val = int(match.group(1))
                n_dirs.append((n_val, d))

    n_dirs.sort(key=lambda x: x[0])
    if not n_dirs:
        print(f"No 'n_<N>' directories found in '{exp_dir}'.")
        return

    print(f"Found {len(n_dirs)} training set sizes: {[n for n, _ in n_dirs]}")

    records = []
    run_histories = {}

    for n_val, n_dir in n_dirs:
        log_versions = sorted(
            list(n_dir.glob("lightning_logs/version_*")),
            key=lambda p: int(p.name.split("_")[-1])
            if p.name.split("_")[-1].isdigit()
            else p.name,
        )
        if not log_versions:
            if (n_dir / "metrics.csv").exists():
                metrics_csv = n_dir / "metrics.csv"
            elif (n_dir / "lightning_logs" / "metrics.csv").exists():
                metrics_csv = n_dir / "lightning_logs" / "metrics.csv"
            else:
                print(f"Warning: No metrics.csv found for {n_dir.name}")
                continue
        else:
            metrics_csv = log_versions[-1] / "metrics.csv"

        best_m = extract_best_metrics(str(metrics_csv))
        if best_m:
            records.append(
                {
                    "n_samples": n_val,
                    "target_dir": n_dir.name,
                    **best_m,
                }
            )

            # Store history dataframe for learning curves
            try:
                hist_df = pd.read_csv(metrics_csv)
                run_histories[n_val] = hist_df
            except Exception:
                pass

    if not records:
        print("No completed run metrics found.")
        return

    summary_df = pd.DataFrame(records)
    summary_df = summary_df.sort_values("n_samples").reset_index(drop=True)

    # Save summary CSV
    csv_path = out_path / "experiment_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved summary CSV: {csv_path}")

    # Print summary table to console
    display_cols = [
        "n_samples",
        "val_loss",
        "val_auc",
        "val_ece",
        "ecuas_0",
        "ecuas_1",
        "ecuas_128",
        "generalization_gap",
        "best_epoch",
        "best_step",
        "elapsed_time_sec",
    ]
    present_cols = [c for c in display_cols if c in summary_df.columns]
    print("\n" + "=" * 105)
    print("DATA SCALING SUMMARY (Experiment 9)")
    print("=" * 105)
    print(summary_df[present_cols].to_string(index=False))
    print("=" * 105 + "\n")

    n_vals = summary_df["n_samples"].values
    primary_color = "#1f77b4"  # Blue
    highlight_color = "#d62728"  # Red
    accent_green = "#2ca02c"
    accent_orange = "#ff7f0e"
    accent_purple = "#9467bd"

    # =========================================================================
    # 1. Multi-Panel Comprehensive Summary Grid (2x3)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    # Panel 1: Validation Cross Entropy (BCE) vs N
    ax1 = axes[0]
    ax1.plot(
        n_vals,
        summary_df["val_cross_entropy"],
        marker="o",
        color=primary_color,
        linewidth=2.5,
        markersize=8,
        label="Val Cross Entropy (BCE)",
    )
    min_ce_idx = summary_df["val_cross_entropy"].idxmin()
    min_ce_n = summary_df.loc[min_ce_idx, "n_samples"]
    min_ce_val = summary_df.loc[min_ce_idx, "val_cross_entropy"]
    ax1.scatter(
        min_ce_n,
        min_ce_val,
        color="gold",
        s=130,
        zorder=5,
        edgecolors=primary_color,
        linewidths=1.5,
    )
    ax1.annotate(
        f"Min: {min_ce_val:.4f}\n(N={min_ce_n})",
        (min_ce_n, min_ce_val),
        textcoords="offset points",
        xytext=(15, -15),
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=primary_color, lw=1.2),
    )
    format_n_ticks(ax1, n_vals)
    ax1.set_title("Validation Cross Entropy (BCE)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Training Set Size N", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10, loc="upper right")

    # Panel 2: Validation AUC Score vs N
    ax2 = axes[1]
    ax2.plot(
        n_vals,
        summary_df["val_auc"],
        marker="s",
        color=accent_green,
        linewidth=2.5,
        markersize=8,
        label="Val AUC Score",
    )
    max_auc_idx = summary_df["val_auc"].idxmax()
    max_auc_n = summary_df.loc[max_auc_idx, "n_samples"]
    max_auc_val = summary_df.loc[max_auc_idx, "val_auc"]
    ax2.scatter(
        max_auc_n,
        max_auc_val,
        color="gold",
        s=130,
        zorder=5,
        edgecolors=accent_green,
        linewidths=1.5,
    )
    ax2.annotate(
        f"Max: {max_auc_val:.4f}\n(N={max_auc_n})",
        (max_auc_n, max_auc_val),
        textcoords="offset points",
        xytext=(-55, -25),
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=accent_green, lw=1.2),
    )
    format_n_ticks(ax2, n_vals)
    ax2.set_title("Validation AUC Score", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Training Set Size N", fontsize=11)
    ax2.set_ylabel("AUC", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10, loc="lower right")

    # Panel 3: Expected Calibration Error (ECE) vs N
    ax3 = axes[2]
    ax3.plot(
        n_vals,
        summary_df["val_ece"],
        marker="^",
        color=accent_orange,
        linewidth=2.5,
        markersize=8,
        label="Expected Calibration Error (ECE)",
    )
    min_ece_idx = summary_df["val_ece"].idxmin()
    min_ece_n = summary_df.loc[min_ece_idx, "n_samples"]
    min_ece_val = summary_df.loc[min_ece_idx, "val_ece"]
    ax3.scatter(
        min_ece_n,
        min_ece_val,
        color="gold",
        s=130,
        zorder=5,
        edgecolors=accent_orange,
        linewidths=1.5,
    )
    ax3.annotate(
        f"Min: {min_ece_val:.4f}\n(N={min_ece_n})",
        (min_ece_n, min_ece_val),
        textcoords="offset points",
        xytext=(15, 15),
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=accent_orange, lw=1.2),
    )
    format_n_ticks(ax3, n_vals)
    ax3.set_title("Expected Calibration Error (ECE)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Training Set Size N", fontsize=11)
    ax3.set_ylabel("ECE", fontsize=11)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(fontsize=10, loc="upper right")

    # Panel 4: ECUAS Metrics vs N
    ax4 = axes[3]
    ecuas_styles = [
        ("ecuas_0", "ECUAS (n=0)", "#1f77b4", "o"),
        ("ecuas_1", "ECUAS (n=1)", "#ff7f0e", "s"),
        ("ecuas_128", "ECUAS (n=128)", "#2ca02c", "^"),
    ]
    for m_col, m_label, m_col_color, m_marker in ecuas_styles:
        if m_col in summary_df.columns:
            ax4.plot(
                n_vals,
                summary_df[m_col],
                marker=m_marker,
                color=m_col_color,
                linewidth=2,
                markersize=7,
                label=m_label,
            )
    format_n_ticks(ax4, n_vals)
    ax4.set_title("ECUAS", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Training Set Size N", fontsize=11)
    ax4.set_ylabel("ECUAS Loss", fontsize=11)
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(fontsize=10, loc="upper right")

    # Panel 5: Overfitting & Generalization Gap
    ax5 = axes[4]
    ax5.plot(
        n_vals,
        summary_df["val_loss"],
        marker="o",
        color=highlight_color,
        linewidth=2.5,
        markersize=8,
        label="Val Loss (at best epoch)",
    )
    if "train_loss_at_best" in summary_df.columns:
        ax5.plot(
            n_vals,
            summary_df["train_loss_at_best"],
            marker="s",
            color=primary_color,
            linestyle="--",
            linewidth=2.2,
            markersize=7,
            label="Train Loss (at best step)",
        )
        ax5.fill_between(
            n_vals,
            summary_df["train_loss_at_best"],
            summary_df["val_loss"],
            alpha=0.15,
            color=highlight_color,
            label="Generalization Gap (Overfitting)",
        )
    format_n_ticks(ax5, n_vals)
    ax5.set_title("Generalization Gap & Overfitting", fontsize=13, fontweight="bold")
    ax5.set_xlabel("Training Set Size N", fontsize=11)
    ax5.set_ylabel("Loss", fontsize=11)
    ax5.grid(True, linestyle=":", alpha=0.6)
    ax5.legend(fontsize=9.5, loc="upper right")

    # Panel 6: Convergence Steps & Training Time
    ax6 = axes[5]
    color_steps = accent_purple
    color_time = "#8c564b"
    ax6.plot(
        n_vals,
        summary_df["best_step"],
        marker="D",
        color=color_steps,
        linewidth=2.2,
        markersize=7,
        label="Steps to Best Checkpoint",
    )
    ax6.set_ylabel(
        "Steps to Best Val Loss", color=color_steps, fontsize=11, fontweight="bold"
    )
    ax6.tick_params(axis="y", labelcolor=color_steps)
    format_n_ticks(ax6, n_vals)

    if "elapsed_time_sec" in summary_df.columns:
        ax6_t = ax6.twinx()
        elapsed_min = summary_df["elapsed_time_sec"] / 60.0
        ax6_t.plot(
            n_vals,
            elapsed_min,
            marker="x",
            color=color_time,
            linestyle=":",
            linewidth=2,
            markersize=8,
            label="Total Time (min)",
        )
        ax6_t.set_ylabel(
            "Wall Clock Time (minutes)",
            color=color_time,
            fontsize=11,
            fontweight="bold",
        )
        ax6_t.tick_params(axis="y", labelcolor=color_time)

    ax6.set_title(
        "Training Convergence & Compute Scaling", fontsize=13, fontweight="bold"
    )
    ax6.set_xlabel("Training Set Size N", fontsize=11)
    ax6.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        "Experiment 9: Data Scaling & Overfitting Analysis\n"
        "AttentivePooling Probe (Layer 17) Swept Across N ∈ [100, 40000] (Val Size = 10,000)",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()

    grid_png = out_path / "experiment_summary.png"
    grid_pdf = out_path / "experiment_summary.pdf"
    plt.savefig(grid_png, dpi=300, bbox_inches="tight")
    plt.savefig(grid_pdf, bbox_inches="tight")
    plt.close()
    print("Saved summary grid plot:")
    print(f"  - {grid_png}")
    print(f"  - {grid_pdf}")

    # =========================================================================
    # 2. Individual Metric Plots in plots/
    # =========================================================================
    metric_configs = [
        {
            "key": "val_cross_entropy",
            "title": "Validation Cross Entropy (BCE) vs Training Set Size",
            "ylabel": "Cross Entropy Loss",
            "find_best": np.argmin,
            "best_prefix": "Min",
            "color": primary_color,
            "marker": "o",
        },
        {
            "key": "val_auc",
            "title": "Validation AUC Score vs Training Set Size",
            "ylabel": "AUC Score",
            "find_best": np.argmax,
            "best_prefix": "Max",
            "color": accent_green,
            "marker": "s",
        },
        {
            "key": "val_ece",
            "title": "Expected Calibration Error (ECE) vs Training Set Size",
            "ylabel": "ECE",
            "find_best": np.argmin,
            "best_prefix": "Min",
            "color": accent_orange,
            "marker": "^",
        },
        {
            "key": "ecuas_0",
            "title": "ECUAS (n=0) vs Training Set Size",
            "ylabel": "ECUAS (n=0)",
            "find_best": np.argmin,
            "best_prefix": "Min",
            "color": "#1f77b4",
            "marker": "o",
        },
        {
            "key": "ecuas_1",
            "title": "ECUAS (n=1) vs Training Set Size",
            "ylabel": "ECUAS (n=1)",
            "find_best": np.argmin,
            "best_prefix": "Min",
            "color": "#ff7f0e",
            "marker": "s",
        },
        {
            "key": "ecuas_128",
            "title": "ECUAS (n=128) vs Training Set Size",
            "ylabel": "ECUAS (n=128)",
            "find_best": np.argmin,
            "best_prefix": "Min",
            "color": "#2ca02c",
            "marker": "^",
        },
    ]

    for cfg in metric_configs:
        key = cfg["key"]
        if key not in summary_df.columns:
            continue
        vals = summary_df[key].dropna().values
        if len(vals) == 0:
            continue

        fig_ind, ax_ind = plt.subplots(figsize=(10, 6))
        ax_ind.plot(
            n_vals,
            summary_df[key],
            marker=cfg["marker"],
            color=cfg["color"],
            linewidth=2.5,
            markersize=9,
            label=key,
        )

        # Annotate each point with its exact value
        for x_pt, y_pt in zip(n_vals, summary_df[key]):
            ax_ind.annotate(
                f"{y_pt:.4f}",
                (x_pt, y_pt),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9.5,
                fontweight="semibold",
            )

        best_idx = cfg["find_best"](summary_df[key].values)
        best_n = n_vals[best_idx]
        best_val = summary_df[key].values[best_idx]
        ax_ind.scatter(
            best_n,
            best_val,
            color="gold",
            s=150,
            zorder=5,
            edgecolors=cfg["color"],
            linewidths=2,
        )

        format_n_ticks(ax_ind, n_vals)
        ax_ind.set_title(cfg["title"], fontsize=14, fontweight="bold")
        ax_ind.set_xlabel("Training Set Size N (log scale)", fontsize=12)
        ax_ind.set_ylabel(cfg["ylabel"], fontsize=12)
        ax_ind.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        single_png = plots_dir / f"summary_{key}.png"
        single_pdf = plots_dir / f"summary_{key}.pdf"
        plt.savefig(single_png, dpi=300, bbox_inches="tight")
        plt.savefig(single_pdf, bbox_inches="tight")
        plt.close()

    # Individual combined ECUAS plot
    fig_ecuas, ax_ecuas = plt.subplots(figsize=(10, 6))
    for m_col, m_label, m_col_color, m_marker in ecuas_styles:
        if m_col in summary_df.columns:
            ax_ecuas.plot(
                n_vals,
                summary_df[m_col],
                marker=m_marker,
                color=m_col_color,
                linewidth=2.5,
                markersize=8,
                label=m_label,
            )
    format_n_ticks(ax_ecuas, n_vals)
    ax_ecuas.set_title(
        "ECUAS Calibration Across Abstention Tolerances vs Training Size",
        fontsize=14,
        fontweight="bold",
    )
    ax_ecuas.set_xlabel("Training Set Size N (log scale)", fontsize=12)
    ax_ecuas.set_ylabel("ECUAS Metric Value", fontsize=12)
    ax_ecuas.grid(True, linestyle=":", alpha=0.6)
    ax_ecuas.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig(plots_dir / "summary_ecuas.png", dpi=300, bbox_inches="tight")
    plt.savefig(plots_dir / "summary_ecuas.pdf", bbox_inches="tight")
    plt.close()

    # Individual Generalization Gap plot
    if (
        "train_loss_at_best" in summary_df.columns
        and "generalization_gap" in summary_df.columns
    ):
        fig_gap, (ax_g1, ax_g2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Train vs Val loss curves
        ax_g1.plot(
            n_vals,
            summary_df["val_loss"],
            marker="o",
            color=highlight_color,
            linewidth=2.5,
            markersize=8,
            label="Validation Loss",
        )
        ax_g1.plot(
            n_vals,
            summary_df["train_loss_at_best"],
            marker="s",
            color=primary_color,
            linewidth=2.5,
            markersize=8,
            linestyle="--",
            label="Training Loss (at Best Step)",
        )
        ax_g1.fill_between(
            n_vals,
            summary_df["train_loss_at_best"],
            summary_df["val_loss"],
            alpha=0.18,
            color=highlight_color,
            label="Generalization Gap",
        )
        format_n_ticks(ax_g1, n_vals)
        ax_g1.set_title(
            "Train Loss vs Validation Loss Across Data Scales",
            fontsize=13,
            fontweight="bold",
        )
        ax_g1.set_xlabel("Training Set Size N (log scale)", fontsize=11)
        ax_g1.set_ylabel("Loss (Cross Entropy)", fontsize=11)
        ax_g1.grid(True, linestyle=":", alpha=0.6)
        ax_g1.legend(fontsize=10, loc="upper right")

        # Right: Delta gap curve
        ax_g2.plot(
            n_vals,
            summary_df["generalization_gap"],
            marker="D",
            color="#9467bd",
            linewidth=2.5,
            markersize=8,
            label=r"$\Delta = \mathrm{ValLoss} - \mathrm{TrainLoss}$",
        )
        for x_pt, y_pt in zip(n_vals, summary_df["generalization_gap"]):
            ax_g2.annotate(
                f"{y_pt:.4f}",
                (x_pt, y_pt),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9.5,
                fontweight="semibold",
            )
        ax_g2.axhline(0, color="gray", linestyle=":", alpha=0.7)
        format_n_ticks(ax_g2, n_vals)
        ax_g2.set_title(
            r"Generalization Gap ($\Delta$) vs Training Set Size",
            fontsize=13,
            fontweight="bold",
        )
        ax_g2.set_xlabel("Training Set Size N (log scale)", fontsize=11)
        ax_g2.set_ylabel(r"Gap ($\Delta$ Loss)", fontsize=11)
        ax_g2.grid(True, linestyle=":", alpha=0.6)
        ax_g2.legend(fontsize=10, loc="upper right")

        plt.tight_layout()
        plt.savefig(
            plots_dir / "summary_generalization_gap.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(plots_dir / "summary_generalization_gap.pdf", bbox_inches="tight")
        plt.close()

    # =========================================================================
    # 3. Learning Curves Comparison Grid across all N
    # =========================================================================
    if run_histories:
        n_plots = len(run_histories)
        cols = 4 if n_plots >= 7 else 3
        rows = int(np.ceil(n_plots / cols))
        fig_lc, axes_lc = plt.subplots(
            rows, cols, figsize=(cols * 4.5, rows * 3.8), sharey=False
        )
        axes_lc = axes_lc.flatten()

        for idx, (n_val, h_df) in enumerate(
            sorted(run_histories.items(), key=lambda x: x[0])
        ):
            ax = axes_lc[idx]
            train_steps = h_df[["step", "train_loss_step"]].dropna().sort_values("step")
            val_steps = h_df[["step", "val_loss_epoch"]].dropna().sort_values("step")

            if not train_steps.empty:
                # Raw and EMA smoothed train loss
                ax.plot(
                    train_steps["step"],
                    train_steps["train_loss_step"],
                    color="royalblue",
                    alpha=0.25,
                    linewidth=1,
                    label="Train (Step)",
                )
                smoothed = (
                    train_steps["train_loss_step"].ewm(span=15, adjust=False).mean()
                )
                ax.plot(
                    train_steps["step"],
                    smoothed,
                    color="blue",
                    linewidth=1.8,
                    label="Train (EMA)",
                )

            if not val_steps.empty:
                ax.plot(
                    val_steps["step"],
                    val_steps["val_loss_epoch"],
                    color="crimson",
                    marker="o",
                    markersize=4,
                    linewidth=1.8,
                    label="Val Loss",
                )
                best_v_row = val_steps.loc[val_steps["val_loss_epoch"].idxmin()]
                ax.scatter(
                    best_v_row["step"],
                    best_v_row["val_loss_epoch"],
                    color="gold",
                    s=60,
                    zorder=5,
                    edgecolors="crimson",
                )

            ax.set_title(f"N = {n_val}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Steps", fontsize=10)
            ax.set_ylabel("Loss", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.5)
            if idx == 0:
                ax.legend(fontsize=8, loc="upper right")

        # Hide extra unused subplots
        for j in range(idx + 1, len(axes_lc)):
            fig_lc.delaxes(axes_lc[j])

        fig_lc.suptitle(
            "Training & Validation Loss Trajectories Across Training Sizes N",
            fontsize=15,
            fontweight="bold",
            y=1.00,
        )
        plt.tight_layout()
        plt.savefig(
            plots_dir / "summary_learning_curves.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(plots_dir / "summary_learning_curves.pdf", bbox_inches="tight")
        plt.close()

    print("All summary plots successfully saved to:")
    print(f"  - {out_path}/experiment_summary.png (.pdf)")
    print(f"  - {plots_dir}/summary_*.png (.pdf)")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize data scaling experiment (exp9_data_scaling) across training sizes N."
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="results/exp9_data_scaling",
        help="Path to experiment 9 directory (default: results/exp9_data_scaling)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional destination directory for plots and summary CSV",
    )
    args = parser.parse_args()
    summarize_data_scaling(args.exp_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
