#!/usr/bin/env python3
"""
Summarize Probe Sweep Results across Scopes (prompt, answer, both) and Layers (0-32 + weighted).

Reads metrics from:
  <exp_dir>/<scope>/<layer_dir>/lightning_logs/version_*/metrics.csv
And outputs:
  - <exp_dir>/experiment_summary.csv
  - <exp_dir>/experiment_summary.png
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


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

    metrics = {
        "best_epoch": int(best_row.get("epoch", -1)),
        "val_loss": float(best_row.get("val_loss_epoch", float("nan"))),
        "val_accuracy": float(best_row.get("val_accuracy", float("nan"))),
        "val_auc": float(best_row.get("val_auc", float("nan"))),
        "val_ece": float(best_row.get("val_ece", float("nan"))),
        "val_brier_score": float(best_row.get("val_brier_score", float("nan"))),
        "val_cross_entropy": float(best_row.get("val_cross_entropy", float("nan"))),
    }

    # Dynamically extract all ECUAS metrics present in metrics.csv
    for col in best_row.index:
        if col.startswith("val_ecuas") or col == "val_ccas":
            metric_name = col.replace("val_", "")
            if metric_name == "ccas":
                metric_name = "ecuas"
            metrics[metric_name] = float(best_row[col])

    return metrics


def summarize_experiment(exp_dir: str, custom_scopes: list = None):
    exp_path = Path(exp_dir)
    if not exp_path.exists():
        print(f"Error: Experiment directory {exp_dir} does not exist.")
        return

    if custom_scopes:
        scopes = custom_scopes
    else:
        # Auto-detect scopes / token window directories
        scopes = []
        for d in sorted(exp_path.iterdir()):
            if d.is_dir() and d.name != "plots":
                if list(d.glob("layer_*")) or (d / "weighted").exists():
                    scopes.append(d.name)

        preferred_order = [
            "prompt",
            "answer",
            "both",
            "exact_answer",
            "exact_answer_before_first_token",
            "exact_answer_first_token",
            "exact_answer_last_token",
            "exact_answer_after_last_token",
        ]
        scopes = sorted(
            scopes,
            key=lambda s: preferred_order.index(s) if s in preferred_order else 999,
        )

    if not scopes:
        print(f"No scope subdirectories found in {exp_dir}")
        return

    print(f"Summarizing experiment across scopes/windows: {scopes}")
    records = []

    for scope in scopes:
        scope_dir = exp_path / scope
        if not scope_dir.exists():
            continue

        # Look for layer directories (layer_0, layer_1, ..., layer_32, weighted)
        layer_dirs = sorted(list(scope_dir.glob("layer_*")))
        weighted_dir = scope_dir / "weighted"
        if weighted_dir.exists():
            layer_dirs.append(weighted_dir)

        for l_dir in layer_dirs:
            target_name = l_dir.name  # e.g., 'layer_0' or 'weighted'
            is_weighted = target_name == "weighted"
            layer_num = -1 if is_weighted else int(target_name.replace("layer_", ""))

            log_versions = sorted(
                list(l_dir.glob("lightning_logs/version_*")),
                key=lambda p: int(p.name.split("_")[-1]),
            )
            if not log_versions:
                continue

            latest_version = log_versions[-1]
            metrics_csv = latest_version / "metrics.csv"

            best_m = extract_best_metrics(str(metrics_csv))
            if best_m:
                records.append(
                    {
                        "scope": scope,
                        "target": target_name,
                        "layer": layer_num,
                        "is_weighted": is_weighted,
                        **best_m,
                    }
                )

    if not records:
        print("No completed probe results found to summarize.")
        return

    summary_df = pd.DataFrame(records)
    summary_csv = exp_path / "experiment_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved experiment summary CSV to: {summary_csv}")

    import numpy as np

    # Generate 3x2 grid of subplots comparing metrics across prompt, answer, and both scopes
    metrics_config = [
        {
            "key": "val_cross_entropy",
            "fallback_key": "val_loss",
            "title": "Validation Cross Entropy (BCE)",
            "find_best": np.argmin,
            "best_label": "Min",
        },
        {
            "key": "val_auc",
            "title": "Validation AUC Score",
            "find_best": np.argmax,
            "best_label": "Max",
        },
        {
            "key": "val_ece",
            "title": "Expected Calibration Error (ECE)",
            "find_best": np.argmin,
            "best_label": "Min",
        },
        {
            "key": "ecuas_0",
            "title": "ECUAS (n=0)",
            "find_best": np.argmin,
            "best_label": "Min",
        },
        {
            "key": "ecuas_1",
            "title": "ECUAS (n=1)",
            "find_best": np.argmin,
            "best_label": "Min",
        },
        {
            "key": "ecuas_128",
            "title": "ECUAS (n=128)",
            "find_best": np.argmin,
            "best_label": "Min",
        },
    ]

    known_styles = {
        "prompt": {"color": "#1f77b4", "marker": "o", "label": "Prompt"},
        "answer": {"color": "#d62728", "marker": "s", "label": "Answer"},
        "both": {"color": "#2ca02c", "marker": "^", "label": "Prompt + Answer"},
        "exact_answer": {
            "color": "#1f77b4",
            "marker": "o",
            "label": "Exact Answer (All)",
        },
        "exact_answer_first_token": {
            "color": "#ff7f0e",
            "marker": "s",
            "label": "First Token",
        },
        "exact_answer_last_token": {
            "color": "#2ca02c",
            "marker": "^",
            "label": "Last Token",
        },
        "exact_answer_before_first_token": {
            "color": "#d62728",
            "marker": "v",
            "label": "Token Before First",
        },
        "exact_answer_after_last_token": {
            "color": "#9467bd",
            "marker": "D",
            "label": "Token After Last",
        },
    }

    default_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    default_markers = ["o", "s", "^", "v", "D", "p", "*", "X"]

    scope_styles = {}
    for idx, scope in enumerate(scopes):
        if scope in known_styles:
            scope_styles[scope] = known_styles[scope]
        else:
            scope_styles[scope] = {
                "color": default_colors[idx % len(default_colors)],
                "marker": default_markers[idx % len(default_markers)],
                "label": scope.replace("_", " ").title(),
            }

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()

    for idx, cfg in enumerate(metrics_config):
        ax = axes[idx]
        key = cfg["key"]
        fallback_key = cfg.get("fallback_key")

        has_data = False

        for scope in scopes:
            style = scope_styles.get(
                scope, {"color": "black", "marker": "x", "label": scope}
            )
            color = style["color"]
            marker = style["marker"]
            label = style["label"]

            # Per-layer data
            scope_df = summary_df[
                (summary_df["scope"] == scope) & (~summary_df["is_weighted"])
            ].sort_values("layer")
            weighted_df = summary_df[
                (summary_df["scope"] == scope) & (summary_df["is_weighted"])
            ]

            target_col = (
                key
                if key in scope_df.columns
                else (
                    fallback_key
                    if fallback_key and fallback_key in scope_df.columns
                    else None
                )
            )
            if target_col is None:
                continue

            sub_df = scope_df.dropna(subset=[target_col])
            if sub_df.empty:
                continue

            has_data = True
            layers = sub_df["layer"].values
            values = sub_df[target_col].values

            # Plot layer curve
            ax.plot(
                layers,
                values,
                color=color,
                marker=marker,
                linewidth=2,
                label=f"{label} (per-layer)",
            )

            # Highlight best layer for this scope
            best_idx = cfg["find_best"](values)
            best_layer = layers[best_idx]
            best_val = values[best_idx]
            ax.scatter(
                best_layer,
                best_val,
                color="gold",
                s=100,
                zorder=5,
                edgecolors=color,
                linewidths=1.5,
            )

            # Plot weighted average horizontal line if present
            if not weighted_df.empty and target_col in weighted_df.columns:
                w_val = weighted_df[target_col].values[0]
                if not np.isnan(w_val):
                    ax.axhline(
                        y=w_val,
                        color=color,
                        linestyle="--",
                        alpha=0.7,
                        label=f"{label} (weighted: {w_val:.4f})",
                    )

        ax.set_title(cfg["title"], fontsize=14, fontweight="bold")
        ax.set_xlabel("Layer Number", fontsize=12)
        ax.set_ylabel("Metric Value", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.6)
        if has_data:
            ax.legend(fontsize=8 if len(scopes) > 3 else 9, loc="best")
        else:
            ax.text(0.5, 0.5, f"No data for {key}", ha="center", va="center")

    plt.suptitle(
        f"Probe Performance Across Model Layers ({exp_path.name})",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()

    summary_png = exp_path / "experiment_summary.png"
    summary_pdf = exp_path / "experiment_summary.pdf"
    plt.savefig(summary_png, dpi=300)
    plt.savefig(summary_pdf)
    plt.close()
    print("Saved 3x2 experiment summary grid plot to:")
    print(f"  - {summary_png}")
    print(f"  - {summary_pdf}")

    # Also save individual metric plots in a plots/ directory inside exp_path
    plots_dir = exp_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    for cfg in metrics_config:
        key = cfg["key"]
        fallback_key = cfg.get("fallback_key")
        fig_ind, ax_ind = plt.subplots(figsize=(10, 6))
        has_ind_data = False

        for scope in scopes:
            style = scope_styles.get(
                scope, {"color": "black", "marker": "x", "label": scope}
            )
            color = style["color"]
            marker = style["marker"]
            label = style["label"]

            scope_df = summary_df[
                (summary_df["scope"] == scope) & (~summary_df["is_weighted"])
            ].sort_values("layer")
            weighted_df = summary_df[
                (summary_df["scope"] == scope) & (summary_df["is_weighted"])
            ]

            target_col = (
                key
                if key in scope_df.columns
                else (
                    fallback_key
                    if fallback_key and fallback_key in scope_df.columns
                    else None
                )
            )
            if target_col is None:
                continue

            sub_df = scope_df.dropna(subset=[target_col])
            if sub_df.empty:
                continue

            has_ind_data = True
            layers = sub_df["layer"].values
            values = sub_df[target_col].values

            ax_ind.plot(
                layers,
                values,
                color=color,
                marker=marker,
                linewidth=2,
                label=f"{label} (per-layer)",
            )

            best_idx = cfg["find_best"](values)
            best_layer = layers[best_idx]
            best_val = values[best_idx]
            ax_ind.scatter(
                best_layer,
                best_val,
                color="gold",
                s=120,
                zorder=5,
                edgecolors=color,
                linewidths=1.5,
            )

            if not weighted_df.empty and target_col in weighted_df.columns:
                w_val = weighted_df[target_col].values[0]
                if not np.isnan(w_val):
                    ax_ind.axhline(
                        y=w_val,
                        color=color,
                        linestyle="--",
                        alpha=0.7,
                        label=f"{label} (weighted: {w_val:.4f})",
                    )

        ax_ind.set_title(
            f"{cfg['title']} by Layer & Scope", fontsize=14, fontweight="bold"
        )
        ax_ind.set_xlabel("Layer Number", fontsize=12)
        ax_ind.set_ylabel("Metric Value", fontsize=12)
        ax_ind.grid(True, linestyle=":", alpha=0.6)
        if has_ind_data:
            ax_ind.legend(fontsize=9, loc="best")

        plt.tight_layout()
        single_png = plots_dir / f"summary_{key}.png"
        single_pdf = plots_dir / f"summary_{key}.pdf"
        plt.savefig(single_png, dpi=300)
        plt.savefig(single_pdf)
        plt.close()
    print(f"Saved individual metric plots to: {plots_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize experiment results across scopes and layers."
    )
    parser.add_argument(
        "--exp_dir", type=str, required=True, help="Directory of the experiment results"
    )
    parser.add_argument(
        "--scopes",
        nargs="*",
        type=str,
        default=None,
        help="Optional explicit list of scopes/windows to summarize",
    )
    args = parser.parse_args()
    summarize_experiment(args.exp_dir, custom_scopes=args.scopes)


if __name__ == "__main__":
    main()
