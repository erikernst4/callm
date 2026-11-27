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
import json
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


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

    # Check for training_time.json if not present in metrics.csv
    json_path = os.path.join(os.path.dirname(metrics_csv), "training_time.json")
    if total_time is None and os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                t_data = json.load(f)
                total_time = float(
                    t_data.get("total_elapsed_time_seconds", float("nan"))
                )
        except Exception:
            pass

    if total_time is not None:
        metrics["elapsed_time_sec"] = total_time

    return metrics


def generate_layer_weight_gifs(exp_dir: str):
    """
    Find all layer_weights_history.json files in the experiment directory
    and generate an animated GIF for each showing the evolution of layer weights.
    """
    exp_path = Path(exp_dir)
    history_files = list(exp_path.rglob("layer_weights_history.json"))

    if not history_files:
        return

    print(f"\nFound {len(history_files)} weight history files. Generating GIFs...")

    for history_file in history_files:
        try:
            with open(history_file, "r") as f:
                data = json.load(f)

            snapshots = data.get("snapshots", [])
            if not snapshots:
                continue

            layer_indices = data.get(
                "layer_indices", list(range(len(snapshots[0]["weights"])))
            )

            frames = []

            # Find global max weight for consistent y-axis scaling
            max_weight = max([max(s["weights"]) for s in snapshots])
            y_max = min(1.0, max_weight * 1.1)  # cap at 1.0, add 10% headroom

            for snapshot in snapshots:
                step = snapshot["step"]
                weights = snapshot["weights"]

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(range(len(weights)), weights, color="#1f77b4")

                # Set custom x-axis labels if we have layer indices
                ax.set_xticks(range(len(weights)))
                ax.set_xticklabels(
                    [str(idx) for idx in layer_indices], rotation=45, ha="right"
                )

                ax.set_ylim(0, y_max)
                ax.set_title(
                    f"Layer Weights Evolution - Step {step}",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.set_xlabel("Layer Index", fontsize=12)
                ax.set_ylabel("Softmax Weight", fontsize=12)
                ax.grid(axis="y", linestyle=":", alpha=0.6)

                plt.tight_layout()

                # Save plot to in-memory buffer
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=100)
                plt.close(fig)
                buf.seek(0)

                # Open with PIL and append to frames
                frames.append(Image.open(buf))

            # Save as GIF
            gif_path = history_file.parent / "layer_weights_evolution.gif"
            if frames:
                frames[0].save(
                    gif_path,
                    format="GIF",
                    append_images=frames[1:],
                    save_all=True,
                    duration=200,  # 200ms per frame = 5fps
                    loop=0,  # infinite loop
                )
                print(f"  Created GIF: {gif_path.relative_to(exp_path)}")

        except Exception as e:
            print(f"  Failed to generate GIF for {history_file}: {e}")


def summarize_experiment(exp_dir: str, custom_scopes: list = None):
    exp_path = Path(exp_dir)
    if not exp_path.exists():
        print(f"Error: Experiment directory {exp_dir} does not exist.")
        return

    if custom_scopes:
        scopes = custom_scopes
    else:
        # Auto-detect scopes / token window directories / pooling method directories
        scopes = []
        for d in sorted(exp_path.iterdir()):
            if d.is_dir() and d.name != "plots" and not d.name.startswith("."):
                if (
                    list(d.glob("layer_*"))
                    or (d / "weighted").exists()
                    or any(
                        (sub / "lightning_logs").exists()
                        or (sub / "metrics.csv").exists()
                        for sub in d.iterdir()
                        if sub.is_dir()
                    )
                ):
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
            "full_answer",
            "baseline",
            "global_znorm",
            "layerwise_znorm",
            "layernorm",
            "attentive",
            "multi_head_attentive",
            "query_attentive",
            "query_multi_head_attentive",
            "transformer",
            "mean",
            "single_layer",
        ]

        known_inits_order = ["uniform", "normal", "reverse_chi", "random"]
        known_methods_order = [
            "baseline",
            "global_znorm",
            "layerwise_znorm",
            "layernorm",
        ]

        def scope_sort_key(s: str):
            for i_idx, init_k in enumerate(known_inits_order):
                if s.startswith(init_k + "_"):
                    rem = s[len(init_k) + 1 :]
                    m_idx = (
                        known_methods_order.index(rem)
                        if rem in known_methods_order
                        else 99
                    )
                    return (0, i_idx, m_idx, s)
            if s in preferred_order:
                return (1, preferred_order.index(s), s)
            return (2, 0, 999, s)

        scopes = sorted(scopes, key=scope_sort_key)

    if not scopes:
        print(f"No scope subdirectories found in {exp_dir}")
        return

    print(f"Summarizing experiment across scopes/windows: {scopes}")
    records = []

    def dir_sort_key(p: Path):
        name = p.name
        if name == "weighted":
            return (2, 0, name)
        if name.startswith("layer_"):
            suffix = name[6:]
            if suffix.isdigit():
                return (0, int(suffix), name)
            return (1, 0, name)
        if name.isdigit():
            return (0, int(name), name)
        return (1, 0, name)

    for scope in scopes:
        scope_dir = exp_path / scope
        if not scope_dir.exists():
            continue

        # Look for layer/target directories (layer_*, weighted, or any subdir with lightning_logs/metrics.csv)
        layer_dirs_set = set(scope_dir.glob("layer_*"))
        if (scope_dir / "weighted").exists():
            layer_dirs_set.add(scope_dir / "weighted")

        for sub in scope_dir.iterdir():
            if sub.is_dir() and (
                (sub / "lightning_logs").exists() or (sub / "metrics.csv").exists()
            ):
                layer_dirs_set.add(sub)

        layer_dirs = sorted(list(layer_dirs_set), key=dir_sort_key)

        for l_dir in layer_dirs:
            target_name = l_dir.name  # e.g., 'layer_0', 'layer_all', 'weighted'
            is_weighted = target_name == "weighted"
            if is_weighted:
                layer_num = -1
            else:
                raw_suffix = (
                    target_name[6:] if target_name.startswith("layer_") else target_name
                )
                try:
                    layer_num = int(raw_suffix)
                except ValueError:
                    layer_num = raw_suffix

            log_versions = sorted(
                list(l_dir.glob("lightning_logs/version_*")),
                key=lambda p: int(p.name.split("_")[-1])
                if p.name.split("_")[-1].isdigit()
                else p.name,
            )
            if not log_versions:
                if (l_dir / "metrics.csv").exists():
                    metrics_csv = l_dir / "metrics.csv"
                elif (l_dir / "lightning_logs" / "metrics.csv").exists():
                    metrics_csv = l_dir / "lightning_logs" / "metrics.csv"
                else:
                    continue
            else:
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

    # Generate 3x2 grid of subplots comparing metrics across scopes
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

    # Style mapping rules
    known_inits = {
        "uniform": {"marker": "o", "linestyle": "-", "name": "Uniform"},
        "normal": {"marker": "s", "linestyle": "-", "name": "Normal"},
        "reverse_chi": {"marker": "^", "linestyle": "-", "name": "Reverse Chi"},
        "random": {"marker": "X", "linestyle": "-", "name": "Random"},
    }

    known_methods = {
        "baseline": {"color": "#1f77b4", "name": "Baseline"},
        "global_znorm": {"color": "#ff7f0e", "name": "Global Z-Norm"},
        "layerwise_znorm": {"color": "#2ca02c", "name": "Layer-wise Z-Norm"},
        "layernorm": {"color": "#d62728", "name": "LayerNorm"},
    }

    known_styles = {
        "prompt": {
            "color": "#1f77b4",
            "marker": "o",
            "linestyle": "-",
            "label": "Prompt",
        },
        "answer": {
            "color": "#d62728",
            "marker": "s",
            "linestyle": "-",
            "label": "Answer",
        },
        "both": {
            "color": "#2ca02c",
            "marker": "^",
            "linestyle": "-",
            "label": "Prompt + Answer",
        },
        "exact_answer": {
            "color": "#1f77b4",
            "marker": "o",
            "linestyle": "-",
            "label": "Exact Answer (All)",
        },
        "exact_answer_first_token": {
            "color": "#ff7f0e",
            "marker": "s",
            "linestyle": "-",
            "label": "First Token",
        },
        "exact_answer_last_token": {
            "color": "#2ca02c",
            "marker": "^",
            "linestyle": "-",
            "label": "Last Token",
        },
        "exact_answer_before_first_token": {
            "color": "#d62728",
            "marker": "v",
            "linestyle": "-",
            "label": "Token Before First",
        },
        "exact_answer_after_last_token": {
            "color": "#9467bd",
            "marker": "D",
            "linestyle": "-",
            "label": "Token After Last",
        },
        "full_answer": {
            "color": "#8c564b",
            "marker": "p",
            "linestyle": "-",
            "label": "Full Answer",
        },
        "baseline": {
            "color": "#1f77b4",
            "marker": "o",
            "linestyle": "-",
            "label": "Baseline",
        },
        "global_znorm": {
            "color": "#ff7f0e",
            "marker": "s",
            "linestyle": "-",
            "label": "Global Z-Norm",
        },
        "layerwise_znorm": {
            "color": "#2ca02c",
            "marker": "^",
            "linestyle": "-",
            "label": "Layer-wise Z-Norm",
        },
        "layernorm": {
            "color": "#d62728",
            "marker": "v",
            "linestyle": "-",
            "label": "LayerNorm",
        },
        "single_layer": {
            "color": "#4b5563",
            "marker": "D",
            "linestyle": "-",
            "label": "Single Layer",
        },
        # Attentive sequence pooling methods (Exp 3)
        "attentive": {
            "color": "#1f77b4",
            "marker": "o",
            "linestyle": "-",
            "label": "Attentive",
        },
        "multi_head_attentive": {
            "color": "#ff7f0e",
            "marker": "s",
            "linestyle": "-",
            "label": "Multi-Head Attentive",
        },
        "query_attentive": {
            "color": "#2ca02c",
            "marker": "^",
            "linestyle": "-",
            "label": "Query Attentive",
        },
        "query_multi_head_attentive": {
            "color": "#d62728",
            "marker": "X",
            "linestyle": "-",
            "label": "Query Multi-Head Attentive",
        },
        "transformer": {
            "color": "#9467bd",
            "marker": "D",
            "linestyle": "-",
            "label": "Transformer",
        },
        "mean": {
            "color": "#8c564b",
            "marker": "v",
            "linestyle": "-",
            "label": "Mean Pooling",
        },
    }

    default_colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Yellow-Green
        "#17becf",  # Teal / Cyan
        "#393b79",  # Dark Indigo
        "#636363",  # Dark Gray
        "#8c6d31",  # Dark Amber
        "#845ec2",  # Vivid Violet
        "#ff6f91",  # Soft Red / Coral
        "#008e97",  # Deep Teal
    ]
    default_markers = [
        "o",  # Circle
        "s",  # Square
        "^",  # Triangle Up
        "X",  # Filled X
        "D",  # Diamond
        "v",  # Triangle Down
        "P",  # Filled Plus
        "p",  # Pentagon
        "*",  # Star
        "h",  # Hexagon
        "<",  # Triangle Left
        ">",  # Triangle Right
        "d",  # Thin Diamond
        "H",  # Hexagon 2
    ]
    default_linestyles = ["-"]

    # Fallback styles skip the colors/markers of known scopes in this plot,
    # so unknown scopes never look like a known one
    reserved = [known_styles[s] for s in scopes if s in known_styles]
    fallback_colors = [
        c for c in default_colors if c not in {ks.get("color") for ks in reserved}
    ] or default_colors
    fallback_markers = [
        m for m in default_markers if m not in {ks.get("marker") for ks in reserved}
    ] or default_markers

    scope_styles = {}
    fallback_counter = 0
    for scope in scopes:
        if scope in known_styles:
            ks = known_styles[scope]
            scope_styles[scope] = {
                "color": ks.get("color", "#1f77b4"),
                "marker": ks.get("marker", "o"),
                "linestyle": ks.get("linestyle", "-"),
                "label": ks.get("label", scope),
            }
        else:
            # Check for compound scope: <init>_<method>
            matched_compound = False
            for init_k, init_info in known_inits.items():
                if scope.startswith(init_k + "_"):
                    rem = scope[len(init_k) + 1 :]
                    if rem in known_methods:
                        method_info = known_methods[rem]
                        scope_styles[scope] = {
                            "color": method_info["color"],
                            "marker": init_info["marker"],
                            "linestyle": init_info["linestyle"],
                            "label": f"{init_info['name']}: {method_info['name']}",
                        }
                        matched_compound = True
                        break
                    else:
                        formatted_rem = (
                            rem.replace("znorm", "z_norm").replace("_", " ").title()
                        )
                        scope_styles[scope] = {
                            "color": fallback_colors[
                                fallback_counter % len(fallback_colors)
                            ],
                            "marker": init_info["marker"],
                            "linestyle": init_info["linestyle"],
                            "label": f"{init_info['name']}: {formatted_rem}",
                        }
                        fallback_counter += 1
                        matched_compound = True
                        break

            if not matched_compound:
                combo_color = fallback_colors[fallback_counter % len(fallback_colors)]
                combo_marker = fallback_markers[
                    fallback_counter % len(fallback_markers)
                ]
                combo_ls = default_linestyles[
                    (fallback_counter // len(fallback_markers))
                    % len(default_linestyles)
                ]
                fallback_counter += 1
                formatted_label = (
                    scope.replace("znorm", "z_norm").replace("_", " ").title()
                )
                scope_styles[scope] = {
                    "color": combo_color,
                    "marker": combo_marker,
                    "linestyle": combo_ls,
                    "label": formatted_label,
                }

    def layer_sort_key(val):
        if isinstance(val, (int, float, np.number)):
            return (0, val)
        try:
            return (0, int(val))
        except (ValueError, TypeError):
            return (1, str(val))

    non_weighted_df = summary_df[~summary_df["is_weighted"]]
    has_any_weighted = bool(summary_df["is_weighted"].any())
    all_numeric_layers = True
    if not non_weighted_df.empty:
        all_numeric_layers = all(
            isinstance(layer, (int, float, np.number))
            or (isinstance(layer, str) and layer.lstrip("-").isdigit())
            for layer in non_weighted_df["layer"]
        )

    x_axis_label = "Layer Number" if all_numeric_layers else "Layer / Subset"

    fig, axes = plt.subplots(3, 2, figsize=(18, 20))
    axes = axes.flatten()

    for idx, cfg in enumerate(metrics_config):
        ax = axes[idx]
        key = cfg["key"]
        fallback_key = cfg.get("fallback_key")

        has_data = False

        for scope in scopes:
            style = scope_styles.get(
                scope,
                {"color": "black", "marker": "x", "linestyle": "-", "label": scope},
            )
            color = style["color"]
            marker = style["marker"]
            linestyle = style.get("linestyle", "-")
            label = style["label"]
            curve_label = f"{label} (per-layer)" if has_any_weighted else label

            # Per-layer data
            scope_df = summary_df[
                (summary_df["scope"] == scope) & (~summary_df["is_weighted"])
            ]
            scope_df = scope_df.sort_values(
                by="layer", key=lambda col: col.map(layer_sort_key)
            )

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

            plot_x = layers if all_numeric_layers else [str(layer) for layer in layers]

            # Plot layer curve with custom linestyle and marker
            ax.plot(
                plot_x,
                values,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=7,
                label=curve_label,
            )

            # Highlight best layer for this scope
            best_idx = cfg["find_best"](values)
            best_layer = plot_x[best_idx]
            best_val = values[best_idx]
            scatter_size = 40 if len(scopes) > 6 else 100
            ax.scatter(
                best_layer,
                best_val,
                color="gold",
                s=scatter_size,
                zorder=5,
                edgecolors=color,
                linewidths=1.2,
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
        ax.set_xlabel(x_axis_label, fontsize=12)
        ax.set_ylabel("Metric Value", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.6)
        if not all_numeric_layers:
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        # Position legend inside only if scopes <= 6
        if has_data and len(scopes) <= 6:
            ax.legend(fontsize=9, loc="best")
        elif not has_data:
            ax.text(0.5, 0.5, f"No data for {key}", ha="center", va="center")

    # For > 6 scopes, place one unified reference table / legend outside on the right side of the figure
    if len(scopes) > 6:
        handles, labels = [], []
        for ax in axes:
            h, label = ax.get_legend_handles_labels()
            if len(h) > len(handles):
                handles, labels = h, label
        if handles:
            fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=9,
                frameon=True,
                title="Scopes / Configurations",
                title_fontsize=11,
            )

    plt.suptitle(
        f"Probe Performance Across Model Layers ({exp_path.name})",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()

    summary_png = exp_path / "experiment_summary.png"
    summary_pdf = exp_path / "experiment_summary.pdf"
    plt.savefig(summary_png, dpi=300, bbox_inches="tight")
    plt.savefig(summary_pdf, bbox_inches="tight")
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
        fig_ind, ax_ind = plt.subplots(figsize=(14, 7))
        has_ind_data = False

        for scope in scopes:
            style = scope_styles.get(
                scope,
                {"color": "black", "marker": "x", "linestyle": "-", "label": scope},
            )
            color = style["color"]
            marker = style["marker"]
            linestyle = style.get("linestyle", "-")
            label = style["label"]
            curve_label = f"{label} (per-layer)" if has_any_weighted else label

            scope_df = summary_df[
                (summary_df["scope"] == scope) & (~summary_df["is_weighted"])
            ]
            scope_df = scope_df.sort_values(
                by="layer", key=lambda col: col.map(layer_sort_key)
            )

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

            plot_x = layers if all_numeric_layers else [str(layer) for layer in layers]

            ax_ind.plot(
                plot_x,
                values,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=7,
                label=curve_label,
            )

            best_idx = cfg["find_best"](values)
            best_layer = plot_x[best_idx]
            best_val = values[best_idx]
            scatter_size = 40 if len(scopes) > 6 else 100
            ax_ind.scatter(
                best_layer,
                best_val,
                color="gold",
                s=scatter_size,
                zorder=5,
                edgecolors=color,
                linewidths=1.2,
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
        ax_ind.set_xlabel(x_axis_label, fontsize=12)
        ax_ind.set_ylabel("Metric Value", fontsize=12)
        ax_ind.grid(True, linestyle=":", alpha=0.6)
        if not all_numeric_layers:
            plt.setp(ax_ind.get_xticklabels(), rotation=30, ha="right")
        if has_ind_data:
            if len(scopes) > 6:
                ax_ind.legend(
                    fontsize=8.5,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    frameon=True,
                    title="Scopes / Configurations",
                    title_fontsize=10,
                )
            else:
                ax_ind.legend(fontsize=9, loc="best")

        plt.tight_layout()
        single_png = plots_dir / f"summary_{key}.png"
        single_pdf = plots_dir / f"summary_{key}.pdf"
        plt.savefig(single_png, dpi=300, bbox_inches="tight")
        plt.savefig(single_pdf, bbox_inches="tight")
        plt.close()
    print(f"Saved individual metric plots to: {plots_dir}/")

    # Generate GIFs for layer weight evolution
    generate_layer_weight_gifs(exp_dir)


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
