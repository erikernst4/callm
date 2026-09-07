import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(metrics_csv, output_dir):
    if not os.path.exists(metrics_csv):
        print(f"Error: {metrics_csv} does not exist.")
        return

    # Load metrics
    df = pd.read_csv(metrics_csv)

    # Make output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Prepare loss data
    # Drop rows with NaN for the respective column and sort by step/epoch
    train_step = df[["step", "train_loss_step"]].dropna().sort_values("step")

    # Map validation epoch loss to corresponding step to plot them in the same step-level x-axis
    # Each validation epoch runs at the end of that epoch
    val_epoch_with_step = df[["step", "val_loss_epoch"]].dropna().sort_values("step")

    # 2. Prepare metrics data
    val_auc = (
        df[["epoch", "val_auc"]].dropna().sort_values("epoch")
        if "val_auc" in df.columns
        else pd.DataFrame()
    )
    val_ece = (
        df[["epoch", "val_ece"]].dropna().sort_values("epoch")
        if "val_ece" in df.columns
        else pd.DataFrame()
    )
    val_ce = (
        df[["epoch", "val_cross_entropy"]].dropna().sort_values("epoch")
        if "val_cross_entropy" in df.columns
        else pd.DataFrame()
    )

    # Create figure with 2 subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Losses ---
    ax1 = axes[0]
    if not train_step.empty:
        ax1.plot(
            train_step["step"],
            train_step["train_loss_step"],
            label="Train Loss (Step)",
            color="royalblue",
            alpha=0.4,
            linestyle="--",
        )
        # Apply exponential moving average to smooth step-level train loss
        smoothed_train = train_step["train_loss_step"].ewm(span=20, adjust=False).mean()
        ax1.plot(
            train_step["step"],
            smoothed_train,
            label="Train Loss (EMA Smoothed)",
            color="blue",
            linewidth=2,
        )

    if not val_epoch_with_step.empty:
        ax1.plot(
            val_epoch_with_step["step"],
            val_epoch_with_step["val_loss_epoch"],
            label="Val Loss (Epoch End)",
            color="crimson",
            marker="o",
            markersize=6,
            linewidth=2,
        )

    ax1.set_title("Loss Curves", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Training Steps", fontsize=12)
    ax1.set_ylabel("Loss (Binary Cross Entropy)", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    # --- Plot 2: Validation Metrics ---
    ax2 = axes[1]
    if not val_auc.empty:
        ax2.plot(
            val_auc["epoch"],
            val_auc["val_auc"],
            label="AUC Score",
            color="darkorange",
            marker="s",
            linewidth=2,
        )
    if not val_ece.empty:
        ax2.plot(
            val_ece["epoch"],
            val_ece["val_ece"],
            label="Expected Calibration Error (ECE)",
            color="forestgreen",
            marker="^",
            linewidth=2,
        )

    # Plot ECUAS metrics if present
    ecuas_colors = ["purple", "magenta", "orchid", "mediumpurple"]
    color_idx = 0
    for col in df.columns:
        if col.startswith("val_ecuas") or col == "val_ccas":
            metric_data = df[["epoch", col]].dropna().sort_values("epoch")
            if not metric_data.empty:
                label_name = col.replace("val_", "").upper()
                ax2.plot(
                    metric_data["epoch"],
                    metric_data[col],
                    label=label_name,
                    color=ecuas_colors[color_idx % len(ecuas_colors)],
                    marker="d",
                    linewidth=2,
                )
                color_idx += 1

    if not val_ce.empty:
        ax2.plot(
            val_ce["epoch"],
            val_ce["val_cross_entropy"],
            label="Cross Entropy",
            color="brown",
            marker="x",
            linewidth=1.5,
            linestyle=":",
        )

    ax2.set_title("Validation Calibration Metrics", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Metric Value", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10)

    # Check if elapsed training time is available
    elapsed_str = ""
    json_path = os.path.join(os.path.dirname(metrics_csv), "training_time.json")
    if os.path.exists(json_path):
        try:
            import json

            with open(json_path, "r", encoding="utf-8") as f:
                t_data = json.load(f)
                elapsed_str = t_data.get("total_elapsed_time_formatted", "")
        except Exception:
            pass
    if not elapsed_str and "total_elapsed_time" in df.columns:
        tot = df["total_elapsed_time"].dropna()
        if not tot.empty:
            from datetime import timedelta

            elapsed_str = str(timedelta(seconds=int(tot.iloc[-1])))

    if elapsed_str:
        plt.suptitle(
            f"Training Progress (Total Elapsed: {elapsed_str})",
            fontsize=15,
            fontweight="bold",
            y=1.02,
        )

    plt.tight_layout()

    # Save plots as PNG and PDF
    png_path = os.path.join(output_dir, "probe_loss_plots.png")
    pdf_path = os.path.join(output_dir, "probe_loss_plots.pdf")

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print("Generated loss and metric plots:")
    print(f"  - PNG: {png_path}")
    print(f"  - PDF: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training/validation losses and calibration metrics from metrics.csv"
    )
    parser.add_argument(
        "--metrics_csv",
        type=str,
        default="activations_cache/qwen_triviaqa/probe_layer_12/lightning_logs/version_2/metrics.csv",
        help="Path to the metrics.csv file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="activations_cache/qwen_triviaqa/probe_layer_12/plots",
        help="Directory to save the generated plots",
    )
    args = parser.parse_args()

    plot_metrics(args.metrics_csv, args.output_dir)


if __name__ == "__main__":
    main()
