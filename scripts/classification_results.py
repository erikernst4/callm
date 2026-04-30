from pathlib import Path

import numpy as np
import torch
from callm.metrics import get_metric_from_id
from callm.data.classification import DATASETS
import pandas as pd
import subprocess
import tempfile
import os
import matplotlib.pyplot as plt
from expected_cost.calibration import calibration_with_crossval

# ── Standard table layout ─────────────────────────────────────────────

TABLE_METRICS = [
    "cls_ner",
    "cls_nbs",
    "cls_nce",
    "cls_auc",
    "cls_aurc",
    "cls_ece_nbins=10",
    "cls_norm_n-ccas_n=0",
    "cls_norm_n-ccas_n=1",
    "cls_norm_n-ccas_n=128",
]

# KEEP fisrt 3 first items from oredered dict DATASETS
# DATASETS = OrderedDict(list(DATASETS.items())[:3])


def generate_results_table(
    logs_dir: Path, table_metrics: list[str], output_filename: Path, seed: int = 42
) -> str:
    if (
        (output_filename.with_suffix(".csv").exists())
        and (output_filename.with_suffix(".pdf").exists())
        and (output_filename.with_suffix(".tex").exists())
    ):
        print(f"Results already exist at {output_filename}, skipping generation.")
        df = pd.read_csv(output_filename.with_suffix(".csv"), index_col=False)
        df = df.set_index(["dataset", "model", "proc"])
        return df

    results = []
    unique_metrics = {}
    for dataset in DATASETS:
        print("Processing dataset:", dataset)
        logits, labels = load_scores(logs_dir / dataset)

        # Calibrate and add metric value for calibrated scores
        logpost_raw = torch.log_softmax(logits, dim=1)
        calibrated_logprobs = calibration_with_crossval(
            logpost_raw, labels, seed=seed, calparams={"bias": True}
        )
        calibrated_logprobs = torch.from_numpy(calibrated_logprobs).float()

        for metric in table_metrics:
            metric_info = get_metric_from_id(metric)
            results.append(
                {
                    "dataset_model": dataset,
                    "dataset": DATASETS[dataset]["dataset"],
                    "model": DATASETS[dataset]["model"],
                    "proc": "raw",
                    "metric": metric_info["display"],
                    "value": metric_info["function"](logits, labels),
                }
            )
            # Append calibrated results
            results.append(
                {
                    "dataset_model": dataset,
                    "dataset": DATASETS[dataset]["dataset"],
                    "model": DATASETS[dataset]["model"],
                    "proc": "cal",
                    "metric": metric_info["display"],
                    "value": metric_info["function"](calibrated_logprobs, labels),
                }
            )
            unique_metrics[metric] = metric_info["display"]

    df = (
        pd.DataFrame(results)
        .pivot_table(
            index=["dataset_model", "dataset", "model", "proc"],
            columns="metric",
            values="value",
        )
        .rename_axis(columns=None)
        .reset_index()
        .set_index("dataset_model")
    )
    df = (
        df.loc[DATASETS.keys()]
        .reset_index(drop=True)
        .set_index(["dataset", "model", "proc"])
    )
    df = df.groupby(level=["dataset", "model"], sort=False, group_keys=False).apply(
        lambda g: g.sort_index(level="proc", ascending=False)
    )
    df = df.loc[
        :, [unique_metrics[metric] for metric in table_metrics]
    ]  # Ensure columns are in the same order as table_metrics
    df.columns = [r"\textbf{" + col + r"}" for col in df.columns]
    df.reset_index().to_csv(output_filename.with_suffix(".csv"), index=False)
    return df


def generate_latex(df: pd.DataFrame, output_filename: Path):
    latex_doc = df.to_latex(
        float_format="%.3f",
        multirow=True,
        index_names=False,
        column_format="lll" + "c" * df.shape[1],
        escape=False,
    )

    tex_doc = (
        r"\begin{table}[h]" + "\n"
        r"\centering" + "\n"
        r"\resizebox{\columnwidth}{!}{%" + "\n"
        f"{latex_doc}"
        r"}" + "\n"
        r"\caption{Standard Calibration Metrics}" + "\n"
        r"\label{tab:standard}" + "\n"
        r"\end{table}" + "\n"
    )

    standalone_pdf_doc = (
        r"\documentclass{standalone}" + "\n"
        r"\usepackage{booktabs}" + "\n"
        r"\usepackage{multirow}" + "\n"
        r"\begin{document}" + "\n"
        f"{latex_doc}"
        r"\end{document}"
    )

    # Write the LaTeX code for the table (without standalone document) to the output directory
    with open(output_filename.with_suffix(".tex"), "w") as f:
        f.write(tex_doc)

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "table.tex")

        # Write LaTeX file
        with open(tex_path, "w") as f:
            f.write(standalone_pdf_doc)

        # Compile LaTeX → PDF
        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "table.tex"],
                cwd=tmpdir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            raise RuntimeError("LaTeX compilation failed")

        # Move resulting PDF
        generated_pdf = os.path.join(tmpdir, "table.pdf")
        if not os.path.exists(generated_pdf):
            raise RuntimeError("PDF was not generated")

        os.replace(
            generated_pdf, output_filename.with_suffix(".pdf")
        )  # Move generated PDF to desired location


def load_scores(scores_dir: Path):
    logits = torch.from_numpy(np.load(scores_dir / "scores.npy")).float()
    labels = torch.from_numpy(np.load(scores_dir / "targets.npy")).long()
    return logits, labels


def plot_ecuas(
    logs_dir: Path, output_path: Path, ns: list[int], normalize: bool = True
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for dataset in DATASETS:
        logits, labels = load_scores(logs_dir / dataset)
        results = {}
        for n in ns:
            norm_str = "norm_" if normalize else ""
            metric_info = get_metric_from_id(f"cls_{norm_str}n-ccas_n={n}")
            results[n] = metric_info["function"](logits, labels)

        ax.plot(
            ns,
            [results[n] for n in ns],
            marker="o",
            linestyle="-",
            label=f"{DATASETS[dataset]['dataset']} - {DATASETS[dataset]['model']}",
        )

    ax.set_xlabel("n")
    title = "ECUAS" if normalize else "n-ECUAS"
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.set_xticks(ns)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlim(left=0)
    ax.grid()
    # set legend outside the plot
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)


def plot_gamma_ccas(
    logs_dir: Path, output_path: Path, gammas: list[float], normalize: bool = False
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for dataset in DATASETS:
        logits, labels = load_scores(logs_dir / dataset)
        results = {}
        for gamma in gammas:
            norm_str = "norm_" if normalize else ""
            metric_info = get_metric_from_id(f"cls_{norm_str}gamma-ccas_gamma={gamma}")
            results[gamma] = metric_info["function"](logits, labels)

        ax.plot(
            gammas,
            [results[gamma] for gamma in gammas],
            marker="o",
            linestyle="-",
            label=f"{DATASETS[dataset]['dataset']} - {DATASETS[dataset]['model']}",
        )

    ax.set_xlabel("γ")
    ax.set_ylabel("γ-ECUAS")
    ax.set_title("γ-ECUAS vs γ")
    ax.grid()
    # set legend outside the plot
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)


def main(gammas, ns, table_metrics, logs_dir, output_dir, seed):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_results_table(
        logs_dir, table_metrics, output_dir / "classification_results", seed=seed
    )
    generate_latex(df, output_dir / "classification_results")
    plot_ecuas(
        logs_dir, output_dir / "classification_ecuas_plot.pdf", ns=ns, normalize=False
    )
    plot_gamma_ccas(
        logs_dir,
        output_dir / "classification_gamma_ccas_plot.pdf",
        gammas=gammas,
        normalize=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate selective classification results table"
    )
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0],
        help="List of gamma values for γ-ECUAS computation",
    )
    parser.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=[0, 1, 2, 4, 8, 16, 32, 64, 128],
        help="List of n values for n-ECUAS computation",
    )
    parser.add_argument(
        "--table-metrics",
        type=str,
        nargs="+",
        default=TABLE_METRICS,
        help="List of metric IDs to include in the results table",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="scores/classification",
        help="Directory containing the logs with scores and targets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for calibration (if applicable)",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)

    main(args.gammas, args.ns, args.table_metrics, logs_dir, output_dir, args.seed)
