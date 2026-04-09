
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from callm.metrics.constants import CLASSIFICATION_METRICS
from callm.data.classification import DATASETS
import pandas as pd
import subprocess
import tempfile
import os
import matplotlib.pyplot as plt

# ── Standard table layout ─────────────────────────────────────────────
STANDARD_METRICS = ["error_rate", "brier", "cross_entropy", "auroc", "ece"]
CNCAG_NS = [0, 1, 128]

def generate_results_table(logs_dir: Path, output_filename: Path) -> str:

    results = []
    for dataset in DATASETS:
        logits, labels = load_scores(logs_dir / dataset)

        metric2name, dataset_results = compute_standard_metrics(logits, labels)
        for metric, result in dataset_results.items():
            results.append({
                "dataset_model": dataset,
                "dataset": DATASETS[dataset]["dataset"],
                "model": DATASETS[dataset]["model"],
                "metric": metric2name[metric],
                "value": result,
            })

        cncags_results = compute_cncag(logits, labels, ns=CNCAG_NS, normalize=True)
        for n, cncag_result in cncags_results.items():
            metric_name = "N" + CLASSIFICATION_METRICS["cncag"]["display"].format(n=n)
            metric2name[f"cncag_n{n}"] = metric_name
            results.append({
                "dataset_model": dataset,
                "dataset": DATASETS[dataset]["dataset"],
                "model": DATASETS[dataset]["model"],
                "metric": metric_name,
                "value": cncag_result,
            })

    df = pd.DataFrame(results).pivot_table(index=["dataset_model", "dataset", "model"], columns="metric", values="value").rename_axis(columns=None).reset_index().set_index("dataset_model")
    df = df.loc[DATASETS.keys()].reset_index(drop=True).set_index(["dataset", "model"])
    all_metrics = STANDARD_METRICS + [f"cncag_n{n}" for n in CNCAG_NS]
    df = df.loc[:, [metric2name[metric] for metric in all_metrics if metric in metric2name]]
    df.columns = [r"\textbf{" + col + r"}" for col in df.columns]

    latex_doc = df.to_latex(
        float_format="%.3f",
        multirow=True,
        index_names=False,
        column_format="ll" + "c" * df.shape[1],
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

        os.replace(generated_pdf, output_filename.with_suffix(".pdf"))  # Move generated PDF to desired location

    # Write the LaTeX code for the table (without standalone document) to the output directory
    with open(output_filename.with_suffix(".tex"), "w") as f:
        f.write(tex_doc)


def load_scores(scores_dir: Path):
    logits = torch.from_numpy(np.load(scores_dir / f"scores.npy")).float()
    labels = torch.from_numpy(np.load(scores_dir / f"targets.npy")).long()
    return logits, labels


def compute_standard_metrics(logits: torch.Tensor, labels: torch.Tensor):
    
    results = {}
    metric2name = {}

    # PSRs
    for metric in ["error_rate", "brier", "cross_entropy"]:
        m = CLASSIFICATION_METRICS[metric]["function"](normalize=True)
        m.update(logits, labels)
        metric_name = "N" + CLASSIFICATION_METRICS[metric]["display"]
        results[metric] = m.compute().item()
        metric2name[metric] = metric_name
    
    # AUROC
    m = CLASSIFICATION_METRICS["auroc"]["function"]()
    m.update(logits, labels)
    results["auroc"] = m.compute().item()
    metric2name["auroc"] = CLASSIFICATION_METRICS["auroc"]["display"]

    # ECE
    nbins = 10
    m = CLASSIFICATION_METRICS["ece"]["function"](n_bins=nbins)
    m.update(logits, labels)
    results["ece"] = m.compute().item()
    metric2name["ece"] = CLASSIFICATION_METRICS["ece"]["display"]

    return metric2name, results


def compute_cncag(logits: torch.Tensor, labels: torch.Tensor, ns: list[int], normalize=True):
    results = {}
    for n in ns:
        m = CLASSIFICATION_METRICS["cncag"]["function"](n=n, normalize=normalize)
        m.update(logits, labels)
        results[n] = m.compute().item()
    return results


def plot_cncag(
    logs_dir: Path,
    output_path: Path, 
    ns: list[int], 
    normalize: bool = True
):

    fig, ax = plt.subplots(1,1, figsize=(10, 5))
    for dataset in DATASETS:
        logits, labels = load_scores(logs_dir / dataset)
        cncags_results = compute_cncag(logits, labels, ns=ns, normalize=normalize)
        ax.plot(ns, [cncags_results[n] for n in ns], 
            marker="o", 
            linestyle="-", 
            label=f"{DATASETS[dataset]['dataset']} - {DATASETS[dataset]['model']}"
        )

    ax.set_xlabel("n")
    title = "NCnCAG" if normalize else "CnCAG"
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

def plot_gamma_ccag(
    logs_dir: Path,
    output_path: Path,
    gammas: list[float],
):
    fig, ax = plt.subplots(1,1, figsize=(10, 5))
    for dataset in DATASETS:
        logits, labels = load_scores(logs_dir / dataset)
        gamma_ccag_results = {}
        for gamma in gammas:
            m = CLASSIFICATION_METRICS["gamma_ccag"]["function"](gamma=gamma)
            m.update(logits, labels)
            gamma_ccag_results[gamma] = m.compute().item()

        ax.plot(gammas, [gamma_ccag_results[gamma] for gamma in gammas], 
            marker="o", 
            linestyle="-", 
            label=f"{DATASETS[dataset]['dataset']} - {DATASETS[dataset]['model']}"
        )

    ax.set_xlabel("γ")
    ax.set_ylabel("γ-CCAG")
    ax.set_title("γ-CCAG vs γ")
    ax.grid()
    # set legend outside the plot
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)


def main(gammas, ns, logs_dir, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)

    generate_results_table(logs_dir, output_dir / "classification_results")
    plot_cncag(logs_dir, output_dir / "classification_cncag_plot.pdf", ns=ns, normalize=False)
    plot_gamma_ccag(logs_dir, output_dir / "classification_gamma_ccag_plot.pdf", gammas=gammas)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate selective classification results table")
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0], help="List of gamma values for γ-CCAG computation")
    parser.add_argument("--ns", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64, 128], help="List of n values for CNCAG computation")
    parser.add_argument("--logs_dir", type=str, default="scores/classification", help="Directory containing the logs with scores and targets")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save the output files")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)

    main(args.gammas, args.ns, logs_dir, output_dir)