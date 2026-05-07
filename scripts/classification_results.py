from itertools import product
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from callm.metrics import get_metric_from_id
from callm.data.classification import DATASETS
from callm.data import SimulationDataset
import pandas as pd
import subprocess
import tempfile
import os
import matplotlib.pyplot as plt
from expected_cost.calibration import calibration_with_crossval
import matplotlib.cm as cm

# ── Standard table layout ─────────────────────────────────────────────

TABLE_METRICS = [
    "cls_ner",
    "cls_ece_nbins=10",
    "cls_auc",
    "conf_norm_cross_entropy",
    "conf_norm_brier",
    "cls_nce",
    "cls_nbs",
    "cls_aurc",
    "cls_norm_n-ecuas_n=0",
    "cls_norm_n-ecuas_n=1",
    "cls_norm_n-ecuas_n=128",
]

SIM_METRICS = [
    "conf_error_rate",
    "conf_brier",
    "conf_cross_entropy",
    "conf_ece_nbins=10",
    "conf_auc",
    "conf_aurc",
    "conf_n-ecuas_n=0",
    "conf_n-ecuas_n=1",
    "conf_n-ecuas_n=128",
]


def generate_results_table(
    logs_dir: Path, table_metrics: list[str], output_filename: Path, seed: int = 42
) -> str:
    # if (
    #     (output_filename.with_suffix(".csv").exists())
    #     and (output_filename.with_suffix(".pdf").exists())
    #     and (output_filename.with_suffix(".tex").exists())
    # ):
    #     print(f"Results already exist at {output_filename}, skipping generation.")
    #     df = pd.read_csv(output_filename.with_suffix(".csv"), index_col=False)
    #     df = df.set_index(["dataset", "model", "proc"])
    #     return df

    results = []
    unique_metrics = {}
    higher_is_better = {}
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
            if "cls" in metric and "conf" not in metric:
                inpt, tgt = logits, labels
                cal_inpt, cal_tgt = calibrated_logprobs, labels
            elif "conf" in metric and "cls" not in metric:
                inpt, idx = torch.softmax(logits, dim=1).max(dim=1)
                tgt = (idx == labels).long()
                cal_inpt, idx = torch.softmax(calibrated_logprobs, dim=1).max(dim=1)
                cal_tgt = (idx == labels).long()
            else:
                raise ValueError(f"Metric ID {metric} is not recognized as classification or confidence metric.")
            results.append(
                {
                    "dataset_model": dataset,
                    "dataset": DATASETS[dataset]["dataset"],
                    "model": DATASETS[dataset]["model"],
                    "proc": "raw",
                    "metric": metric_info["display"],
                    "value": metric_info["function"](inpt, tgt),
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
                    "value": metric_info["function"](cal_inpt, cal_tgt),
                }
            )
            unique_metrics[metric] = metric_info["display"]
            higher_is_better[metric] = metric_info["higher_is_better"]

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

    # compute the higher_is_better dict for the metrics in the table
    higher_is_better = {r"\textbf{" + unique_metrics[metric] + r"}" : higher_is_better[metric] for metric in table_metrics}
    # import pdb; pdb.set_trace()  # --- IGNORE ---
    df = highlight_best_systems(df, higher_is_better)

    return df


def highlight_best_systems(df: pd.DataFrame, higher_is_better: dict) -> pd.DataFrame:
    # for each dataset, find the best value for each metric (considering all models and both raw and cal) and bold it
    df_str = df.copy()
    df_str = df_str.astype(str)
    for dataset in df.index.get_level_values("dataset").unique():
        dataset_df = df.xs(dataset, level="dataset")
        for metric in df.columns:
            if metric not in higher_is_better:
                continue
            elif not higher_is_better[metric]:
                best_value = dataset_df[metric].min()
            else:
                best_value = dataset_df[metric].max()
            best_mask = dataset_df[metric] == best_value

            for model, proc in best_mask.index:
                if best_mask.loc[(model, proc)]:
                    df_str.loc[(dataset, model, proc), metric] = r"\textbf{" + f"{float(df.loc[(dataset, model, proc), metric]):.4f}" + r"}"
                else:
                    df_str.loc[(dataset, model, proc), metric] = f"{float(df.loc[(dataset, model, proc), metric]):.4f}"
    return df_str

def generate_latex(df: pd.DataFrame, output_filename: Path):
    latex_doc = df.to_latex(
        float_format="%.3f",
        multirow=True,
        index_names=False,
        column_format="l" * df.index.nlevels + "c" * df.shape[1],
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
            metric_info = get_metric_from_id(f"cls_{norm_str}n-ecuas_n={n}")
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


def plot_gamma_ecuas(
    logs_dir: Path, output_path: Path, gammas: list[float], normalize: bool = False
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for dataset in DATASETS:
        logits, labels = load_scores(logs_dir / dataset)
        results = {}
        for gamma in gammas:
            norm_str = "norm_" if normalize else ""
            metric_info = get_metric_from_id(f"cls_{norm_str}gamma-ecuas_gamma={gamma}")
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


def plot_temperature_ecuas(
    logs_dir: Path, output_path: Path, temperatures: list[float], nseeds: int = 5
):
    from collections import OrderedDict
    DATASETS2 = OrderedDict(list(DATASETS.items())[:10])  # Only use the first 10 datasets for this plot to avoid clutter
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    ns = [0, 1]
    for i, n in enumerate(ns):
        for dataset in DATASETS2:
            logits, labels = load_scores(logs_dir / dataset)
            calibrated_logprobs = calibration_with_crossval(
                torch.log_softmax(logits, dim=1),
                labels,
                calparams={"bias": True},
                seed=nseeds,
            )
            calibrated_logprobs = torch.from_numpy(calibrated_logprobs).float()
            results = []
            for temp in temperatures:
                seed_results = []
                for seed in range(nseeds):  # Average over multiple runs for stability
                    if temp == 0:
                        pred = torch.argmax(calibrated_logprobs, dim=1)
                    else:
                        pred = torch.distributions.Categorical(logits=calibrated_logprobs / temp).sample()
                    probs = torch.softmax(calibrated_logprobs, dim=1)
                    confidence = probs[torch.arange(probs.size(0)), pred]
                    correctness = (pred == labels).float()
                    metric_info = get_metric_from_id(f"conf_n-ecuas_n={n}")
                    seed_results.append(metric_info["function"](confidence, correctness))
                results.append({
                    "dataset": DATASETS2[dataset]["dataset"],
                    "model": DATASETS2[dataset]["model"],
                    "temp": temp,
                    "median": np.median(seed_results),
                    "q1": np.percentile(seed_results, 25),
                    "q3": np.percentile(seed_results, 75),
                })
            df = pd.DataFrame(results)
            ax[i].plot(
                df["temp"],
                df["median"],
                # marker="o",
                linestyle="-",
                label=f"{DATASETS[dataset]['dataset']} - {DATASETS[dataset]['model']}",
            )
            ax[i].fill_between(df["temp"], df["q1"], df["q3"], alpha=0.2)

    ax[0].set_ylabel("ECUAS (n=0)", fontsize=14)
    ax[0].set_xlabel("Temperature", fontsize=14)
    ax[0].grid()
    ax[1].set_xlabel("Temperature", fontsize=14)
    ax[1].set_ylabel("ECUAS (n=1)", fontsize=14)
    ax[1].grid()

    # set global legend outside the plot
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)



def main(gammas, ns, temperatures, table_metrics, logs_dir, output_dir, seed, nseeds = 5):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating results table...")
    df = generate_results_table(logs_dir, table_metrics, output_dir / "classification_results", seed=seed)
    generate_latex(df, output_dir / "classification_results")
    
    # print("Generating n-ECUAS plot...")
    # plot_ecuas(logs_dir, output_dir / "classification_ecuas_plot.pdf", ns=ns, normalize=False)

    # print("Generating γ-ECUAS plot...")
    # plot_gamma_ecuas(
    #     logs_dir,
    #     output_dir / "classification_gamma_ecuas_plot.pdf",
    #     gammas=gammas,
    #     normalize=False,
    # )
    print("Generating temperature-ECUAS plot...")
    plot_temperature_ecuas(
        logs_dir,
        output_dir / "classification_temperature_ecuas_plot.pdf",
        temperatures=temperatures,
        nseeds = nseeds,
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
        "--temps",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 10.0, 20.0, 40.0],
        help="List of temperature values for Temperature-ECUAS computation",
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
    parser.add_argument(
        "--nseeds",
        type=int,
        default=5,
        help="Number of seeds to average over for temperature-ECUAS stability",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)

    main(args.gammas, args.ns, args.temps, args.table_metrics, logs_dir, output_dir, args.seed, args.nseeds)
