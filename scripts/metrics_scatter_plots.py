
from collections import OrderedDict
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import torch

from callm.metrics.classification import (
    ClassificationCrossEntropy,
    ClassificationBrierScore,
    ClassificationErrorRate,
    ClassificationnCCAS
)
from callm.data.classification import DATASETS

METRICS = OrderedDict([
    ("error_rate", {
        "cls": ClassificationErrorRate,
        "args": {"normalize": False},
        "display": "NER"
        }),
    ("cross_entropy", {
        "cls": ClassificationCrossEntropy,
        "args": {"normalize": False},
        "display": "NCE"
        }),
    ("brier_score", {
        "cls": ClassificationBrierScore,
        "args": {"normalize": False},
        "display": "NBS"
    }),
])

CCAS_METRICS = OrderedDict([
    ("0-ccas", {
        "cls": ClassificationnCCAS,
        "args": {"n": 0, "normalize": False},
        "display": "n-NCCAS (n=0)"
    }),
    ("1-ccas", {
        "cls": ClassificationnCCAS,
        "args": {"n": 1, "normalize": False},
        "display": "n-NCCAS (n=1)"
    }),
])

def load_scores(scores_dir: Path):
    logits = torch.from_numpy(np.load(scores_dir / f"scores.npy")).float()
    labels = torch.from_numpy(np.load(scores_dir / f"targets.npy")).long()
    return logits, labels


def plot_scatter(ccas_metrics, metrics, logs_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        print(f"Processing {dataset}...")
        logits, labels = load_scores(logs_dir / dataset)
        prior = torch.bincount(labels.long(), minlength=logits.size(1)).float() / labels.size(0)
        logpriors = torch.log(prior).unsqueeze(0).expand(logits.size(0), -1)

        fig, ax = plt.subplots(len(ccas_metrics), len(metrics), figsize=(5 * len(metrics), 4 * len(ccas_metrics)), sharex=True, sharey=True)
        for i, y_metric in enumerate(ccas_metrics):
            for j, x_metric in enumerate(metrics):

                # Compute x metric without reduction                
                x_metric_obj = METRICS[x_metric]["cls"](reduction="none", **METRICS[x_metric]["args"])
                x_metric_obj.update(logits, labels)
                x_results = x_metric_obj.compute().float().numpy()
                # Compute x metric of dummy model to normalize
                x_metric_dummy_obj = METRICS[x_metric]["cls"](reduction="mean", **METRICS[x_metric]["args"])
                x_metric_dummy_obj.update(logpriors, labels)
                x_dummy_result = x_metric_dummy_obj.compute().float().item()
                x_results = x_results / x_dummy_result
                
                # Compute y metric without reduction
                y_metric_obj = CCAS_METRICS[y_metric]["cls"](reduction="none", **CCAS_METRICS[y_metric]["args"])
                y_metric_obj.update(logits, labels)
                y_results = y_metric_obj.compute().float().numpy()
                # Compute y metric of dummy model to normalize
                y_metric_dummy_obj = CCAS_METRICS[y_metric]["cls"](reduction="mean", **CCAS_METRICS[y_metric]["args"])
                y_metric_dummy_obj.update(logpriors, labels)
                y_dummy_result = y_metric_dummy_obj.compute().float().item()
                y_results = y_results / y_dummy_result

                corrcoef = np.corrcoef(x_results, y_results)[0, 1]
                ax[i, j].scatter(x_results, y_results, rasterized=True, alpha=0.5)
                ax[i, j].set_title(f"{DATASETS[dataset]['dataset']} - {DATASETS[dataset]['model']} (r={corrcoef:.2f})")
                ax[i, j].grid()

        for i, m in enumerate(ccas_metrics):
            ax[i, 0].set_ylabel(CCAS_METRICS[m]["display"])
        for j, m in enumerate(metrics):
            ax[-1, j].set_xlabel(METRICS[m]["display"])

        fig.tight_layout()
        fig.savefig(output_dir / f"{dataset}_scatter_plot.pdf", dpi=100)
        plt.close(fig)

def main(ccas_metrics, metrics, logs_dir, output_dir):
    plot_scatter(ccas_metrics, metrics, logs_dir, output_dir / "scatter_plots")
            



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate scatter plots for metrics.")
    parser.add_argument("--metrics", nargs="+", help="List of metrics to plot.", default=METRICS)
    parser.add_argument("--ccas_metrics", nargs="+", help="List of CCAS metrics to plot.", default=CCAS_METRICS)
    parser.add_argument("--logs_dir", type=str, help="Directory containing the log files.", default=f"{str(Path(__file__).parent.parent)}/scores/classification")
    parser.add_argument("--output_dir", type=str, help="Directory to save the scatter plots.", default=f"{str(Path(__file__).parent.parent)}/outputs")
    args = parser.parse_args()
    main(
        ccas_metrics=args.ccas_metrics,
        metrics=args.metrics,
        logs_dir=Path(args.logs_dir),
        output_dir=Path(args.output_dir)
    )