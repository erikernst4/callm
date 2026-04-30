from collections import OrderedDict
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import torch

from callm.metrics.classification import (
    ClassificationCrossEntropy,
    ClassificationBrierScore,
    ClassificationErrorRate,
    ClassificationNCCAS,
)
from callm.data.classification import DATASETS

METRICS = OrderedDict(
    [
        (
            "error_rate",
            {
                "cls": ClassificationErrorRate,
                "args": {"normalize": False},
                "display": "NER",
            },
        ),
        (
            "cross_entropy",
            {
                "cls": ClassificationCrossEntropy,
                "args": {"normalize": False},
                "display": "NCE",
            },
        ),
        (
            "brier_score",
            {
                "cls": ClassificationBrierScore,
                "args": {"normalize": False},
                "display": "NBS",
            },
        ),
    ]
)

CCAS_METRICS = OrderedDict(
    [
        (
            "0-ccas",
            {
                "cls": ClassificationNCCAS,
                "args": {"n": 0, "normalize": False},
                "display": "n-NCCAS (n=0)",
            },
        ),
        (
            "1-ccas",
            {
                "cls": ClassificationNCCAS,
                "args": {"n": 1, "normalize": False},
                "display": "n-NCCAS (n=1)",
            },
        ),
    ]
)


def load_scores(scores_dir: Path):
    logits = torch.from_numpy(np.load(scores_dir / "scores.npy")).float()
    labels = torch.from_numpy(np.load(scores_dir / "targets.npy")).long()
    return logits, labels


def compute_sample_metric(logits, labels, metric, logpriors=None):
    if metric in CCAS_METRICS:
        metrics_dict = CCAS_METRICS
    elif metric in METRICS:
        metrics_dict = METRICS
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Compute x metric without reduction
    metric_obj = metrics_dict[metric]["cls"](
        reduction="none", **metrics_dict[metric]["args"]
    )
    metric_obj.update(logits, labels)
    results = metric_obj.compute().float().numpy()

    if logpriors is not None:
        # Compute x metric of dummy model to normalize
        metric_dummy_obj = metrics_dict[metric]["cls"](
            reduction="mean", **metrics_dict[metric]["args"]
        )
        metric_dummy_obj.update(logpriors, labels)
        dummy_result = metric_dummy_obj.compute().float().item()
        results = results / dummy_result

    return results

def autoscale_grid(axes, by_col_x=True, by_row_y=True):
    """
    Autoscale axes in a subplot grid.
    
    Parameters:
        axes     : 2D array of matplotlib Axes
        by_col_x : if True, each column shares the same fitted x range
        by_row_y : if True, each row shares the same fitted y range
    """
    axes = np.array(axes)
    n_rows, n_cols = axes.shape

    def get_data_range(ax, axis='x'):
        vals_min, vals_max = np.inf, -np.inf
        for line in ax.get_lines():
            data = line.get_xdata() if axis == 'x' else line.get_ydata()
            if len(data):
                vals_min = min(vals_min, np.nanmin(data))
                vals_max = max(vals_max, np.nanmax(data))
        for coll in ax.collections:
            offsets = coll.get_offsets()
            if len(offsets):
                idx = 0 if axis == 'x' else 1
                vals_min = min(vals_min, np.nanmin(offsets[:, idx]))
                vals_max = max(vals_max, np.nanmax(offsets[:, idx]))
        return vals_min, vals_max

    # Fix x per column
    if by_col_x:
        for col in range(n_cols):
            x_min, x_max = np.inf, -np.inf
            for row in range(n_rows):
                lo, hi = get_data_range(axes[row, col], axis='x')
                x_min, x_max = min(x_min, lo), max(x_max, hi)
            if x_min < x_max:
                margin = (x_max - x_min) * 0.05
                for row in range(n_rows):
                    axes[row, col].set_xlim(x_min - margin, x_max + margin)

    # Fix y per row
    if by_row_y:
        for row in range(n_rows):
            y_min, y_max = np.inf, -np.inf
            for col in range(n_cols):
                lo, hi = get_data_range(axes[row, col], axis='y')
                y_min, y_max = min(y_min, lo), max(y_max, hi)
            if y_min < y_max:
                margin = (y_max - y_min) * 0.05
                for col in range(n_cols):
                    axes[row, col].set_ylim(y_min - margin, y_max + margin)


def plot_scatter(ccas_metrics, metrics, logs_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        print(f"Processing {dataset}...")
        logits, labels = load_scores(logs_dir / dataset)
        prior = torch.bincount(
            labels.long(), minlength=logits.size(1)
        ).float() / labels.size(0)
        logpriors = torch.log(prior).unsqueeze(0).expand(logits.size(0), -1)

        fig, ax = plt.subplots(
            len(ccas_metrics),
            len(metrics),
            figsize=(5 * len(metrics), 4 * len(ccas_metrics)),
        )
        for i, y_metric in enumerate(ccas_metrics):
            for j, x_metric in enumerate(metrics):
                x_results = compute_sample_metric(logits, labels, x_metric, logpriors)
                y_results = compute_sample_metric(logits, labels, y_metric, logpriors)
                correct = labels == logits.argmax(dim=1)
                incorrect = ~correct
                # corrcoef = np.corrcoef(x_results, y_results)[0, 1]
                ax[i, j].scatter(x_results[correct], y_results[correct], rasterized=True, alpha=0.5, label="Correct")
                ax[i, j].scatter(x_results[incorrect], y_results[incorrect], rasterized=True, alpha=0.5, label="Incorrect")
                # ax[i, j].set_title(
                #     f"{DATASETS[dataset]['dataset']} - {DATASETS[dataset]['model']} (r={corrcoef:.2f})"
                # )
                ax[i, j].grid()
        for i, m in enumerate(ccas_metrics):
            ax[i, 0].set_ylabel(CCAS_METRICS[m]["display"])
            for j in range(1, len(metrics)):
                ax[i, j].set_yticklabels([])
        for j, m in enumerate(metrics):
            ax[-1, j].set_xlabel(METRICS[m]["display"])
            for i in range(len(ccas_metrics) - 1):
                ax[i, j].set_xticklabels([])

        autoscale_grid(ax, by_col_x=True, by_row_y=True)

        fig.suptitle(f"{DATASETS[dataset]['dataset']} - {DATASETS[dataset]['model']}", fontsize=16)

        # unified legend
        ax[-1,-1].legend(loc="lower right")

        fig.tight_layout()
        fig.savefig(output_dir / f"{dataset}_scatter_plot.pdf", dpi=100)
        plt.close(fig)


def main(ccas_metrics, metrics, logs_dir, output_dir):
    plot_scatter(ccas_metrics, metrics, logs_dir, output_dir / "scatter_plots")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate scatter plots for metrics.")
    parser.add_argument(
        "--metrics", nargs="+", help="List of metrics to plot.", default=METRICS
    )
    parser.add_argument(
        "--ccas_metrics",
        nargs="+",
        help="List of CCAS metrics to plot.",
        default=CCAS_METRICS,
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        help="Directory containing the log files.",
        default=f"{str(Path(__file__).parent.parent)}/scores/classification",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the scatter plots.",
        default=f"{str(Path(__file__).parent.parent)}/outputs",
    )
    args = parser.parse_args()
    main(
        ccas_metrics=args.ccas_metrics,
        metrics=args.metrics,
        logs_dir=Path(args.logs_dir),
        output_dir=Path(args.output_dir),
    )
