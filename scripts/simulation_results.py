from itertools import product
from pathlib import Path

import torch
from tqdm import tqdm

from callm.data import SimulationDataset, SimulationDataset1D
from ecuas import get_metric_from_id
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def relative_difference(d):
    answer = d.set_index("proc").loc["answer", "value"]
    eq_group = d.set_index("proc").loc["eq-group", "value"]
    result = (answer - eq_group) / answer if answer != 0 else 0.0
    return result


def plot_heatmaps(K, N, metrics, output_dir, unidim, seed):
    results = []

    if unidim:
        SimulationClass = SimulationDataset1D
    else:
        SimulationClass = SimulationDataset

    torch.manual_seed(seed + 1)
    min_sigma_K = 1e-10
    max_sigma_K = 2
    min_sigma_N = 1e-10
    max_sigma_N = 1
    num_grid_points = 40
    sigma_K_values = torch.linspace(min_sigma_K, max_sigma_K, num_grid_points).tolist()
    sigma_N_values = torch.linspace(min_sigma_N, max_sigma_N, num_grid_points).tolist()
    for sigma_K, sigma_N in tqdm(product(sigma_K_values, sigma_N_values)):
        conf_eqclass, conf_answer, correctness, _, _ = SimulationClass(
            num_samples=1000,
            num_eqclasses=K,
            samples_per_eqclass=N,
            sigma_K=sigma_K,
            sigma_N=sigma_N,
            suboptimal_T=1.0,
            seed=seed,
        ).generate_confidences()

        unique_metrics = {}
        for metric in metrics + ["conf_error_rate"]:
            metric_info = get_metric_from_id(metric)
            unique_metrics[metric] = metric_info["display"]
            results.extend(
                [
                    {
                        "proc": "answer",
                        "sigma_K": sigma_K,
                        "sigma_N": sigma_N,
                        "metric": metric,
                        "value": metric_info["function"](conf_answer, correctness),
                    },
                    {
                        "proc": "eq-group",
                        "sigma_K": sigma_K,
                        "sigma_N": sigma_N,
                        "metric": metric,
                        "value": metric_info["function"](conf_eqclass, correctness),
                    },
                ]
            )

    error_rate_results = (
        pd.DataFrame(results)
        .loc[lambda df: (df["metric"] == "conf_error_rate") & (df["proc"] == "answer")]
        .pivot(index="sigma_K", columns="sigma_N", values="value")
        .sort_index(axis=0, ascending=False)
        .sort_index(axis=1, ascending=True)
    )
    rel_diff = (
        pd.DataFrame(results)
        .loc[lambda df: df["metric"] != "conf_error_rate"]
        .groupby(["sigma_K", "sigma_N", "metric"])
        .apply(relative_difference)
        .reset_index()
    )

    fig, ax = plt.subplots(
        1, len(metrics) + 1, figsize=(5 * len(metrics), 5), sharey=True, sharex=True
    )
    vmin = min(rel_diff[0].min(), error_rate_results.min().min())
    vmax = max(rel_diff[0].max(), error_rate_results.max().max())
    cmap = cm.Reds
    norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
    ax[0].imshow(
        error_rate_results,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        extent=[min_sigma_N, max_sigma_N, min_sigma_K, max_sigma_K],
    )
    ax[0].set_title("ER", fontsize=14)
    ax[0].set_xlabel(r"$\sigma_N$", fontsize=14)
    ax[0].set_ylabel(r"$\sigma_K$", fontsize=14)

    for j, (metric, group) in enumerate(rel_diff.groupby("metric"), start=1):
        this_heatmap = (
            group.pivot(index="sigma_K", columns="sigma_N", values=0)
            .sort_index(axis=0, ascending=False)
            .sort_index(axis=1, ascending=True)
        )
        ax[j].imshow(
            this_heatmap,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=[min_sigma_N, max_sigma_N, min_sigma_K, max_sigma_K],
        )
        ax[j].set_title(f"Relative {unique_metrics[metric]} change", fontsize=14)
        ax[j].set_xlabel(r"$\sigma_N$", fontsize=14)

    pos_left = ax[0].get_position()  # bounding box del primer plot
    pos_right = ax[-1].get_position()  # bounding box del último plot
    cbar_ax = fig.add_axes(
        [
            pos_right.x1 + 0.01,  # justo a la derecha del último plot
            pos_left.y0,  # mismo bottom
            0.02,  # ancho de la barra
            pos_left.height,  # mismo alto que los plots
        ]
    )
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, cax=cbar_ax)
    if unidim:
        filename = f"heatmap_1D_K={K}_N={N}_seed={seed}.pdf"
    else:
        filename = f"heatmap_K={K}_N={N}_seed={seed}.pdf"
    plt.savefig(output_dir / filename, bbox_inches="tight", dpi=300)


def main(K, N, metrics, output_dir, unidim, seed):
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_heatmaps(K, N, metrics, output_dir, unidim=unidim, seed=seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate selective classification results table"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "conf_aurc",
            "conf_n-ecuas_n=0",
            "conf_n-ecuas_n=1",
        ],
        help="List of metric identifiers to include in the heatmap",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save the heatmaps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=100,
        help="Number of eq-classes (K)",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=5,
        help="Number of samples per eq-class (N)",
    )
    parser.add_argument(
        "--unidim",
        action="store_true",
        help="Whether to use the 1D version of the simulation dataset (instead of K-D)",
    )
    args = parser.parse_args()
    main(
        K=args.K,
        N=args.N,
        metrics=args.metrics,
        output_dir=Path(args.output_dir),
        unidim=args.unidim,
        seed=args.seed,
    )
