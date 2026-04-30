import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import argparse


def compute_cost(q, c, K, n):
    """
    Computes the cost given:
      q: confidence
      c: base cost \tilde{C}^* (0 for correct, 1 for incorrect)
      K: number of classes (used to compute max uncertainty u_M)
      n: parameter defining the weight function w(gamma)
    """
    u = 1 - q
    u_M = 1 - 1 / K

    # Handle q=1 (u=0) carefully to avoid log(0) for n=0
    u = np.maximum(u, 1e-12)

    if n == 0:
        base_cost = u + c * (np.log(u_M) - np.log(u))
    else:
        base_cost = u ** (n + 1) + c * ((n + 1) / n) * (u_M**n - u**n)
    factor = 1.0 / (u_M ** (n + 1))
    return base_cost * factor


def main():
    parser = argparse.ArgumentParser(description="Plot q vs Cost for different n.")
    parser.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=[0, 1, 2, 4, 8, 64],
        help="List of n values to plot.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cost_vs_q_k_comparison.pdf",
        help="Output filename for the plot.",
    )
    args = parser.parse_args()

    K_values = [2, 4, 16, float("inf")]
    fig, axes = plt.subplots(1, 4, figsize=(28, 6), sharey=True)

    # Pre-defined colors
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, K in zip(axes, K_values):
        # Minimum confidence is 1/K
        q_min = 1.0 / K if K != float("inf") else 0.0
        # Generate values for q
        q = np.linspace(q_min, 0.9999, 1000)

        for i, n in enumerate(args.ns):
            color = colors[i % len(colors)]

            # Correct prediction (c=0) -> Solid line
            cost_0 = compute_cost(q, c=0, K=K, n=n)
            ax.plot(
                q, cost_0, label=f"n={n}", color=color, linestyle="-", linewidth=2.5
            )

            # Incorrect prediction (c=1) -> Dashed line
            cost_1 = compute_cost(q, c=1, K=K, n=n)
            ax.plot(q, cost_1, color=color, linestyle="--", linewidth=2.5)

        ax.set_xlabel(r"$q_e$", fontsize=26)
        if ax == axes[0]:
            ax.set_ylabel(r"ECUAS$_n$", fontsize=26)

        K_title = r"\infty" if K == float("inf") else int(K)
        ax.set_title(rf"$K = {K_title}$", fontsize=28)
        ax.tick_params(axis="both", which="major", labelsize=20)

        # Cut the x-axis starting from q_min
        ax.set_xlim(q_min - 0.02, 1.02)
        ax.set_ylim(-0.01, 3.0)

    # Tight layout leaving space on the right for legends
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    # Get handles for n from the first axis
    handles, labels = axes[0].get_legend_handles_labels()

    # Create custom handles for line styles
    correct_line = mlines.Line2D(
        [], [], color="gray", linestyle="-", linewidth=2.5, label="Correct"
    )
    incorrect_line = mlines.Line2D(
        [], [], color="gray", linestyle="--", linewidth=2.5, label="Incorrect"
    )
    style_handles = [correct_line, incorrect_line]

    # Add legends to the right side of the figure
    fig.legend(
        handles=handles,
        labels=labels,
        loc="center left",
        bbox_to_anchor=(0.88, 0.65),
        fontsize=20,
        title="Value of $n$",
        title_fontsize=22,
    )
    fig.legend(
        handles=style_handles,
        loc="center left",
        bbox_to_anchor=(0.88, 0.25),
        fontsize=20,
        title="Prediction",
        title_fontsize=22,
    )

    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
