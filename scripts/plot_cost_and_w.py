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


def plot_W(ns, Ks, fig, ax):

    num_points = 1000
    gamma = np.linspace(0.001, 0.999, num_points)

    # legend_handles = None
    # legend_labels = None

    for k, K in enumerate(Ks):
        if K == float("inf"):
            u_M = 1.0
            label = r'$K=\infty$ $(u_M=1)$'
        else:
            u_M = (1 - 1/K)
            label = r'$K={}$ $(u_M={})$'.format(K, np.round(u_M, 2))
        for n in ns:
            alpha_n = (n + 1) / (u_M ** (n + 1))
            W = alpha_n * np.power(gamma, n - 1)
            line, = ax[k].plot(gamma, W, label=f'n={n}', linewidth=2.5)

        # if legend_handles is None:
        #     legend_handles, legend_labels = ax[k].get_legend_handles_labels()

        # Scale ylim with the n=1 level (w_1 = 2 / u_M^2, constant in gamma)
        # so that line sits at the same relative height in every panel
        ax[k].set_ylim([0, 2.5 * 2 / u_M**2])
        ax[k].set_xlim([0, u_M])
        ax[k].set_title(label, fontsize=28)
        ax[k].grid(True)
        ax[k].tick_params(axis='both', labelsize=20)
        ax[k].set_xlabel(r'$\gamma$', fontsize=26)
        if k == 0:
            ax[k].set_ylabel(r'$w_n$', fontsize=26)


    # fig.suptitle(r'')
    # fig.supylabel(r'$w_n(\gamma)$', fontsize=14)
    # fig.legend(legend_handles, legend_labels, fontsize=14, loc='lower center', ncol=len(ns), frameon=False, bbox_to_anchor=(0.5, -0.1))
    # fig.tight_layout(pad=1.5)
    # fig.savefig('../outputs/W_gamma_plot.pdf', bbox_inches='tight', dpi=300)

    # plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot q vs Cost for different n.")
    parser.add_argument(
        "--ns",
        type=float,
        nargs="+",
        default=[0, 1, 2, 4, 8, 64],
        help="List of n values to plot.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/cost_vs_qe_and_w_vs_gamma.pdf",
        help="Output filename for the plot.",
    )
    args = parser.parse_args()

    K_values = [2, 4, 16, float("inf")]
    fig, axes = plt.subplots(2, 4, figsize=(28, 12))
    # Share y-axis across the bottom row only; the top row gets per-K ylims
    for ax in axes[1][1:]:
        ax.sharey(axes[1][0])
        # ax.tick_params(labelleft=False)

    # Pre-defined colors
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plot_W(args.ns, K_values, fig, axes[0])

    for ax_idx, (ax, K) in enumerate(zip(axes[1], K_values)):
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
        if ax_idx == 0:
            ax.set_ylabel(r"$C_n^*$", fontsize=26)

        # K_title = r"\infty" if K == float("inf") else int(K)
        # ax.set_title(rf"$K = {K_title}$", fontsize=28)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.grid(True)

        # Cut the x-axis starting from q_min
        ax.set_xlim(q_min - 0.02, 1.02)
        ax.set_ylim(-0.01, 3.0)

    # Tight layout leaving space on the right for legends
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    # Get handles for n from the first axis
    handles, labels = axes[1][0].get_legend_handles_labels()

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
        bbox_to_anchor=(0.88, 0.55),
        fontsize=20,
        title="Value of $n$",
        title_fontsize=22,
    )
    fig.legend(
        handles=style_handles,
        loc="center left",
        bbox_to_anchor=(0.88, 0.35),
        fontsize=20,
        title="Prediction",
        title_fontsize=22,
    )

    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
