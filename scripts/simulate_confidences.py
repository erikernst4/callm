from pathlib import Path

import torch
import matplotlib.pyplot as plt

from callm.metrics import (
    ExpectedCalibrationError,
    BrierScore,
    CrossEntropy,
    AUCScore,
    # CnCAG,
)


def compute_accuracy(correctness: torch.Tensor) -> float:
    return correctness.float().mean().item()


def compute_confidence(
    cls, confidences: torch.Tensor, correctness: torch.Tensor, **kwargs
) -> float:
    metric = cls(**kwargs)
    metric.update(confidences, correctness)
    return metric.compute().item()


CONFIDENCE_METRICS = {
    "ece": lambda confidences, correctness: compute_confidence(
        ExpectedCalibrationError, confidences, correctness, n_bins=10
    ),
    "brier": lambda confidences, correctness: compute_confidence(
        BrierScore, confidences, correctness
    ),
    "cross-entropy": lambda confidences, correctness: compute_confidence(
        CrossEntropy, confidences, correctness
    ),
    "auc": lambda confidences, correctness: compute_confidence(
        AUCScore, confidences, correctness
    ),
    # "c0cag": lambda confidences, correctness: compute_confidence(CnCAG, confidences, correctness, n=0),
    # "c1cag": lambda confidences, correctness: compute_confidence(CnCAG, confidences, correctness, n=1),
    # "c128cag": lambda confidences, correctness: compute_confidence(CnCAG, confidences, correctness, n=128),
}


def generate_confidences(num_samples, priors, N=5, sigma_expand=1.0, suboptimal_T=1.0):
    K = priors.shape[0]
    mu, sigma = [], []
    for k in range(K):
        one_hot = torch.zeros(K)
        one_hot[k] = 1.0
        mu_n = torch.rand(N, K) - 0.5 + one_hot
        mu.append(mu_n)
        min_diff = float("inf")
        # compute minimum distance between mu_n points
        for i in range(N):
            for j in range(i + 1, N):
                diff = torch.norm(mu_n[i] - mu_n[j])
                min_diff = min(min_diff, diff.item())
        sigma_n = min_diff / 2 / 3 * sigma_expand
        sigma.append(sigma_n)

    confidences = []
    correctness = []
    x_sampled = []
    for k_sampled in torch.randint(0, K, (num_samples,)):
        k = k_sampled.item()
        n = torch.randint(0, N, (1,)).item()
        x = torch.normal(mu[k][n], sigma[k])
        x_sampled.append(x)
        p_k = priors
        p_n_given_k = 1.0 / N / K
        p_x_given_n_k = torch.zeros(N, K)
        for kk in range(K):
            for nn in range(N):
                p_x_given_n_k[nn, kk] = torch.exp(
                    -0.5 * (torch.sum((x - mu[kk][nn]) ** 2) / sigma[kk] ** 2)
                ) / (sigma[kk] * torch.sqrt(torch.tensor(2.0) * torch.pi))
        p_n_k_given_x = (
            p_x_given_n_k
            * p_n_given_k
            * p_k
            / (p_x_given_n_k * p_n_given_k * p_k).sum()
        )
        p_k_given_x = p_n_k_given_x.sum(dim=0)
        if suboptimal_T is None:
            k_pred = torch.argmax(p_k_given_x).item()
        else:
            p_k_given_x_T = p_k_given_x ** (1.0 / suboptimal_T)
            p_k_given_x_T /= p_k_given_x_T.sum()
            k_pred = torch.multinomial(p_k_given_x_T, 1).item()

        conf = p_n_k_given_x[n, k].item()
        corr = 1.0 if k_pred == k else 0.0
        confidences.append(conf)
        correctness.append(corr)

    return (
        torch.tensor(confidences),
        torch.tensor(correctness),
        torch.stack(x_sampled),
        mu,
        sigma,
    )


def plot_distribution(mu, sigma, x, confidences, correctness, output_dir):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # plot circles for each sample
    for k, (m, s) in enumerate(zip(mu, sigma)):
        for i in range(m.shape[0]):
            ax[0].scatter(
                m[i, 0],
                m[i, 1],
                color=f"C{k}",
                label=f"Gaussian {k}" if i == 0 else None,
                alpha=1,
                marker="x",
            )
            circle = plt.Circle(
                (m[i, 0], m[i, 1]), s, color=f"C{k}", fill=False, alpha=0.5
            )
            ax[0].add_artist(circle)
    ax[0].scatter(
        x[:, 0], x[:, 1], color="black", label="Samples", alpha=0.5, marker="x", s=1
    )
    ax[0].set_xlabel("Mu 1")
    ax[0].set_ylabel("Mu 2")
    ax[0].grid(True)
    ax[0].legend()
    ax[0].set_aspect("equal", adjustable="datalim")

    # Compute accuracy
    acc_result = compute_accuracy(correctness)
    ax[1].text(
        0.05,
        0.50,
        f"Accuracy: {acc_result:.4f}",
        transform=ax[1].transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    # Compute confidence metrics
    for name, metric in CONFIDENCE_METRICS.items():
        result = metric(confidences, correctness)
        ax[1].text(
            0.05,
            0.95 - 0.05 * list(CONFIDENCE_METRICS.keys()).index(name),
            f"{name}: {result:.4f}",
            transform=ax[1].transAxes,
            fontsize=12,
            verticalalignment="top",
        )

    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "simulated_results.pdf", bbox_inches="tight", dpi=300)


def main(num_samples, priors, N, sigma_expand, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    confidences, correctness, x, mu, sigma = generate_confidences(
        num_samples=num_samples, priors=priors, N=N, sigma_expand=sigma_expand
    )

    if priors.shape[0] == 2:
        plot_distribution(mu, sigma, x, confidences, correctness, output_dir)
    else:
        # Compute accuracy
        acc_result = compute_accuracy(correctness)
        print(f"Accuracy: {acc_result:.4f}")

        # Compute confidence metrics
        for name, metric in CONFIDENCE_METRICS.items():
            result = metric(confidences, correctness)
            print(f"{name}: {result:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--priors", type=float, nargs="+", default=None)
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--sigma-expand", type=float, default=50.0)
    parser.add_argument(
        "--output-dir", type=str, default=f"{str(Path(__file__).parent.parent)}/outputs"
    )
    args = parser.parse_args()

    if args.priors is None and args.K is not None:
        priors = torch.tensor([1.0 / args.K] * args.K)
    elif args.priors is not None:
        priors = torch.tensor(args.priors)
    else:
        raise ValueError("Either --priors or --K must be provided.")

    output_dir = Path(args.output_dir)

    main(
        num_samples=args.num_samples,
        priors=priors,
        N=args.N,
        sigma_expand=args.sigma_expand,
        output_dir=output_dir,
    )
