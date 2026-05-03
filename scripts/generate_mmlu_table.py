"""
Generate LaTeX tables with calibration metrics and ECUAS line plots from MMLU results.

Produces:
  1. Standard metrics table: Acc, AUROC, ECE, BS, CE, ECUAS (table_standard.tex)
  2. Gamma-ECUAS Line Plots: One plot per LLM showing γ-ECUAS variants vs. Gamma
  3. ECUAS Line Plots: One plot per LLM showing ECUAS vs n.

Usage:
    python generate_mmlu_table.py [--gammas 0.05 0.1 0.2 0.5 0.8 0.9 0.95] [--ns 0 1 2 4 8 16 32 64 128] [--output-dir .]
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path so we can import callm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from callm.metrics import (
    ExpectedCalibrationError,
    ConfidenceBrierScore as BrierScore,
    ConfidenceCrossEntropy as CrossEntropy,
    ConfidenceAUCScore as AUCScore,
    ConfidenceGammaECUAS as GammaCCAS,
    ConfidenceECUAS as CCAS,
    ConfidenceAURC as AURC,
)

# ── Directory → (Method, LLM) mapping ──────────────────────────────────

# Display-name mapping for LLMs
LLM_DISPLAY_NAMES = {
    "Qwen3.5-4B": "Qwen3.5 4B",
    "Qwen3.5-9B": "Qwen3.5 9B",
    "GLM-4.6V-Flash": "GLM-4.6V-Flash",
    "Ministral-3-8B-Instruct-2512": "Ministral-3-8B-Instruct-2512",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-pro": "Gemini 2.5 pro",
}

# Method display names
METHOD_DISPLAY_NAMES = {
    "label_prob": "Sequence Posterior",
    "is_true": "Is True",
    "verbalized": "Verbalized",
}

# Method ordering for table rows
METHOD_ORDER = ["Sequence Posterior", "Is True", "Verbalized"]

# LLM ordering for table groups
LLM_ORDER = [
    "Qwen3.5 4B",
    "Qwen3.5 9B",
    "GLM-4.6V-Flash",
    "Ministral-3-8B-Instruct-2512",
    "Gemini 2.5 Flash Lite",
    "Gemini 2.5 Flash",
    "Gemini 2.5 pro",
]


def find_csv_for_experiment(exp_dir: Path) -> Path | None:
    dir_name = exp_dir.name
    if dir_name.startswith("is_true_"):
        csv_path = exp_dir / "merged_results.csv"
        if csv_path.exists():
            return csv_path
    else:
        # Base runs like label_prob and verbalized
        csv_path = exp_dir / "llm_outputs.csv"
        if csv_path.exists():
            return csv_path
    return None


def parse_experiment_dir(dir_name: str) -> tuple[str, str] | None:
    clean_name = (
        dir_name[: -len("_evaluation")]
        if dir_name.endswith("_evaluation")
        else dir_name
    )
    method_key = None
    remainder = clean_name
    for prefix in ["is_true", "label_prob", "verbalized"]:
        if clean_name.startswith(prefix + "_"):
            method_key = prefix
            remainder = clean_name[len(prefix) + 1 :]
            break
    if method_key is None:
        return None

    model_name = remainder
    model_name = (
        model_name.replace("zero_shot_", "")
        .replace("zero_shot", "")
        .replace("-no-thinking", "")
        .rstrip("_- ")
    )

    method_display = METHOD_DISPLAY_NAMES.get(method_key, method_key)
    llm_display = LLM_DISPLAY_NAMES.get(model_name, model_name)
    return method_display, llm_display


def load_csv_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    confidences = []
    correctness = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                conf_str = (
                    row.get("Confidence")
                    if "Confidence" in row
                    else row.get("confidence")
                )
                conf = float(conf_str)
            except (ValueError, TypeError, KeyError):
                conf = float("nan")

            if "Correct" in row:
                is_correct = row.get("Correct", "").strip().lower() == "yes"
            else:
                pred = str(row.get("pred_answer", "")).strip().lower()
                gold = str(row.get("gold_answers", "")).strip().lower()
                is_correct = pred == gold

            confidences.append(conf)
            correctness.append(float(is_correct))

    return np.array(confidences), np.array(correctness)


def compute_standard_metrics(
    confidences: np.ndarray, correctness: np.ndarray
) -> dict[str, float]:
    conf_t = torch.tensor(confidences, dtype=torch.float32)
    corr_t = torch.tensor(correctness, dtype=torch.float32)
    accuracy = float(corr_t.mean()) if len(corr_t) > 0 else 0.0

    ece = ExpectedCalibrationError(n_bins=10)
    bs = BrierScore()
    ce = CrossEntropy()
    auc = AUCScore()
    aurc = AURC()
    ccas_0 = CCAS(n=0)
    ccas_1 = CCAS(n=1)

    # NaN confidences are handled internally by the metrics (fallback to 0.5)
    for metric in [ece, bs, ce, auc, aurc, ccas_0, ccas_1]:
        metric.update(conf_t, corr_t)

    return {
        "Acc": accuracy,
        "AUROC": auc.compute().item(),
        "ECE": ece.compute().item(),
        "BS": bs.compute().item(),
        "CE": ce.compute().item(),
        "AURC": aurc.compute().item(),
        "ECUAS_0": ccas_0.compute().item(),
        "ECUAS_1": ccas_1.compute().item(),
    }


def compute_ecuas_metrics(
    confidences: np.ndarray, correctness: np.ndarray, ns: list[int]
) -> dict[str, float]:
    conf_t = torch.tensor(confidences, dtype=torch.float32)
    corr_t = torch.tensor(correctness, dtype=torch.float32)

    results = {}
    for n in ns:
        metric = CCAS(n=n)
        metric.update(conf_t, corr_t)
        results[f"ECUAS(n={n})"] = metric.compute().item()

    return results


def compute_gamma_ccas_metrics(
    confidences: np.ndarray, correctness: np.ndarray, gammas: list[float]
) -> dict[str, float]:
    conf_t = torch.tensor(confidences, dtype=torch.float32)
    corr_t = torch.tensor(correctness, dtype=torch.float32)

    results = {}

    for gamma in gammas:
        metric = GammaCCAS(gamma=gamma)
        metric.update(conf_t, corr_t)
        col_name = f"γ-ECUAS({gamma})"
        results[col_name] = metric.compute().item()

    return results


# ── Standard table layout ─────────────────────────────────────────────

STANDARD_COLUMNS = ["Acc", "AUROC", "ECE", "BS", "CE", "AURC", "ECUAS_0", "ECUAS_1"]
STANDARD_DIRECTION = {
    "Acc": True,
    "AUROC": True,
    "ECE": False,
    "BS": False,
    "CE": False,
    "AURC": False,
    "ECUAS_0": False,
    "ECUAS_1": False,
}
STANDARD_LATEX_HEADER = {
    "Acc": r"\textbf{Acc}",
    "AUROC": r"\textbf{AUROC}",
    "ECE": r"\textbf{ECE}",
    "BS": r"\textbf{BS}",
    "CE": r"\textbf{CE}",
    "AURC": r"\textbf{AURC}",
    "ECUAS_0": r"\textbf{ECUAS$_0$}",
    "ECUAS_1": r"\textbf{ECUAS$_1$}",
}


def format_val(val: float, is_best: bool, precision: int = 3) -> str:
    s = f"{val:.{precision}f}"
    return r"\textbf{" + s + "}" if is_best else s


def find_best_values(
    llm_methods: dict, columns: list[str], directions: dict[str, bool]
) -> dict[str, float]:
    best = {}
    for col in columns:
        vals = [
            llm_methods[m][col]
            for m in METHOD_ORDER
            if m in llm_methods
            and col in llm_methods[m]
            and not np.isnan(llm_methods[m][col])
        ]
        if vals:
            best[col] = max(vals) if directions.get(col, False) else min(vals)
    return best


def generate_standard_table(results: dict[str, dict[str, dict[str, float]]]) -> str:
    n_metrics = len(STANDARD_COLUMNS)
    col_spec = "cc|" + "c" * n_metrics
    header_cols = " & ".join(STANDARD_LATEX_HEADER[m] for m in STANDARD_COLUMNS)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        r"\textbf{LLM} & \textbf{Method} & " + header_cols + r"\\",
        r"\hline",
    ]

    for llm in LLM_ORDER:
        if llm not in results:
            continue
        llm_methods = results[llm]
        n_methods = sum(1 for m in METHOD_ORDER if m in llm_methods)
        if n_methods == 0:
            continue

        best = find_best_values(llm_methods, STANDARD_COLUMNS, STANDARD_DIRECTION)

        first = True
        for method in METHOD_ORDER:
            if method not in llm_methods:
                continue

            row_data = llm_methods[method]
            row_prefix = (
                r"\multirow{" + str(n_methods) + r"}{*}{" + llm + r"} & "
                if first
                else " & "
            )
            row_prefix += method

            row_cells = []
            for col in STANDARD_COLUMNS:
                if col in row_data and not np.isnan(row_data[col]):
                    is_best = row_data[col] == best[col]
                    row_cells.append(format_val(row_data[col], is_best))
                else:
                    row_cells.append("-")

            lines.append(row_prefix + " & " + " & ".join(row_cells) + r" \\")
            first = False
        lines.append(r"\hline")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{Standard Calibration Metrics (MMLU)}",
            r"\label{tab:standard_mmlu}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def generate_gamma_ecuas_plots(results: dict, gammas: list[float], output_dir: Path):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    for llm in LLM_ORDER:
        if llm not in results:
            continue

        llm_methods = results[llm]

        plt.figure(figsize=(8, 5))
        for method in METHOD_ORDER:
            if method not in llm_methods:
                continue

            y_vals = []
            for gamma in gammas:
                col_name = f"γ-ECUAS({gamma})"
                y_vals.append(llm_methods[method].get(col_name, np.nan))

            plt.plot(gammas, y_vals, marker="o", label=method)

        plt.title(f"γ-ECUAS vs γ for {llm}")
        plt.xlabel("γ")
        plt.ylabel("γ-ECUAS")
        plt.legend()
        plt.grid(True)

        out_path = (
            plots_dir
            / f"{llm.replace(' ', '_').replace('.', '_')}_gamma-ecuas_plots.png"
        )
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()


def generate_ecuas_plots(results: dict, ns: list[int], output_dir: Path):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    for llm in LLM_ORDER:
        if llm not in results:
            continue

        llm_methods = results[llm]

        plt.figure(figsize=(8, 5))
        for method in METHOD_ORDER:
            if method not in llm_methods:
                continue

            y_vals = []
            for n in ns:
                col_name = f"ECUAS(n={n})"
                y_vals.append(llm_methods[method].get(col_name, np.nan))

            plt.plot(ns, y_vals, marker="o", label=method)

        plt.title(f"ECUAS vs n for {llm}")
        plt.xlabel("n")
        plt.xscale("symlog", linthresh=1.0)
        plt.xticks(ns, labels=[str(n) for n in ns])
        plt.ylabel("ECUAS")
        plt.legend()
        plt.grid(True)

        out_path = (
            plots_dir / f"{llm.replace(' ', '_').replace('.', '_')}_ecuas_plots.png"
        )
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gammas", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
    )
    parser.add_argument(
        "--ns", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64, 128]
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/eernst/callm/lightning_logs/paper_results/mmlu",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="/home/eernst/callm/lightning_logs/paper_results/mmlu",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = defaultdict(lambda: defaultdict(dict))

    for exp_dir in logs_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        parsed = parse_experiment_dir(exp_dir.name)
        if not parsed:
            continue
        method, llm = parsed

        csv_path = find_csv_for_experiment(exp_dir)
        if not csv_path:
            continue

        confidences, correctness = load_csv_data(csv_path)
        if len(confidences) == 0:
            continue

        std_metrics = compute_standard_metrics(confidences, correctness)
        g_metrics = compute_gamma_ccas_metrics(confidences, correctness, args.gammas)
        ECUAS_metrics = compute_ecuas_metrics(confidences, correctness, args.ns)

        results[llm][method].update(std_metrics)
        results[llm][method].update(g_metrics)
        results[llm][method].update(ECUAS_metrics)

    tex_table = generate_standard_table(results)
    with open(out_dir / "table_standard.tex", "w") as f:
        f.write(tex_table)

    generate_gamma_ecuas_plots(results, args.gammas, out_dir)
    generate_ecuas_plots(results, args.ns, out_dir)
    print(f"Generated results in {out_dir}")


if __name__ == "__main__":
    main()
