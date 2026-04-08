"""
Generate LaTeX tables with calibration metrics and CCAG line plots from paper results.

Produces:
  1. Standard metrics table: Acc, AUROC, ECE, BS, CE, CCAG (table_standard.tex)
  2. CCAG Line Plots: One plot per LLM showing CCAG variants vs. Gamma
  3. CnCAG Line Plots: One plot per LLM showing CnCAG vs n.

Usage:
    python generate_paper_table.py [--gammas 0.05 0.1 0.2 0.5 0.8 0.9 0.95] [--ns 0 1 2 4 8 16 32 64 128] [--output-dir .]
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import tempfile

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path so we can import callm
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from callm.metrics import (
    ExpectedCalibrationError,
    BrierScore,
    ConfidenceCrossEntropy,
    AUCScore,
    CnCAG,
    GammaCCAG,
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

# CCAG variants
CnCAG_VARIANTS = [0, 1, 128]

def find_csv_for_experiment(exp_dir: Path) -> Path | None:
    dir_name = exp_dir.name
    if dir_name.startswith("is_true_"):
        csv_path = exp_dir / "merged_results.csv"
        if csv_path.exists():
            return csv_path
    else:
        if dir_name.endswith("_evaluation"):
            csv_path = exp_dir / "version_0" / "evaluation_results.csv"
            if csv_path.exists():
                return csv_path
    return None

def parse_experiment_dir(dir_name: str) -> tuple[str, str] | None:
    clean_name = dir_name[:-len("_evaluation")] if dir_name.endswith("_evaluation") else dir_name
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
    model_name = model_name.replace("zero_shot_", "").replace("zero_shot", "").replace("-no-thinking", "").rstrip("_- ")
    
    method_display = METHOD_DISPLAY_NAMES.get(method_key, method_key)
    llm_display = LLM_DISPLAY_NAMES.get(model_name, model_name)
    return method_display, llm_display

def load_csv_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    confidences = []
    correctness = []
    all_correctness = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                conf = float(row["Confidence"])
            except (ValueError, KeyError):
                conf = float("nan")
            is_correct = row.get("Correct", "").strip().lower() == "yes"
            all_correctness.append(float(is_correct))
            if not np.isnan(conf):
                confidences.append(conf)
                correctness.append(float(is_correct))
    return np.array(confidences), np.array(correctness), np.array(all_correctness)

def compute_standard_metrics(confidences: np.ndarray, correctness: np.ndarray, all_correctness: np.ndarray) -> dict[str, float]:
    conf_t = torch.tensor(confidences, dtype=torch.float32)
    corr_t = torch.tensor(correctness, dtype=torch.float32)
    all_corr_t = torch.tensor(all_correctness, dtype=torch.float32)
    accuracy = float(all_corr_t.mean()) if len(all_corr_t) > 0 else 0.0

    ece = ExpectedCalibrationError(n_bins=10)
    bs = BrierScore()
    ce = ConfidenceCrossEntropy()
    auc = AUCScore()

    for metric in [ece, bs, ce, auc]:
        metric.update(conf_t, corr_t)

    return {
        "Acc": accuracy,
        "AUROC": auc.compute().item(),
        "ECE": ece.compute().item(),
        "BS": bs.compute().item(),
        "CE": ce.compute().item(),
    }

def compute_cncag_metrics(confidences: np.ndarray, correctness: np.ndarray, ns: list[int]) -> dict[str, float]:
    conf_t = torch.tensor(confidences, dtype=torch.float32)
    corr_t = torch.tensor(correctness, dtype=torch.float32)

    results = {}
    for n in ns:
        metric = CnCAG(n=n)
        metric.update(conf_t, corr_t)
        results[f"CnCAG(n={n})"] = metric.compute().item()

    return results

def compute_gamma_ccag_metrics(confidences: np.ndarray, correctness: np.ndarray, gammas: list[float]) -> dict[str, float]:
    conf_t = torch.tensor(confidences, dtype=torch.float32)
    corr_t = torch.tensor(correctness, dtype=torch.float32)

    results = {}

    for gamma in gammas:
        metric = GammaCCAG(gamma=gamma)
        metric.update(conf_t, corr_t)
        col_name = f"γ-CCAG({gamma})"
        results[col_name] = metric.compute().item()

    return results

# ── Standard table layout ─────────────────────────────────────────────

STANDARD_COLUMNS = ["Acc", "AUROC", "ECE", "BS", "CE"]
STANDARD_DIRECTION = {
    "Acc": True, "AUROC": True, "ECE": False, "BS": False, "CE": False
}
STANDARD_LATEX_HEADER = {
    "Acc": r"\textbf{Acc}", "AUROC": r"\textbf{AUROC}", "ECE": r"\textbf{ECE}",
    "BS": r"\textbf{BS}", "CE": r"\textbf{CE}"
}

def format_val(val: float, is_best: bool, precision: int = 3) -> str:
    s = f"{val:.{precision}f}"
    return r"\textbf{" + s + "}" if is_best else s

def find_best_values(llm_methods: dict, columns: list[str], directions: dict[str, bool]) -> dict[str, float]:
    best = {}
    for col in columns:
        vals = [llm_methods[m][col] for m in METHOD_ORDER if m in llm_methods and col in llm_methods[m] and not np.isnan(llm_methods[m][col])]
        if vals:
            best[col] = max(vals) if directions.get(col, False) else min(vals)
    return best

def generate_standard_table(results: dict[str, dict[str, dict[str, float]]], output_filename: Path) -> str:
    n_metrics = len(STANDARD_COLUMNS)
    col_spec = "cc|" + "c" * n_metrics
    header_cols = " & ".join(STANDARD_LATEX_HEADER[m] for m in STANDARD_COLUMNS)

    lines = [
        r"\begin{tabular}{" + col_spec + "}", r"\toprule",
        r"\textbf{LLM} & \textbf{Method} & " + header_cols + r"\\", r"\hline"
    ]

    for llm in LLM_ORDER:
        if llm not in results: continue
        llm_methods = results[llm]
        n_methods = sum(1 for m in METHOD_ORDER if m in llm_methods)
        if n_methods == 0: continue

        best = find_best_values(llm_methods, STANDARD_COLUMNS, STANDARD_DIRECTION)

        first = True
        for method in METHOD_ORDER:
            if method not in llm_methods:
                continue

            row_data = llm_methods[method]
            row_prefix = r"\multirow{" + str(n_methods) + r"}{*}{" + llm + r"} & " if first else " & "
            row_prefix += method

            row_cells = []
            for col in STANDARD_COLUMNS:
                if col in row_data and not np.isnan(row_data[col]):
                    is_best = (row_data[col] == best[col])
                    row_cells.append(format_val(row_data[col], is_best))
                else:
                    row_cells.append("-")

            lines.append(row_prefix + " & " + " & ".join(row_cells) + r" \\")
            first = False
        lines.append(r"\hline")

    lines.extend([r"\bottomrule", r"\end{tabular}"])

    tex_lines = [
        r"\begin{table}[h]", r"\centering", r"\resizebox{\columnwidth}{!}{%",
    ] + lines + [
        r"}",
        r"\caption{Standard Calibration Metrics}", r"\label{tab:standard}",
        r"\end{table}",
    ]
    tex_doc = "\n".join(tex_lines)

    standalone_pdf_lines = [
        r"\documentclass{standalone}",
        r"\usepackage{booktabs}",
        r"\usepackage{multirow}",
        r"\begin{document}",
    ] + lines + [
        r"\end{document}"
    ]
    standalone_pdf_doc = "\n".join(standalone_pdf_lines)

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

        

def generate_ccag_plots(results: dict, gammas: list[float], output_dir: Path):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    for llm in LLM_ORDER:
        if llm not in results: continue

        llm_methods = results[llm]
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharey=True)
        for method in METHOD_ORDER:
            if method not in llm_methods: continue
            
            y_vals = [llm_methods[method].get(f"γ-CCAG({g})", np.nan) for g in gammas]
            ax.plot(gammas, y_vals, marker='o', linestyle='-', label=method)
            
            ax.set_title(f"γ-CCAG vs γ for {llm}")
            ax.set_xlabel("γ (Gamma)")
            ax.set_ylabel("γ-CCAG")
            ax.grid(True)
            ax.legend()
            
        plt.tight_layout()
        out_path = plots_dir / f"{llm.replace(' ', '_').replace('.', '_')}_ccag_plots.pdf"
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

def generate_cncag_plots(results: dict, ns: list[int], output_dir: Path):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    for llm in LLM_ORDER:
        if llm not in results: continue

        llm_methods = results[llm]
        
        plt.figure(figsize=(8, 5))
        for method in METHOD_ORDER:
            if method not in llm_methods: continue
            
            y_vals = []
            for n in ns:
                col_name = f"CnCAG(n={n})"
                y_vals.append(llm_methods[method].get(col_name, np.nan))
                
            plt.plot(ns, y_vals, marker='o', label=method)
            
        plt.title(f"CnCAG vs n for {llm}")
        plt.xlabel("n")
        plt.xscale('symlog', linthresh=1.0)
        plt.xticks(ns, labels=[str(n) for n in ns])
        plt.ylabel("CnCAG")
        plt.legend()
        plt.grid(True)
        
        out_path = plots_dir / f"{llm.replace(' ', '_').replace('.', '_')}_cncag_plots.pdf"
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95])
    parser.add_argument("--ns", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--output-dir", type=str, default=f"{str(Path(__file__).parent.parent)}/outputs")
    parser.add_argument("--logs-dir", type=str, default=f"{str(Path(__file__).parent.parent)}/paper_results")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    results = defaultdict(lambda: defaultdict(dict))

    for exp_dir in logs_dir.iterdir():
        if not exp_dir.is_dir(): continue
        
        parsed = parse_experiment_dir(exp_dir.name)
        if not parsed: continue
        method, llm = parsed
        
        csv_path = find_csv_for_experiment(exp_dir)
        if not csv_path: continue
        
        confidences, correctness, all_correctness = load_csv_data(csv_path)
        if len(confidences) == 0: continue
        
        std_metrics = compute_standard_metrics(confidences, correctness, all_correctness)
        g_metrics = compute_gamma_ccag_metrics(confidences, correctness, args.gammas)
        cncag_metrics = compute_cncag_metrics(confidences, correctness, args.ns)
        
        results[llm][method].update(std_metrics)
        results[llm][method].update(g_metrics)
        results[llm][method].update(cncag_metrics)

    generate_standard_table(results, out_dir / "table_standard")
    generate_ccag_plots(results, args.gammas, out_dir)
    generate_cncag_plots(results, args.ns, out_dir)
    print(f"Generated results in {out_dir}")

if __name__ == "__main__":
    main()