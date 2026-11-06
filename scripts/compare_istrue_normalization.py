"""
Compare raw vs normalized IsTrue confidences on the paper's evaluation subsets.

Raw: the paper's merged_results.csv confidences (P(True) from the full-vocab
posterior of the generated answer), taken from paper_results/triviaqa_strict
(curated 455 questions) and paper_results/mmlu. These files also give correctness.
Normalized: the llm_outputs.csv of scripts/istrue_normalization.sh runs
(<results-dir>/<ds>/is_true_<suffix>/), aligned to the merged files and
converted to P(True) = p(True) / (p(True) + p(False)).

Rows with an unmatched normalized output are dropped from both columns, so raw
and normalized are always scored on the same samples.

Writes <results-dir>/comparison.csv, comparison.tex (needs \\usepackage[table]{xcolor})
and a compiled comparison_preview.pdf (if pdflatex is available).

Usage:
    uv run python scripts/compare_istrue_normalization.py [--results-dir DIR]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from callm.extractors import IsTrueExtractor  # noqa: E402
from check_istrue_confidences import align  # noqa: E402
from generate_consolidated_csv import parse_experiment_dir, LLM_ORDER  # noqa: E402
from callm.utils import impute_missing_confidences  # noqa: E402
from generate_mmlu_table import (  # noqa: E402
    compute_standard_metrics,
    STANDARD_COLUMNS,
    STANDARD_DIRECTION,
)

# Dataset -> paper_results subdirectory holding the merged (scored) raw results
MERGED_SUBDIRS = {"triviaqa": "triviaqa_strict", "mmlu": "mmlu"}

# The paper table's metrics, in its column order: key -> (LaTeX name, higher is better)
DISPLAY_NAMES = {
    "ER": "ER",
    "ECE": "ECE",
    "AUROC": "AUC",
    "CE": r"CE$_{q_e}$",
    "BS": r"BS$_{q_e}$",
    "AURC": "AURC",
    "ECUAS_0": r"ECUAS$_0$",
    "ECUAS_1": r"ECUAS$_1$",
    "ECUAS_128": r"ECUAS$_{128}$",
}
METRICS = {k: (DISPLAY_NAMES[k], STANDARD_DIRECTION[k]) for k in STANDARD_COLUMNS}


def _aligned_p_true(
    merged: pd.DataFrame, outputs_csv: Path
) -> tuple[pd.DataFrame, str]:
    """Align an IsTrue llm_outputs.csv to the merged file; add its P(True) as `p_true`."""
    outputs = pd.read_csv(outputs_csv)
    outputs.columns = [c.lower() for c in outputs.columns]
    df, how = align(merged, outputs)
    df["p_true"] = np.where(
        df["_matched"],
        IsTrueExtractor.to_p_true(df["it_answer"], df["raw_confidence"]),
        np.nan,
    )
    return df, how


def compare_run(
    merged_csv: Path, normalized_csv: Path, raw_csv: Path | None = None
) -> list[dict]:
    """Score raw vs normalized P(True) against the merged file's correctness.

    Raw is the merged file's Confidence (the paper run), or `raw_csv` (a raw
    IsTrue rerun's llm_outputs.csv) when given.
    """
    merged = pd.read_csv(merged_csv)
    norm_df, how = _aligned_p_true(merged, normalized_csv)
    keep = norm_df["_matched"].to_numpy(dtype=bool, copy=True)
    raw = merged["Confidence"].astype(float).values
    if raw_csv is not None:
        raw_df, _ = _aligned_p_true(merged, raw_csv)
        keep &= raw_df["_matched"].to_numpy(dtype=bool)
        raw = raw_df["p_true"].values

    correct = (merged["Correct"].astype(str).str.strip() == "Yes").astype(float).values
    correct = correct[keep]
    confidences = {"raw": raw[keep], "normalized": norm_df["p_true"].values[keep]}
    n = int(keep.sum())

    rows = []
    for variant, conf in confidences.items():
        row = {
            "variant": variant,
            "n": n,
            "raw_source": "rerun" if raw_csv is not None else "paper",
            "n_nan": int(np.isnan(conf).sum()),
            "align": how,
        }
        # Same as the paper table: missing confidences are imputed with 0.5
        row.update(compute_standard_metrics(impute_missing_confidences(conf), correct))
        rows.append(row)
    return rows


def collect(results_dir: Path, paper_dir: Path, raw_dir: Path) -> pd.DataFrame:
    rows = []
    for ds, merged_subdir in MERGED_SUBDIRS.items():
        for run_dir in sorted((results_dir / ds).glob("is_true_*")):
            normalized_csv = run_dir / "llm_outputs.csv"
            merged_csv = paper_dir / merged_subdir / run_dir.name / "merged_results.csv"
            if not normalized_csv.exists() or not merged_csv.exists():
                print(
                    f"Skipping {ds}/{run_dir.name}: missing {normalized_csv} or {merged_csv}"
                )
                continue
            parsed = parse_experiment_dir(run_dir.name)
            model = parsed[1] if parsed else run_dir.name
            raw_csv = raw_dir / ds / run_dir.name / "llm_outputs.csv"
            raw_csv = raw_csv if raw_csv.exists() else None
            for row in compare_run(merged_csv, normalized_csv, raw_csv):
                rows.append({"dataset": ds, "model": model, **row})
    return pd.DataFrame(rows)


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(
        index=["dataset", "model", "raw_source"],
        columns="variant",
        values=list(METRICS),
    )
    order = {m: i for i, m in enumerate(LLM_ORDER)}
    keys = sorted(
        wide.index, key=lambda k: (k[0] != "triviaqa", order.get(k[1], 99), k[1])
    )
    return wide.loc[keys]


COLOR_THRESHOLD = 0.05  # relative change vs raw needed to color the normalized cell


def format_pair(raw: float, norm: float, higher_is_better: bool) -> tuple[str, str]:
    """LaTeX cells for a raw/normalized metric pair.

    The better value is bold, comparing the displayed (3-decimal) values so that
    ties bold both. The normalized cell is green/red when it is more than
    COLOR_THRESHOLD better/worse than raw, relative to raw.
    """
    raw_s, norm_s = f"{raw:.3f}", f"{norm:.3f}"
    raw_r, norm_r = float(raw_s), float(norm_s)
    best = max(raw_r, norm_r) if higher_is_better else min(raw_r, norm_r)
    raw_cell = rf"\textbf{{{raw_s}}}" if raw_r == best else raw_s
    norm_cell = rf"\textbf{{{norm_s}}}" if norm_r == best else norm_s

    gain = (norm - raw) if higher_is_better else (raw - norm)
    if raw != 0 and gain / abs(raw) > COLOR_THRESHOLD:
        norm_cell = r"\cellcolor{green!25}" + norm_cell
    elif raw != 0 and gain / abs(raw) < -COLOR_THRESHOLD:
        norm_cell = r"\cellcolor{red!25}" + norm_cell
    return raw_cell, norm_cell


def to_latex(wide: pd.DataFrame) -> str:
    """Tabular comparing raw vs normalized; needs \\usepackage[table]{xcolor}."""
    cols = "ll" + "rr" * len(METRICS)
    up, down = r"$\uparrow$", r"$\downarrow$"
    header1 = " & ".join(
        ["Dataset", "Model"]
        + [
            rf"\multicolumn{{2}}{{c}}{{{name}{up if hib else down}}}"
            for name, hib in METRICS.values()
        ]
    )
    header2 = " & ".join(["", ""] + ["Raw", "Norm."] * len(METRICS))
    lines = [
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        header1 + r" \\",
        header2 + r" \\",
        r"\midrule",
    ]
    prev_ds = None
    for (ds, model, raw_source), row in wide.iterrows():
        if raw_source == "rerun":
            model += r"$^\dagger$"
        if prev_ds is not None and ds != prev_ds:
            lines.append(r"\midrule")
        cells = ["TriviaQA" if ds == "triviaqa" else "MMLU", model]
        for key, (_, higher_is_better) in METRICS.items():
            cells.extend(
                format_pair(
                    row[(key, "raw")], row[(key, "normalized")], higher_is_better
                )
            )
        lines.append(" & ".join(cells) + r" \\")
        prev_ds = ds
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def to_markdown(wide: pd.DataFrame) -> str:
    names = [f"{k} raw \\| norm" for k in METRICS]
    lines = [
        "| Dataset | Model | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for (ds, model, raw_source), row in wide.iterrows():
        if raw_source == "rerun":
            model += " †"
        cells = [
            f"{row[(k, 'raw')]:.3f} \\| {row[(k, 'normalized')]:.3f}" for k in METRICS
        ]
        lines.append(f"| {ds} | {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


PREVIEW_TEX = r"""\documentclass[border=8pt]{standalone}
\usepackage{booktabs}
\usepackage[table]{xcolor}
\begin{document}
\begin{tabular}{c}
\textbf{IsTrue: raw vs normalized $P(\mathrm{True})$} \\[2pt]
\small Bold: better (ties bold both). Green/red: normalized $>$5\% better/worse than raw. \\
\small $^\dagger$ raw rerun (with MMLU choices) instead of the paper run. \\[4pt]
\input{comparison.tex}
\end{tabular}
\end{document}
"""


def compile_preview(results_dir: Path) -> None:
    """Write and compile a standalone PDF preview of comparison.tex."""
    (results_dir / "comparison_preview.tex").write_text(PREVIEW_TEX)
    if shutil.which("pdflatex") is None:
        print("pdflatex not found: skipping comparison_preview.pdf")
        return
    proc = subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "comparison_preview.tex",
        ],
        cwd=results_dir,
        capture_output=True,
        text=True,
    )
    for ext in ("aux", "log"):
        (results_dir / f"comparison_preview.{ext}").unlink(missing_ok=True)
    if proc.returncode != 0:
        print(f"pdflatex failed:\n{proc.stdout[-2000:]}")
    else:
        print(f"Saved {results_dir / 'comparison_preview.pdf'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "lightning_logs" / "istrue_normalization",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "lightning_logs" / "istrue_normalization_raw",
        help="Raw IsTrue reruns (<ds>/is_true_<suffix>/llm_outputs.csv) that replace the paper raw run",
    )
    parser.add_argument(
        "--paper-dir", type=Path, default=ROOT / "lightning_logs" / "paper_results"
    )
    args = parser.parse_args()

    df = collect(args.results_dir, args.paper_dir, args.raw_dir)
    if df.empty:
        print(f"No normalized runs found under {args.results_dir}")
        return

    csv_path = args.results_dir / "comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    wide = to_wide(df)
    tex_path = args.results_dir / "comparison.tex"
    tex_path.write_text(to_latex(wide))
    print(f"Saved {tex_path}")
    compile_preview(args.results_dir)
    print()

    print(
        df[
            ["dataset", "model", "variant", "raw_source", "n", "n_nan", "align"]
        ].to_string(index=False)
    )
    print()
    print(to_markdown(wide))


if __name__ == "__main__":
    main()
