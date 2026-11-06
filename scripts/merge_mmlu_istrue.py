"""
Merge MMLU IsTrue confidences with the answers of their base (label_prob) run.

For every is_true_<model> directory under --mmlu-dir, writes merged_results.csv
(base run answers + correctness, IsTrue confidences, in base run order) and
metrics.csv (the paper table's metrics). Existing files are backed up first as
<name>.bak-<timestamp>.

Rows are matched by question, never by position: IsTrue runs are shuffled
relative to their base run, because subsample_dataset shuffles with the seed
even when it keeps every row. Repeated questions can't be matched, so they are
an error.

The IsTrue llm_outputs.csv stores the sequence posterior of the generated
answer, i.e. P(False) when the model answered False. Confidences are converted
to P(True) here (1 - raw for False/B answers, via IsTrueExtractor.to_p_true)
so they score the base answer. Run scripts/check_istrue_confidences.py
afterwards to verify.

Usage:
    uv run python scripts/merge_mmlu_istrue.py [--mmlu-dir DIR] [--base-dir DIR]
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from callm.extractors import IsTrueExtractor  # noqa: E402
from callm.utils import impute_missing_confidences  # noqa: E402
from generate_mmlu_table import compute_standard_metrics  # noqa: E402


def base_dir_name(is_true_dir_name: str) -> str:
    """Name of the base (label_prob) run judged by an is_true_<model> run."""
    suffix = is_true_dir_name[len("is_true_") :]
    if "GLM-4.6V-Flash-no-thinking" in suffix:
        return f"label_prob_zero_shot_{suffix}"
    return f"label_prob_{suffix}"


def merge_is_true(df_base: pd.DataFrame, df_is_true: pd.DataFrame) -> pd.DataFrame:
    """Merged results (evaluate_csv format) in base run order.

    Raises ValueError if the runs have different row counts, the base run has
    repeated questions, or any question does not match.
    """
    df_base = df_base.rename(columns=str.lower)
    df_is_true = df_is_true.rename(columns=str.lower)
    if len(df_base) != len(df_is_true):
        raise ValueError(
            f"base has {len(df_base)} rows but is_true has {len(df_is_true)} rows"
        )

    duplicated = int(df_base["question"].duplicated().sum())
    if duplicated:
        raise ValueError(f"{duplicated} duplicate questions: rows can't be matched")

    df_is_true = df_is_true.assign(
        p_true=IsTrueExtractor.to_p_true(
            df_is_true["pred_answer"], df_is_true["confidence"]
        ),
    )
    merged = df_base.merge(
        df_is_true[["question", "pred_answer", "p_true", "raw_output"]],
        on="question",
        how="left",
        suffixes=("", "_istrue"),
        indicator=True,
    )
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        raise ValueError(f"{unmatched} base questions unmatched in the is_true run")

    is_correct = (
        merged["pred_answer"].astype(str).str.strip().str.lower()
        == merged["gold_answers"].astype(str).str.strip().str.lower()
    )
    return pd.DataFrame(
        {
            "Question": merged["question"]
            .fillna("")
            .apply(lambda x: str(x).replace("\n", "\\n")),
            "Gold Answers": merged["gold_answers"],
            "Predicted Answer": merged["pred_answer"],
            # P(True) of the IsTrue run, i.e. 1 - raw for False answers
            "Confidence": merged["p_true"],
            "Correct": is_correct.map({True: "Yes", False: "No"}),
            "Evaluator Response": "Merged from base run",
            "Raw Output": merged["raw_output_istrue"],
            "Is True Predicted Answer": merged["pred_answer_istrue"],
        }
    )


def backup(path: Path) -> Path | None:
    """Copy an existing file to <name>.bak-<timestamp>; return the backup path."""
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    bak = path.with_name(f"{path.name}.bak-{stamp}")
    shutil.copy2(path, bak)
    return bak


def compute_metrics(merged: pd.DataFrame) -> dict[str, float]:
    """The paper table's metrics (missing confidences imputed with 0.5)."""
    confidences = impute_missing_confidences(merged["Confidence"].astype(float).values)
    correctness = (merged["Correct"] == "Yes").astype(float).values
    return compute_standard_metrics(confidences, correctness)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--mmlu-dir",
        type=Path,
        default=ROOT / "lightning_logs" / "paper_results" / "mmlu",
        help="Directory with the is_true_<model> runs (outputs are written there)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Directory with the label_prob_<model> base runs (default: --mmlu-dir)",
    )
    args = parser.parse_args()
    base_root = args.base_dir or args.mmlu_dir

    for exp_dir in sorted(args.mmlu_dir.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("is_true_"):
            continue

        base_dir = base_root / base_dir_name(exp_dir.name)
        is_true_csv = exp_dir / "llm_outputs.csv"
        base_csv = base_dir / "llm_outputs.csv"
        if not is_true_csv.exists() or not base_csv.exists():
            print(f"Skipping {exp_dir.name}: missing {is_true_csv} or {base_csv}")
            continue

        print(f"Processing {exp_dir.name} using base {base_dir}...")
        merged = merge_is_true(pd.read_csv(base_csv), pd.read_csv(is_true_csv))

        merged_csv = exp_dir / "merged_results.csv"
        if bak := backup(merged_csv):
            print(f"  Backed up {bak.name}")
        merged.to_csv(merged_csv, index=False)
        print(f"  Saved {merged_csv}")

        metrics_csv = exp_dir / "metrics.csv"
        if bak := backup(metrics_csv):
            print(f"  Backed up {bak.name}")
        pd.DataFrame([compute_metrics(merged)]).to_csv(metrics_csv, index=False)
        print(f"  Saved {metrics_csv}")


if __name__ == "__main__":
    main()
