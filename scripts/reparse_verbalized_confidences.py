"""
Re-parse verbalized confidences of finished runs with the current extractor.

The verbalized extractor used to miss markdown-formatted probabilities
("Probability: **0.95**"), leaving NaN confidences or picking a later probability
from a correction block. Raw outputs are saved truncated to 200 characters, so a
re-parsed value replaces the stored one only when the saved text contains a
probability; otherwise the stored value (parsed on the full text) is kept.

Updates in place, for every verbalized run under the given results directories:
  - <run>/llm_outputs.csv                         (confidence / raw_output)
  - <run>_evaluation/version_0/evaluation_results.csv (Confidence / Raw Output)

Usage:
    uv run python scripts/reparse_verbalized_confidences.py [--results-dirs ...]
"""

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from callm.extractors.base import extract_verbalized_probability


def reparse_confidence(raw_output: str, old: float) -> float:
    """Return the re-parsed confidence, or `old` when the saved text has none."""
    new = extract_verbalized_probability(raw_output)
    return old if math.isnan(new) else new


def _to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def reparse_csv(csv_path: Path) -> int:
    """Re-parse the confidence column of one CSV in place; return the changed rows."""
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    conf_col = "confidence" if "confidence" in fieldnames else "Confidence"
    raw_col = "raw_output" if "raw_output" in fieldnames else "Raw Output"

    changed = 0
    for row in rows:
        old = _to_float(row[conf_col])
        new = reparse_confidence(row[raw_col], old)
        if not (math.isnan(old) and math.isnan(new)) and new != old:
            row[conf_col] = f"{new:.6f}"
            changed += 1

    if changed:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return changed


def find_verbalized_csvs(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("verbalized*/llm_outputs.csv")) + sorted(
        results_dir.glob("verbalized*_evaluation/version_0/evaluation_results.csv")
    )


def main():
    root_dir = Path(__file__).resolve().parents[1]
    paper_results = root_dir / "lightning_logs" / "paper_results"

    parser = argparse.ArgumentParser(description="Re-parse verbalized confidences")
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        default=[str(paper_results / "triviaqa"), str(paper_results / "mmlu")],
        help="Directories containing verbalized_* runs",
    )
    args = parser.parse_args()

    for results_dir in args.results_dirs:
        for csv_path in find_verbalized_csvs(Path(results_dir)):
            changed = reparse_csv(csv_path)
            print(f"{csv_path.relative_to(results_dir)}: {changed} confidences updated")


if __name__ == "__main__":
    main()
