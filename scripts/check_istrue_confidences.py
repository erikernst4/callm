"""
Check that IsTrue confidences in merged_results.csv files are P(True).

The IsTrue run's llm_outputs.csv stores the raw sequence posterior of the
generated answer token, i.e. P(True) when the model answered True/A and
P(False) when it answered False/B. The merged_results.csv used for metrics
must hold P(True), so False/B rows must be 1 - raw.

For every is_true_*/merged_results.csv under the logs directory, rows are
aligned with the raw IsTrue llm_outputs.csv (positionally when the question
columns match row by row, otherwise by question and occurrence index) and
classified:
  - True/A rows:  expected confidence == raw
  - False/B rows: expected confidence == 1 - raw ("flipped")
  - other rows:   unparseable answer, confidence should be NaN

TriviaQA cleaned/strict subsets have no llm_outputs.csv of their own; the
raw outputs of the matching paper_results/triviaqa run are used instead.

Exits with status 1 if any file has False/B rows that are not flipped.
"""

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

TRUE_ANSWERS = {"true", "a"}
FALSE_ANSWERS = {"false", "b"}
SUBSETS_WITHOUT_RAW = ("triviaqa_cleaned", "triviaqa_strict")
ATOL = 1e-6


def normalize_question(q) -> str:
    return str(q).replace("\\n", "\n").strip()


def question_key(q) -> str:
    """Matching key robust to merged files truncating long questions with '...'."""
    return normalize_question(q).removesuffix("...")[:150]


def find_raw_outputs(merged_csv: Path) -> Path | None:
    raw_csv = merged_csv.parent / "llm_outputs.csv"
    if raw_csv.exists():
        return raw_csv
    for subset in SUBSETS_WITHOUT_RAW:
        if merged_csv.parent.parent.name == subset:
            raw_csv = (
                merged_csv.parent.parent.parent
                / "triviaqa"
                / merged_csv.parent.name
                / "llm_outputs.csv"
            )
            if raw_csv.exists():
                return raw_csv
    return None


def align(merged: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return merged with raw IsTrue answer/confidence columns attached."""
    merged = merged.copy()
    merged["_q"] = merged["Question"].map(normalize_question)
    raw = raw.copy()
    raw["_q"] = raw["question"].map(normalize_question)

    if len(merged) == len(raw) and (merged["_q"].values == raw["_q"].values).all():
        merged["it_answer"] = raw["pred_answer"].values
        merged["raw_confidence"] = raw["confidence"].values
        merged["_matched"] = True
        return merged, "positional"

    # Row order is never trusted when the questions differ: IsTrue runs are
    # shuffled relative to their base run (subsample_dataset shuffles with the seed).
    # Pass 1 matches the full question; pass 2 matches the rows left over by
    # question_key (merged files may truncate long questions with '...'; the key
    # alone would collide on MMLU questions sharing long openings).
    # Repeated questions: the k-th occurrence matches the k-th unused occurrence.
    raw_row = [None] * len(merged)
    used = set()
    for key_fn in (normalize_question, question_key):
        pool = defaultdict(deque)
        for j, q in enumerate(raw["question"]):
            if j not in used:
                pool[key_fn(q)].append(j)
        for i, q in enumerate(merged["Question"]):
            if raw_row[i] is None and pool[key_fn(q)]:
                raw_row[i] = pool[key_fn(q)].popleft()
                used.add(raw_row[i])

    def pick(col):
        return [raw[col].iloc[j] if j is not None else None for j in raw_row]

    merged["it_answer"] = pick("pred_answer")
    merged["raw_confidence"] = pd.to_numeric(
        pd.Series(pick("confidence"), index=merged.index), errors="coerce"
    )
    merged["_matched"] = [j is not None for j in raw_row]
    return merged, "by question"


def check_file(merged_csv: Path, raw_csv: Path) -> dict:
    merged = pd.read_csv(merged_csv)
    raw = pd.read_csv(raw_csv)
    raw.columns = [c.lower() for c in raw.columns]

    df, how = align(merged, raw)
    answers = df["it_answer"].astype(str).str.strip().str.lower()
    is_true = answers.isin(TRUE_ANSWERS)
    is_false = answers.isin(FALSE_ANSWERS)
    is_other = ~is_true & ~is_false

    conf = df["Confidence"].astype(float)
    raw_conf = df["raw_confidence"].astype(float)
    equals_raw = np.isclose(conf, raw_conf, atol=ATOL, equal_nan=True)
    equals_flip = np.isclose(conf, 1 - raw_conf, atol=ATOL, equal_nan=True)

    return {
        "file": merged_csv,
        "raw": raw_csv,
        "align": how,
        "n": len(df),
        "unmatched": int((~df["_matched"]).sum()),
        "true_n": int(is_true.sum()),
        "true_raw": int((is_true & equals_raw).sum()),
        "false_n": int(is_false.sum()),
        "false_flipped": int((is_false & equals_flip).sum()),
        "false_raw": int((is_false & equals_raw & ~equals_flip).sum()),
        "other_n": int(is_other.sum()),
        "other_nan": int((is_other & conf.isna()).sum()),
    }


def verdict(r: dict) -> str:
    if r["false_n"] == 0:
        return "OK (no False rows)"
    if r["false_flipped"] == r["false_n"]:
        return "OK"
    if r["false_raw"] == r["false_n"]:
        return "NOT FLIPPED"
    return "MIXED"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "lightning_logs"),
    )
    args = parser.parse_args()
    logs_dir = Path(args.logs_dir)

    results = []
    for merged_csv in sorted(logs_dir.rglob("is_true_*/merged_results.csv")):
        raw_csv = find_raw_outputs(merged_csv)
        if raw_csv is None:
            print(f"SKIP {merged_csv.relative_to(logs_dir)}: no raw llm_outputs.csv")
            continue
        results.append(check_file(merged_csv, raw_csv))

    header = (
        f"{'file':<75} {'align':<14} {'n':>6} {'unm':>4} "
        f"{'True ok/n':>11} {'False flip/raw/n':>18} {'other NaN/n':>12}  verdict"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        rel = str(r["file"].parent.relative_to(logs_dir))
        print(
            f"{rel:<75} {r['align'][:14]:<14} {r['n']:>6} {r['unmatched']:>4} "
            f"{r['true_raw']:>5}/{r['true_n']:<5} "
            f"{r['false_flipped']:>6}/{r['false_raw']}/{r['false_n']:<6} "
            f"{r['other_nan']:>5}/{r['other_n']:<6}  {verdict(r)}"
        )

    bad = [r for r in results if not verdict(r).startswith("OK")]
    print(f"\n{len(results)} files checked, {len(bad)} with unflipped False rows.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
