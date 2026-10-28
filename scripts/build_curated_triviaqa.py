"""
Build a new revision of the curated TriviaQA subset (ErikErnst/triviaqa-extended-subset)
by adding the human-validated answers from equivalence_labeling.csv as gold aliases.

An answer is added when, for one of the 455 curated questions, it was labeled
correct_human=Yes and should_change_gold_answers=Yes, and it does not already exact-match
the question's gold answers. Correct answers flagged should_change_gold_answers=No are
never added (they are formatting variants handled by the exact-match normalizer).

The cleaned (lowercased) answer is appended to both `answer.aliases` and
`answer.normalized_aliases`, the same way the first revision extended the gold answers.

Usage:
    uv run python scripts/build_curated_triviaqa.py                # dry run: print added aliases
    uv run python scripts/build_curated_triviaqa.py --write-local  # update triviaqa_annotated_subset.{csv,parquet}
    uv run python scripts/build_curated_triviaqa.py --push         # push to HF, tag old/new revisions
"""

import argparse
import ast
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import Dataset, load_dataset

from callm.utils import check_exact_match
from generate_consolidated_csv import build_triviaqa_id_map

REPO_ID = "ErikErnst/triviaqa-extended-subset"
BASE_REVISION = "15bb481cfc22a9d82d08c58b7dcb5e362dcac564"
BASE_TAG = "v1.0"
NEW_TAG = "v1.1"
CHANGELOG = f"""
## Changelog

- **{NEW_TAG}**: Added gold aliases for answers given by Gemini 2.5 Flash, Flash Lite and
  Pro that were manually validated as correct, plus a few corrections to earlier labels.
  Exact match should normalize case, markdown emphasis, punctuation, articles and
  parenthetical content before comparing answers.
- **{BASE_TAG}**: Initial release (455 questions, gold aliases extended with manually
  validated LLM answers).
"""


def clean_alias(answer: str) -> str:
    """Turn a raw model answer into an alias: lowercase, no markdown/parentheticals/quotes."""
    t = answer.lower().replace("*", "")
    t = re.sub(r"\s*\([^)]*\)", "", t)
    t = " ".join(t.split())
    return t.strip("\"'“”‘’ ").rstrip(".:;,!").strip("\"'“”‘’ ")


def gold_answers(answer: dict) -> list[str]:
    return (
        list(answer["aliases"]) + list(answer["normalized_aliases"]) + [answer["value"]]
    )


def collect_new_aliases(rows: list[dict], labeling_csv: Path) -> dict[str, list[str]]:
    """Map question_id -> aliases to add, checking that no wrong answer starts matching."""
    id_matcher = build_triviaqa_id_map(
        [
            {
                "id": r["question_id"],
                "question": r["question"],
                "norm_q": r["question"].strip().strip("'").strip('"'),
            }
            for r in rows
        ]
    )
    gold = {r["question_id"]: gold_answers(r["answer"]) for r in rows}

    new_aliases: dict[str, list[str]] = defaultdict(list)
    wrong_answers: dict[str, list[str]] = defaultdict(list)
    with open(labeling_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta = id_matcher(row["question"])
            if meta is None or row.get("erase_question", "").strip() == "Yes":
                continue
            qid, pred = meta["id"], row["predicted answer"].strip()
            label = row.get("correct_human", "").strip()
            if label == "No":
                wrong_answers[qid].append(pred)
                continue
            if (
                label != "Yes"
                or row.get("should_change_gold_answers", "").strip() != "Yes"
            ):
                continue
            if check_exact_match(pred, gold[qid] + new_aliases.get(qid, [])):
                continue
            new_aliases[qid].append(clean_alias(pred))

    collisions = [
        (qid, pred)
        for qid, aliases in new_aliases.items()
        for pred in wrong_answers[qid]
        if check_exact_match(pred, gold[qid] + aliases)
    ]
    if collisions:
        raise ValueError(f"New aliases would make wrong answers match: {collisions}")
    return dict(new_aliases)


def write_local_subset(
    rows: list[dict], subset_dir: Path, new_aliases: dict[str, list[str]]
):
    """Update triviaqa_annotated_subset.{parquet,csv} so local scripts see the new gold."""
    Dataset.from_list(rows).to_parquet(subset_dir / "triviaqa_annotated_subset.parquet")

    csv_path = subset_dir / "triviaqa_annotated_subset.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        csv_rows = list(reader)
    aliases_by_id = {r["question_id"]: list(r["answer"]["aliases"]) for r in rows}
    for row in csv_rows:
        qid = row["Question ID"]
        row["Gold Answers (aliases)"] = str(aliases_by_id[qid])
        if qid in new_aliases:
            additional = ast.literal_eval(row["Additional Correct Answers"] or "[]")
            row["Additional Correct Answers"] = str(additional + new_aliases[qid])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


def push(dataset: Dataset):
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    tags = {t.name for t in api.list_repo_refs(REPO_ID, repo_type="dataset").tags}
    if NEW_TAG in tags:
        raise ValueError(f"{REPO_ID} already has tag {NEW_TAG}")
    if BASE_TAG not in tags:
        api.create_tag(
            REPO_ID, tag=BASE_TAG, revision=BASE_REVISION, repo_type="dataset"
        )

    dataset.push_to_hub(
        REPO_ID, commit_message=f"Add manually validated Gemini answers ({NEW_TAG})"
    )

    readme = Path(
        hf_hub_download(REPO_ID, "README.md", repo_type="dataset")
    ).read_text()
    if "## Changelog" not in readme:
        api.upload_file(
            path_or_fileobj=(readme.rstrip("\n") + "\n" + CHANGELOG).encode(),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Add changelog",
        )
    api.create_tag(REPO_ID, tag=NEW_TAG, repo_type="dataset")
    print(f"Pushed {REPO_ID} and tagged {BASE_TAG} ({BASE_REVISION[:7]}) / {NEW_TAG}")


def main():
    root_dir = Path(__file__).resolve().parents[1]
    subset_dir = root_dir / "lightning_logs" / "paper_results" / "triviaqa_strict"

    parser = argparse.ArgumentParser(
        description="Build a new curated TriviaQA revision"
    )
    parser.add_argument(
        "--labeling-csv",
        type=str,
        default=str(root_dir / "equivalence_labeling.csv"),
        help="Path to equivalence_labeling.csv",
    )
    parser.add_argument(
        "--write-local",
        action="store_true",
        help="Update triviaqa_annotated_subset.{csv,parquet} with the new aliases",
    )
    parser.add_argument(
        "--push", action="store_true", help=f"Push to {REPO_ID} and tag it"
    )
    args = parser.parse_args()

    base = load_dataset(REPO_ID, revision=BASE_REVISION)["train"]
    rows = base.to_list()
    new_aliases = collect_new_aliases(rows, Path(args.labeling_csv))

    by_id = {r["question_id"]: r for r in rows}
    for qid, aliases in new_aliases.items():
        answer = by_id[qid]["answer"]
        answer["aliases"] = list(answer["aliases"]) + aliases
        answer["normalized_aliases"] = list(answer["normalized_aliases"]) + aliases
        print(f"{qid}: {by_id[qid]['question'][:70]!r}")
        for alias in aliases:
            print(f"    + {alias!r}")
    print(
        f"\n{sum(len(a) for a in new_aliases.values())} aliases added to "
        f"{len(new_aliases)} of {len(rows)} questions"
    )

    if args.write_local:
        write_local_subset(rows, subset_dir, new_aliases)
        print(f"Updated local subset files in {subset_dir}")
    if args.push:
        push(Dataset.from_list(rows, features=base.features))


if __name__ == "__main__":
    main()
