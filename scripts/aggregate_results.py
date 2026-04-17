import argparse

import pandas as pd
from pathlib import Path
import ast


def is_exact_match(row):
    pred = str(row["predicted answer"]).strip()
    gold_str = str(row["gold answers"])
    try:
        gold_list = ast.literal_eval(gold_str)
        if isinstance(gold_list, list):
            # Check exact match
            if pred in gold_list:
                return True
            # Also check case-insensitive match
            if pred.lower() in [str(g).strip().lower() for g in gold_list]:
                return True
            return False
    except (ValueError, SyntaxError):
        pass

    # Fallback to simple string inclusion
    return pred in gold_str or pred.lower() in gold_str.lower()


def extract_system_from_path(file_path, logs_dir):
    exp_dir = file_path.relative_to(logs_dir).parts[0]
    if exp_dir.endswith("_evaluation"):
        exp_dir = exp_dir[: -len("_evaluation")]

    prompt = "unknown"
    model = exp_dir

    for p in ["is_true", "verbalized", "label_prob"]:
        if exp_dir.startswith(p + "_"):
            prompt = p
            model = exp_dir[len(p + "_") :]
            break

    # Remove 'zero_shot' and 'no-thinking'
    model = model.replace("zero_shot_", "")
    model = model.replace("zero_shot", "")
    model = model.replace("-no-thinking", "")
    model = model.replace("_no-thinking", "")
    model = model.replace("no-thinking", "")

    return f"{prompt}-{model}"


def main(out_dir, logs_dir):
    out_dir.mkdir(exist_ok=True, parents=True)

    # Find all evaluation_results.csv and merged_results.csv
    files = list(logs_dir.rglob("evaluation_results.csv")) + list(
        logs_dir.rglob("merged_results.csv")
    )

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        # Check if expected columns are present
        cols = ["Question", "Gold Answers", "Predicted Answer", "Correct"]
        if all(col in df.columns for col in cols):
            df = df[cols]
            # Rename columns to lowercase as requested
            df.columns = ["question", "gold answers", "predicted answer", "correct"]
            df["system"] = extract_system_from_path(file, logs_dir)
            dfs.append(df)
        else:
            # Check if they are already lowercase
            lower_cols = ["question", "gold answers", "predicted answer", "correct"]
            if all(col in df.columns for col in lower_cols):
                df = df[lower_cols]
                df["system"] = extract_system_from_path(file, logs_dir)
                dfs.append(df)
            else:
                print(f"Skipping {file} due to missing columns")

    if not dfs:
        print("No valid CSV files found.")
        return

    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Normalize gold answers for deduplication so order doesn't matter
    def normalize_gold_answers(gold_str):
        gold_str_val = str(gold_str)
        try:
            gold_list = ast.literal_eval(gold_str_val)
            if isinstance(gold_list, list):
                # Sort the list to ensure order doesn't affect equality
                sorted_list = sorted([str(g).strip().lower() for g in gold_list])
                return str(sorted_list)
        except (ValueError, SyntaxError):
            pass
        return gold_str_val.strip().lower()

    # Create a temporary column for deduplication
    combined_df["norm_gold"] = combined_df["gold answers"].apply(normalize_gold_answers)

    # Group by the unique combinations to collect the models, keeping the first 'gold answers'
    grp_cols = ["question", "norm_gold", "predicted answer", "correct"]

    # Capture the original question order before grouping
    categories = combined_df["question"].drop_duplicates().tolist()

    dedup_df = combined_df.groupby(grp_cols, as_index=False).agg(
        {"gold answers": "first", "system": lambda x: sorted(list(set(x)))}
    )

    # Remove the temporary column
    dedup_df = dedup_df.drop(columns=["norm_gold"])

    # Remove rows that have an exact match
    mask = dedup_df.apply(is_exact_match, axis=1)
    filtered_df = dedup_df[~mask].copy()

    # Sort by the original question appearance order
    filtered_df["question"] = pd.Categorical(
        filtered_df["question"], categories=categories, ordered=True
    )
    filtered_df = filtered_df.sort_values("question")

    # Reorder columns
    filtered_df = filtered_df[
        ["question", "gold answers", "predicted answer", "correct", "system"]
    ]

    output_file = out_dir / "aggregated_results.csv"
    filtered_df.to_csv(output_file, index=False)
    print(
        f"Saved aggregated results to {output_file} w/ {len(filtered_df)} rows (from {len(combined_df)} raw)."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=str, default=f"{str(Path(__file__).parent.parent)}/outputs"
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=f"{str(Path(__file__).parent.parent)}/paper_results",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.output_dir)
    main(logs_dir=logs_dir, out_dir=out_dir)
