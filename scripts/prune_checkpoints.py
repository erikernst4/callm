"""
Prune probe checkpoints under results/ to save disk space.

For each experiment, runs (lightning_logs/version_K dirs with a checkpoints/ folder)
are grouped by method (first path component under the experiment dir) and scored by
the minimum validation loss in their metrics.csv. The best method keeps its top
`--top_best` runs, every other method keeps its top `--top_other` runs, and the
.ckpt files of all remaining runs are deleted. metrics.csv, configs and summaries
are never touched. Dry run by default.

Usage:
    uv run python scripts/prune_checkpoints.py                 # dry run, all experiments
    uv run python scripts/prune_checkpoints.py --exp exp11_mmlu_attentive_pooling
    uv run python scripts/prune_checkpoints.py --delete
"""

import argparse
import math
import os
from collections import defaultdict
from glob import glob

import pandas as pd

# Experiments whose top-level dirs are not methods (e.g. data-scaling sizes)
SINGLE_METHOD_PREFIXES = ("exp9_",)
VAL_LOSS_COLUMNS = ("val_loss_epoch", "val_loss")


def run_score(run_dir: str) -> float:
    metrics_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        return math.nan
    df = pd.read_csv(metrics_path)
    for col in VAL_LOSS_COLUMNS:
        if col in df.columns and df[col].notna().any():
            return float(df[col].min())
    return math.nan


def find_runs(exp_dir: str) -> list[dict]:
    exp_name = os.path.basename(exp_dir)
    single_method = exp_name.startswith(SINGLE_METHOD_PREFIXES)
    runs = []
    # os.walk does not follow symlinks, so runs symlinked into other dirs
    # (e.g. exp4 cross_method_*) are only counted at their real location.
    ckpt_dirs = sorted(
        os.path.join(root, "checkpoints")
        for root, dirs, _ in os.walk(exp_dir)
        if "checkpoints" in dirs
    )
    for ckpt_dir in ckpt_dirs:
        ckpts = sorted(glob(os.path.join(ckpt_dir, "*.ckpt")))
        if not ckpts:
            continue
        run_dir = os.path.dirname(ckpt_dir)
        rel = os.path.relpath(run_dir, exp_dir)
        runs.append(
            {
                "run_dir": run_dir,
                "rel": rel,
                "method": "all" if single_method else rel.split(os.sep)[0],
                "score": run_score(run_dir),
                "ckpts": ckpts,
                "bytes": sum(os.path.getsize(c) for c in ckpts),
            }
        )
    return runs


def select_runs(
    runs: list[dict], top_best: int, top_other: int
) -> tuple[set[str], str | None]:
    """Return the run_dirs to keep and the best method."""
    by_method = defaultdict(list)
    keep = set()
    for r in runs:
        if math.isnan(r["score"]):
            keep.add(r["run_dir"])  # never delete what we can't rank
        else:
            by_method[r["method"]].append(r)
    if not by_method:
        return keep, None

    for method_runs in by_method.values():
        method_runs.sort(key=lambda r: r["score"])
    best_method = min(by_method, key=lambda m: by_method[m][0]["score"])
    for method, method_runs in by_method.items():
        k = top_best if method == best_method else top_other
        keep.update(r["run_dir"] for r in method_runs[:k])
    return keep, best_method


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--exp", nargs="*", help="Experiment dir names (default: all)")
    parser.add_argument("--top_best", type=int, default=3)
    parser.add_argument("--top_other", type=int, default=1)
    parser.add_argument("--delete", action="store_true", help="Actually delete")
    args = parser.parse_args()

    exp_names = args.exp or sorted(
        d
        for d in os.listdir(args.results_dir)
        if os.path.isdir(os.path.join(args.results_dir, d))
    )

    total_freed = total_deleted = total_kept = 0
    for exp_name in exp_names:
        exp_dir = os.path.join(args.results_dir, exp_name)
        runs = find_runs(exp_dir)
        if not runs:
            continue
        keep, best_method = select_runs(runs, args.top_best, args.top_other)
        to_delete = [r for r in runs if r["run_dir"] not in keep]
        freed = sum(r["bytes"] for r in to_delete)
        n_deleted = sum(len(r["ckpts"]) for r in to_delete)
        n_kept = sum(len(r["ckpts"]) for r in runs) - n_deleted

        print(f"\n=== {exp_name} (best method: {best_method})")
        for r in sorted(runs, key=lambda r: (r["method"], r["score"])):
            if r["run_dir"] in keep:
                print(
                    f"  keep  {r['score']:.4f}  {r['bytes'] / 1e9:6.2f}GB  {r['rel']}"
                )
        print(f"  -> delete {n_deleted} ckpts ({freed / 1e9:.2f} GB), keep {n_kept}")

        if args.delete and to_delete:
            log_path = os.path.join(exp_dir, "pruned_checkpoints.txt")
            with open(log_path, "a") as f:
                for r in to_delete:
                    for c in r["ckpts"]:
                        os.remove(c)
                        f.write(f"{r['score']:.6f}\t{os.path.relpath(c, exp_dir)}\n")

        total_freed += freed
        total_deleted += n_deleted
        total_kept += n_kept

    action = "Deleted" if args.delete else "Would delete"
    print(
        f"\n{action} {total_deleted} ckpts ({total_freed / 1e9:.1f} GB), "
        f"keeping {total_kept}."
    )


if __name__ == "__main__":
    main()
