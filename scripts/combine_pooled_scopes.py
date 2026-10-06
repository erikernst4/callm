#!/usr/bin/env python3
"""
Combine the pooled prompt and answer scopes written by scripts/pool_activations.py
into a single ``split`` scope, for training SplitConfidenceAugmentedLLM on
pre-pooled activations.

Reads ``<pooled_dir>/prompt`` and ``<pooled_dir>/answer`` shard by shard and
writes ``<pooled_dir>/split/activations_XXXX.pt``. Each layer tensor has shape
(2, hidden_dim), the pooled prompt vector followed by the pooled answer vector,
with ``prompt_length == 1`` and ``seq_len == 2``. Train on them with
``ActivationsDataModule(activations_to_use="split", pool_strategy="mean")``,
which returns the two vectors as the prompt and answer activations.

Usage:
    uv run python3 scripts/combine_pooled_scopes.py \\
        --pooled_dir activations_cache/mmlu_auxiliary_train_50k_pooled_mean
"""

import os
import glob
import json
import argparse

import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pooled_dir", required=True)
    args = parser.parse_args()

    prompt_dir = os.path.join(args.pooled_dir, "prompt")
    answer_dir = os.path.join(args.pooled_dir, "answer")
    split_dir = os.path.join(args.pooled_dir, "split")

    prompt_shards = sorted(glob.glob(os.path.join(prompt_dir, "activations_*.pt")))
    answer_shards = sorted(glob.glob(os.path.join(answer_dir, "activations_*.pt")))
    if not prompt_shards:
        raise FileNotFoundError(f"No pooled shards found in {prompt_dir}")
    if [os.path.basename(p) for p in prompt_shards] != [
        os.path.basename(p) for p in answer_shards
    ]:
        raise ValueError(f"Shard files differ between {prompt_dir} and {answer_dir}")

    os.makedirs(split_dir, exist_ok=True)
    num_samples = 0
    for prompt_path, answer_path in tqdm(
        list(zip(prompt_shards, answer_shards)), desc="Combining shards"
    ):
        prompt_samples = torch.load(prompt_path, map_location="cpu", weights_only=False)
        answer_samples = torch.load(answer_path, map_location="cpu", weights_only=False)
        if len(prompt_samples) != len(answer_samples):
            raise ValueError(f"Sample count differs in {os.path.basename(prompt_path)}")

        combined = []
        for p, a in zip(prompt_samples, answer_samples):
            if p["sample_idx"] != a["sample_idx"]:
                raise ValueError(
                    f"sample_idx mismatch: {p['sample_idx']} vs {a['sample_idx']}"
                )
            if p["activations"].keys() != a["activations"].keys():
                raise ValueError(f"Layer mismatch for sample {p['sample_idx']}")
            combined.append(
                {
                    "activations": {
                        layer: torch.cat(
                            [p["activations"][layer], a["activations"][layer]]
                        )
                        for layer in p["activations"]
                    },
                    "seq_len": 2,
                    "prompt_length": 1,
                    "sample_idx": p["sample_idx"],
                }
            )
        torch.save(combined, os.path.join(split_dir, os.path.basename(prompt_path)))
        num_samples += len(combined)

    with open(os.path.join(prompt_dir, "metadata.json")) as f:
        metadata = json.load(f)
    metadata["scope"] = "split"
    metadata["num_samples"] = num_samples
    with open(os.path.join(split_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Done: {num_samples} samples, {len(prompt_shards)} shards -> {split_dir}")


if __name__ == "__main__":
    main()
