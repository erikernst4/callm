#!/usr/bin/env python3
"""
Mean-pool cached activations once so probe sweeps don't re-read the full
token-level activations on every run.

Streams the raw activation files (consolidated ``activations_XXXX.pt`` shards
or unconsolidated ``activations_chunk_rank*_N.pt`` chunks) from --input_dir and
writes, for each scope (prompt, answer, both), pooled shards to
``<output_dir>/<scope>/activations_XXXX.pt``.

Every pooled sample keeps the original shard format, but each layer tensor has
shape (1, hidden_dim) (the mean over the scope's tokens) and
``prompt_length == seq_len == 1``. Train on them with
``ActivationsDataModule(activations_to_use="both", pool_strategy="mean")``,
which just returns that single pooled vector.

Usage:
    uv run python3 scripts/pool_activations.py \\
        --input_dir activations_cache/mmlu_auxiliary_train_50k/lightning_logs/version_1 \\
        --output_dir activations_cache/mmlu_auxiliary_train_50k_pooled_mean
"""

import os
import re
import glob
import json
import argparse
from concurrent.futures import ProcessPoolExecutor

import torch
from tqdm import tqdm

SCOPES = ("prompt", "answer", "both")
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def find_activation_files(input_dir: str) -> list[str]:
    """Return chunk files (if any) or consolidated shards, in sample order."""
    files = glob.glob(os.path.join(input_dir, "activations_chunk_rank*_*.pt"))
    if not files:
        files = glob.glob(os.path.join(input_dir, "activations_[0-9]*.pt"))

    def file_num(p: str) -> int:
        return int(re.search(r"(\d+)\.pt$", os.path.basename(p)).group(1))

    return sorted(files, key=file_num)


def pool_file(args) -> tuple[list[int], list[int], dict[str, torch.Tensor]]:
    """Load one activation file and mean-pool every sample for every scope.

    Returns (sample_idxs, layers, {scope: (num_samples, num_layers, dim)}).
    Results are stacked into one tensor per scope because tensors returned from
    a worker are sent through shared memory, one mapping per tensor: returning
    thousands of small tensors per file exhausts vm.max_map_count.
    """
    path, save_dtype = args
    # Parallelism comes from the process pool; avoid thread oversubscription
    torch.set_num_threads(1)
    samples = torch.load(path, map_location="cpu", weights_only=False, mmap=True)

    layers = sorted(samples[0]["activations"].keys())
    sample_idxs = []
    pooled = {scope: [] for scope in SCOPES}
    for sample in samples:
        prompt_len = sample.get("prompt_length", sample["seq_len"])
        per_layer = {scope: [] for scope in SCOPES}
        for layer in layers:
            act = sample["activations"][layer]
            slices = {
                "prompt": act[:prompt_len],
                "answer": act[prompt_len:],
                "both": act,
            }
            for scope, sl in slices.items():
                # Same fallback as ActivationsDataModule for empty slices
                if sl.shape[0] == 0:
                    vec = torch.zeros(act.shape[-1], dtype=torch.float32)
                else:
                    vec = sl.float().mean(dim=0)
                per_layer[scope].append(vec.to(save_dtype))
        for scope in SCOPES:
            pooled[scope].append(torch.stack(per_layer[scope]))
        sample_idxs.append(sample["sample_idx"])
    del samples
    return sample_idxs, layers, {s: torch.stack(v) for s, v in pooled.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shard_size", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_dtype", default="bfloat16", choices=list(DTYPES))
    args = parser.parse_args()

    files = find_activation_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f"No activation files found in {args.input_dir}")
    print(f"Pooling {len(files)} files from {args.input_dir} -> {args.output_dir}")

    save_dtype = DTYPES[args.save_dtype]
    for scope in SCOPES:
        os.makedirs(os.path.join(args.output_dir, scope), exist_ok=True)

    buffers = {scope: [] for scope in SCOPES}
    shard_idx = 0
    num_samples = 0
    layer_indices = None
    hidden_size = None

    def flush():
        nonlocal shard_idx
        for scope in SCOPES:
            path = os.path.join(
                args.output_dir, scope, f"activations_{shard_idx:04d}.pt"
            )
            torch.save(buffers[scope], path)
            buffers[scope] = []
        shard_idx += 1

    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        results = pool.map(pool_file, [(f, save_dtype) for f in files])
        for sample_idxs, layers, pooled in tqdm(
            results, total=len(files), desc="Pooling"
        ):
            if layer_indices is None:
                layer_indices = layers
                hidden_size = pooled["both"].shape[-1]
            elif layers != layer_indices:
                raise ValueError(f"Layer mismatch: {layers} vs {layer_indices}")
            for i, sample_idx in enumerate(sample_idxs):
                # Labels are joined by CSV row position == sample_idx
                if sample_idx != num_samples:
                    raise ValueError(
                        f"Expected sample_idx {num_samples}, got {sample_idx}"
                    )
                for scope in SCOPES:
                    # clone() so each vector owns its storage: torch.save of a
                    # view would serialize the whole per-file tensor
                    buffers[scope].append(
                        {
                            "activations": {
                                layer: pooled[scope][i, j].clone().unsqueeze(0)
                                for j, layer in enumerate(layers)
                            },
                            "seq_len": 1,
                            "prompt_length": 1,
                            "sample_idx": sample_idx,
                        }
                    )
                num_samples += 1
                if len(buffers["both"]) >= args.shard_size:
                    flush()
    if buffers["both"]:
        flush()

    for scope in SCOPES:
        with open(os.path.join(args.output_dir, scope, "metadata.json"), "w") as f:
            json.dump(
                {
                    "source_dir": os.path.abspath(args.input_dir),
                    "scope": scope,
                    "pool_strategy": "mean",
                    "layer_indices": layer_indices,
                    "hidden_size": hidden_size,
                    "num_samples": num_samples,
                    "dtype": args.save_dtype,
                },
                f,
                indent=4,
            )
    print(f"Done: {num_samples} samples, {shard_idx} shards per scope.")


if __name__ == "__main__":
    main()
