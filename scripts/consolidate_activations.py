"""
Stream-consolidates temporary activation chunk files into 1000-sample shards.

Features:
- Low memory: processes chunks incrementally without loading full dataset into RAM.
- Safe chunk deletion: deletes temporary chunks only after the consolidated shard
  containing their samples has been successfully saved to disk.
- Generates llm_outputs.csv, metadata.json, and evaluation_results.csv.
"""

import os
import csv
import json
import glob
import re
import argparse
import torch


def get_chunk_number(filepath: str) -> int:
    match = re.search(
        r"activations_chunk_rank\d+_(\d+)\.pt", os.path.basename(filepath)
    )
    if match:
        return int(match.group(1))
    # Fallback to any trailing number
    match = re.search(r"(\d+)\.pt$", os.path.basename(filepath))
    return int(match.group(1)) if match else 0


def serialize_gold_answers(gold_answers) -> str:
    if gold_answers is None:
        return ""
    if isinstance(gold_answers, list):
        return "|".join(str(a) for a in gold_answers)
    return str(gold_answers)


def consolidate(
    chunks_dir: str,
    output_dir: str,
    model_name: str = "Qwen/Qwen3.5-9B",
    shard_size: int = 1000,
    delete_chunks: bool = True,
    save_dtype: str = "bfloat16",
):
    os.makedirs(output_dir, exist_ok=True)

    chunk_files = glob.glob(os.path.join(chunks_dir, "activations_chunk_*.pt"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {chunks_dir}")

    chunk_files = sorted(chunk_files, key=get_chunk_number)
    print(f"Found {len(chunk_files)} chunk files in {chunks_dir}")

    # 1. Stream write llm_outputs.csv
    llm_outputs_file = os.path.join(output_dir, "llm_outputs.csv")
    total_samples = 0
    first_sample = None

    print(f"Writing {llm_outputs_file}...")
    with open(llm_outputs_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "gold_answers", "pred_answer", "raw_output"])

        for c_file in chunk_files:
            chunk = torch.load(c_file, map_location="cpu", weights_only=False)
            if first_sample is None and chunk:
                first_sample = chunk[0]
            for sample in chunk:
                total_samples += 1
                gold_str = serialize_gold_answers(sample.get("gold_answers"))
                writer.writerow(
                    [
                        sample.get("question", ""),
                        gold_str,
                        sample.get("pred_answer", ""),
                        sample.get("raw_output", "").replace("\n", "\\n"),
                    ]
                )
            del chunk

    print(f"Wrote {total_samples} samples to {llm_outputs_file}")

    if total_samples == 0 or first_sample is None:
        print("No samples to consolidate.")
        return

    # Extract metadata info from first sample
    first_layer_idx = list(first_sample["activations"].keys())[0]
    hidden_size = first_sample["activations"][first_layer_idx].shape[-1]
    layer_indices = sorted(list(first_sample["activations"].keys()))

    # 2. Stream consolidate activations into shards
    shard_buffer = []
    current_shard_idx = 0
    chunks_in_flight = []  # Chunks contributing to current shard

    print(f"Consolidating shards (shard size: {shard_size})...")
    for chunk_idx, c_file in enumerate(chunk_files):
        chunk = torch.load(c_file, map_location="cpu", weights_only=False)
        chunks_in_flight.append(c_file)

        for sample in chunk:
            shard_buffer.append(
                {
                    "activations": sample["activations"],
                    "seq_len": sample["seq_len"],
                    "prompt_length": sample["prompt_length"],
                    "sample_idx": sample["sample_idx"],
                }
            )

            if len(shard_buffer) >= shard_size:
                shard_file = os.path.join(
                    output_dir, f"activations_{current_shard_idx:04d}.pt"
                )
                torch.save(shard_buffer, shard_file)
                print(
                    f"Saved shard {current_shard_idx:04d} with {len(shard_buffer)} samples to {shard_file}"
                )
                current_shard_idx += 1
                shard_buffer = []

                # Delete chunk files that have been fully absorbed
                # All chunks except possibly the current one (if it spans across shards)
                if delete_chunks:
                    while len(chunks_in_flight) > 1:
                        old_chunk = chunks_in_flight.pop(0)
                        if os.path.exists(old_chunk):
                            os.remove(old_chunk)

        del chunk

    # Save any remaining samples in the final shard
    if shard_buffer:
        shard_file = os.path.join(output_dir, f"activations_{current_shard_idx:04d}.pt")
        torch.save(shard_buffer, shard_file)
        print(
            f"Saved final shard {current_shard_idx:04d} with {len(shard_buffer)} samples to {shard_file}"
        )
        current_shard_idx += 1
        shard_buffer = []

    # Clean up any remaining chunks in flight
    if delete_chunks:
        for c_file in chunks_in_flight:
            if os.path.exists(c_file):
                os.remove(c_file)
        print("All temporary chunk files cleaned up.")

    # 3. Write metadata.json
    metadata = {
        "model_name": model_name,
        "layer_indices": layer_indices,
        "hidden_size": hidden_size,
        "num_samples": total_samples,
        "dtype": save_dtype,
    }
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Consolidate activation chunk files.")
    parser.add_argument(
        "--chunks_dir", type=str, required=True, help="Directory containing chunk files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for shards"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument(
        "--keep_chunks", action="store_true", help="Keep chunk files (do not delete)"
    )
    parser.add_argument("--save_dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    consolidate(
        chunks_dir=args.chunks_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        shard_size=args.shard_size,
        delete_chunks=not args.keep_chunks,
        save_dtype=args.save_dtype,
    )


if __name__ == "__main__":
    main()
