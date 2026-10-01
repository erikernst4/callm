#!/bin/bash
# ==============================================================================
# Prepare MMLU activations for probe training (run once before exp10).
#
#   1. Creates evaluation_results.csv (correctness labels) for auxiliary_train
#      from its llm_outputs.csv, if missing.
#   2. Mean-pools the token-level activations of the train chunks and the val
#      shards for each scope (prompt, answer, both) with
#      scripts/pool_activations.py. This reads the ~1.4TB of raw train chunks
#      only once instead of once per probe run. Output: ~11GB (train, bf16)
#      and ~0.4GB (val).
#
# The train activations are read directly from the unconsolidated chunks of
# version_1 (the resumed run, which hard-links version_0's chunks), so no
# consolidation is needed. The partial activations_0000.pt left in the train
# root by the failed consolidation is ignored.
#
# Usage:
#   bash scripts/prepare_mmlu_probe_data.sh
#
# Environment variables:
#   NUM_WORKERS: Parallel processes for pooling (default: 8)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

NUM_WORKERS="${NUM_WORKERS:-8}"

TRAIN_ROOT="activations_cache/mmlu_auxiliary_train_50k"
TRAIN_CHUNKS_DIR="${TRAIN_ROOT}/lightning_logs/version_1"
TRAIN_LLM_OUTPUTS="${TRAIN_ROOT}/llm_outputs.csv"
TRAIN_EVAL_CSV="${TRAIN_ROOT}/evaluation_results.csv"
TRAIN_POOLED_DIR="activations_cache/mmlu_auxiliary_train_50k_pooled_mean"

VAL_SHARDS_DIR="activations_cache/mmlu_val"
VAL_POOLED_DIR="activations_cache/mmlu_val_pooled_mean"

# 1. Correctness labels for the train split
if [[ -f "$TRAIN_EVAL_CSV" ]]; then
    echo "--> ${TRAIN_EVAL_CSV} already exists. Skipping evaluation."
else
    uv run python3 scripts/evaluate_mmlu_outputs.py \
        --llm_outputs_csv "$TRAIN_LLM_OUTPUTS" \
        --output_dir "$TRAIN_ROOT"
fi

# 2. Mean-pool activations per scope (skipped if already pooled)
pool() {
    local input_dir="$1"
    local output_dir="$2"
    if [[ -f "${output_dir}/both/metadata.json" ]]; then
        echo "--> ${output_dir} already pooled. Skipping."
        return
    fi
    echo "=========================================================================="
    echo " Pooling ${input_dir} -> ${output_dir}"
    echo "=========================================================================="
    uv run python3 scripts/pool_activations.py \
        --input_dir "$input_dir" \
        --output_dir "$output_dir" \
        --num_workers "$NUM_WORKERS"
}

pool "$VAL_SHARDS_DIR" "$VAL_POOLED_DIR"
pool "$TRAIN_CHUNKS_DIR" "$TRAIN_POOLED_DIR"

echo "MMLU probe data ready:"
echo "  Train: ${TRAIN_POOLED_DIR}/{prompt,answer,both} + ${TRAIN_EVAL_CSV}"
echo "  Val:   ${VAL_POOLED_DIR}/{prompt,answer,both}"
