#!/bin/bash
# ==============================================================================
# Extract Activations for MMLU Dataset using Qwen3.5-9B
#
# Usage:
#   ./scripts/run_extract_mmlu_activations.sh [DEVICE_IDX] [SPLIT]
#
# Arguments:
#   DEVICE_IDX: GPU index (default: 0)
#   SPLIT: "val", "test", "train_half", or "all" (default: "all")
#
# Environment variables:
#   MAX_SAMPLES: Override sample limit (e.g. MAX_SAMPLES=10 for smoke test)
#   RESUME: Set to 1 to resume from existing chunks in the output_dir
#   RESUME_FROM: Override resume directory (e.g. RESUME_FROM=activations_cache/mmlu_auxiliary_train_50k)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

DEVICE_IDX="${1:-0}"
TARGET_SPLIT="${2:-all}"

extract_split() {
    local split_name="$1"
    local hf_split="$2"
    local default_samples="$3"
    local output_dir="$4"

    local num_samples="${MAX_SAMPLES:-$default_samples}"

    local resume_arg=""
    if [[ -n "$RESUME_FROM" ]]; then
        resume_arg="--resume_from ${RESUME_FROM}"
    elif [[ "${RESUME:-0}" == "1" ]]; then
        resume_arg="--resume_from ${output_dir}"
    fi

    echo "=========================================================================="
    echo " Extracting MMLU Activations: ${split_name}"
    echo "   HF Split:    ${hf_split}"
    echo "   Max Samples: ${num_samples}"
    echo "   GPU:         ${DEVICE_IDX}"
    echo "   Output Dir:  ${output_dir}"
    if [[ -n "$resume_arg" ]]; then
        echo "   Resume:      ${resume_arg}"
    fi
    echo "   Layers:      12, 14, 16, 17, 18, 19, 20, 21, 22, 24"
    echo "=========================================================================="

    local sample_arg=""
    if [[ "$num_samples" != "null" && -n "$num_samples" ]]; then
        sample_arg="--data.init_args.max_samples ${num_samples}"
    else
        sample_arg="--data.init_args.max_samples null"
    fi

    uv run main.py extract_activations \
        --config configs/config_extract_activations_mmlu.yaml \
        --trainer.devices "[${DEVICE_IDX}]" \
        --output_dir "${output_dir}" \
        --trainer.default_root_dir "${output_dir}" \
        --data.init_args.split "${hf_split}" \
        $sample_arg \
        $resume_arg

    # Automatically generate evaluation_results.csv with correctness labels
    local llm_outputs_csv
    llm_outputs_csv=$(find "${output_dir}" -name "llm_outputs.csv" 2>/dev/null | sort -V | tail -n 1)
    if [[ -n "$llm_outputs_csv" && -f "$llm_outputs_csv" ]]; then
        echo "Evaluating MMLU outputs to produce evaluation_results.csv..."
        uv run python3 scripts/evaluate_mmlu_outputs.py --llm_outputs_csv "${llm_outputs_csv}"
    fi

    echo "Finished extraction and evaluation for ${split_name} -> ${output_dir}"
    echo ""
}

if [[ "$TARGET_SPLIT" == "val" || "$TARGET_SPLIT" == "all" ]]; then
    extract_split "validation" "validation" "null" "activations_cache/mmlu_val"
fi

if [[ "$TARGET_SPLIT" == "test" || "$TARGET_SPLIT" == "all" ]]; then
    extract_split "test" "test" "null" "activations_cache/mmlu_test"
fi

if [[ "$TARGET_SPLIT" == "train_half" || "$TARGET_SPLIT" == "all" ]]; then
    extract_split "auxiliary_train (half)" "auxiliary_train" "50000" "activations_cache/mmlu_auxiliary_train_50k"
fi

echo "All requested MMLU activation extractions completed successfully."
