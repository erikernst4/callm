#!/bin/bash
# ==============================================================================
# Extract Activations and Exact Tokens for Curated TriviaQA Subset (N=455)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

DEVICE_IDX="${1:-0}"
OUTPUT_DIR="activations_cache/curated_triviaqa"

echo "=========================================================================="
echo " Step 1: Extract Activations from Qwen3.5-9B (GPU ${DEVICE_IDX})"
echo "   Output Dir: ${OUTPUT_DIR}"
echo "   Layers:     12, 14, 16, 17, 18, 19, 20, 21, 22, 24"
echo "=========================================================================="

uv run main.py extract_activations \
    --config configs/config_extract_activations_curated.yaml \
    --trainer.devices "[${DEVICE_IDX}]" \
    --output_dir "${OUTPUT_DIR}" \
    --trainer.default_root_dir "${OUTPUT_DIR}"

# Locate the generated llm_outputs.csv
LLM_OUTPUTS_CSV=$(find "${OUTPUT_DIR}" -name "llm_outputs.csv" 2>/dev/null | sort -V | tail -n 1)
if [[ -z "$LLM_OUTPUTS_CSV" || ! -f "$LLM_OUTPUTS_CSV" ]]; then
    echo "Error: llm_outputs.csv not found in ${OUTPUT_DIR}"
    exit 1
fi
echo "Found LLM outputs at: ${LLM_OUTPUTS_CSV}"

echo ""
echo "=========================================================================="
echo " Step 2: Evaluate Exact Match & Export Incorrect Cases CSV"
echo "=========================================================================="
uv run python3 scripts/evaluate_curated_and_export_incorrect.py \
    --llm_outputs_csv "${LLM_OUTPUTS_CSV}" \
    --output_dir "results/curated_benchmark"

echo ""
echo "=========================================================================="
echo " Step 3: Extract Exact Answer Entities (Gemini)"
echo "=========================================================================="
uv run main.py extract_exact_answers \
    --config configs/config_extract_exact_answers.yaml \
    --trainer.default_root_dir "${OUTPUT_DIR}" \
    --data.init_args.llm_outputs_path "${LLM_OUTPUTS_CSV}" \
    --data.init_args.batch_size 1 \
    --data.init_args.max_samples null

EXACT_CSV=$(find "${OUTPUT_DIR}" -name "evaluation_results_exact.csv" 2>/dev/null | sort -V | tail -n 1)
echo "Exact token extraction complete: ${EXACT_CSV}"
echo "=========================================================================="


uv run python3 scripts/benchmark_curated_checkpoints.py
