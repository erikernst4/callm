#!/bin/bash
# ==============================================================================
# Pre-compute Correctness Labels for On-the-Fly Probe Training
#
# Usage:
#   bash scripts/precompute_labels.sh <N> [DEVICE] [SPLIT] [PREVIOUS_EVAL_CSV]
#
# Arguments:
#   N                 Number of samples to pre-compute labels for (e.g. 50000)
#   DEVICE            GPU device index for Step 1 (default: 0)
#   SPLIT             Dataset split to use (default: train)
#   PREVIOUS_EVAL_CSV Path to previous evaluation CSV for incremental labeling.
#                     Defaults to existing 5k evaluation CSV if present.
#
# This script:
#   1. Runs LLM inference (batch_size=1) to generate answers for N samples
#   2. Runs correctness evaluation using Gemini (GCP) with incremental caching:
#      - Exact matches skip API calls
#      - Previously evaluated samples from PREVIOUS_EVAL_CSV skip API calls
#
# Output:
#   <OUTPUT_DIR>/lightning_logs/version_<V>/llm_outputs.csv     — questions + full raw outputs
#   <OUTPUT_DIR>/lightning_logs/version_<V>_evaluation/version_<E>/evaluation_results.csv
#                                                                — correctness labels
# ==============================================================================

set -e

# Parse arguments
N="${1:?Usage: $0 <N> [DEVICE] [SPLIT] [PREVIOUS_EVAL_CSV]}"
DEVICE="${2:-0}"
SPLIT="${3:-train}"

# Default previous eval CSV to the existing 5k run if available
DEFAULT_PREV_CSV="activations_cache/qwen_triviaqa/lightning_logs/version_5_evaluation/version_9/evaluation_results.csv"
if [[ -n "$4" ]]; then
    PREVIOUS_EVAL_CSV="$4"
elif [[ -f "$DEFAULT_PREV_CSV" ]]; then
    PREVIOUS_EVAL_CSV="$DEFAULT_PREV_CSV"
else
    PREVIOUS_EVAL_CSV=""
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_DIR="activations_cache/qwen_triviaqa_${N}"

echo "=========================================================================="
echo " Pre-computing labels for N=${N} TriviaQA samples"
echo "=========================================================================="
echo " Project Root      : ${PROJECT_ROOT}"
echo " Output Dir        : ${OUTPUT_DIR}"
echo " Device (Step 1)   : ${DEVICE}"
echo " Split             : ${SPLIT}"
echo " Previous Eval CSV : ${PREVIOUS_EVAL_CSV:-None (no cache)}"
echo "=========================================================================="

# Step 1: Run LLM inference (batch_size=1 for deterministic outputs)
echo ""
echo "[Step 1/2] Running LLM inference (batch_size=1)..."
echo ""

uv run main.py validate \
    --trainer.evaluate_correctness false \
    --trainer.accelerator gpu \
    --trainer.strategy auto \
    --trainer.devices "[${DEVICE}]" \
    --trainer.deterministic true \
    --trainer.inference_mode true \
    --trainer.default_root_dir "$OUTPUT_DIR" \
    --model.class_path callm.models.LLM \
    --model.init_args.model_name "Qwen/Qwen3.5-9B" \
    --model.init_args.train false \
    --model.init_args.return_logits false \
    --model.init_args.extractor.class_path callm.extractors.SequencePosteriorExtractor \
    --model.init_args.extractor.init_args.model_name "Qwen/Qwen3.5-9B" \
    --model.init_args.save_outputs false \
    --model.init_args.flush_outputs_every_n_steps 1000 \
    --model.init_args.max_new_tokens 200 \
    --model.init_args.raw_output_max_length null \
    --data.class_path callm.data.TriviaQADataModule \
    --data.init_args.batch_size 1 \
    --data.init_args.model_name "Qwen/Qwen3.5-9B" \
    --data.init_args.prompt "callm.prompts.TRANSPARENT_PROMPT_ZERO_SHOT" \
    --data.init_args.max_samples "$N" \
    --data.init_args.seed 42 \
    --data.init_args.num_workers 24 \
    --data.init_args.disable_thinking true \
    --data.init_args.split "$SPLIT"

# Find the latest version directory
LATEST_VERSION=$(ls -vd "${OUTPUT_DIR}/lightning_logs/version_"[0-9]* 2>/dev/null | grep -v "_evaluation" | tail -n 1)
if [[ -z "$LATEST_VERSION" ]]; then
    echo "Error: No version directory found in ${OUTPUT_DIR}/lightning_logs/"
    exit 1
fi

LLM_OUTPUTS="${LATEST_VERSION}/llm_outputs.csv"
if [[ ! -f "$LLM_OUTPUTS" ]]; then
    echo "Error: llm_outputs.csv not found at ${LLM_OUTPUTS}"
    exit 1
fi

echo ""
echo "LLM outputs saved to: ${LLM_OUTPUTS}"
echo "$(wc -l < "$LLM_OUTPUTS") lines (including header)"

# Step 2: Run correctness evaluation using Gemini via GCP
echo ""
echo "[Step 2/2] Running correctness evaluation (Gemini via GCP)..."
echo ""

EXTRA_EVAL_ARGS=()
if [[ -n "$PREVIOUS_EVAL_CSV" && -f "$PREVIOUS_EVAL_CSV" ]]; then
    echo "Using previous evaluation cache: $PREVIOUS_EVAL_CSV"
    EXTRA_EVAL_ARGS+=("--previous_eval_csv" "$PREVIOUS_EVAL_CSV")
fi

uv run main.py evaluation \
    --config configs/config_gcp_evaluation.yaml \
    --trainer.default_root_dir "$OUTPUT_DIR" \
    --llm_outputs_path "$LLM_OUTPUTS" \
    "${EXTRA_EVAL_ARGS[@]}"

# Find evaluation results
EVAL_DIR=$(ls -vd "${LATEST_VERSION}_evaluation/version_"[0-9]* 2>/dev/null | tail -n 1)
if [[ -z "$EVAL_DIR" ]]; then
    echo "Error: No evaluation directory found"
    exit 1
fi

EVAL_CSV="${EVAL_DIR}/evaluation_results.csv"
if [[ ! -f "$EVAL_CSV" ]]; then
    echo "Error: evaluation_results.csv not found at ${EVAL_CSV}"
    exit 1
fi

echo ""
echo "=========================================================================="
echo " Pre-computation complete!"
echo "=========================================================================="
echo " LLM Outputs    : ${LLM_OUTPUTS}"
echo " Evaluation CSV : ${EVAL_CSV}"
echo ""
echo " To train probes with on-the-fly activations, use:"
echo "   uv run main.py fit \\"
echo "     --config configs/config_train_probe_on_the_fly.yaml \\"
echo "     --data.init_args.llm_outputs_csv '${LLM_OUTPUTS}' \\"
echo "     --data.init_args.evaluation_csv '${EVAL_CSV}'"
echo "=========================================================================="
