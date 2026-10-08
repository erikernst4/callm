#!/bin/bash
# ==============================================================================
# Experiment 8: Exact-Answer Token Windows with Linear Probes (N=50,000)
# Experiment Name: exp8_exact_tokens_probes_50k
#
# Two-phase experiment:
#   Phase 1 — Extract exact answer entities from the 50k LLM outputs using
#             Gemini (via GCP). Produces evaluation_results_exact.csv.
#   Phase 2 — Train per-layer probes (and weighted average probes) on 5 exact
#             answer token windows, replicating Experiment 2 at N=50k scale.
#
# Token windows probed (matching LLMsKnow paper):
#   1. exact_answer                    (all tokens of the exact answer entity)
#   2. exact_answer_first_token        (first token of exact answer)
#   3. exact_answer_last_token         (last token of exact answer)
#   4. exact_answer_before_first_token (token immediately before exact answer)
#   5. exact_answer_after_last_token   (token immediately after exact answer)
#
# Fallback: When no valid exact answer is found, falls back to the full answer
# without the prompt (i.e. mean-pooled answer tokens), guaranteeing 100% sample
# retention.
#
# Uses:
#   - Activations from version_27 (Qwen/Qwen3.5-9B, 50,000 TriviaQA samples,
#     layers [12, 14, 16, 17, 18, 19, 20, 21, 22, 24])
#   - LLM outputs from version_0 (50k inference run)
#   - Evaluation from version_0_evaluation/version_0
#
# Outputs are organized under:
#   results/exp8_exact_tokens_probes_50k/<window>/layer_<N>
#   results/exp8_exact_tokens_probes_50k/<window>/weighted
#
# Usage:
#   bash scripts/run_exp8_exact_tokens_probes_50k.sh [DEVICE_IDX]
# Example:
#   bash scripts/run_exp8_exact_tokens_probes_50k.sh 0
# ==============================================================================

set -e

# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

DEVICE_IDX="${1:-0}"
DEVICE="[${DEVICE_IDX}]"

# Configuration
EXP_NAME="${EXP_NAME:-exp8_exact_tokens_probes_50k}"
RESULTS_DIR="results/${EXP_NAME}"

# 50k activations (version_27) — same cache used by exp7
ACTIVATIONS_DIR="activations_cache/qwen_triviaqa/lightning_logs/version_27"

# 50k LLM outputs and evaluation results
LLM_OUTPUTS_CSV="activations_cache/qwen_triviaqa_50000/lightning_logs/version_0/llm_outputs.csv"
EVALUATION_CSV="activations_cache/qwen_triviaqa_50000/lightning_logs/version_0_evaluation/version_0/evaluation_results.csv"

# Where Phase 1 saves the exact answer extraction results
EXACT_EXTRACTION_ROOT="activations_cache/qwen_triviaqa_50000"
# The exact CSV will be written by the ExactAnswerExtractorModule into trainer.log_dir

CONFIG_SINGLE="configs/config_train_probe.yaml"
CONFIG_WEIGHTED="configs/config_train_probe_weighted.yaml"

# Model and prompt used during activation extraction (needed to reconstruct token indices)
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"
PROMPT="${PROMPT:-callm.prompts.base.TRANSPARENT_PROMPT_ZERO_SHOT}"
MODEL_DIM=4096

# 10 Extracted Layers available in version_27 (N=50k)
INDIVIDUAL_LAYERS=(12 14 16 17 18 19 20 21 22 24)
WEIGHTED_LAYERS="[12, 14, 16, 17, 18, 19, 20, 21, 22, 24]"

# Exact-answer token windows to probe (matching LLMsKnow paper)
WINDOWS=(
    "exact_answer"
    "exact_answer_first_token"
    "exact_answer_last_token"
    "exact_answer_before_first_token"
    "exact_answer_after_last_token"
)

echo "=========================================================================="
echo " Starting Experiment: ${EXP_NAME}"
echo "=========================================================================="
echo " Project Root     : ${PROJECT_ROOT}"
echo " Results Dir      : ${RESULTS_DIR}"
echo " Activations Dir  : ${ACTIVATIONS_DIR}"
echo " LLM Outputs CSV  : ${LLM_OUTPUTS_CSV}"
echo " Evaluation CSV   : ${EVALUATION_CSV}"
echo " Model Name       : ${MODEL_NAME}"
echo " Prompt           : ${PROMPT}"
echo " GPU Device       : ${DEVICE}"
echo " Individual Layers: ${INDIVIDUAL_LAYERS[*]}"
echo " Weighted Layers  : ${WEIGHTED_LAYERS}"
echo " Token Windows    : ${WINDOWS[*]}"
echo "=========================================================================="

# Check prerequisites
if [[ ! -d "$ACTIVATIONS_DIR" ]]; then
    echo "Error: Activations directory not found at $ACTIVATIONS_DIR"
    exit 1
fi

if [[ ! -f "$LLM_OUTPUTS_CSV" ]]; then
    echo "Error: LLM outputs CSV not found at $LLM_OUTPUTS_CSV"
    exit 1
fi

if [[ ! -f "$EVALUATION_CSV" ]]; then
    echo "Error: Evaluation CSV not found at $EVALUATION_CSV"
    exit 1
fi

# ==============================================================================
# Phase 1: Extract Exact Answers
# ==============================================================================
# Check if an exact CSV already exists (from a previous run) to allow resumption
EXACT_CSV=""
EXACT_CSV_CANDIDATES=$(find "${EXACT_EXTRACTION_ROOT}/lightning_logs" -name "evaluation_results_exact.csv" 2>/dev/null || true)
if [[ -n "$EXACT_CSV_CANDIDATES" ]]; then
    # Use the most recently modified one
    EXACT_CSV=$(echo "$EXACT_CSV_CANDIDATES" | xargs ls -t 2>/dev/null | head -n 1)
fi

if [[ -n "$EXACT_CSV" && -f "$EXACT_CSV" ]]; then
    EXACT_LINE_COUNT=$(wc -l < "$EXACT_CSV")
    echo ""
    echo "=========================================================================="
    echo " Phase 1: SKIPPED — Exact answer CSV already exists"
    echo "   ${EXACT_CSV} (${EXACT_LINE_COUNT} lines)"
    echo "=========================================================================="
else
    echo ""
    echo "=========================================================================="
    echo " Phase 1: Extracting Exact Answers from 50k LLM Outputs"
    echo "=========================================================================="
    echo ""

    uv run main.py extract_exact_answers \
        --config configs/config_extract_exact_answers.yaml \
        --trainer.default_root_dir "$EXACT_EXTRACTION_ROOT" \
        --data.init_args.llm_outputs_path "$EVALUATION_CSV" \
        --data.init_args.max_samples null \
        --data.init_args.batch_size 1

    # Find the newly created exact CSV
    LATEST_EXACT_VERSION=$(ls -vd "${EXACT_EXTRACTION_ROOT}/lightning_logs/version_"[0-9]* 2>/dev/null | tail -n 1)
    if [[ -z "$LATEST_EXACT_VERSION" ]]; then
        echo "Error: No version directory found after exact answer extraction"
        exit 1
    fi

    EXACT_CSV="${LATEST_EXACT_VERSION}/evaluation_results_exact.csv"
    if [[ ! -f "$EXACT_CSV" ]]; then
        echo "Error: evaluation_results_exact.csv not found at ${EXACT_CSV}"
        exit 1
    fi

    EXACT_LINE_COUNT=$(wc -l < "$EXACT_CSV")
    echo ""
    echo "Phase 1 complete: ${EXACT_CSV} (${EXACT_LINE_COUNT} lines)"
fi

echo ""
echo "=========================================================================="
echo " Phase 2: Training Probes on Exact Answer Token Windows"
echo "   Exact CSV: ${EXACT_CSV}"
echo "=========================================================================="

# Count weighted layers
NUM_WEIGHTED_LAYERS=$(echo "$WEIGHTED_LAYERS" | tr -cd ',' | wc -c)
NUM_WEIGHTED_LAYERS=$((NUM_WEIGHTED_LAYERS + 1))

# Loop over token windows
for window in "${WINDOWS[@]}"; do
    echo ""
    echo "##########################################################################"
    echo " RUNNING SWEEP FOR WINDOW: ${window}"
    echo "##########################################################################"

    # 1. Per-layer probe training
    for layer in "${INDIVIDUAL_LAYERS[@]}"; do
        TARGET_DIR="${RESULTS_DIR}/${window}/layer_${layer}"
        echo ""
        echo "--------------------------------------------------------------------------"
        echo " [Window: ${window}] Training Probe for Layer ${layer}"
        echo " Target Dir: ${TARGET_DIR}"
        echo "--------------------------------------------------------------------------"

        # Skip if already trained
        LATEST_METRICS=$(ls -vd "${TARGET_DIR}"/lightning_logs/version_*/metrics.csv 2>/dev/null | tail -n 1 || true)
        if [[ -n "$LATEST_METRICS" && -f "$LATEST_METRICS" ]]; then
            echo "--> Already finished (${LATEST_METRICS}). Skipping."
            continue
        fi

        uv run main.py fit \
            --config "$CONFIG_SINGLE" \
            --trainer.default_root_dir "$TARGET_DIR" \
            --trainer.devices "$DEVICE" \
            --model.init_args.layer_idx "$layer" \
            --model.init_args.model_dim "$MODEL_DIM" \
            --data.class_path "callm.data.SelectedTokensActivationsDataModule" \
            --data.init_args.activations_dir "$ACTIVATIONS_DIR" \
            --data.init_args.evaluation_csv "$EXACT_CSV" \
            --data.init_args.layer_idx "$layer" \
            --data.init_args.activations_to_use "$window" \
            --data.init_args.pool_strategy "mean" \
            --data.init_args.model_name "$MODEL_NAME" \
            --data.init_args.prompt "$PROMPT"

        # Generate plot for this layer probe
        LATEST_LOG=$(ls -vd "${TARGET_DIR}"/lightning_logs/version_* 2>/dev/null | tail -n 1 || true)
        if [[ -n "$LATEST_LOG" && -f "${LATEST_LOG}/metrics.csv" ]]; then
            uv run python3 scripts/plot_probe_training.py \
                --metrics_csv "${LATEST_LOG}/metrics.csv" \
                --output_dir "${TARGET_DIR}/plots" || true
        fi
    done

    # 2. Weighted Average Probe training across all available layers
    WEIGHTED_TARGET_DIR="${RESULTS_DIR}/${window}/weighted"
    echo ""
    echo "--------------------------------------------------------------------------"
    echo " [Window: ${window}] Training Weighted Average Probe across layers ${WEIGHTED_LAYERS}"
    echo " Target Dir: ${WEIGHTED_TARGET_DIR}"
    echo "--------------------------------------------------------------------------"

    # Skip if already trained
    LATEST_WEIGHTED_METRICS=$(ls -vd "${WEIGHTED_TARGET_DIR}"/lightning_logs/version_*/metrics.csv 2>/dev/null | tail -n 1 || true)
    if [[ -n "$LATEST_WEIGHTED_METRICS" && -f "$LATEST_WEIGHTED_METRICS" ]]; then
        echo "--> Already finished (${LATEST_WEIGHTED_METRICS}). Skipping."
        continue
    fi

    uv run main.py fit \
        --config "$CONFIG_WEIGHTED" \
        --trainer.default_root_dir "$WEIGHTED_TARGET_DIR" \
        --trainer.devices "$DEVICE" \
        --model.init_args.layer_idx "$WEIGHTED_LAYERS" \
        --model.init_args.num_layers "$NUM_WEIGHTED_LAYERS" \
        --model.init_args.model_dim "$MODEL_DIM" \
        --data.class_path "callm.data.SelectedTokensActivationsDataModule" \
        --data.init_args.activations_dir "$ACTIVATIONS_DIR" \
        --data.init_args.evaluation_csv "$EXACT_CSV" \
        --data.init_args.layer_idx "$WEIGHTED_LAYERS" \
        --data.init_args.activations_to_use "$window" \
        --data.init_args.pool_strategy "mean" \
        --data.init_args.model_name "$MODEL_NAME" \
        --data.init_args.prompt "$PROMPT"

    # Generate plot for weighted probe
    LATEST_WEIGHTED_LOG=$(ls -vd "${WEIGHTED_TARGET_DIR}"/lightning_logs/version_* 2>/dev/null | tail -n 1 || true)
    if [[ -n "$LATEST_WEIGHTED_LOG" && -f "${LATEST_WEIGHTED_LOG}/metrics.csv" ]]; then
        uv run python3 scripts/plot_probe_training.py \
            --metrics_csv "${LATEST_WEIGHTED_LOG}/metrics.csv" \
            --output_dir "${WEIGHTED_TARGET_DIR}/plots" || true
    fi
done

echo ""
echo "=========================================================================="
echo " Summarizing Experiment Results for ${EXP_NAME}..."
echo "=========================================================================="

uv run python3 scripts/summarize_experiment_results.py \
    --exp_dir "$RESULTS_DIR" \
    --scopes "${WINDOWS[@]}"

echo ""
echo "=========================================================================="
echo " Experiment ${EXP_NAME} finished successfully!"
echo " All results, metrics, and plots saved under:"
echo "   ${RESULTS_DIR}"
echo " Summary CSV  : ${RESULTS_DIR}/experiment_summary.csv"
echo " Summary Plot : ${RESULTS_DIR}/experiment_summary.png"
echo "=========================================================================="
