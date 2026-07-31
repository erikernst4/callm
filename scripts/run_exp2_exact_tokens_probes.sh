#!/bin/bash
# ==============================================================================
# Experiment 2: Exact-Answer Token Windows with Linear Probes
# Experiment Name: exp2_exact_tokens_linear_probe
#
# Replicates the LLMsKnow probing approach using exact answer token positions.
# Trains per-layer probes (0..32) and weighted average probes across 5 windows:
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
#   - Activations from version_6 (Qwen/Qwen3.5-9B, 5000 TriviaQA samples)
#   - Exact answer extraction from version_19 (evaluation_results_exact.csv)
#
# Outputs are organized under:
#   results/exp2_exact_tokens_linear_probe/<window>/layer_<N>
#   results/exp2_exact_tokens_linear_probe/<window>/weighted
# ==============================================================================

set -e

# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
EXP_NAME="${EXP_NAME:-exp2_exact_tokens_linear_probe}"
RESULTS_DIR="results/${EXP_NAME}"
ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-/home/eernst/doctorado/callm-dev/activations_cache/qwen_triviaqa/lightning_logs/version_6}"
EVALUATION_CSV="${1:-${EVALUATION_CSV:-/home/eernst/doctorado/callm-dev/activations_cache/qwen_triviaqa/lightning_logs/version_19/evaluation_results_exact.csv}}"

CONFIG_SINGLE="configs/config_train_probe.yaml"
CONFIG_WEIGHTED="configs/config_train_probe_weighted.yaml"

# Model and prompt used during activation extraction (needed to reconstruct token indices)
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"
PROMPT="${PROMPT:-callm.prompts.base.TRANSPARENT_PROMPT_ZERO_SHOT}"

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
echo " Evaluation CSV   : ${EVALUATION_CSV}"
echo " Model Name       : ${MODEL_NAME}"
echo " Prompt           : ${PROMPT}"
echo " Token Windows    : ${WINDOWS[*]}"
echo "=========================================================================="

# Check prerequisites
if [[ ! -d "$ACTIVATIONS_DIR" ]]; then
    echo "Error: Activations directory not found at $ACTIVATIONS_DIR"
    exit 1
fi

if [[ ! -f "$EVALUATION_CSV" ]]; then
    echo "Error: Evaluation CSV not found at $EVALUATION_CSV"
    exit 1
fi

# Detect number of layers in saved activations
NUM_LAYERS=$(uv run python3 -c "
import glob, torch
files = glob.glob('${ACTIVATIONS_DIR}/activations_chunk_*.pt')
if files:
    d = torch.load(files[0], map_location='cpu', weights_only=False)
    print(len(d[0]['activations']))
else:
    print('0')
")

if [[ "$NUM_LAYERS" -eq 0 ]]; then
    echo "Error: Could not detect activation layers in ${ACTIVATIONS_DIR}"
    exit 1
fi

echo "Detected ${NUM_LAYERS} layer hidden states (0 to $((NUM_LAYERS-1)))"
echo "=========================================================================="

# Loop over token windows
for window in "${WINDOWS[@]}"; do
    echo ""
    echo "##########################################################################"
    echo " RUNNING SWEEP FOR WINDOW: ${window}"
    echo "##########################################################################"

    MAX_LAYER=$((NUM_LAYERS - 1))

    # 1. Per-layer probe training (layer 0 to NUM_LAYERS-1)
    for ((layer=0; layer<NUM_LAYERS; layer++)); do
        TARGET_DIR="${RESULTS_DIR}/${window}/layer_${layer}"
        echo "--------------------------------------------------------------------------"
        echo " [Window: ${window}] Training Probe for Layer ${layer}/${MAX_LAYER}"
        echo " Target Dir: ${TARGET_DIR}"
        echo "--------------------------------------------------------------------------"

        uv run main.py fit \
            --config "$CONFIG_SINGLE" \
            --trainer.default_root_dir "$TARGET_DIR" \
            --model.init_args.layer_idx "$layer" \
            --model.init_args.model_dim 4096 \
            --data.class_path "callm.data.SelectedTokensActivationsDataModule" \
            --data.init_args.activations_dir "$ACTIVATIONS_DIR" \
            --data.init_args.evaluation_csv "$EVALUATION_CSV" \
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
                --output_dir "${TARGET_DIR}/plots"
        fi
    done

    # 2. Weighted Average Probe training across all layers
    WEIGHTED_TARGET_DIR="${RESULTS_DIR}/${window}/weighted"
    echo "--------------------------------------------------------------------------"
    echo " [Window: ${window}] Training Weighted Average Probe across all ${NUM_LAYERS} layers"
    echo " Target Dir: ${WEIGHTED_TARGET_DIR}"
    echo "--------------------------------------------------------------------------"

    uv run main.py fit \
        --config "$CONFIG_WEIGHTED" \
        --trainer.default_root_dir "$WEIGHTED_TARGET_DIR" \
        --model.init_args.layer_idx "all" \
        --model.init_args.num_layers "$NUM_LAYERS" \
        --model.init_args.model_dim 4096 \
        --data.class_path "callm.data.SelectedTokensActivationsDataModule" \
        --data.init_args.activations_dir "$ACTIVATIONS_DIR" \
        --data.init_args.evaluation_csv "$EVALUATION_CSV" \
        --data.init_args.layer_idx "all" \
        --data.init_args.activations_to_use "$window" \
        --data.init_args.pool_strategy "mean" \
        --data.init_args.model_name "$MODEL_NAME" \
        --data.init_args.prompt "$PROMPT"

    # Generate plot for weighted probe
    LATEST_WEIGHTED_LOG=$(ls -vd "${WEIGHTED_TARGET_DIR}"/lightning_logs/version_* 2>/dev/null | tail -n 1 || true)
    if [[ -n "$LATEST_WEIGHTED_LOG" && -f "${LATEST_WEIGHTED_LOG}/metrics.csv" ]]; then
        uv run python3 scripts/plot_probe_training.py \
            --metrics_csv "${LATEST_WEIGHTED_LOG}/metrics.csv" \
            --output_dir "${WEIGHTED_TARGET_DIR}/plots"
    fi
done

echo ""
echo "=========================================================================="
echo " Summarizing Experiment Results for ${EXP_NAME}..."
echo "=========================================================================="

uv run python3 scripts/summarize_experiment_results.py --exp_dir "$RESULTS_DIR"

echo ""
echo "=========================================================================="
echo " Experiment ${EXP_NAME} finished successfully!"
echo " All results, metrics, and plots saved under:"
echo "   ${RESULTS_DIR}"
echo " Summary CSV  : ${RESULTS_DIR}/experiment_summary.csv"
echo " Summary Plot : ${RESULTS_DIR}/experiment_summary.png"
echo "=========================================================================="
