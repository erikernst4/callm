#!/bin/bash
# ==============================================================================
# Experiment 7: Attentive Sequence Pooling Probes (N=50,000) with Cosine Annealing
# Experiment Name: exp7_attentive_pooling_50k
#
# Re-runs Experiment 3 scaled to N=50,000 samples with a Warmup + Cosine
# Annealing learning rate schedule.
#
# Single-layer probes: sweep available layers in the 50k activation cache:
#   [12, 14, 16, 17, 18, 19, 20, 21, 22, 24]
# Weighted average probe: trained across all 10 available layers with learned
# softmax weights.
#
# Evaluates 6 sequence pooling strategies:
#   1. mean                       (MeanPooling)
#   2. attentive                  (AttentivePooling)
#   3. query_attentive            (QueryAttentivePooling)
#   4. multi_head_attentive       (MultiHeadAttentivePooling)
#   5. query_multi_head_attentive (QueryMultiHeadAttentivePooling)
#   6. transformer                (TransformerPooling)
#
# All pooling strategies are trained on the 'both' scope (prompt + answer)
# with pool_strategy: none to preserve full sequence token representations.
#
# LR schedule: WarmupCosineAnnealingLR
#   - warmup_steps: 100
#   - max_steps: 62500 (40k train samples / batch_size 64 = 625 steps/epoch * 100 epochs)
#   - eta_min: 1e-6
#
# Usage:
#   bash scripts/run_exp7_attentive_pooling_50k.sh [DEVICE_IDX]
# Example:
#   bash scripts/run_exp7_attentive_pooling_50k.sh 0
# ==============================================================================

set -e

# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

DEVICE_IDX="${1:-0}"
DEVICE="[${DEVICE_IDX}]"
DTYPE="${2:-${DTYPE:-bfloat16}}"

# Configuration
EXP_NAME="${EXP_NAME:-exp7_attentive_pooling_50k}"
RESULTS_DIR="results/${EXP_NAME}"
ACTIVATIONS_DIR="activations_cache/qwen_triviaqa/lightning_logs/version_27"
EVALUATION_CSV="activations_cache/qwen_triviaqa_50000/lightning_logs/version_0_evaluation/version_0/evaluation_results.csv"

CONFIG_BASE="configs/config_train_probe_attentive_cosine.yaml"
SCOPE="both"
MODEL_DIM=4096

# 10 Extracted Layers available in version_27 (N=50k)
INDIVIDUAL_LAYERS=(12 14 16 17 18 19 20 21 22 24)
WEIGHTED_LAYERS="[12, 14, 16, 17, 18, 19, 20, 21, 22, 24]"

POOLING_METHODS=(
    "mean"
    "attentive"
    "query_attentive"
    # "multi_head_attentive"
    # "query_multi_head_attentive"
    "transformer"
)

echo "=========================================================================="
echo " Starting Experiment: ${EXP_NAME}"
echo "=========================================================================="
echo " Project Root    : ${PROJECT_ROOT}"
echo " Results Dir     : ${RESULTS_DIR}"
echo " Activations Dir : ${ACTIVATIONS_DIR}"
echo " Evaluation CSV  : ${EVALUATION_CSV}"
echo " GPU Device      : ${DEVICE}"
echo " Base Config     : ${CONFIG_BASE}"
echo " Activation Dtype: ${DTYPE}"
echo " LR Scheduler    : WarmupCosineAnnealingLR (warmup=100, max_steps=62500, eta_min=1e-6)"
echo " Single Layers   : ${INDIVIDUAL_LAYERS[*]}"
echo " Weighted Layers : ${WEIGHTED_LAYERS}"
echo " Pooling Methods : ${POOLING_METHODS[*]}"
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

if [[ ! -f "$CONFIG_BASE" ]]; then
    echo "Error: Base config not found at $CONFIG_BASE"
    exit 1
fi

# Count weighted layers
NUM_WEIGHTED_LAYERS=$(echo "$WEIGHTED_LAYERS" | tr -cd ',' | wc -c)
NUM_WEIGHTED_LAYERS=$((NUM_WEIGHTED_LAYERS + 1))

# Function to get class_path for a pooling method
get_pooling_classpath() {
    case "$1" in
        "mean")                        echo "callm.models.MeanPooling" ;;
        "attentive")                   echo "callm.models.AttentivePooling" ;;
        "query_attentive")             echo "callm.models.QueryAttentivePooling" ;;
        "multi_head_attentive")        echo "callm.models.MultiHeadAttentivePooling" ;;
        "query_multi_head_attentive")  echo "callm.models.QueryMultiHeadAttentivePooling" ;;
        "transformer")                 echo "callm.models.TransformerPooling" ;;
        *)                             echo "" ;;
    esac
}

# Loop over pooling methods
for method in "${POOLING_METHODS[@]}"; do
    CLASSPATH=$(get_pooling_classpath "$method")
    echo ""
    echo "##########################################################################"
    echo " RUNNING SWEEP FOR POOLING METHOD: ${method} (${CLASSPATH})"
    echo "##########################################################################"

    # # 1. Single-layer probe training
    # for layer in "${INDIVIDUAL_LAYERS[@]}"; do
    #     TARGET_DIR="${RESULTS_DIR}/${method}/layer_${layer}"
    #     echo ""
    #     echo "--------------------------------------------------------------------------"
    #     echo " [Method: ${method}] Training Probe for Layer ${layer}"
    #     echo " Target Dir: ${TARGET_DIR}"
    #     echo "--------------------------------------------------------------------------"

    #     # Skip if already trained
    #     LATEST_METRICS=$(ls -vd "${TARGET_DIR}"/lightning_logs/version_*/metrics.csv 2>/dev/null | tail -n 1 || true)
    #     if [[ -n "$LATEST_METRICS" && -f "$LATEST_METRICS" ]]; then
    #         echo "--> Already finished (${LATEST_METRICS}). Skipping."
    #         continue
    #     fi

    #     uv run main.py fit \
    #         --config "$CONFIG_BASE" \
    #         --trainer.default_root_dir "$TARGET_DIR" \
    #         --model.init_args.layer_idx "$layer" \
    #         --model.init_args.model_dim "$MODEL_DIM" \
    #         --model.init_args.seq_pooling.class_path "$CLASSPATH" \
    #         --model.init_args.seq_pooling.init_args.model_dim "$MODEL_DIM" \
    #         --data.init_args.activations_dir "$ACTIVATIONS_DIR" \
    #         --data.init_args.evaluation_csv "$EVALUATION_CSV" \
    #         --data.init_args.layer_idx "$layer" \
    #         --data.init_args.activations_to_use "$SCOPE" \
    #         --data.init_args.pool_strategy "none" \
    #         --data.init_args.dtype "$DTYPE" \
    #         --trainer.devices "$DEVICE"

    #     # Generate plot for this layer probe
    #     LATEST_LOG=$(ls -vd "${TARGET_DIR}"/lightning_logs/version_* 2>/dev/null | tail -n 1 || true)
    #     if [[ -n "$LATEST_LOG" && -f "${LATEST_LOG}/metrics.csv" ]]; then
    #         uv run python3 scripts/plot_probe_training.py \
    #             --metrics_csv "${LATEST_LOG}/metrics.csv" \
    #             --output_dir "${TARGET_DIR}/plots" || true
    #     fi
    # done

    # # 2. Weighted Average Probe training across all available layers
    # WEIGHTED_TARGET_DIR="${RESULTS_DIR}/${method}/weighted"
    # echo ""
    # echo "--------------------------------------------------------------------------"
    # echo " [Method: ${method}] Training Weighted Average Probe across layers ${WEIGHTED_LAYERS}"
    # echo " Target Dir: ${WEIGHTED_TARGET_DIR}"
    # echo "--------------------------------------------------------------------------"

    # # Skip if already trained
    # LATEST_WEIGHTED_METRICS=$(ls -vd "${WEIGHTED_TARGET_DIR}"/lightning_logs/version_*/metrics.csv 2>/dev/null | tail -n 1 || true)
    # if [[ -n "$LATEST_WEIGHTED_METRICS" && -f "$LATEST_WEIGHTED_METRICS" ]]; then
    #     echo "--> Already finished (${LATEST_WEIGHTED_METRICS}). Skipping."
    #     continue
    # fi

    # uv run main.py fit \
    #     --config "$CONFIG_BASE" \
    #     --trainer.default_root_dir "$WEIGHTED_TARGET_DIR" \
    #     --model.init_args.layer_idx "$WEIGHTED_LAYERS" \
    #     --model.init_args.num_layers "$NUM_WEIGHTED_LAYERS" \
    #     --model.init_args.model_dim "$MODEL_DIM" \
    #     --model.init_args.seq_pooling.class_path "$CLASSPATH" \
    #     --model.init_args.seq_pooling.init_args.model_dim "$MODEL_DIM" \
    #     --data.init_args.activations_dir "$ACTIVATIONS_DIR" \
    #     --data.init_args.evaluation_csv "$EVALUATION_CSV" \
    #     --data.init_args.layer_idx "$WEIGHTED_LAYERS" \
    #     --data.init_args.activations_to_use "$SCOPE" \
    #     --data.init_args.pool_strategy "none" \
    #     --data.init_args.dtype "$DTYPE" \
    #     --trainer.devices "$DEVICE"

    # # Generate plot for weighted probe
    # LATEST_WEIGHTED_LOG=$(ls -vd "${WEIGHTED_TARGET_DIR}"/lightning_logs/version_* 2>/dev/null | tail -n 1 || true)
    # if [[ -n "$LATEST_WEIGHTED_LOG" && -f "${LATEST_WEIGHTED_LOG}/metrics.csv" ]]; then
    #     uv run python3 scripts/plot_probe_training.py \
    #         --metrics_csv "${LATEST_WEIGHTED_LOG}/metrics.csv" \
    #         --output_dir "${WEIGHTED_TARGET_DIR}/plots" || true
    # fi
done

echo ""
echo "=========================================================================="
echo " Summarizing Experiment Results for ${EXP_NAME}..."
echo "=========================================================================="

uv run python3 scripts/summarize_experiment_results.py \
    --exp_dir "$RESULTS_DIR" \
    --scopes "${POOLING_METHODS[@]}"

echo ""
echo "=========================================================================="
echo " Experiment ${EXP_NAME} finished successfully!"
echo " All results, metrics, and plots saved under:"
echo "   ${RESULTS_DIR}"
echo " Summary CSV  : ${RESULTS_DIR}/experiment_summary.csv"
echo " Summary Plot : ${RESULTS_DIR}/experiment_summary.png"
echo "=========================================================================="
