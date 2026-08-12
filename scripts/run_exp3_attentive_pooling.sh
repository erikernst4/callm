#!/bin/bash
# ==============================================================================
# Experiment 3: Attentive Sequence Pooling Probes
# Experiment Name: exp3_attentive_pooling
#
# Single-layer probes: sweep layers 14 to 23 (since individual layers outside
# this range are not as effective for the task).
# Weighted average probe: trained across ALL layers (0..32) with learned weights.
#
# Evaluates 6 pooling strategies:
#   1. mean                       (MeanPooling)
#   2. attentive                  (AttentivePooling)
#   3. query_attentive            (QueryAttentivePooling)
#   4. multi_head_attentive        (MultiHeadAttentivePooling)
#   5. query_multi_head_attentive  (QueryMultiHeadAttentivePooling)
#   6. transformer                (TransformerPooling)
#
# All pooling strategies are trained on the 'both' scope (prompt + answer).
# LR schedule: warmup (100 steps) + exponential decay (gamma=0.98 per step).
#
# Outputs are organized under:
#   results/exp3_attentive_pooling/<pooling_method>/layer_<N>
#   results/exp3_attentive_pooling/<pooling_method>/weighted
# ==============================================================================

set -e

# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
EXP_NAME="exp3_attentive_pooling"
RESULTS_DIR="results/${EXP_NAME}"
ACTIVATIONS_DIR="/home/eernst/doctorado/callm-dev/activations_cache/qwen_triviaqa/lightning_logs/version_6"
EVALUATION_CSV="/home/eernst/doctorado/callm-dev/activations_cache/qwen_triviaqa/lightning_logs/version_5_evaluation/version_9/evaluation_results.csv"

CONFIG_BASE="configs/config_train_probe_attentive.yaml"
DEVICE="[0]"
START_LAYER=14
END_LAYER=23
SCOPE="both"

POOLING_METHODS=(
    "mean"
    "attentive"
    "query_attentive"
    "multi_head_attentive"
    "query_multi_head_attentive"
    "transformer"
)

echo "=========================================================================="
echo " Starting Experiment: ${EXP_NAME}"
echo "=========================================================================="
echo " Project Root    : ${PROJECT_ROOT}"
echo " Results Dir     : ${RESULTS_DIR}"
echo " Activations Dir : ${ACTIVATIONS_DIR}"
echo " Evaluation CSV  : ${EVALUATION_CSV}"
echo " Single Layers   : ${START_LAYER} to ${END_LAYER}"
echo " Weighted Layers : All layers"
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

    # 1. Single-layer probe training for layers 14 to 23
    for ((layer=START_LAYER; layer<=END_LAYER; layer++)); do
        TARGET_DIR="${RESULTS_DIR}/${method}/layer_${layer}"
        echo "--------------------------------------------------------------------------"
        echo " [Method: ${method}] Training Probe for Layer ${layer}"
        echo " Target Dir: ${TARGET_DIR}"
        echo "--------------------------------------------------------------------------"

        uv run main.py fit \
            --config "$CONFIG_BASE" \
            --trainer.default_root_dir "$TARGET_DIR" \
            --model.init_args.layer_idx "$layer" \
            --model.init_args.model_dim 4096 \
            --model.init_args.seq_pooling.class_path "$CLASSPATH" \
            --model.init_args.seq_pooling.init_args.model_dim 4096 \
            --data.init_args.activations_dir "$ACTIVATIONS_DIR" \
            --data.init_args.evaluation_csv "$EVALUATION_CSV" \
            --data.init_args.layer_idx "$layer" \
            --data.init_args.activations_to_use "$SCOPE" \
            --data.init_args.pool_strategy "none" \
            --trainer.devices "$DEVICE"

        # Generate plot for this layer probe
        LATEST_LOG=$(ls -vd "${TARGET_DIR}"/lightning_logs/version_* 2>/dev/null | tail -n 1 || true)
        if [[ -n "$LATEST_LOG" && -f "${LATEST_LOG}/metrics.csv" ]]; then
            uv run python3 scripts/plot_probe_training.py \
                --metrics_csv "${LATEST_LOG}/metrics.csv" \
                --output_dir "${TARGET_DIR}/plots"
        fi
    done

    # 2. Weighted Average Probe training across layers 20%
    WEIGHTED_TARGET_DIR="${RESULTS_DIR}/${method}/weighted"
    echo "--------------------------------------------------------------------------"
    echo " [Method: ${method}] Training Weighted Average Probe across layers 20%"
    echo " Target Dir: ${WEIGHTED_TARGET_DIR}"
    echo "--------------------------------------------------------------------------"

    uv run main.py fit \
        --config "$CONFIG_BASE" \
        --trainer.default_root_dir "$WEIGHTED_TARGET_DIR" \
        --model.init_args.layer_idx "[6, 11, 16, 21, 26, 31]" \
        --model.init_args.num_layers "$NUM_LAYERS" \
        --model.init_args.model_dim 4096 \
        --model.init_args.seq_pooling.class_path "$CLASSPATH" \
        --model.init_args.seq_pooling.init_args.model_dim 4096 \
        --data.init_args.activations_dir "$ACTIVATIONS_DIR" \
        --data.init_args.evaluation_csv "$EVALUATION_CSV" \
        --data.init_args.layer_idx "[6, 11, 16, 21, 26, 31]" \
        --data.init_args.activations_to_use "$SCOPE" \
        --data.init_args.pool_strategy "none" \
        --trainer.devices "$DEVICE"

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
