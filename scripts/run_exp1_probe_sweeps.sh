#!/bin/bash
# ==============================================================================
# Experiment 1: Linear Probe Sweeps (Input Sequence Mean-Pooling)
# Experiment Name: exp1_linear_probe_input_seq-mean
#
# Trains per-layer probes (0..32) and weighted average probes across 3 scopes:
#   1. prompt (activations_to_use=prompt)
#   2. answer (activations_to_use=answer)
#   3. both   (activations_to_use=both, prompt+answer)
#
# Uses new activations (version_6) with old evaluation results (version_5_evaluation/version_9).
# Outputs are organized under clean directory structure:
#   results/exp1_linear_probe_input_seq-mean/<scope>/layer_<N>
#   results/exp1_linear_probe_input_seq-mean/<scope>/weighted
# ==============================================================================

set -e

# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
EXP_NAME="exp1_linear_probe_input_seq-mean"
RESULTS_DIR="results/${EXP_NAME}"
ACTIVATIONS_DIR="/home/eernst/doctorado/callm-dev/activations_cache/qwen_triviaqa/lightning_logs/version_6"
EVALUATION_CSV="/home/eernst/doctorado/callm-dev/activations_cache/qwen_triviaqa/lightning_logs/version_5_evaluation/version_9/evaluation_results.csv"

CONFIG_SINGLE="configs/config_train_probe.yaml"
CONFIG_WEIGHTED="configs/config_train_probe_weighted.yaml"
SCOPES=("prompt" "answer" "both")
DEVICE="[0]"
echo "=========================================================================="
echo " Starting Experiment: ${EXP_NAME}"
echo "=========================================================================="
echo " Project Root    : ${PROJECT_ROOT}"
echo " Results Dir     : ${RESULTS_DIR}"
echo " Activations Dir : ${ACTIVATIONS_DIR}"
echo " Evaluation CSV  : ${EVALUATION_CSV}"
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
echo "Target Scopes to train: ${SCOPES[*]}"
echo "=========================================================================="

# Loop over scopes (prompt, answer, both)
for scope in "${SCOPES[@]}"; do
    echo ""
    echo "##########################################################################"
    echo " RUNNING SWEEP FOR SCOPE: ${scope^^}"
    echo "##########################################################################"

    MAX_LAYER=$((NUM_LAYERS - 1))

    # 1. Per-layer probe training (layer 0 to NUM_LAYERS-1)
    for ((layer=0; layer<NUM_LAYERS; layer++)); do
        TARGET_DIR="${RESULTS_DIR}/${scope}/layer_${layer}"
        echo "--------------------------------------------------------------------------"
        echo " [Scope: ${scope}] Training Probe for Layer ${layer}/${MAX_LAYER}"
        echo " Target Dir: ${TARGET_DIR}"
        echo "--------------------------------------------------------------------------"

        uv run main.py fit \
            --config "$CONFIG_SINGLE" \
            --trainer.default_root_dir "$TARGET_DIR" \
            --model.init_args.layer_idx "$layer" \
            --model.init_args.model_dim 4096 \
            --data.init_args.activations_dir "$ACTIVATIONS_DIR" \
            --data.init_args.evaluation_csv "$EVALUATION_CSV" \
            --data.init_args.layer_idx "$layer" \
            --data.init_args.activations_to_use "$scope" \
            --data.init_args.pool_strategy "mean" \
            --trainer.devices "$DEVICE"

        # Generate plot for this layer probe
        LATEST_LOG=$(ls -vd "${TARGET_DIR}"/lightning_logs/version_* 2>/dev/null | tail -n 1 || true)
        if [[ -n "$LATEST_LOG" && -f "${LATEST_LOG}/metrics.csv" ]]; then
            uv run python3 scripts/plot_probe_training.py \
                --metrics_csv "${LATEST_LOG}/metrics.csv" \
                --output_dir "${TARGET_DIR}/plots"
        fi
    done

    # 2. Weighted Average Probe training across all layers
    WEIGHTED_TARGET_DIR="${RESULTS_DIR}/${scope}/weighted"
    echo "--------------------------------------------------------------------------"
    echo " [Scope: ${scope}] Training Weighted Average Probe across all ${NUM_LAYERS} layers"
    echo " Target Dir: ${WEIGHTED_TARGET_DIR}"
    echo "--------------------------------------------------------------------------"

    uv run main.py fit \
        --config "$CONFIG_WEIGHTED" \
        --trainer.default_root_dir "$WEIGHTED_TARGET_DIR" \
        --model.init_args.layer_idx "all" \
        --model.init_args.num_layers "$NUM_LAYERS" \
        --model.init_args.model_dim 4096 \
        --data.init_args.activations_dir "$ACTIVATIONS_DIR" \
        --data.init_args.evaluation_csv "$EVALUATION_CSV" \
        --data.init_args.layer_idx "all" \
        --data.init_args.activations_to_use "$scope" \
        --data.init_args.pool_strategy "mean" \
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
