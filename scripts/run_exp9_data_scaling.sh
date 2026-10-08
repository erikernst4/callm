#!/bin/bash
# ==============================================================================
# Experiment 9: Data Scaling — How Training Set Size Affects Probe Overfitting
# Experiment Name: exp9_data_scaling
#
# Investigates how training set size N affects overfitting in attentive pooling
# probes (layer 17, AttentivePooling) by sweeping N while keeping a fixed
# validation set of 10,000 samples.
#
# Sweep: N ∈ {100, 1000, 2000, 5000, 10000, 20000, 40000}
#
# Key design choices:
#   - Fixed validation set (val_size=10000) ensures fair comparison across N
#   - val_check_interval=100 monitors validation every 100 training steps
#   - max_epochs=1000 with patience=20 early stopping
#   - LR schedule max_steps auto-computed from trainer.estimated_stepping_batches
#
# Usage:
#   bash scripts/run_exp9_data_scaling.sh [DEVICE_IDX]
# Example:
#   bash scripts/run_exp9_data_scaling.sh 0
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
EXP_NAME="${EXP_NAME:-exp9_data_scaling_transformer}"
RESULTS_DIR="results/${EXP_NAME}"
CONFIG_BASE="configs/config_train_probe_data_scaling_transformer.yaml"

# Training set sizes to sweep
TRAIN_SIZES=(100 1000 2000 5000 10000 20000 40000)

echo "=========================================================================="
echo " Starting Experiment: ${EXP_NAME}"
echo "=========================================================================="
echo " Project Root    : ${PROJECT_ROOT}"
echo " Results Dir     : ${RESULTS_DIR}"
echo " GPU Device      : ${DEVICE}"
echo " Base Config     : ${CONFIG_BASE}"
echo " Activation Dtype: ${DTYPE}"
echo " Training Sizes  : ${TRAIN_SIZES[*]}"
echo " Validation Size : 10000 (fixed)"
echo " Layer           : 17"
echo " Pooling         : TransformerPooling"
echo "=========================================================================="

# Check prerequisites
if [[ ! -f "$CONFIG_BASE" ]]; then
    echo "Error: Base config not found at $CONFIG_BASE"
    exit 1
fi

# Loop over training sizes
for N in "${TRAIN_SIZES[@]}"; do
    TARGET_DIR="${RESULTS_DIR}/n_${N}"
    echo ""
    echo "--------------------------------------------------------------------------"
    echo " Training Probe with N=${N} training samples"
    echo " Target Dir: ${TARGET_DIR}"
    echo "--------------------------------------------------------------------------"

    # Skip if already trained
    LATEST_METRICS=$(ls -vd "${TARGET_DIR}"/lightning_logs/version_*/metrics.csv 2>/dev/null | tail -n 1 || true)
    if [[ -n "$LATEST_METRICS" && -f "$LATEST_METRICS" ]]; then
        echo "--> Already finished (${LATEST_METRICS}). Skipping."
        continue
    fi

    uv run main.py fit \
        --config "$CONFIG_BASE" \
        --trainer.default_root_dir "$TARGET_DIR" \
        --data.init_args.max_samples "$N" \
        --data.init_args.dtype "$DTYPE" \
        --trainer.devices "$DEVICE"

    # Generate plot for this run
    LATEST_LOG=$(ls -vd "${TARGET_DIR}"/lightning_logs/version_* 2>/dev/null | tail -n 1 || true)
    if [[ -n "$LATEST_LOG" && -f "${LATEST_LOG}/metrics.csv" ]]; then
        uv run python3 scripts/plot_probe_training.py \
            --metrics_csv "${LATEST_LOG}/metrics.csv" \
            --output_dir "${TARGET_DIR}/plots" || true
    fi
done

echo ""
echo "=========================================================================="
echo " Summarizing Experiment Results for ${EXP_NAME}..."
echo "=========================================================================="

uv run python3 scripts/summarize_data_scaling.py \
    --exp_dir "$RESULTS_DIR"

echo ""
echo "=========================================================================="
echo " Experiment ${EXP_NAME} finished successfully!"
echo " All results saved under: ${RESULTS_DIR}"
echo " Summary CSV  : ${RESULTS_DIR}/experiment_summary.csv"
echo " Summary Plot : ${RESULTS_DIR}/experiment_summary.png"
echo "=========================================================================="
