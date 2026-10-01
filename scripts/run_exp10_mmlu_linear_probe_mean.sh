#!/bin/bash
# ==============================================================================
# Experiment 10: MMLU Linear Probes (Sequence Mean-Pooling)
# Experiment Name: exp10_mmlu_linear_probe_seq-mean
#
# Trains one linear probe per extracted layer for each scope:
#   1. prompt (mean over prompt tokens)
#   2. answer (mean over generated answer tokens)
#   3. both   (mean over prompt + answer tokens)
#
# Train: MMLU auxiliary_train (44,300 samples extracted before the run stopped)
# Val:   MMLU validation split (1,531 samples)
#
# Requires the pooled activations from scripts/prepare_mmlu_probe_data.sh
# (run automatically if missing).
#
# Outputs:
#   results/exp10_mmlu_linear_probe_seq-mean/<scope>/layer_<N>
#
# Usage:
#   bash scripts/run_exp10_mmlu_linear_probe_mean.sh [DEVICE_IDX]
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

DEVICE_IDX="${1:-0}"
DEVICE="[${DEVICE_IDX}]"

EXP_NAME="${EXP_NAME:-exp10_mmlu_linear_probe_seq-mean}"
RESULTS_DIR="results/${EXP_NAME}"
CONFIG="configs/config_train_probe_mmlu.yaml"

TRAIN_POOLED_DIR="activations_cache/mmlu_auxiliary_train_50k_pooled_mean"
TRAIN_EVAL_CSV="activations_cache/mmlu_auxiliary_train_50k/evaluation_results.csv"
VAL_POOLED_DIR="activations_cache/mmlu_val_pooled_mean"
VAL_EVAL_CSV="activations_cache/mmlu_val/lightning_logs/version_2/evaluation_results.csv"

MODEL_DIM=4096
LAYERS=(12 14 16 17 18 19 20 21 22 24)
SCOPES=("prompt" "answer" "both")

echo "=========================================================================="
echo " Starting Experiment: ${EXP_NAME}"
echo "=========================================================================="
echo " Results Dir   : ${RESULTS_DIR}"
echo " Train Pooled  : ${TRAIN_POOLED_DIR}"
echo " Train Labels  : ${TRAIN_EVAL_CSV}"
echo " Val Pooled    : ${VAL_POOLED_DIR}"
echo " Val Labels    : ${VAL_EVAL_CSV}"
echo " GPU Device    : ${DEVICE}"
echo " Layers        : ${LAYERS[*]}"
echo " Scopes        : ${SCOPES[*]}"
echo "=========================================================================="

if [[ ! -f "${TRAIN_POOLED_DIR}/both/metadata.json" || ! -f "${VAL_POOLED_DIR}/both/metadata.json" || ! -f "$TRAIN_EVAL_CSV" ]]; then
    echo "Pooled activations or labels missing. Running prepare_mmlu_probe_data.sh..."
    bash scripts/prepare_mmlu_probe_data.sh
fi

if [[ ! -f "$VAL_EVAL_CSV" ]]; then
    echo "Error: Val evaluation CSV not found at $VAL_EVAL_CSV"
    exit 1
fi

for scope in "${SCOPES[@]}"; do
    echo ""
    echo "##########################################################################"
    echo " RUNNING SWEEP FOR SCOPE: ${scope^^}"
    echo "##########################################################################"

    for layer in "${LAYERS[@]}"; do
        TARGET_DIR="${RESULTS_DIR}/${scope}/layer_${layer}"
        echo ""
        echo "--------------------------------------------------------------------------"
        echo " [Scope: ${scope}] Training Probe for Layer ${layer}"
        echo " Target Dir: ${TARGET_DIR}"
        echo "--------------------------------------------------------------------------"

        LATEST_METRICS=$(ls -vd "${TARGET_DIR}"/lightning_logs/version_*/metrics.csv 2>/dev/null | tail -n 1 || true)
        if [[ -n "$LATEST_METRICS" && -f "$LATEST_METRICS" ]]; then
            echo "--> Already finished (${LATEST_METRICS}). Skipping."
            continue
        fi

        # Pooled shards hold one vector per sample: the scope is selected by
        # directory, and activations_to_use=both returns that vector as-is.
        uv run main.py fit \
            --config "$CONFIG" \
            --trainer.default_root_dir "$TARGET_DIR" \
            --trainer.devices "$DEVICE" \
            --model.init_args.layer_idx "$layer" \
            --model.init_args.model_dim "$MODEL_DIM" \
            --data.init_args.activations_dir "${TRAIN_POOLED_DIR}/${scope}" \
            --data.init_args.evaluation_csv "$TRAIN_EVAL_CSV" \
            --data.init_args.val_activations_dir "${VAL_POOLED_DIR}/${scope}" \
            --data.init_args.val_evaluation_csv "$VAL_EVAL_CSV" \
            --data.init_args.layer_idx "$layer" \
            --data.init_args.activations_to_use "both" \
            --data.init_args.pool_strategy "mean"

        LATEST_LOG=$(ls -vd "${TARGET_DIR}"/lightning_logs/version_* 2>/dev/null | tail -n 1 || true)
        if [[ -n "$LATEST_LOG" && -f "${LATEST_LOG}/metrics.csv" ]]; then
            uv run python3 scripts/plot_probe_training.py \
                --metrics_csv "${LATEST_LOG}/metrics.csv" \
                --output_dir "${TARGET_DIR}/plots" || true
        fi
    done
done

echo ""
echo "=========================================================================="
echo " Summarizing Experiment Results for ${EXP_NAME}..."
echo "=========================================================================="

uv run python3 scripts/summarize_experiment_results.py \
    --exp_dir "$RESULTS_DIR" \
    --scopes "${SCOPES[@]}"

echo ""
echo "=========================================================================="
echo " Experiment ${EXP_NAME} finished successfully!"
echo " Summary CSV  : ${RESULTS_DIR}/experiment_summary.csv"
echo " Summary Plot : ${RESULTS_DIR}/experiment_summary.png"
echo "=========================================================================="
