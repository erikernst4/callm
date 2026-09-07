#!/bin/bash
# ==============================================================================
# Experiment 6: Layer Weight Initializations & Normalization (N=50,000)
# Experiment Name: exp6_layer_weight_init_50k
#
# Re-runs and scales Experiment 5 to N=50,000 samples using the newly extracted
# activations and GCP correctness evaluations.
#
# Sweeps across:
#   1. Single Layer Baselines:
#      - Layers: [12, 14, 16, 17, 18, 19, 20, 21, 22, 24]
#
#   2. Layer weight initialization strategies (for weighted pooling):
#      - Uniform (zeros baseline, softmax = 1/N)
#      - Normal (Gaussian centered at middle layers)
#      - Reverse Chi (Chi distribution, peak at middle, low at first layers)
#      - Random (Gaussian random raw weights sampled from N(0, 1))
#
#   3. Layer pooling normalization methods:
#      - Baseline (LayerWeightedAverage)
#      - Global Z-Norm (GlobalZNormWeightedAverage)
#      - Layer-wise Z-Norm (LayerwiseZNormWeightedAverage)
#      - LayerNorm (LayerNormWeightedAverage)
#
#   4. Subsets for weighted pooling:
#      - middle_30: [12, 14, 16, 17, 18, 19, 20, 21, 22, 24] (all 10 saved layers)
#      - middle_20: [14, 16, 18, 20, 22, 24] (6 middle layers)
#
# Sequence pooling is fixed to 'mean' on 'both' (prompt + answer).
#
# Usage:
#   bash scripts/run_exp6_layer_weight_init.sh [DEVICE_IDX]
# Example:
#   bash scripts/run_exp6_layer_weight_init.sh 0
# ==============================================================================

set -e

# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Device configuration
DEVICE_IDX="${1:-0}"
DEVICE="[${DEVICE_IDX}]"

# Configuration paths
EXP_NAME="exp6_layer_weight_init_50k"
RESULTS_DIR="results/${EXP_NAME}"
ACTIVATIONS_DIR="activations_cache/qwen_triviaqa/lightning_logs/version_27"
EVALUATION_CSV="activations_cache/qwen_triviaqa_50000/lightning_logs/version_0_evaluation/version_0/evaluation_results.csv"

CONFIG_SINGLE="configs/config_train_probe.yaml"
CONFIG_BASE="configs/config_train_probe_attentive.yaml"
SCOPE="both"
SEQ_POOLING="callm.models.MeanPooling"

# 10 Extracted Layers available in version_27
INDIVIDUAL_LAYERS=(12 14 16 17 18 19 20 21 22 24)

INITS=(
    "uniform"
    "normal"
    "reverse_chi"
    "random"
)

METHODS=(
    "baseline|callm.models.LayerWeightedAverage"
    "global_znorm|callm.models.GlobalZNormWeightedAverage"
    "layerwise_znorm|callm.models.LayerwiseZNormWeightedAverage"
    "layernorm|callm.models.LayerNormWeightedAverage"
)

SUBSETS=(
    "middle_30|[12,14,16,17,18,19,20,21,22,24]"
    "middle_20|[14,16,18,20,22,24]"
)

echo "=========================================================================="
echo " Starting Experiment: ${EXP_NAME}"
echo "=========================================================================="
echo " Project Root    : ${PROJECT_ROOT}"
echo " Results Dir     : ${RESULTS_DIR}"
echo " Activations Dir : ${ACTIVATIONS_DIR}"
echo " Evaluation CSV  : ${EVALUATION_CSV}"
echo " GPU Device      : ${DEVICE}"
echo " Sequence Pooling: mean"
echo " Initializations : ${INITS[*]}"
echo " Subsets         : middle_30, middle_20"
echo " Individual Ls   : ${INDIVIDUAL_LAYERS[*]}"
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

ALL_SCOPES=()

# ==============================================================================
# Part 1: Single Layer Probes (Individual Layer Baselines)
# ==============================================================================
echo ""
echo "##########################################################################"
echo " PART 1: TRAINING INDIVIDUAL LAYER PROBES"
echo "##########################################################################"
ALL_SCOPES+=("single_layer")

for layer in "${INDIVIDUAL_LAYERS[@]}"; do
    TARGET_DIR="${RESULTS_DIR}/single_layer/layer_${layer}"
    echo ""
    echo "--------------------------------------------------------------------------"
    echo " [Single Layer] Training Probe for Layer ${layer}"
    echo " Target Dir: ${TARGET_DIR}"
    echo "--------------------------------------------------------------------------"

    # Skip if already trained
    LATEST_METRICS=$(ls -vd "${TARGET_DIR}"/lightning_logs/version_*/metrics.csv 2>/dev/null | tail -n 1 || true)
    if [[ -n "$LATEST_METRICS" && -f "$LATEST_METRICS" ]]; then
        echo "--> Already finished. Skipping."
        continue
    fi

    uv run main.py fit \
        --config "$CONFIG_SINGLE" \
        --trainer.default_root_dir "$TARGET_DIR" \
        --model.init_args.layer_idx "$layer" \
        --model.init_args.model_dim 4096 \
        --data.init_args.activations_dir "$ACTIVATIONS_DIR" \
        --data.init_args.evaluation_csv "$EVALUATION_CSV" \
        --data.init_args.layer_idx "$layer" \
        --data.init_args.activations_to_use "$SCOPE" \
        --data.init_args.pool_strategy "mean" \
        --trainer.devices "$DEVICE"

    LATEST_LOG=$(ls -vd "${TARGET_DIR}"/lightning_logs/version_* 2>/dev/null | tail -n 1 || true)
    if [[ -n "$LATEST_LOG" && -f "${LATEST_LOG}/metrics.csv" ]]; then
        uv run python3 scripts/plot_probe_training.py \
            --metrics_csv "${LATEST_LOG}/metrics.csv" \
            --output_dir "${TARGET_DIR}/plots" || true
    fi
done

# ==============================================================================
# Part 2: Weighted Layer Pooling Sweeps (Inits x Normalization Methods x Subsets)
# ==============================================================================
echo ""
echo "##########################################################################"
echo " PART 2: WEIGHTED LAYER POOLING SWEEPS"
echo "##########################################################################"

for INIT_NAME in "${INITS[@]}"; do
    for METHOD_CFG in "${METHODS[@]}"; do
        METHOD_NAME="${METHOD_CFG%%|*}"
        METHOD_CLASS="${METHOD_CFG##*|}"

        SCOPE_NAME="${INIT_NAME}_${METHOD_NAME}"
        ALL_SCOPES+=("$SCOPE_NAME")

        echo ""
        echo "=========================================================================="
        echo " Init Strategy : ${INIT_NAME}"
        echo " Pooling Method: ${METHOD_NAME} (${METHOD_CLASS})"
        echo " Combined Scope: ${SCOPE_NAME}"
        echo "=========================================================================="

        for SUBSET_CFG in "${SUBSETS[@]}"; do
            SUBSET_NAME="${SUBSET_CFG%%|*}"
            LAYER_IDX_VAL="${SUBSET_CFG##*|}"

            TARGET_DIR="${RESULTS_DIR}/${SCOPE_NAME}/layer_${SUBSET_NAME}"
            echo ""
            echo ">>> Running subset: ${SUBSET_NAME} (Layers: ${LAYER_IDX_VAL})"
            echo " Target Dir: ${TARGET_DIR}"

            # Skip if already trained
            LATEST_METRICS=$(ls -vd "${TARGET_DIR}"/lightning_logs/version_*/metrics.csv 2>/dev/null | tail -n 1 || true)
            if [[ -n "$LATEST_METRICS" && -f "$LATEST_METRICS" ]]; then
                echo "--> Already finished. Skipping."
                continue
            fi

            NL=$(echo "$LAYER_IDX_VAL" | tr -cd ',' | wc -c)
            NL=$((NL + 1))

            if [[ "$METHOD_NAME" == "baseline" ]]; then
                POOLING_ARGS="--model.init_args.layer_pooling.init_args.num_layers ${NL} --model.init_args.layer_pooling.init_args.init_strategy ${INIT_NAME}"
            else
                POOLING_ARGS="--model.init_args.layer_pooling.init_args.num_layers ${NL} --model.init_args.layer_pooling.init_args.model_dim 4096 --model.init_args.layer_pooling.init_args.init_strategy ${INIT_NAME}"
            fi

            uv run main.py fit -c "$CONFIG_BASE" \
                --trainer.default_root_dir "$TARGET_DIR" \
                --model.init_args.layer_idx "${LAYER_IDX_VAL}" \
                --model.init_args.seq_pooling.class_path "$SEQ_POOLING" \
                --model.init_args.layer_pooling.class_path "${METHOD_CLASS}" \
                ${POOLING_ARGS} \
                --data.init_args.activations_dir "$ACTIVATIONS_DIR" \
                --data.init_args.evaluation_csv "$EVALUATION_CSV" \
                --data.init_args.layer_idx "${LAYER_IDX_VAL}" \
                --data.init_args.activations_to_use "$SCOPE" \
                --data.init_args.pool_strategy "mean" \
                --trainer.devices "$DEVICE"

            LATEST_LOG=$(ls -vd "${TARGET_DIR}"/lightning_logs/version_* 2>/dev/null | tail -n 1 || true)
            if [[ -n "$LATEST_LOG" && -f "${LATEST_LOG}/metrics.csv" ]]; then
                uv run python3 scripts/plot_probe_training.py \
                    --metrics_csv "${LATEST_LOG}/metrics.csv" \
                    --output_dir "${TARGET_DIR}/plots" || true
            fi
        done
    done
done

# ==============================================================================
# Part 3: Summarize Experiment Results
# ==============================================================================
echo ""
echo "=========================================================================="
echo " Summarizing Experiment Results for ${EXP_NAME}..."
echo "=========================================================================="

uv run python3 scripts/summarize_experiment_results.py --exp_dir "$RESULTS_DIR" --scopes "${ALL_SCOPES[@]}"

echo ""
echo "=========================================================================="
echo " Experiment ${EXP_NAME} finished successfully!"
echo " All results, metrics, and plots saved under: ${RESULTS_DIR}"
echo "=========================================================================="
