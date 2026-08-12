#!/bin/bash
# ==============================================================================
# Experiment 5: Layer Weight Initializations
# Experiment Name: exp5_layer_weight_init
#
# Sweeps across:
#   1. Layer weight initialization strategies:
#      - Uniform (zeros baseline, softmax = 1/N)
#      - Normal (Gaussian centered at middle layers)
#      - Reverse Chi (Chi distribution, peak at middle, low at first layers)
#      - Random (Gaussian random raw weights sampled from N(0, 1))
#
#   2. Layer pooling methods:
#      - Baseline (LayerWeightedAverage)
#      - Global Z-Norm (GlobalZNormWeightedAverage)
#      - Layer-wise Z-Norm (LayerwiseZNormWeightedAverage)
#      - LayerNorm (LayerNormWeightedAverage)
#
#   3. Layer subsets (densities + uniform/middle sampling):
#      - all
#      - uniform_20, middle_20
#      - uniform_30, middle_30
#      - uniform_50, middle_50
#      - uniform_70, middle_70
#
# Sequence pooling is fixed to 'mean' to isolate layer pooling effect.
# Outputs are organized under:
#   results/exp5_layer_weight_init/<init>_<method>/layer_<subset>
# ==============================================================================

set -e

# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
EXP_NAME="exp5_layer_weight_init"
RESULTS_DIR="results/${EXP_NAME}"
ACTIVATIONS_DIR="/home/eernst/doctorado/callm-dev/activations_cache/qwen_triviaqa/lightning_logs/version_6"
EVALUATION_CSV="/home/eernst/doctorado/callm-dev/activations_cache/qwen_triviaqa/lightning_logs/version_5_evaluation/version_9/evaluation_results.csv"

CONFIG_BASE="configs/config_train_probe_attentive.yaml"
DEVICE="[0]"
SCOPE="both"
SEQ_POOLING="callm.models.MeanPooling"

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
    "all|all"
    "uniform_20|[4,9,14,19,24,29]"
    "middle_20|[14,16,18,20,22,24]"
    "uniform_30|[3,6,9,12,16,19,22,25,28,31]"
    "middle_30|[12,14,16,17,18,19,20,21,22,24]"
    "uniform_50|[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]"
    "middle_50|[10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28]"
    "uniform_70|[1,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28,30,31]"
    "middle_70|[6,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]"
)

echo "=========================================================================="
echo " Starting Experiment: ${EXP_NAME}"
echo "=========================================================================="
echo " Results Dir     : ${RESULTS_DIR}"
echo " Activations Dir : ${ACTIVATIONS_DIR}"
echo " Evaluation CSV  : ${EVALUATION_CSV}"
echo " Sequence Pooling: mean"
echo " Initializations : ${INITS[*]}"
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

echo "Detected ${NUM_LAYERS} layer hidden states."
echo "=========================================================================="

ALL_SCOPES=()

# Sweep across initializations, methods, and subsets
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

            echo ""
            echo ">>> Running subset: ${SUBSET_NAME} (Layers: ${LAYER_IDX_VAL})"

            TARGET_DIR="${RESULTS_DIR}/${SCOPE_NAME}/layer_${SUBSET_NAME}"

            if [[ "$LAYER_IDX_VAL" == "all" ]]; then
                NL=$NUM_LAYERS
            else
                NL=$(echo "$LAYER_IDX_VAL" | tr -cd ',' | wc -c)
                NL=$((NL + 1))
            fi

            # Pass kwargs to layer pooling
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

            # Plot weights evolution if the metrics CSV exists
            LATEST_LOG=$(ls -vd "${TARGET_DIR}"/lightning_logs/version_* 2>/dev/null | tail -n 1 || true)
            if [[ -n "$LATEST_LOG" && -f "${LATEST_LOG}/metrics.csv" ]]; then
                uv run python3 scripts/plot_probe_training.py \
                    --metrics_csv "${LATEST_LOG}/metrics.csv" \
                    --output_dir "${TARGET_DIR}/plots"
            fi
        done
    done
done

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
