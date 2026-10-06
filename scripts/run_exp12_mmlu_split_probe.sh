#!/bin/bash
# ==============================================================================
# Experiment 12: MMLU Split Prompt/Answer Probes
# Experiment Name: exp12_mmlu_split_probe
#
# exp10's 'both' scope averages prompt and answer tokens together, so the
# (much longer) prompt dominates. Here the prompt and answer are pooled
# separately (SplitConfidenceAugmentedLLM, with its own pooling weights for each
# part), then merged:
#   - concat:      linear layer over [h_prompt; h_answer]
#   - mean:        linear layer over (h_prompt + h_answer) / 2
#   - mean_logits: one linear layer per part, averaging their logits
#
# Sequence pooling per part:
#   1. mean      (pre-pooled mean vectors, as in exp10)
#   2. attentive (AttentivePooling over the raw tokens, as in exp11)
#
# One probe per layer of the middle_30 subset: [12, 14, 16, 17, 18, 19, 20, 21, 22, 24]
#
# Train: MMLU auxiliary_train (44,300 samples)
# Val:   MMLU validation split (1,531 samples)
#
# The mean runs need <pooled_dir>/split, built from the exp10 pooled prompt and
# answer scopes by scripts/combine_pooled_scopes.py (run automatically if missing).
# The attentive runs stream the raw chunks lazily (~148GB per epoch, like exp11),
# so all mean runs are done first.
#
# Outputs:
#   results/exp12_mmlu_split_probe/<pooling>_<merge>/layer_<N>
#
# Usage:
#   bash scripts/run_exp12_mmlu_split_probe.sh [DEVICE_IDX]
#   POOLINGS=mean MERGES="concat mean" bash scripts/run_exp12_mmlu_split_probe.sh 0
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

DEVICE_IDX="${1:-0}"
DEVICE="[${DEVICE_IDX}]"

EXP_NAME="${EXP_NAME:-exp12_mmlu_split_probe}"
RESULTS_DIR="results/${EXP_NAME}"

TRAIN_POOLED_DIR="activations_cache/mmlu_auxiliary_train_50k_pooled_mean"
VAL_POOLED_DIR="activations_cache/mmlu_val_pooled_mean"
TRAIN_ACTIVATIONS_DIR="activations_cache/mmlu_auxiliary_train_50k/lightning_logs/version_1"
VAL_ACTIVATIONS_DIR="activations_cache/mmlu_val"
TRAIN_EVAL_CSV="activations_cache/mmlu_auxiliary_train_50k/evaluation_results.csv"
VAL_EVAL_CSV="activations_cache/mmlu_val/lightning_logs/version_2/evaluation_results.csv"

MODEL_DIM=4096
LAYERS=(${LAYERS:-12 14 16 17 18 19 20 21 22 24})
POOLINGS=(${POOLINGS:-mean attentive})
MERGES=(${MERGES:-concat mean mean_logits})

echo "=========================================================================="
echo " Starting Experiment: ${EXP_NAME}"
echo "=========================================================================="
echo " Results Dir   : ${RESULTS_DIR}"
echo " Train Pooled  : ${TRAIN_POOLED_DIR}/split"
echo " Train Raw     : ${TRAIN_ACTIVATIONS_DIR}"
echo " Train Labels  : ${TRAIN_EVAL_CSV}"
echo " Val Pooled    : ${VAL_POOLED_DIR}/split"
echo " Val Raw       : ${VAL_ACTIVATIONS_DIR}"
echo " Val Labels    : ${VAL_EVAL_CSV}"
echo " GPU Device    : ${DEVICE}"
echo " Layers        : ${LAYERS[*]}"
echo " Poolings      : ${POOLINGS[*]}"
echo " Merges        : ${MERGES[*]}"
echo "=========================================================================="

for f in "$TRAIN_EVAL_CSV" "$VAL_EVAL_CSV"; do
    if [[ ! -f "$f" ]]; then
        echo "Error: ${f} not found"
        exit 1
    fi
done

if [[ " ${POOLINGS[*]} " == *" mean "* ]]; then
    for pooled_dir in "$TRAIN_POOLED_DIR" "$VAL_POOLED_DIR"; do
        if [[ ! -f "${pooled_dir}/split/metadata.json" ]]; then
            echo "Split activations missing in ${pooled_dir}. Running combine_pooled_scopes.py..."
            uv run python3 scripts/combine_pooled_scopes.py --pooled_dir "$pooled_dir"
        fi
    done
fi

SCOPES=()
for pooling in "${POOLINGS[@]}"; do
    case "$pooling" in
        "mean")
            CONFIG="configs/config_train_probe_mmlu_split.yaml"
            DATA_ARGS=(
                --data.init_args.activations_dir "${TRAIN_POOLED_DIR}/split"
                --data.init_args.val_activations_dir "${VAL_POOLED_DIR}/split"
            )
            ;;
        "attentive")
            CONFIG="configs/config_train_probe_mmlu_split_attentive.yaml"
            DATA_ARGS=(
                --data.init_args.activations_dir "$TRAIN_ACTIVATIONS_DIR"
                --data.init_args.val_activations_dir "$VAL_ACTIVATIONS_DIR"
            )
            ;;
        *)
            echo "Error: unknown pooling ${pooling}"
            exit 1
            ;;
    esac

    for merge in "${MERGES[@]}"; do
        scope="${pooling}_${merge}"
        SCOPES+=("$scope")
        echo ""
        echo "##########################################################################"
        echo " RUNNING SWEEP FOR: ${scope} (${CONFIG})"
        echo "##########################################################################"

        for layer in "${LAYERS[@]}"; do
            TARGET_DIR="${RESULTS_DIR}/${scope}/layer_${layer}"
            echo ""
            echo "--------------------------------------------------------------------------"
            echo " [${scope}] Training Probe for Layer ${layer}"
            echo " Target Dir: ${TARGET_DIR}"
            echo "--------------------------------------------------------------------------"

            LATEST_METRICS=$(ls -vd "${TARGET_DIR}"/lightning_logs/version_*/metrics.csv 2>/dev/null | tail -n 1 || true)
            if [[ -n "$LATEST_METRICS" && -f "$LATEST_METRICS" ]]; then
                echo "--> Already finished (${LATEST_METRICS}). Skipping."
                continue
            fi

            uv run main.py fit \
                --config "$CONFIG" \
                --trainer.default_root_dir "$TARGET_DIR" \
                --trainer.devices "$DEVICE" \
                --model.init_args.merge "$merge" \
                --model.init_args.layer_idx "$layer" \
                --model.init_args.model_dim "$MODEL_DIM" \
                --data.init_args.evaluation_csv "$TRAIN_EVAL_CSV" \
                --data.init_args.val_evaluation_csv "$VAL_EVAL_CSV" \
                --data.init_args.layer_idx "$layer" \
                "${DATA_ARGS[@]}"

            LATEST_LOG=$(ls -vd "${TARGET_DIR}"/lightning_logs/version_* 2>/dev/null | tail -n 1 || true)
            if [[ -n "$LATEST_LOG" && -f "${LATEST_LOG}/metrics.csv" ]]; then
                uv run python3 scripts/plot_probe_training.py \
                    --metrics_csv "${LATEST_LOG}/metrics.csv" \
                    --output_dir "${TARGET_DIR}/plots" || true
            fi
        done
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
