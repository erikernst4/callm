#!/bin/bash
# ==============================================================================
# Experiment 13: MMLU Attentive Sequence Pooling Probes with a Step Budget
# Experiment Name: exp13_attentive_pooling_w_step_budget_fix
#
# Reruns Experiment 11 with an LR schedule that actually completes. In exp11
# the cosine horizon was max_epochs (100) x steps/epoch, but early stopping
# ended runs after 6-14 epochs, so the LR stayed at ~99% of its peak.
#
# Changes vs exp11 (configs/config_train_probe_mmlu_attentive_step_budget.yaml):
#   - fixed budget of 10 epochs (6,930 optimizer steps), no early stopping;
#     the best checkpoint is still selected by val_loss
#   - cosine horizon = the real budget (max_steps from max_epochs)
#   - warmup = 5% of the budget (~350 steps) instead of a fixed 100 steps
#   - lr logged every 10 steps
#
# One probe per layer of the middle_30 subset (the 10 extracted layers):
#   [12, 14, 16, 17, 18, 19, 20, 21, 22, 24]
# over the full prompt + answer token sequence (scope 'both', pool_strategy
# none), for each sequence pooling method:
#   1. mean             (MeanPooling, masked mean baseline)
#   2. attentive        (AttentivePooling)
#   3. query_attentive  (QueryAttentivePooling)
#   4. transformer      (TransformerPooling)
# The multi-head variants are left out (they peaked at epoch 0 in exp11) but
# can be added with POOLING_METHODS.
#
# Train: MMLU auxiliary_train (44,300 samples), raw chunks of version_1
# Val:   MMLU validation split (1,531 samples)
# Samples are streamed from the memory-mapped chunks (lazy: true), reading
# ~148GB per epoch for one layer.
#
# TransformerPooling uses batch 32 with 2-step gradient accumulation (same
# effective batch and number of optimizer steps), as in exp11.
#
# Outputs:
#   results/exp13_attentive_pooling_w_step_budget_fix/<method>/layer_<N>
#
# Usage:
#   bash scripts/run_exp13_attentive_pooling_w_step_budget_fix.sh [DEVICE_IDX]
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

DEVICE_IDX="${1:-0}"
DEVICE="[${DEVICE_IDX}]"

EXP_NAME="${EXP_NAME:-exp13_attentive_pooling_w_step_budget_fix}"
RESULTS_DIR="results/${EXP_NAME}"
CONFIG="configs/config_train_probe_mmlu_attentive_step_budget.yaml"

TRAIN_ACTIVATIONS_DIR="activations_cache/mmlu_auxiliary_train_50k/lightning_logs/version_1"
TRAIN_EVAL_CSV="activations_cache/mmlu_auxiliary_train_50k/evaluation_results.csv"
VAL_ACTIVATIONS_DIR="activations_cache/mmlu_val"
VAL_EVAL_CSV="activations_cache/mmlu_val/lightning_logs/version_2/evaluation_results.csv"

MODEL_DIM=4096
LAYERS=(${LAYERS:-12 14 16 17 18 19 20 21 22 24})
POOLING_METHODS=(${POOLING_METHODS:-mean attentive query_attentive transformer})

echo "=========================================================================="
echo " Starting Experiment: ${EXP_NAME}"
echo "=========================================================================="
echo " Results Dir     : ${RESULTS_DIR}"
echo " Train Acts      : ${TRAIN_ACTIVATIONS_DIR}"
echo " Train Labels    : ${TRAIN_EVAL_CSV}"
echo " Val Acts        : ${VAL_ACTIVATIONS_DIR}"
echo " Val Labels      : ${VAL_EVAL_CSV}"
echo " GPU Device      : ${DEVICE}"
echo " Layers          : ${LAYERS[*]}"
echo " Pooling Methods : ${POOLING_METHODS[*]}"
echo "=========================================================================="

for f in "$TRAIN_EVAL_CSV" "$VAL_EVAL_CSV" "$CONFIG"; do
    if [[ ! -f "$f" ]]; then
        echo "Error: ${f} not found"
        exit 1
    fi
done

for d in "$TRAIN_ACTIVATIONS_DIR" "$VAL_ACTIVATIONS_DIR"; do
    if ! compgen -G "${d}/activations_*.pt" > /dev/null; then
        echo "Error: no raw activation chunks (activations_*.pt) in ${d}"
        echo "       Extract the MMLU activations first (see scripts/prepare_mmlu_probe_data.sh)."
        exit 1
    fi
done

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

for method in "${POOLING_METHODS[@]}"; do
    CLASSPATH=$(get_pooling_classpath "$method")
    if [[ -z "$CLASSPATH" ]]; then
        echo "Error: unknown pooling method ${method}"
        exit 1
    fi

    BATCH_ARGS=()
    if [[ "$method" == "transformer" ]]; then
        BATCH_ARGS=(--data.init_args.batch_size 32 --trainer.accumulate_grad_batches 2)
    fi

    echo ""
    echo "##########################################################################"
    echo " RUNNING SWEEP FOR POOLING METHOD: ${method} (${CLASSPATH})"
    echo "##########################################################################"

    for layer in "${LAYERS[@]}"; do
        TARGET_DIR="${RESULTS_DIR}/${method}/layer_${layer}"
        echo ""
        echo "--------------------------------------------------------------------------"
        echo " [Method: ${method}] Training Probe for Layer ${layer}"
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
            --model.init_args.layer_idx "$layer" \
            --model.init_args.model_dim "$MODEL_DIM" \
            --model.init_args.seq_pooling.class_path "$CLASSPATH" \
            --model.init_args.seq_pooling.init_args.model_dim "$MODEL_DIM" \
            --data.init_args.activations_dir "$TRAIN_ACTIVATIONS_DIR" \
            --data.init_args.evaluation_csv "$TRAIN_EVAL_CSV" \
            --data.init_args.val_activations_dir "$VAL_ACTIVATIONS_DIR" \
            --data.init_args.val_evaluation_csv "$VAL_EVAL_CSV" \
            --data.init_args.layer_idx "$layer" \
            "${BATCH_ARGS[@]}"

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
    --scopes "${POOLING_METHODS[@]}"

echo ""
echo "=========================================================================="
echo " Experiment ${EXP_NAME} finished successfully!"
echo " Summary CSV  : ${RESULTS_DIR}/experiment_summary.csv"
echo " Summary Plot : ${RESULTS_DIR}/experiment_summary.png"
echo "=========================================================================="
