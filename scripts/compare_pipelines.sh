#!/bin/bash
# ==============================================================================
# Compare Pipelines: Cached Activations vs On-the-Fly
#
# Trains the same probe configuration on 5k samples using both:
#   1. Pre-saved activations (existing ActivationsDataModule)
#   2. On-the-fly LLM forward pass (new OnTheFlyActivationsDataModule)
#
# Compares val_loss, val_ece, val_cross_entropy, val_auc to verify
# the two pipelines produce similar results.
#
# Usage:
#   bash scripts/compare_pipelines.sh [DEVICE]
#
# Prerequisites:
#   - Pre-computed activations in activations_cache/qwen_triviaqa/
#   - Pre-computed labels (run scripts/precompute_labels.sh 5000 first)
# ==============================================================================

set -e

DEVICE="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Configuration ──────────────────────────────────────────────────────
RESULTS_DIR="results/pipeline_comparison"
LAYER=19
SCOPE="prompt"  # Use "prompt" for fair comparison (no generation needed)
POOL_STRATEGY="mean"

# Cached activations paths (existing 5k data)
ACTIVATIONS_DIR="activations_cache/qwen_triviaqa/lightning_logs/version_6"
EVALUATION_CSV="activations_cache/qwen_triviaqa/lightning_logs/version_5_evaluation/version_9/evaluation_results.csv"

# On-the-fly paths (from precompute_labels.sh 5000)
# Auto-detect latest if available, or fall back to default
LATEST_5K_VER=$(ls -vd "activations_cache/qwen_triviaqa_5000/lightning_logs/version_"[0-9]* 2>/dev/null | grep -v "_evaluation" | tail -n 1)
if [[ -n "$LATEST_5K_VER" ]]; then
    ON_THE_FLY_LLM_CSV="${LATEST_5K_VER}/llm_outputs.csv"
    LATEST_5K_EVAL=$(ls -vd "${LATEST_5K_VER}_evaluation/version_"[0-9]* 2>/dev/null | tail -n 1)
    ON_THE_FLY_EVAL_CSV="${LATEST_5K_EVAL}/evaluation_results.csv"
else
    ON_THE_FLY_LLM_CSV="activations_cache/qwen_triviaqa_5000/lightning_logs/version_0/llm_outputs.csv"
    ON_THE_FLY_EVAL_CSV="activations_cache/qwen_triviaqa_5000/lightning_logs/version_0_evaluation/version_0/evaluation_results.csv"
fi

echo "=========================================================================="
echo " Pipeline Comparison: Cached vs On-the-Fly"
echo "=========================================================================="
echo " Layer       : ${LAYER}"
echo " Scope       : ${SCOPE}"
echo " Pool        : ${POOL_STRATEGY}"
echo " Device      : ${DEVICE}"
echo "=========================================================================="

# ── Pipeline 1: Cached Activations ────────────────────────────────────
CACHED_DIR="${RESULTS_DIR}/cached"
echo ""
echo "──────────────────────────────────────────────────────────────────────────"
echo " [Pipeline 1] Training with CACHED activations"
echo " Target: ${CACHED_DIR}"
echo "──────────────────────────────────────────────────────────────────────────"

if [[ ! -d "$ACTIVATIONS_DIR" ]]; then
    echo "Error: Activations directory not found at $ACTIVATIONS_DIR"
    echo "       Skipping cached pipeline."
else
    uv run main.py fit \
        --config configs/config_train_probe.yaml \
        --trainer.default_root_dir "$CACHED_DIR" \
        --model.init_args.layer_idx "$LAYER" \
        --model.init_args.model_dim 4096 \
        --data.init_args.activations_dir "$ACTIVATIONS_DIR" \
        --data.init_args.evaluation_csv "$EVALUATION_CSV" \
        --data.init_args.layer_idx "$LAYER" \
        --data.init_args.activations_to_use "$SCOPE" \
        --data.init_args.pool_strategy "$POOL_STRATEGY" \
        --trainer.devices "[$DEVICE]"
fi

# ── Pipeline 2: On-the-Fly ────────────────────────────────────────────
ON_THE_FLY_DIR="${RESULTS_DIR}/on_the_fly"
echo ""
echo "──────────────────────────────────────────────────────────────────────────"
echo " [Pipeline 2] Training with ON-THE-FLY activations"
echo " Target: ${ON_THE_FLY_DIR}"
echo "──────────────────────────────────────────────────────────────────────────"

if [[ ! -f "$ON_THE_FLY_LLM_CSV" ]]; then
    echo "Error: LLM outputs CSV not found at $ON_THE_FLY_LLM_CSV"
    echo "       Run 'scripts/precompute_labels.sh 5000' first."
    exit 1
fi

if [[ ! -f "$ON_THE_FLY_EVAL_CSV" ]]; then
    echo "Error: Evaluation CSV not found at $ON_THE_FLY_EVAL_CSV"
    echo "       Run 'scripts/precompute_labels.sh 5000' first."
    exit 1
fi

uv run main.py fit \
    --config configs/config_train_probe_on_the_fly.yaml \
    --trainer.default_root_dir "$ON_THE_FLY_DIR" \
    --model.init_args.layer_idx "$LAYER" \
    --model.init_args.model_dim 4096 \
    --model.init_args.pool_strategy "$POOL_STRATEGY" \
    --model.init_args.activations_to_use "$SCOPE" \
    --data.init_args.llm_outputs_csv "$ON_THE_FLY_LLM_CSV" \
    --data.init_args.evaluation_csv "$ON_THE_FLY_EVAL_CSV" \
    --trainer.devices "[$DEVICE]"

# ── Compare Results ───────────────────────────────────────────────────
echo ""
echo "=========================================================================="
echo " Results Comparison"
echo "=========================================================================="

CACHED_METRICS=$(ls -vd "${CACHED_DIR}/lightning_logs/version_"* 2>/dev/null | tail -n 1)
ON_THE_FLY_METRICS=$(ls -vd "${ON_THE_FLY_DIR}/lightning_logs/version_"* 2>/dev/null | tail -n 1)

echo ""
echo "Cached pipeline metrics:"
if [[ -n "$CACHED_METRICS" && -f "${CACHED_METRICS}/metrics.csv" ]]; then
    head -2 "${CACHED_METRICS}/metrics.csv"
else
    echo "  (not available)"
fi

echo ""
echo "On-the-fly pipeline metrics:"
if [[ -n "$ON_THE_FLY_METRICS" && -f "${ON_THE_FLY_METRICS}/metrics.csv" ]]; then
    head -2 "${ON_THE_FLY_METRICS}/metrics.csv"
else
    echo "  (not available)"
fi

echo ""
echo "=========================================================================="
echo " Comparison complete. Check results under: ${RESULTS_DIR}/"
echo "=========================================================================="
