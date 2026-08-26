#!/bin/bash
# ==============================================================================
# Significance Re-Run Script
#
# Re-runs an existing experiment with multiple random seeds to measure the
# statistical significance of results. For each subset (sub-experiment) in the
# given experiment directory, this script:
#   1. Reads the saved config.yaml from the original run
#   2. Re-trains the probe with N different seeds
#   3. Saves per-sample validation predictions (for bootstrap analysis)
#   4. Logs training time per run
#   5. Runs bootstrap_evaluation_metrics.py to compute CIs and significance tests
#
# Usage:
#   bash scripts/run_significance_rerun.sh <exp_dir> [options]
#
# Arguments:
#   <exp_dir>  Path to the experiment directory containing subset subdirs
#              (e.g., results/exp4_layer_pooling/layernorm)
#
# Options:
#   --num_seeds N            Number of seeds to run (default: 5)
#   --device D               GPU device specification (default: [0])
#   --reference_subset NAME  Subset for paired significance tests (default: layer_middle_30)
#   --n_bootstraps N         Bootstrap iterations (default: 10000)
#   --dry_run                Print commands without executing
#
# Output structure:
#   <exp_dir>/
#     <subset>/
#       seed_0/lightning_logs/version_0/{metrics.csv, val_predictions.csv}
#       seed_1/...
#       ...
#     significance_results/
#       timing.csv
#       seed_metrics.csv
#       bootstrap_summary.csv
#       pairwise_significance.csv
#       bootstrap_summary.png
# ==============================================================================

set -e

# ── Argument parsing ────────────────────────────────────────────────────

if [[ $# -lt 1 ]]; then
    echo "Usage: bash $0 <exp_dir> [--num_seeds N] [--device D] [--reference_subset NAME] [--n_bootstraps N] [--dry_run]"
    exit 1
fi

EXP_DIR="$1"
shift

NUM_SEEDS=5
DEVICE="[0]"
REFERENCE_SUBSET="layer_middle_30"
DRY_RUN=false
N_BOOTSTRAPS=10000

while [[ $# -gt 0 ]]; do
    case $1 in
        --num_seeds) NUM_SEEDS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --reference_subset) REFERENCE_SUBSET="$2"; shift 2 ;;
        --n_bootstraps) N_BOOTSTRAPS="$2"; shift 2 ;;
        --dry_run) DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Navigate to project root ────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Validate inputs ─────────────────────────────────────────────────────

if [[ ! -d "$EXP_DIR" ]]; then
    echo "Error: Experiment directory not found: $EXP_DIR"
    exit 1
fi

echo "=========================================================================="
echo " Significance Re-Run"
echo "=========================================================================="
echo " Experiment Dir  : ${EXP_DIR}"
echo " Num Seeds       : ${NUM_SEEDS} (0 to $((NUM_SEEDS - 1)))"
echo " Device          : ${DEVICE}"
echo " Reference Subset: ${REFERENCE_SUBSET}"
echo " Bootstrap Iters : ${N_BOOTSTRAPS}"
echo " Dry Run         : ${DRY_RUN}"
echo "=========================================================================="

# ── Setup timing log ────────────────────────────────────────────────────

RESULTS_DIR="${EXP_DIR}/significance_results"
mkdir -p "$RESULTS_DIR"

TIMING_LOG="${RESULTS_DIR}/timing.csv"
echo "subset,seed,duration_seconds,status" > "$TIMING_LOG"

TOTAL_START=$SECONDS

# ── Discover subsets with saved configs ──────────────────────────────────

SUBSETS=()
declare -A SUBSET_CONFIGS

for subset_dir in "$EXP_DIR"/*/; do
    [[ ! -d "$subset_dir" ]] && continue
    subset_name=$(basename "$subset_dir")

    # Skip non-experiment directories
    [[ "$subset_name" == "significance_results" ]] && continue
    [[ "$subset_name" == "plots" ]] && continue
    [[ "$subset_name" == ".git" ]] && continue

    # Find saved config from the original run (latest version)
    config_file=""
    for vdir in $(ls -vd "${subset_dir}lightning_logs/version_"* 2>/dev/null | tail -n 1); do
        if [[ -f "${vdir}/config.yaml" ]]; then
            config_file="${vdir}/config.yaml"
        fi
    done

    if [[ -n "$config_file" ]]; then
        SUBSETS+=("$subset_name")
        SUBSET_CONFIGS["$subset_name"]="$config_file"
    else
        echo "  Warning: No config.yaml found for subset '${subset_name}', skipping."
    fi
done

if [[ ${#SUBSETS[@]} -eq 0 ]]; then
    echo "Error: No subsets with saved configs found in ${EXP_DIR}"
    exit 1
fi

echo ""
echo "Found ${#SUBSETS[@]} subsets:"
for subset in "${SUBSETS[@]}"; do
    echo "  - ${subset} (config: ${SUBSET_CONFIGS[$subset]})"
done
echo ""

# ── Run each subset with each seed ───────────────────────────────────────

TOTAL_RUNS=$((${#SUBSETS[@]} * NUM_SEEDS))
COMPLETED=0
SKIPPED=0
FAILED=0

for subset in "${SUBSETS[@]}"; do
    SUBSET_DIR="${EXP_DIR}/${subset}"
    SAVED_CONFIG="${SUBSET_CONFIGS[$subset]}"

    echo "=========================================================================="
    echo " Subset: ${subset}"
    echo " Config: ${SAVED_CONFIG}"
    echo "=========================================================================="

    for ((seed=0; seed<NUM_SEEDS; seed++)); do
        SEED_DIR="${SUBSET_DIR}/seed_${seed}"

        # Check if already completed (has val_predictions.csv)
        existing_pred=$(find "$SEED_DIR" -name "val_predictions.csv" -print -quit 2>/dev/null || true)
        if [[ -n "$existing_pred" ]]; then
            echo "  [SKIP] seed_${seed} — val_predictions.csv already exists"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        echo ""
        echo "  >>> Running seed_${seed} → ${SEED_DIR}"

        if [[ "$DRY_RUN" == "true" ]]; then
            echo "    DRY RUN command:"
            echo "      uv run main.py fit \\"
            echo "        --config \"${SAVED_CONFIG}\" \\"
            echo "        --seed_everything ${seed} \\"
            echo "        --data.init_args.seed ${seed} \\"
            echo "        --trainer.default_root_dir \"${SEED_DIR}\" \\"
            echo "        --trainer.devices \"${DEVICE}\" \\"
            echo "        --model.init_args.save_val_predictions true"
            echo ""
            continue
        fi

        RUN_START=$SECONDS
        STATUS="success"

        # Run training with the saved config, overriding seeds and output dir.
        # The model's on_fit_end hook will automatically:
        #   1. Load the best checkpoint
        #   2. Run a validation pass
        #   3. Save per-sample predictions to val_predictions.csv
        uv run main.py fit \
            --config "$SAVED_CONFIG" \
            --seed_everything "$seed" \
            --data.init_args.seed "$seed" \
            --trainer.default_root_dir "$SEED_DIR" \
            --trainer.devices "$DEVICE" \
            --model.init_args.save_val_predictions true \
        || STATUS="failed"

        DURATION=$((SECONDS - RUN_START))
        echo "${subset},${seed},${DURATION},${STATUS}" >> "$TIMING_LOG"

        if [[ "$STATUS" == "success" ]]; then
            COMPLETED=$((COMPLETED + 1))
            echo "  [DONE] seed_${seed} — ${DURATION}s"
        else
            FAILED=$((FAILED + 1))
            echo "  [FAIL] seed_${seed} — ${DURATION}s"
        fi
    done
done

TOTAL_DURATION=$((SECONDS - TOTAL_START))

echo ""
echo "=========================================================================="
echo " Re-Run Summary"
echo "=========================================================================="
echo " Total time      : ${TOTAL_DURATION}s"
echo " Completed       : ${COMPLETED}"
echo " Skipped (cached): ${SKIPPED}"
echo " Failed          : ${FAILED}"
echo " Timing log      : ${TIMING_LOG}"
echo "=========================================================================="

# ── Run bootstrap analysis ───────────────────────────────────────────────

if [[ "$DRY_RUN" != "true" && $COMPLETED -gt 0 ]]; then
    echo ""
    echo ">>> Running bootstrap evaluation metrics..."
    echo ""

    uv run python scripts/bootstrap_evaluation_metrics.py \
        --exp_dir "$EXP_DIR" \
        --reference_subset "$REFERENCE_SUBSET" \
        --n_bootstraps "$N_BOOTSTRAPS"
elif [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "DRY RUN: Would run bootstrap analysis:"
    echo "  uv run python scripts/bootstrap_evaluation_metrics.py \\"
    echo "    --exp_dir \"${EXP_DIR}\" \\"
    echo "    --reference_subset \"${REFERENCE_SUBSET}\" \\"
    echo "    --n_bootstraps ${N_BOOTSTRAPS}"
fi

echo ""
echo "=========================================================================="
echo " Significance analysis complete!"
echo " Results: ${RESULTS_DIR}/"
echo "=========================================================================="
