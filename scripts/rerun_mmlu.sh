#!/bin/bash

# Script to re-run MMLU label_prob experiments (excluding GLM)

# Navigate to the callm project root directory
cd "$(dirname "$0")/.." || exit 1

MMLU_LOGS_DIR="lightning_logs/mmlu"

# Find all config.yaml files in label_prob experiments, excluding those with GLM
for exp_dir in "$MMLU_LOGS_DIR"/is_true_*; do
    # Skip if not a directory
    if [[ ! -d "$exp_dir" ]]; then
        continue
    fi

    # Exclude GLM experiments
    if [[ "$exp_dir" == *"GLM"* ]]; then
        echo "Skipping GLM experiment: $exp_dir"
        continue
    fi

    config_file="$exp_dir/config.yaml"
    if [[ -f "$config_file" ]]; then
        echo "=========================================================================="
        echo "Running validation for: $config_file"
        echo "=========================================================================="
        uv run main.py validate --config "$config_file"
    else
        echo "Warning: No config.yaml found in $exp_dir"
    fi
done
