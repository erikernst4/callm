#!/bin/bash
# Run the normalized IsTrue method, p(answer) / (p(True) + p(False)), for the paper models.
#
# Each run reuses the paper run's config (lightning_logs/paper_results/<ds>/is_true_<suffix>/config.yaml),
# so prompts, datamodules and generation settings are identical; only the extractor is
# switched to normalize=true and the model keeps the top-k logprobs of each step.
# The raw (not normalized) results are the existing paper runs and are not rerun.
#
# Usage: scripts/istrue_normalization.sh <group> [gpu]
#   group: triviaqa-local | mmlu-local | triviaqa-gemini | mmlu-gemini
#   gpu:   GPU index for local groups (default 0)
# Env:
#   OUT          output root (default lightning_logs/istrue_normalization)
#   MAX_SAMPLES  limit samples per run (sandbox checks)
#   NORMALIZE    true (default) | false (sandbox check of the raw code path)
#   MODELS       space-separated suffixes to run instead of the whole group
#
# Outputs: $OUT/<ds>/is_true_<suffix>/llm_outputs.csv
# Then compare: uv run python scripts/compare_istrue_normalization.py --results-dir $OUT
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
P=lightning_logs/paper_results
OUT=${OUT:-lightning_logs/istrue_normalization}
NORMALIZE=${NORMALIZE:-true}
GROUP=${1:?usage: $0 <triviaqa-local|mmlu-local|triviaqa-gemini|mmlu-gemini> [gpu]}
GPU=${2:-0}

case $GROUP in
    triviaqa-local)
        DS=triviaqa
        DEFAULT_MODELS="zero_shot_Qwen3.5-4B-no-thinking zero_shot_Qwen3.5-9B-no-thinking zero_shot_GLM-4.6V-Flash-no-thinking Ministral-3-8B-Instruct-2512" ;;
    mmlu-local)
        DS=mmlu
        DEFAULT_MODELS="Qwen3.5-4B Qwen3.5-9B GLM-4.6V-Flash-no-thinking Ministral-3-8B-Instruct-2512" ;;
    triviaqa-gemini)
        DS=triviaqa
        DEFAULT_MODELS="gemini-2.5-flash-lite gemini-2.5-flash gemini-2.5-pro" ;;
    mmlu-gemini)
        DS=mmlu
        DEFAULT_MODELS="gemini-2.5-flash-lite gemini-2.5-flash gemini-2.5-pro" ;;
    *) echo "unknown group: $GROUP" >&2; exit 1 ;;
esac
MODELS=${MODELS:-$DEFAULT_MODELS}

for suffix in $MODELS; do
    run_dir=$OUT/$DS/is_true_$suffix
    if [[ -f $run_dir/llm_outputs.csv ]]; then
        echo "Skipping $run_dir (llm_outputs.csv exists)"
        continue
    fi

    # Base (label_prob) answers that IsTrue judges; MMLU GLM base has a zero_shot prefix
    base=label_prob_$suffix
    [[ $DS == mmlu && $suffix == GLM-4.6V-Flash-no-thinking ]] && base=label_prob_zero_shot_$suffix

    base_csv=$ROOT/$P/$DS/$base/llm_outputs.csv

    # Old MMLU base runs (GLM) lack the choices column the IsTrue prompt needs:
    # copy it from the Qwen3.5-4B base run, which has the same questions in order.
    if [[ $DS == mmlu ]] && ! head -1 "$base_csv" | grep -q choices; then
        patched=$OUT/mmlu/bases/$base/llm_outputs.csv
        mkdir -p "$(dirname "$patched")"
        uv run python - "$base_csv" "$ROOT/$P/mmlu/label_prob_Qwen3.5-4B/llm_outputs.csv" "$patched" <<'EOF'
import sys
import pandas as pd

base, donor, out = sys.argv[1:]
b, d = pd.read_csv(base), pd.read_csv(donor)
assert (b["question"].values == d["question"].values).all(), "questions differ"
b.insert(2, "choices", d["choices"])
b.to_csv(out, index=False)
print(f"Wrote {out} with choices from {donor}")
EOF
        base_csv=$patched
    fi

    if [[ $GROUP == *-gemini ]]; then
        extractor=callm.extractors.GCPIsTruePosteriorExtractor
    else
        extractor=callm.extractors.IsTruePosteriorExtractor
    fi

    args=(
        -c "$P/$DS/is_true_$suffix/config.yaml"
        --model.init_args.extractor.class_path="$extractor"
        --model.init_args.extractor.init_args.normalize="$NORMALIZE"
        --model.init_args.top_logprobs=20
        --data.init_args.llm_outputs_path="$base_csv"
        --trainer.logger=lightning.pytorch.loggers.TensorBoardLogger
        --trainer.logger.save_dir="$OUT"
        --trainer.logger.name="$DS"
        --trainer.logger.version="is_true_$suffix"
    )
    [[ -n ${MAX_SAMPLES:-} ]] && args+=(--data.init_args.max_samples="$MAX_SAMPLES")

    if [[ $GROUP == *-gemini ]]; then
        echo "=== $DS $suffix (Gemini)"
        uv run python main.py validate "${args[@]}"
    else
        args+=(--trainer.devices=1)
        echo "=== $DS $suffix (GPU $GPU)"
        CUDA_VISIBLE_DEVICES=$GPU uv run python main.py validate "${args[@]}"
    fi
done
