#!/bin/bash -ex
#SBATCH --job-name=ood_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9          # num_workers + 1 for main process
#SBATCH --output=logs/ood_analysis_%A_%a.out
#SBATCH --error=logs/ood_analysis_%A_%a.err
#SBATCH --account=zva@a100
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-0


declare -a CONFIGS=(
    128
)

batch_size="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
srun uv run python scripts/ood_analysis.py --batch_size $batch_size

