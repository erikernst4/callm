#!/bin/bash -ex
#SBATCH --job-name=image_clsf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9          # num_workers + 1 for main process
#SBATCH --output=logs/image_clsf_%A_%a.out
#SBATCH --error=logs/image_clsf_%A_%a.err
#SBATCH --account=zva@a100
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-5


declare -a CONFIGS=(
    "cifar10 resnet18 cls_cross_entropy"
    "cifar10 resnet18 cls_loglog"
    "cifar10 vgg16 cls_cross_entropy"
    "cifar10 vgg16 cls_loglog"
    "cifar10 densenet121 cls_cross_entropy"
    "cifar10 densenet121 cls_loglog"
    # "cifar100 resnet18 cls_cross_entropy"
    # "cifar100 resnet18 cls_loglog"
    # "cifar100 vgg16 cls_cross_entropy"
    # "cifar100 vgg16 cls_loglog"
    # "cifar100 densenet121 cls_cross_entropy"
    # "cifar100 densenet121 cls_loglog"
)

read -r dataset model loss <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Running image classification with dataset: $dataset, model: $model, loss: $loss"
srun uv run python scripts/image_classification.py \
    --dataset $dataset \
    --model $model \
    --max-epochs 60 \
    --batch-size 128 \
    --learning-rate 0.0002 \
    --save-scores-every-n-steps 128 \
    --log-train-loss-every-n-steps 16 \
    --loss $loss

