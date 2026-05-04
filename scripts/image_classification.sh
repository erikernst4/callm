
TRAIN_LOSSES=(
    "cls_cross_entropy"
    "cls_loglog"
)

DATASETS=(
    "cifar10"
    "cifar100"
)

MODELS=(
    "resnet18"
    "vgg16"
    "densenet121"
)

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for loss in "${TRAIN_LOSSES[@]}"; do
            echo "Running image classification with dataset: $dataset, model: $model, loss: $loss"
            python scripts/image_classification.py \
                --dataset $dataset \
                --model $model \
                --max-epochs 60 \
                --batch-size 128 \
                --learning-rate 0.0002 \
                --save-scores-every-n-steps 128 \
                --log-train-loss-every-n-steps 16 \
                --loss $loss
        done
    done
done
