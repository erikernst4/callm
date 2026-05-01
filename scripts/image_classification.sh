
TRAIN_LOSSES=(
    "cls_cross_entropy"
    "cls_brier"
    "cls_n-ccas_n=0"
    "cls_n-ccas_n=1"
)

DATASETS=(
    "cifar10"
    "cifar100"
)

MODELS=(
    "resnet18"
    "vgg16"
    "vgg19"
    "densenet121"
)

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for loss in "${TRAIN_LOSSES[@]}"; do
            python scripts/image_classification.py \
                --dataset $dataset \
                --model $model \
                --max-epochs 6 \
                --batch-size 64 \
                --learning-rate 0.0001 \
                --save-scores-every-n-steps 32 \
                --log-train-loss-every-n-steps 16 \
                --loss $loss
        done
    done
done
