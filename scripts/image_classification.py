from pathlib import Path

import pandas as pd
import torch
from callm.metrics import get_metric_from_id
import matplotlib.pyplot as plt
import yaml
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch_uncertainty.datamodules import CIFAR10DataModule, CIFAR100DataModule

dataset2display = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
}

model2display = {
    "resnet18": "ResNet-18",
    "vgg16": "VGG-16",
    "vgg19": "VGG-19",
    "densenet121": "DenseNet-121",
}


EVAL_METRICS = [
    "cls_ner",
    "cls_nbs",
    "cls_nce",
    # "cls_auc",
    # "cls_aurc",
    # "cls_ece_nbins=10",
    "cls_norm_n-ccas_n=0",
    "cls_norm_n-ccas_n=1",
]


def save_yaml(data, path):
    path = Path(path)
    with open(path, "w") as f:
        yaml.dump(data, f)


def load_dataset(dataset, batch_size):
    # Select dataset
    if dataset.lower() == "cifar10":
        num_classes = 10
        datamodule = CIFAR10DataModule(root="./data", batch_size=batch_size, num_workers=8, eval_ood=True)
    elif dataset.lower() == "cifar100":
        num_classes = 100
        datamodule = CIFAR100DataModule(root="./data", batch_size=batch_size, num_workers=8, eval_ood=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    datamodule.setup(stage="test")
    id_test_loader, ood_test_loader = datamodule.test_dataloader()

    return train_loader, id_test_loader, ood_test_loader, num_classes


def load_model(model, num_classes):
    # Select model
    if model.lower() == "resnet18":
        net = torchvision.models.resnet18(num_classes=num_classes)
    elif model.lower() == "vgg16":
        net = torchvision.models.vgg16(num_classes=num_classes)
    elif model.lower() == "vgg19":
        net = torchvision.models.vgg19(num_classes=num_classes)
    elif model.lower() == "densenet121":
        net = torchvision.models.densenet121(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    return net, device


def _compute_test_scores(net, device, test_loader):
    net.eval()
    test_logits = []
    with torch.no_grad():
        for test_inputs, _ in test_loader:
            test_inputs = test_inputs.to(device)
            logits = net(test_inputs)
            test_logits.append(logits.cpu())
    test_logits = torch.cat(test_logits, dim=0)
    net.train()

    # Save scores to disk
    return pd.DataFrame(test_logits.numpy())


def _train_and_inference(
    dataset,
    model,
    max_epochs,
    batch_size,
    learning_rate,
    save_scores_every_n_steps,
    log_train_loss_every_n_steps,
    loss,
    scores_dir: Path,
):
    # Load dataset and save validation labels
    train_loader, id_test_loader, ood_test_loader, num_classes = load_dataset(dataset, batch_size)
    id_test_labels = torch.cat([item[1] for item in id_test_loader], dim=0)
    id_test_labels_df = pd.DataFrame(id_test_labels.numpy(), columns=["label"])
    id_test_labels_df.to_csv(scores_dir / "id_test_labels.csv", index=False, header=False)
    ood_test_labels = torch.cat([item[1] for item in ood_test_loader], dim=0)
    ood_test_labels_df = pd.DataFrame(ood_test_labels.numpy(), columns=["label"])
    ood_test_labels_df.to_csv(scores_dir / "ood_test_labels.csv", index=False, header=False)

    # Load model
    net, device = load_model(model, num_classes=num_classes)

    # Loss function and optimizer
    loss_dict = get_metric_from_id(loss)
    criterion = loss_dict["obj"]
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Train loop
    id_scores = {}
    ood_scores = {}
    train_step = 0
    net.train()
    for epoch in range(max_epochs):
        for inputs, targets in train_loader:
            # Save validation scores every n steps
            if train_step % save_scores_every_n_steps == 0:
                id_scores[train_step] = _compute_test_scores(net, device, id_test_loader)
                id_scores[train_step].to_csv(
                    scores_dir / f"id_scores_step={train_step}.csv",
                    index=False,
                    header=False,
                )
                ood_scores[train_step] = _compute_test_scores(net, device, ood_test_loader)
                ood_scores[train_step].to_csv(
                    scores_dir / f"ood_scores_step={train_step}.csv",
                    index=False,
                    header=False,
                )

            # Train step
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            train_loss = criterion(outputs, targets)
            train_loss.backward()
            optimizer.step()

            # Log training loss every n steps
            if train_step % log_train_loss_every_n_steps == 0:
                print(
                    f"Epoch: {epoch}, Train step: {train_step}, Train Loss: {train_loss.item():.4f}"
                )

            # Update training step
            train_step += 1

    # Add final validation scores after training is complete
    if train_step not in id_scores:
        id_scores[train_step] = _compute_test_scores(net, device, id_test_loader)
        id_scores[train_step].to_csv(
            scores_dir / f"id_scores_step={train_step}.csv", index=False, header=False
        )
    if train_step not in ood_scores:
        ood_scores[train_step] = _compute_test_scores(net, device, ood_test_loader)
        ood_scores[train_step].to_csv(
            scores_dir / f"ood_scores_step={train_step}.csv", index=False, header=False
        )

    # Save last step results
    last_train_step_id_scores = {train_step: id_scores[train_step]}
    id_results = compute_metrics(last_train_step_id_scores, id_test_labels_df, metrics=[loss])
    id_results = id_results.reset_index(drop=True)
    id_results = (
        id_results.T.reset_index()
        .rename(columns={"index": "metric", 0: "value"})
        .set_index("metric")
    )
    id_results.to_csv(scores_dir / "results_id_last_step.csv", index=False)
    ood_results = compute_metrics({train_step: ood_scores[train_step]}, ood_test_labels_df, metrics=[loss])
    ood_results = ood_results.reset_index(drop=True)
    ood_results = (
        ood_results.T.reset_index()
        .rename(columns={"index": "metric", 0: "value"})
        .set_index("metric")
    )
    ood_results.to_csv(scores_dir / "results_ood_last_step.csv", index=False)

    return id_scores, id_test_labels_df, ood_scores, ood_test_labels_df


def train_and_inference(
    dataset,
    model,
    max_epochs,
    batch_size,
    learning_rate,
    save_scores_every_n_steps,
    log_train_loss_every_n_steps,
    loss,
    logs_dir: Path,
):
    scores_dir = logs_dir / f"me={max_epochs}_bs={batch_size}_lr={learning_rate}"
    scores_dir.mkdir(parents=True, exist_ok=True)

    if (scores_dir / "results_id_last_step.csv").exists() and (scores_dir / "results_ood_last_step.csv").exists():
        print(f"Scores already exist at {scores_dir}. Skipping training.")
        id_scores = {
            int(scores_path.stem.split("step=")[-1]): pd.read_csv(
                scores_path, index_col=None, header=None
            )
            for scores_path in scores_dir.glob("id_scores_step=*.csv")
        }
        id_labels_df = pd.read_csv(scores_dir / "id_test_labels.csv", index_col=None, header=None)
        
        ood_scores = {
            int(scores_path.stem.split("step=")[-1]): pd.read_csv(
                scores_path, index_col=None, header=None
            )
            for scores_path in scores_dir.glob("ood_scores_step=*.csv")
        }
        ood_labels_df = pd.read_csv(scores_dir / "ood_test_labels.csv", index_col=None, header=None)

        return id_scores, id_labels_df, ood_scores, ood_labels_df

    save_yaml(
        {
            "dataset": dataset,
            "model": model,
            "loss": loss,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "save_scores_every_n_steps": save_scores_every_n_steps,
        },
        logs_dir / "config.yaml",
    )

    id_scores, id_labels_df, ood_scores, ood_labels_df = _train_and_inference(
        dataset=dataset,
        model=model,
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_scores_every_n_steps=save_scores_every_n_steps,
        log_train_loss_every_n_steps=log_train_loss_every_n_steps,
        loss=loss,
        scores_dir=scores_dir,
    )

    return id_scores, id_labels_df, ood_scores, ood_labels_df


def compute_metrics(scores, labels, metrics):
    results = []
    for step, group in scores.items():
        for metric in metrics:
            metric_dict = get_metric_from_id(metric)
            metric_fn = metric_dict["function"]
            logits_arr = group.values.copy()
            logits_tensor = torch.from_numpy(logits_arr).float()
            labels_arr = labels.values.reshape(-1).copy()
            labels_tensor = torch.from_numpy(labels_arr).long()
            metric_value = metric_fn(logits_tensor, labels_tensor)
            results.append(
                {"step": step, "metric": metric_dict["display"], "value": metric_value}
            )

    results = pd.DataFrame(results).pivot(
        index="step", columns="metric", values="value"
    )
    results = results.sort_index()
    return results


def plot_results(id_results, ood_results, loss, dataset, model, output_dir: Path):
    steps = id_results.index.values
    fig, ax = plt.subplots(1,2, figsize=(12, 5))
    for metric in id_results.columns:
        metric_results = id_results.loc[:, metric].values
        ax[0].plot(steps, metric_results, label=metric)
    for metric in ood_results.columns:
        metric_results = ood_results.loc[:, metric].values
        ax[1].plot(steps, metric_results, label=metric)
    loss_display = get_metric_from_id(loss)["display"]
    dataset_display = dataset2display[dataset]
    model_display = model2display[model]
    ax[0].set_title(
        f"ID Test curves when trained with {loss_display}\nDataset: {dataset_display}, Model: {model_display}"
    )
    ax[0].set_xlabel("Training Step")
    ax[0].set_ylabel("Metric Value")
    ax[0].grid()
    ax[0].legend()
    ax[1].set_title(
        f"OOD Test curves when trained with {loss_display}\nDataset: {dataset_display}, Model: {model_display}"
    )
    ax[1].set_xlabel("Training Step")
    ax[1].set_ylabel("Metric Value")
    ax[1].grid()
    ax[1].legend()
    fig.savefig(output_dir / "training.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main(
    dataset: str = "cifar10",
    model: str = "resnet18",
    max_epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    save_scores_every_n_steps: int = 10,
    log_train_loss_every_n_steps: int = 10,
    loss: str = "cls_cross_entropy",
    eval_metrics: list[str] = [],
    logs_dir: str = "scores/image_classification",
    output_dir: str = "outputs/image_classification",
):
    logs_dir = Path(logs_dir) / dataset / model / loss
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir) / dataset / model / loss
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train a model on a dataset and save the scores during training
    id_scores, id_labels_df, ood_scores, ood_labels_df = train_and_inference(
        dataset=dataset,
        model=model,
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_scores_every_n_steps=save_scores_every_n_steps,
        log_train_loss_every_n_steps=log_train_loss_every_n_steps,
        loss=loss,
        logs_dir=logs_dir,
    )

    # Compute evaluation metrics from the saved scores and labels
    id_results = compute_metrics(id_scores, id_labels_df, eval_metrics)
    id_results.to_csv(output_dir / "results_id.csv", index=True)
    ood_results = compute_metrics(ood_scores, ood_labels_df, eval_metrics)
    ood_results.to_csv(output_dir / "results_ood.csv", index=True)

    # Plot the results
    plot_results(id_results, ood_results, loss, dataset, model, output_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image Classification")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use")
    parser.add_argument("--model", type=str, default="resnet18", help="Model to use")
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate for training"
    )
    parser.add_argument(
        "--save-scores-every-n-steps",
        type=int,
        default=10,
        help="Save scores every n training steps",
    )
    parser.add_argument(
        "--log-train-loss-every-n-steps",
        type=int,
        default=10,
        help="Log training loss every n training steps",
    )
    parser.add_argument(
        "--loss", type=str, default="cls_cross_entropy", help="Loss function to use"
    )
    parser.add_argument(
        "--eval-metrics",
        nargs="+",
        default=EVAL_METRICS,
        help="Evaluation metrics to compute",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="scores/image_classification",
        help="Directory to save logs and scores",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/image_classification",
        help="Directory to save output plots",
    )
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        model=args.model,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_scores_every_n_steps=args.save_scores_every_n_steps,
        log_train_loss_every_n_steps=args.log_train_loss_every_n_steps,
        loss=args.loss,
        eval_metrics=args.eval_metrics,
        logs_dir=args.logs_dir,
        output_dir=args.output_dir,
    )
