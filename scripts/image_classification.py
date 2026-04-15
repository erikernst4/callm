
from pathlib import Path

import pandas as pd
import torch
from callm.metrics import get_metric_from_id
import matplotlib.pyplot as plt

def train_and_inference(
    dataset,
    model,
    max_epochs,
    batch_size,
    learning_rate,
    save_scores_every_n_steps,
    loss,
    logs_dir: str
):
    logs_dir = Path(logs_dir)
    scores_path = logs_dir / f"final_scores_{dataset}_{model}.json"
    if scores_path.exists():
        print(f"Scores already exist at {scores_path}. Skipping training.")
        all_scores = []
        for scores in logs_dir.glob(f"scores_{dataset}_{model}_step=*.json"):
            print(f"Found scores file: {scores}")
            scores_df = pd.read_csv(scores, index_col=None, header=None)
            scores_df["step"] = int(scores.stem.split("step=")[-1])
            all_scores.append(scores_df)
        if all_scores:
            all_scores_df = pd.concat(all_scores, ignore_index=True)
            labels = pd.read_csv(logs_dir / f"labels_{dataset}_{model}.csv", index_col=None, header=None)

        return all_scores_df, labels
    
    # TODO: If scores do not exist, train the model and save the scores during training
        

def compute_metrics(scores, labels, metrics):
    results = []
    for step, group in scores.groupby("step"):
        for metric in metrics:
            metric_fn = get_metric_from_id(metric)
            logits = torch.from_numpy(group.iloc[:, :-1].values).float()  # Assuming the last column is 'step'
            labels = torch.from_numpy(labels.values).long()
            metric_value = metric_fn(logits, labels)
            print(f"Metric {metric} at step {step}: {metric_value}")
            results.append({
                "step": step,
                "metric": metric,
                "value": metric_value
            })

    results = pd.DataFrame(results).pivot(index="step", columns="metric", values="value").reset_index()
    results = results.sort_index()
    return results


def plot_results(results, loss, dataset, model, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = results.index.values
    fig, ax = plt.subplots()
    for metric in results.columns:
        metric_results = results.loc[:, metric].values
        ax.plot(steps, metric_results, label=metric)
        ax.set_title(f"Training Loss: {loss}, Dataset: {dataset}, Model: {model}")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Metric Value")
        ax.grid()
    fig.savefig(output_dir / f"{dataset}_{model}_{loss}_{metric}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

def main(
    dataset: str = "cifar10",
    model: str = "resnet18",
    max_epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    save_scores_every_n_steps: int = 10,
    loss: str = "cross_entropy",
    eval_metrics: list[str] = [],
    logs_dir: str = "scores/image_classification",
    output_dir: str = "outputs/image_classification",
):
    
    # Train a model on a dataset and save the scores during training
    scores, labels = train_and_inference(
        dataset=dataset,
        model=model,
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_scores_every_n_steps=save_scores_every_n_steps,
        loss=loss,
        logs_dir=logs_dir
    )

    # Compute evaluation metrics from the saved scores and labels
    results = compute_metrics(scores, labels, eval_metrics)

    # Plot the results
    plot_results(results, loss, dataset, model, output_dir)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Image Classification")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use")
    parser.add_argument("--model", type=str, default="resnet18", help="Model to use")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--save_scores_every_n_steps", type=int, default=10, help="Save scores every n training steps")
    parser.add_argument("--loss", type=str, default="cross_entropy", help="Loss function to use")
    parser.add_argument("--eval_metrics", nargs="+", default=[], help="Evaluation metrics to compute")
    parser.add_argument("--logs_dir", type=str, default="scores/image_classification", help="Directory to save logs and scores")
    parser.add_argument("--output_dir", type=str, default="outputs/image_classification", help="Directory to save output plots")
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        model=args.model,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_scores_every_n_steps=args.save_scores_every_n_steps,
        loss=args.loss,
        eval_metrics=args.eval_metrics,
        logs_dir=args.logs_dir,
        output_dir=args.output_dir
    )