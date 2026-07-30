import os
from pathlib import Path
import subprocess
import tempfile

import pandas as pd
import torch
from ecuas import get_metric_from_id
from torch_uncertainty.datamodules import CIFAR10DataModule, CIFAR100DataModule
from tqdm import tqdm

dataset2display = {
    "cifar10": "CIFAR-10-C (OOD)",
    "cifar100": "CIFAR-100-C (OOD)",
}

model2display = {
    "resnet20": "ResNet20",
    "vgg19_bn": "VGG19",
    "repvgg_a2": "RepVGG",
}


EVAL_METRICS = [
    "cls_ner",
    "cls_ece_nbins=10",
    "cls_auc",
    "conf_norm_cross_entropy",
    "conf_norm_brier",
    "cls_nce",
    "cls_nbs",
    "cls_aurc",
    "cls_norm_n-ecuas_n=0",
    "cls_norm_n-ecuas_n=1",
    "cls_norm_n-ecuas_n=128",
]


def compute_test_scores(net, device, test_loader):
    net.eval()
    test_logits = []
    with torch.no_grad():
        for test_inputs, _ in tqdm(test_loader, desc="Computing test scores"):
            test_inputs = test_inputs.to(device)
            logits = net(test_inputs)
            test_logits.append(logits.cpu())
    test_logits = torch.cat(test_logits, dim=0)

    return pd.DataFrame(test_logits.numpy())


def compute_metrics(scores, labels, metrics):
    results = []
    for metric in metrics:
        logits_arr = scores.values.copy()
        logits_tensor = torch.from_numpy(logits_arr).float()
        labels_arr = labels.values.reshape(-1).copy()
        labels_tensor = torch.from_numpy(labels_arr).long()
        if "cls" in metric and "conf" not in metric:
            inpt, tgt = logits_tensor, labels_tensor
        elif "conf" in metric and "cls" not in metric:
            inpt, idx = torch.softmax(logits_tensor, dim=1).max(dim=1)
            tgt = (idx == labels_tensor).long()
        metric_dict = get_metric_from_id(metric)
        metric_value = metric_dict["function"](inpt, tgt)
        results.append({"metric": metric_dict["display"], "value": metric_value})
    results = pd.DataFrame(results)
    return results


def main(
    batch_size: int = 128,
    root_outputs_dir: Path = Path("outputs/ood_analysis"),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = []

    for model in ["resnet20", "vgg19_bn", "repvgg_a2"]:
        for dataset in ["cifar10", "cifar100"]:
            net = torch.hub.load("checkpoints/chenyaofo/pytorch-cifar-models", f"{dataset}_{model}", pretrained=True, trust_repo=True, source="local")
            net = net.to(device)

            if dataset == "cifar10":
                datamodule = CIFAR10DataModule(root="./data", batch_size=batch_size, num_workers=2, eval_shift=True, shift_severity=5, eval_ood=False)
            elif dataset == "cifar100":
                datamodule = CIFAR100DataModule(root="./data", batch_size=batch_size, num_workers=2, eval_shift=True, shift_severity=5, eval_ood=False)
            datamodule.prepare_data()

            # Define the outputs directory for the current model, dataset, and shift severity
            outputs_dir = root_outputs_dir / model / dataset
            outputs_dir.mkdir(parents=True, exist_ok=True)
            scores_path = outputs_dir / f"scores.csv"
            labels_path = outputs_dir / f"labels.csv"

            datamodule.setup(stage="test")
            _, test_loader = datamodule.test_dataloader()

            if scores_path.exists():
                print(
                    f"Scores already exist at {scores_path}. Skipping computation."
                )
                scores = pd.read_csv(scores_path, header=0, index_col=0)
                test_labels_df = pd.read_csv(labels_path, header=0, index_col=0)
            else:
                test_labels = torch.cat([item[1] for item in test_loader], dim=0)
                test_labels_df = pd.DataFrame(test_labels.numpy(), columns=["label"])

                # Compute test scores
                scores = compute_test_scores(net, device, test_loader)
                scores.to_csv(scores_path, index=True)
                test_labels_df.to_csv(labels_path, index=True)

                # Compute metrics and save results
            results = compute_metrics(scores, test_labels_df, metrics=EVAL_METRICS)
            results["model"] = model2display[model]
            results["dataset"] = dataset2display[dataset]
            all_results.append(results)

    df_all_results = pd.concat(all_results, axis=0)
    df_all_results = df_all_results.pivot_table(
        index=["dataset", "model"],
        columns="metric",
        values="value",
    )
    df_all_results = df_all_results[[get_metric_from_id(m)["display"] for m in EVAL_METRICS]]
    df_all_results = df_all_results.reset_index()
    df_all_results.to_csv(root_outputs_dir / f"results.csv", index=False)
    df_all_results.to_markdown(root_outputs_dir / f"results.md", index=True)
    generate_latex_table(df_all_results, root_outputs_dir / f"results")


def generate_latex_table(df_all_results, output_filename: Path):

    latex_doc = df_all_results.to_latex(
        index=True,
        float_format="%.4f",
        multirow=True,
        multicolumn=True,
        index_names=True,
        column_format="l" * (len(df_all_results.columns) + 2),
        escape=False,
    )

    standalone_pdf_doc = (
        r"\documentclass{standalone}" + "\n"
        r"\usepackage{booktabs}" + "\n"
        r"\usepackage{multirow}" + "\n"
        r"\begin{document}" + "\n"
        f"{latex_doc}"
        r"\end{document}"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "table.tex")

        with open(tex_path, "w") as f:
            f.write(standalone_pdf_doc)

        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "table.tex"],
                cwd=tmpdir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            raise RuntimeError("LaTeX compilation failed")

        generated_pdf = os.path.join(tmpdir, "table.pdf")
        if not os.path.exists(generated_pdf):
            raise RuntimeError("PDF was not generated")

        os.replace(generated_pdf, output_filename.with_suffix(".pdf"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OOD Analysis")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--outputs_dir",
        type=Path,
        default=Path("outputs/ood_analysis"),
        help="Directory to save outputs",
    )
    args = parser.parse_args()

    main(
        batch_size=args.batch_size,
        root_outputs_dir=args.outputs_dir,
    )
