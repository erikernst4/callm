import os
from pathlib import Path
import subprocess
import tempfile

import pandas as pd
import torch
from ecuas import get_metric_from_id
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch_uncertainty.datamodules import CIFAR10DataModule, CIFAR100DataModule

from scripts.merge_mmlu_istrue import compute_metrics

dataset2display = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
}

model2display = {
    "resnet20": "ResNet-20",
    "vgg19_bn": "VGG-19",
    "repvgg_a2": "RepVGG-A2",
}


EVAL_METRICS = [
    "cls_ner",
    "cls_nbs",
    "cls_nce",
    "cls_auc",
    "cls_aurc",
    "cls_ece_nbins=10",
    # "cls_norm_loglog",
    "cls_norm_n-ecuas_n=0",
    "cls_norm_n-ecuas_n=1",
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

    # Save scores to disk
    return pd.DataFrame(test_logits.numpy())

def compute_metrics(scores, labels, metrics):
    results = []
    for metric in metrics:
        metric_dict = get_metric_from_id(metric)
        metric_fn = metric_dict["function"]
        logits_arr = scores.values.copy()
        logits_tensor = torch.from_numpy(logits_arr).float()
        labels_arr = labels.values.reshape(-1).copy()
        labels_tensor = torch.from_numpy(labels_arr).long()
        metric_value = metric_fn(logits_tensor, labels_tensor)
        results.append(
            {"metric": metric_dict["display"], "value": metric_value}
        )
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

            for shift_severity in [5]:

                if dataset == "cifar10":
                    datamodule = CIFAR10DataModule(root="./data", batch_size=batch_size, num_workers=8, eval_shift=True, shift_severity=shift_severity, eval_ood=True)
                elif dataset == "cifar100":
                    datamodule = CIFAR100DataModule(root="./data", batch_size=batch_size, num_workers=8, eval_shift=True, shift_severity=shift_severity, eval_ood=True)
                datamodule.prepare_data()

                # Define the outputs directory for the current model, dataset, and shift severity
                outputs_dir = root_outputs_dir / model / dataset
                outputs_dir.mkdir(parents=True, exist_ok=True)

                datamodule.setup(stage="test")
                id_test_loader, ood_test_loader, shift_test_loader = datamodule.test_dataloader()
                test_loaders = {
                    "ID": id_test_loader,
                    "OOD": ood_test_loader,
                    "Shift": shift_test_loader,
                }

                print(f"Computing for model {model} and dataset {dataset} with shift severity {shift_severity}...")

                # for prefix in ["ID", "OOD", "Shift"]:
                for prefix in ["ID", "Shift"]:

                    if prefix == "ID":
                        results_path = outputs_dir / f"results_ID.csv"
                        dataset_base = dataset2display[dataset]
                        dataset_type = "ID"
                    elif prefix == "OOD":
                        results_path = outputs_dir / f"results_OOD.csv"
                        dataset_base = dataset2display[dataset]
                        dataset_type = "OOD"
                    elif prefix == "Shift":
                        results_path = outputs_dir / f"results_Shift_severity={shift_severity}.csv"
                        dataset_base = dataset2display[dataset]
                        dataset_type = f"Shift (severity={shift_severity})"

                    if results_path.exists():
                        print(f"Results already exist at {results_path}. Skipping computation.")
                        results = pd.read_csv(results_path)

                    else:
                        test_loader = test_loaders[prefix]
                        test_labels = torch.cat([item[1] for item in test_loader], dim=0)
                        test_labels_df = pd.DataFrame(test_labels.numpy(), columns=["label"])

                        # Compute test scores
                        scores = compute_test_scores(net, device, test_loader)

                        # Compute metrics and save results
                        results = compute_metrics(scores, test_labels_df, metrics=EVAL_METRICS)
                        results["model"] = model2display[model]
                        results["dataset-base"] = dataset_base
                        results["dataset-type"] = dataset_type

                        results.to_csv(results_path, index=False)

                    all_results.append(results)

    df_all_results = pd.concat(all_results, axis=0)
    df_all_results = df_all_results.pivot_table(
        index=["dataset-base", "model"],
        columns=["metric", "dataset-type"],
        values="value",
    )
    df_all_results.columns.name = None
    df_all_results.to_csv(root_outputs_dir / f"results.csv", index=False)
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

        # Write LaTeX file
        with open(tex_path, "w") as f:
            f.write(standalone_pdf_doc)

        # Compile LaTeX → PDF
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

        # Move resulting PDF
        generated_pdf = os.path.join(tmpdir, "table.pdf")
        if not os.path.exists(generated_pdf):
            raise RuntimeError("PDF was not generated")

        os.replace(
            generated_pdf, output_filename.with_suffix(".pdf")
        )  # Move generated PDF to desired location


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OOD Analysis")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--outputs_dir", type=Path, default=Path("outputs/ood_analysis"), help="Directory to save outputs")
    args = parser.parse_args()
    

    args = parser.parse_args()

    main(
        batch_size=args.batch_size,
        root_outputs_dir=args.outputs_dir,
    )