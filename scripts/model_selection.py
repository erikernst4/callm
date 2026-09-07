
from pathlib import Path

import pandas as pd
import torch
import numpy as np
from ecuas import get_metric_from_id



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
        results.append({"metric": metric, "display": metric_dict["display"], "value": metric_value})
    results = pd.DataFrame(results)
    return results

def _compute_chow(scores, labels, gamma):
    probs = torch.softmax(torch.from_numpy(scores.values.copy()).float(), dim=1)
    labels_tensor = torch.from_numpy(labels.values.reshape(-1).copy()).long()
    pred_val, pred_idx = probs.max(dim=1)
    cost01 = (labels_tensor != pred_idx).float()
    u_C = 1 - pred_val
    cost01[u_C > gamma] = gamma[u_C > gamma]
    return cost01

def compute_chow_cost_0(scores, labels, u_M, zero=1e-8, rs=None):
    alpha_n = 1 / u_M 
    u = rs.rand(len(scores))
    log_a, log_b = np.log(zero), np.log(u_M)
    log_gamma = log_a + (log_b - log_a) * u # = log_a * (1 - u) + log_b * u
    gamma = torch.from_numpy(np.exp(log_gamma)).float()
    cost01 = _compute_chow(scores, labels, gamma) * (torch.log(torch.tensor(u_M)) - torch.log(torch.tensor(zero))) * alpha_n
    return cost01.mean().item()

def compute_chow_cost_1(scores, labels, u_M, zero=1e-8, rs=None):
    alpha_n = 2 / (u_M ** 2)
    gamma = rs.rand(len(scores)) * u_M
    gamma = torch.from_numpy(gamma).float()
    cost01 = _compute_chow(scores, labels, gamma) * u_M * alpha_n
    return cost01.mean().item()

def compute_chow_cost_14(scores, labels, u_M, zero=1e-8, rs=None):
    alpha_n = (1/4 + 1) / (u_M ** (1/4 + 1))
    gamma = rs.rand(len(scores)) ** 4 * u_M
    gamma = torch.from_numpy(gamma).float()
    cost01 = _compute_chow(scores, labels, gamma) * np.power(u_M, 1/4) / 4 * alpha_n
    return cost01.mean().item()

def compute_chow_cost_12(scores, labels, u_M, zero=1e-8, rs=None):
    alpha_n = (1/2 + 1) / (u_M ** (1/2 + 1))
    gamma = rs.rand(len(scores)) ** 2 * u_M
    gamma = torch.from_numpy(gamma).float()
    cost01 = _compute_chow(scores, labels, gamma) * np.power(u_M, 1/2) / 2 * alpha_n
    return cost01.mean().item()

def main():
    dataset = "cifar100"
    metrics = ["cls_aurc", "cls_n-ecuas_n=0", "cls_n-ecuas_n=1"]
    models = ["resnet20", "vgg19_bn", "repvgg_a2"]
    model2name = {
        "resnet20": "ResNet-20",
        "vgg19_bn": "VGG-19",
        "repvgg_a2": "RepVGG-A2",
    }

    results_all = []
    for model in models:

        scores_path = Path(f"outputs/ood_analysis/{model}/{dataset}/scores_OOD.csv")
        labels_path = Path(f"outputs/ood_analysis/{model}/{dataset}/labels_OOD.csv")
        if scores_path.exists() and labels_path.exists():
            print(f"Scores already exist at {scores_path}. Skipping computation.")
            scores = pd.read_csv(scores_path, header=0, index_col=0)
            labels = pd.read_csv(labels_path, header=0, index_col=0)
        else:
            raise FileNotFoundError(f"Scores file not found at {scores_path}. Please run the OOD analysis script first.")

        # randomly split the data into 50% train and 50% test
        rs = np.random.RandomState(seed=42)
        shuffle_idx = rs.permutation(len(scores))
        train_size = int(0.5 * len(scores))
        train_idx = shuffle_idx[:train_size]
        test_idx = shuffle_idx[train_size:]
        train_scores = scores.iloc[train_idx]
        train_labels = labels.iloc[train_idx]
        test_scores = scores.iloc[test_idx]
        test_labels = labels.iloc[test_idx]

        train_results = compute_metrics(train_scores, train_labels, metrics)
        train_results["model"] = model2name[model]
        train_results["split"] = "Validation"
        results_all.append(train_results)

        test_results = compute_metrics(test_scores, test_labels, metrics)
        test_results["model"] = model2name[model]
        test_results["split"] = "Test"
        results_all.append(test_results)

        # Compute Chow's cost for n=0
        zero = 1e-40
        K = 10 if dataset == "cifar10" else 100
        u_M = 1 - 1 / K
        # gamma_values = torch.from_numpy(zero * (u_M / zero) ** rs.rand(len(test_scores))).float()
        cost0 = compute_chow_cost_0(test_scores, test_labels, u_M=u_M, zero=zero, rs=rs)
        cost1 = compute_chow_cost_1(test_scores, test_labels, u_M=u_M, zero=zero, rs=rs)
        cost14 = compute_chow_cost_14(test_scores, test_labels, u_M=u_M, zero=zero, rs=rs)
        cost12 = compute_chow_cost_12(test_scores, test_labels, u_M=u_M, zero=zero, rs=rs)
        results_all.extend([
            pd.DataFrame([{
                "model": model2name[model],
                "metric": "chow_cost0",
                "display": "Chow (n=0)",
                "value": cost0,
                "split": "Test"
            }]),
            pd.DataFrame([{
                "model": model2name[model],
                "metric": "chow_cost1",
                "display": "Chow (n=1)",
                "value": cost1,
                "split": "Test"
            }]),
            pd.DataFrame([{
                "model": model2name[model],
                "metric": "chow_cost14",
                "display": "Chow (n=1/4)",
                "value": cost14,
                "split": "Test"
            }]),
            pd.DataFrame([{
                "model": model2name[model],
                "metric": "chow_cost12",
                "display": "Chow (n=1/2)",
                "value": cost12,
                "split": "Test"
            }])
        ])
        
    results = pd.concat(results_all, ignore_index=True)
    table = results.pivot_table(index="model",columns=["split","display"], values="value")
    table = table.loc[:,[
        ("Validation", "AURC"), ("Validation", "ECUAS (n=0)"), ("Validation", "ECUAS (n=1)"),
        ("Test", "AURC"), ("Test", "ECUAS (n=0)"), ("Test", "ECUAS (n=1)"),
        ("Test", "Chow (n=0)"), ("Test", "Chow (n=1/4)"), ("Test", "Chow (n=1/2)"), ("Test", "Chow (n=1)")
    ]]
    table = table.round(4)
    results_path = Path(f"outputs/model_selection/{dataset}/results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path)
    table.to_markdown(results_path.with_suffix(".md"))


if __name__ == "__main__":
    main()