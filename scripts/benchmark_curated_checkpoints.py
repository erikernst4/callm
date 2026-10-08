"""
High-Speed Benchmark Engine for Curated TriviaQA Probes.

Loads pre-extracted activations once into memory, selects required tokens / scopes,
and runs forward passes for all trained probe checkpoints (exp7 attentive pooling
and exp8 exact tokens), computing full calibration metrics:
- AUC-ROC
- Expected Calibration Error (ECE, 10 bins)
- Brier Score
- Confidence Cross-Entropy
- CCAS
- Accuracy
"""

import os
import csv
import glob
import ast
import argparse
import numpy as np
import torch
import torch.nn as nn

from callm.utils import (
    get_tokenizer_for_model,
    get_indices_of_exact_answer,
    check_exact_match,
)
from callm.models import (
    Probe,
    MeanPooling,
    AttentivePooling,
    QueryAttentivePooling,
    MultiHeadAttentivePooling,
    QueryMultiHeadAttentivePooling,
    TransformerPooling,
    LayerWeightedAverage,
)
from ecuas import (
    ExpectedCalibrationError,
    ConfidenceBrierScore,
    ConfidenceCrossEntropy,
    ConfidenceAUCScore,
    ConfidenceECUAS,
)


def get_seq_pooling_module(method: str, model_dim: int = 4096) -> nn.Module:
    if method == "mean":
        return MeanPooling()
    elif method == "attentive":
        return AttentivePooling(model_dim=model_dim)
    elif method == "query_attentive":
        return QueryAttentivePooling(model_dim=model_dim)
    elif method == "multi_head_attentive":
        return MultiHeadAttentivePooling(model_dim=model_dim)
    elif method == "query_multi_head_attentive":
        return QueryMultiHeadAttentivePooling(model_dim=model_dim)
    elif method == "transformer":
        return TransformerPooling(model_dim=model_dim)
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def parse_gold_answers(gold_str: str) -> list[str]:
    if not gold_str:
        return []
    try:
        gold_list = ast.literal_eval(gold_str)
        if isinstance(gold_list, list):
            return [str(g) for g in gold_list]
    except (ValueError, SyntaxError):
        pass
    if "|" in gold_str:
        return [g.strip() for g in gold_str.split("|")]
    return [gold_str]


def compute_metrics(confidences: np.ndarray, correctness: np.ndarray) -> dict:
    conf_t = torch.tensor(confidences, dtype=torch.float32)
    corr_t = torch.tensor(correctness, dtype=torch.float32)

    ece_fn = ExpectedCalibrationError(n_bins=10)
    brier_fn = ConfidenceBrierScore()
    ce_fn = ConfidenceCrossEntropy()
    auc_fn = ConfidenceAUCScore()
    ecuas_0_fn = ConfidenceECUAS(n=0)
    ecuas_1_fn = ConfidenceECUAS(n=1)
    ecuas_128_fn = ConfidenceECUAS(n=128)

    ece_fn.update(conf_t, corr_t)
    brier_fn.update(conf_t, corr_t)
    ce_fn.update(conf_t, corr_t)
    auc_fn.update(conf_t, corr_t)
    ecuas_0_fn.update(conf_t, corr_t)
    ecuas_1_fn.update(conf_t, corr_t)
    ecuas_128_fn.update(conf_t, corr_t)

    def _val(x):
        return float(x.item()) if hasattr(x, "item") else float(x)

    return {
        "accuracy": float(correctness.mean()),
        "auc": _val(auc_fn.compute()),
        "ece": _val(ece_fn.compute()),
        "brier_score": _val(brier_fn.compute()),
        "cross_entropy": _val(ce_fn.compute()),
        "ECUAS_n=0": _val(ecuas_0_fn.compute()),
        "ECUAS_n=1": _val(ecuas_1_fn.compute()),
        "ECUAS_n=128": _val(ecuas_128_fn.compute()),
    }


def load_probe(
    ckpt_path: str, model_dim: int = 4096, seq_pooling=None, layer_pooling=None
) -> Probe:
    probe = Probe(
        model_dim=model_dim, layer_pooling=layer_pooling, seq_pooling=seq_pooling
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    probe_state = {
        k.replace("probe.", "", 1): v
        for k, v in state_dict.items()
        if k.startswith("probe.")
    }
    probe.load_state_dict(probe_state)
    probe.eval()
    return probe


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark probe checkpoints on curated TriviaQA."
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        default="activations_cache/curated_triviaqa",
        help="Directory containing extracted activations shards",
    )
    parser.add_argument(
        "--exact_csv",
        type=str,
        default=None,
        help="Path to evaluation_results_exact.csv. If not provided, will look in activations_dir.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/curated_benchmark",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3.5-9B",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--generate_table_only",
        action="store_true",
        help="Only generate LaTeX tables from existing benchmark_summary.csv without re-running benchmark",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary_csv = os.path.join(args.output_dir, "benchmark_summary.csv")

    if args.generate_table_only:
        if not os.path.exists(summary_csv):
            raise FileNotFoundError(f"benchmark_summary.csv not found at {summary_csv}")
        out_tex_lt = os.path.join(args.output_dir, "benchmark_table.tex")
        out_tex_tab = os.path.join(args.output_dir, "benchmark_table_tabular.tex")
        generate_latex_table(summary_csv, output_tex=out_tex_lt, table_type="longtable")
        generate_latex_table(summary_csv, output_tex=out_tex_tab, table_type="tabular")
        return
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. Resolve exact CSV
    exact_csv = args.exact_csv
    if exact_csv is None:
        exact_candidates = glob.glob(
            os.path.join(args.activations_dir, "**/evaluation_results_exact.csv"),
            recursive=True,
        )
        if exact_candidates:
            exact_csv = sorted(exact_candidates, key=os.path.getmtime)[-1]
    if exact_csv is None or not os.path.exists(exact_csv):
        # Fallback to checking version_26 if available
        fallback_v26 = "activations_cache/qwen_triviaqa/lightning_logs/version_26/evaluation_results_exact.csv"
        if os.path.exists(fallback_v26):
            print(f"Using exact answers from fallback: {fallback_v26}")
            exact_csv = fallback_v26
        else:
            raise FileNotFoundError(
                f"Could not find evaluation_results_exact.csv in {args.activations_dir}"
            )

    print(f"Exact CSV: {exact_csv}")

    # 2. Load exact CSV metadata
    exact_info = {}
    with open(exact_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        field_map = {col.lower(): col for col in reader.fieldnames or []}
        correct_col = field_map.get("correct")
        ea_col = field_map.get("exact answer") or field_map.get("exact_answer")
        vea_col = field_map.get("valid exact answer") or field_map.get(
            "valid_exact_answer"
        )
        q_col = field_map.get("question")
        r_col = field_map.get("raw output") or field_map.get("raw_output")
        g_col = field_map.get("gold answers") or field_map.get("gold_answers")

        for idx, row in enumerate(reader):
            gold_answers = parse_gold_answers(row.get(g_col, ""))
            pred_ans = row.get(
                field_map.get("predicted answer") or field_map.get("predicted_answer"),
                "",
            )

            # Ground truth correctness by exact match
            if pred_ans and gold_answers:
                is_correct = check_exact_match(pred_ans, gold_answers)
            else:
                is_correct = row.get(correct_col, "").strip().lower() == "yes"

            ea = row.get(ea_col, "") if ea_col else ""
            vea = (
                (row.get(vea_col, "").strip().lower() in ("yes", "true", "1"))
                if vea_col
                else False
            )
            question = row.get(q_col, "") if q_col else ""
            raw_out = row.get(r_col, "") if r_col else ""

            exact_info[idx] = {
                "correct": is_correct,
                "exact_answer": ea,
                "valid_exact_answer": vea,
                "question": question,
                "raw_output": raw_out,
            }

    num_samples = len(exact_info)
    ground_truth = np.array(
        [exact_info[i]["correct"] for i in range(num_samples)], dtype=bool
    )
    print(
        f"Loaded {num_samples} samples from CSV. Accuracy: {ground_truth.mean() * 100:.2f}%"
    )

    # 3. Load activation shards
    shard_files = sorted(
        glob.glob(os.path.join(args.activations_dir, "activations_*.pt"))
    )
    if not shard_files:
        raise FileNotFoundError(f"No activation shards found in {args.activations_dir}")

    print(f"Loading {len(shard_files)} activation shard(s)...")
    raw_samples = []
    for sf in shard_files:
        data = torch.load(sf, map_location="cpu", weights_only=False)
        raw_samples.extend(data)
    print(f"Loaded {len(raw_samples)} raw sample activations.")

    # Check available layers
    sample_0 = raw_samples[0]
    available_layers = sorted(list(sample_0["activations"].keys()))
    print(f"Available layers in activations: {available_layers}")
    model_dim = sample_0["activations"][available_layers[0]].shape[-1]
    print(f"Model dimension: {model_dim}")

    # Tokenizer and prompt for exact token slicing
    tokenizer = get_tokenizer_for_model(args.model_name)
    from callm.prompts import TRANSPARENT_PROMPT_ZERO_SHOT

    # Pre-compute token slices for each window across all samples
    windows = [
        "exact_answer",
        "exact_answer_first_token",
        "exact_answer_last_token",
        "exact_answer_before_first_token",
        "exact_answer_after_last_token",
    ]

    print("Pre-computing exact token indices...")
    sample_window_slices = {w: [] for w in windows}

    for idx, sample in enumerate(raw_samples):
        info = exact_info.get(idx, {})
        prompt_len = sample.get("prompt_length", sample["seq_len"])
        seq_len = sample["seq_len"]

        exact_range = None
        if info.get("valid_exact_answer") and info.get("exact_answer"):
            prompt_text = TRANSPARENT_PROMPT_ZERO_SHOT(question=info["question"])
            prompt_tokens = tokenizer.apply_chat_template(
                prompt_text,
                tokenize=True,
                enable_thinking=False,
                add_generation_prompt=True,
                return_dict=True,
            )
            prompt_ids = prompt_tokens["input_ids"]
            if torch.is_tensor(prompt_ids):
                prompt_ids = prompt_ids.squeeze().tolist()

            answer_ids = tokenizer.encode(info["raw_output"], add_special_tokens=False)
            if torch.is_tensor(answer_ids):
                answer_ids = answer_ids.squeeze().tolist()

            full_ids = prompt_ids + answer_ids
            exact_range = get_indices_of_exact_answer(
                tokenizer, full_ids, info["exact_answer"], len(prompt_ids)
            )

        for w in windows:
            if exact_range is not None:
                lower = exact_range[0]
                higher = exact_range[-1]
                if w == "exact_answer":
                    sl = (lower, higher + 1)
                elif w == "exact_answer_first_token":
                    sl = (lower, lower + 1)
                elif w == "exact_answer_last_token":
                    sl = (higher, higher + 1)
                elif w == "exact_answer_before_first_token":
                    sl = (max(0, lower - 1), max(1, lower))
                elif w == "exact_answer_after_last_token":
                    sl = (min(seq_len - 1, higher + 1), min(seq_len, higher + 2))
            else:
                # Fallback to answer tokens without prompt
                sl = (prompt_len, seq_len)
            sample_window_slices[w].append(sl)

    results_rows = []

    # =========================================================================
    # Benchmark 1: Attentive Pooling Probes (exp7)
    # =========================================================================
    pooling_methods = [
        "mean",
        "attentive",
        "query_attentive",
        "multi_head_attentive",
        "query_multi_head_attentive",
        "transformer",
    ]

    print("\n" + "=" * 80)
    print(" Benchmarking Exp 7: Attentive Pooling Probes")
    print("=" * 80)

    for method in pooling_methods:
        method_dir = os.path.join("results/exp7_attentive_pooling_50k", method)
        if not os.path.exists(method_dir):
            continue

        # Find layer checkpoints
        for layer_idx in available_layers:
            layer_dir = os.path.join(method_dir, f"layer_{layer_idx}")
            ckpts = glob.glob(
                os.path.join(layer_dir, "**/checkpoints/*.ckpt"), recursive=True
            )
            if not ckpts:
                continue
            ckpt_path = sorted(ckpts, key=os.path.getmtime)[-1]

            seq_pooling = get_seq_pooling_module(method, model_dim=model_dim)
            probe = load_probe(
                ckpt_path, model_dim=model_dim, seq_pooling=seq_pooling
            ).to(device)

            confidences = []
            with torch.no_grad():
                for sample in raw_samples:
                    act = (
                        sample["activations"][layer_idx].to(device).unsqueeze(0)
                    )  # (1, seq_len, dim)
                    seq_mask = torch.ones(
                        (1, act.shape[1]), dtype=torch.bool, device=device
                    )
                    conf = probe(act, seq_mask=seq_mask).squeeze().item()
                    confidences.append(conf)

            conf_arr = np.array(confidences)
            m = compute_metrics(conf_arr, ground_truth)
            m.update(
                {
                    "experiment": "exp7_attentive_pooling",
                    "method": method,
                    "window": "both",
                    "layer": layer_idx,
                    "checkpoint": os.path.basename(ckpt_path),
                }
            )
            results_rows.append(m)
            print(
                f"[{method} | Layer {layer_idx:2d}] AUC: {m['auc']:.4f} | ECE: {m['ece']:.4f} | Brier: {m['brier_score']:.4f} | CE: {m['cross_entropy']:.4f} | ECUAS@0: {m['ECUAS_n=0']:.4f} | ECUAS@1: {m['ECUAS_n=1']:.4f} | ECUAS@128: {m['ECUAS_n=128']:.4f}"
            )

    # =========================================================================
    # Benchmark 2: Exact Token Probes (exp8)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" Benchmarking Exp 8: Exact Token Probes")
    print("=" * 80)

    for w in windows:
        window_dir = os.path.join("results/exp8_exact_tokens_probes_50k", w)
        if not os.path.exists(window_dir):
            continue

        # Single layer probes
        for layer_idx in available_layers:
            layer_dir = os.path.join(window_dir, f"layer_{layer_idx}")
            ckpts = glob.glob(
                os.path.join(layer_dir, "**/checkpoints/*.ckpt"), recursive=True
            )
            if not ckpts:
                continue
            ckpt_path = sorted(ckpts, key=os.path.getmtime)[-1]

            probe = load_probe(ckpt_path, model_dim=model_dim, seq_pooling=None).to(
                device
            )

            confidences = []
            with torch.no_grad():
                for idx, sample in enumerate(raw_samples):
                    sl = sample_window_slices[w][idx]
                    act = sample["activations"][layer_idx][
                        sl[0] : sl[1]
                    ]  # (window_len, dim)
                    if act.shape[0] == 0:
                        act = sample["activations"][layer_idx].mean(dim=0, keepdim=True)
                    else:
                        act = act.mean(dim=0, keepdim=True)  # mean pool over window
                    act = act.unsqueeze(0).to(device)  # (1, 1, dim)
                    conf = probe(act).squeeze().item()
                    confidences.append(conf)

            conf_arr = np.array(confidences)
            m = compute_metrics(conf_arr, ground_truth)
            m.update(
                {
                    "experiment": "exp8_exact_tokens",
                    "method": "linear_mean",
                    "window": w,
                    "layer": layer_idx,
                    "checkpoint": os.path.basename(ckpt_path),
                }
            )
            results_rows.append(m)
            print(
                f"[{w:30s} | Layer {layer_idx:2d}] AUC: {m['auc']:.4f} | ECE: {m['ece']:.4f} | Brier: {m['brier_score']:.4f} | CE: {m['cross_entropy']:.4f} | ECUAS@0: {m['ECUAS_n=0']:.4f} | ECUAS@1: {m['ECUAS_n=1']:.4f} | ECUAS@128: {m['ECUAS_n=128']:.4f}"
            )

        # Weighted average probes across layers
        weighted_dir = os.path.join(window_dir, "weighted")
        w_ckpts = glob.glob(
            os.path.join(weighted_dir, "**/checkpoints/*.ckpt"), recursive=True
        )
        if w_ckpts:
            ckpt_path = sorted(w_ckpts, key=os.path.getmtime)[-1]
            num_layers = len(available_layers)
            layer_pooling = LayerWeightedAverage(num_layers=num_layers)
            probe = load_probe(
                ckpt_path, model_dim=model_dim, layer_pooling=layer_pooling
            ).to(device)

            confidences = []
            with torch.no_grad():
                for idx, sample in enumerate(raw_samples):
                    sl = sample_window_slices[w][idx]
                    # Stack across available layers: (num_layers, window_len, dim)
                    layer_tensors = []
                    for layer in available_layers:
                        act_l = sample["activations"][layer][sl[0] : sl[1]]
                        if act_l.shape[0] == 0:
                            act_l = sample["activations"][layer].mean(
                                dim=0, keepdim=True
                            )
                        else:
                            act_l = act_l.mean(dim=0, keepdim=True)
                        layer_tensors.append(act_l)
                    multi_layer_act = (
                        torch.cat(layer_tensors, dim=0).unsqueeze(0).to(device)
                    )  # (1, num_layers, dim)
                    conf = probe(multi_layer_act).squeeze().item()
                    confidences.append(conf)

            conf_arr = np.array(confidences)
            m = compute_metrics(conf_arr, ground_truth)
            m.update(
                {
                    "experiment": "exp8_exact_tokens",
                    "method": "weighted_average",
                    "window": w,
                    "layer": "weighted",
                    "checkpoint": os.path.basename(ckpt_path),
                }
            )
            results_rows.append(m)
            print(
                f"[{w:30s} | Weighted] AUC: {m['auc']:.4f} | ECE: {m['ece']:.4f} | Brier: {m['brier_score']:.4f} | CE: {m['cross_entropy']:.4f} | ECUAS@0: {m['ECUAS_n=0']:.4f} | ECUAS@1: {m['ECUAS_n=1']:.4f} | ECUAS@128: {m['ECUAS_n=128']:.4f}"
            )

    # =========================================================================
    # Save Master Benchmark Results
    # =========================================================================
    summary_csv = os.path.join(args.output_dir, "benchmark_summary.csv")
    if results_rows:
        fieldnames = [
            "method",
            "window",
            "layer",
            "accuracy",
            "auc",
            "ece",
            "brier_score",
            "cross_entropy",
            "ECUAS_n=0",
            "ECUAS_n=1",
            "ECUAS_n=128",
        ]
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results_rows)

        print("\n" + "=" * 80)
        print(
            f" Benchmark Complete! Evaluated {len(results_rows)} probe configurations."
        )
        print(f" Master Summary saved to: {summary_csv}")
        print("=" * 80)

        # Print top 10 models by AUC
        print("\nTop 10 Models by AUC:")
        sorted_by_auc = sorted(results_rows, key=lambda x: x["auc"], reverse=True)[:10]
        for r in sorted_by_auc:
            print(
                f"  {r['method']:26s} | {r['window']:24s} | Layer {str(r['layer']):8s} | AUC: {r['auc']:.4f} | ECE: {r['ece']:.4f} | ECUAS_0: {r['ECUAS_n=0']:.4f}"
            )

        # Print top 10 models by ECE (lowest is best)
        print("\nTop 10 Models by ECE (Calibration):")
        sorted_by_ece = sorted(results_rows, key=lambda x: x["ece"])[:10]
        for r in sorted_by_ece:
            print(
                f"  {r['method']:26s} | {r['window']:24s} | Layer {str(r['layer']):8s} | ECE: {r['ece']:.4f} | AUC: {r['auc']:.4f} | ECUAS_0: {r['ECUAS_n=0']:.4f}"
            )

        out_tex_lt = os.path.join(args.output_dir, "benchmark_table.tex")
        out_tex_tab = os.path.join(args.output_dir, "benchmark_table_tabular.tex")
        generate_latex_table(summary_csv, output_tex=out_tex_lt, table_type="longtable")
        generate_latex_table(summary_csv, output_tex=out_tex_tab, table_type="tabular")


def escape_latex(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def generate_latex_table(
    csv_path: str,
    output_tex: str | None = None,
    table_type: str = "longtable",
) -> str:
    """
    Generate a LaTeX table from benchmark_summary.csv, omitting experiment and checkpoint.
    Columns: Method, Window, Layer, Accuracy, AUC, ECE, Brier Score, Cross Entropy, ECUAS_n=0, ECUAS_n=1, ECUAS_n=128.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return ""

    metric_cols = [
        ("accuracy", r"\textbf{Acc}", True),
        ("auc", r"\textbf{AUC}", True),
        ("ece", r"\textbf{ECE}", False),
        ("brier_score", r"\textbf{Brier}", False),
        ("cross_entropy", r"\textbf{CE}", False),
        ("ECUAS_n=0", r"\textbf{ECUAS$_{n=0}$}", False),
        ("ECUAS_n=1", r"\textbf{ECUAS$_{n=1}$}", False),
        ("ECUAS_n=128", r"\textbf{ECUAS$_{n=128}$}", False),
    ]

    # Find global best values across all rows
    best_vals = {}
    for col_key, _, higher_is_better in metric_cols:
        vals = []
        for r in rows:
            raw_val = r.get(col_key)
            if raw_val is None and col_key.startswith("ECUAS_n="):
                n_str = col_key.split("=")[-1]
                raw_val = r.get(f"ecuas_{n_str}")
                if raw_val is None and n_str == "0":
                    raw_val = r.get("ccas")
            if raw_val is not None and raw_val != "":
                try:
                    vals.append(float(raw_val))
                except ValueError:
                    pass
        if vals:
            best_vals[col_key] = max(vals) if higher_is_better else min(vals)

    def _fmt(col_k: str, row: dict) -> str:
        raw = row.get(col_k)
        if raw is None and col_k.startswith("ECUAS_n="):
            n_str = col_k.split("=")[-1]
            raw = row.get(f"ecuas_{n_str}")
            if raw is None and n_str == "0":
                raw = row.get("ccas")
        if raw is None or raw == "":
            return "-"
        try:
            fv = float(raw)
            formatted = f"{fv:.4f}"
            if col_k in best_vals and abs(fv - best_vals[col_k]) < 1e-6:
                return r"\textbf{" + formatted + "}"
            return formatted
        except ValueError:
            return escape_latex(raw)

    col_spec = "lll" + "c" * len(metric_cols)
    headers = [r"\textbf{Method}", r"\textbf{Window}", r"\textbf{Layer}"] + [
        h for _, h, _ in metric_cols
    ]

    lines = []
    if table_type == "longtable":
        lines.extend(
            [
                r"\begin{longtable}{" + col_spec + "}",
                r"\caption{Curated Benchmark Results for TriviaQA Probes. Best values across all configurations are \textbf{bolded}.} \label{tab:curated_benchmark} \\",
                r"\toprule",
                " & ".join(headers) + r" \\",
                r"\midrule",
                r"\endfirsthead",
                r"\multicolumn{"
                + str(len(headers))
                + r"}{c}{\tablename\ \thetable\ -- \textit{Continued from previous page}} \\",
                r"\toprule",
                " & ".join(headers) + r" \\",
                r"\midrule",
                r"\endhead",
                r"\midrule",
                r"\multicolumn{"
                + str(len(headers))
                + r"}{r}{\textit{Continued on next page}} \\",
                r"\endfoot",
                r"\bottomrule",
                r"\endlastfoot",
            ]
        )
    else:
        lines.extend(
            [
                r"\begin{table*}[t]",
                r"\centering",
                r"\small",
                r"\resizebox{\textwidth}{!}{%",
                r"\begin{tabular}{" + col_spec + "}",
                r"\toprule",
                " & ".join(headers) + r" \\",
                r"\midrule",
            ]
        )

    prev_group = None
    for r in rows:
        method = r.get("method", "")
        window = r.get("window", "")
        layer = r.get("layer", "")

        curr_group = (method, window)
        if prev_group is not None and curr_group != prev_group:
            lines.append(r"\midrule")
        prev_group = curr_group

        row_cells = [
            escape_latex(method),
            escape_latex(window),
            escape_latex(layer),
        ]
        for col_k, _, _ in metric_cols:
            row_cells.append(_fmt(col_k, r))

        lines.append(" & ".join(row_cells) + r" \\")

    if table_type == "longtable":
        lines.append(r"\end{longtable}")
    else:
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"}",
                r"\caption{Curated Benchmark Results for TriviaQA Probes. Best values across all configurations are \textbf{bolded}.}",
                r"\label{tab:curated_benchmark}",
                r"\end{table*}",
            ]
        )

    tex_code = "\n".join(lines) + "\n"

    if output_tex:
        with open(output_tex, "w", encoding="utf-8") as f:
            f.write(tex_code)
        print(f"Saved LaTeX table ({table_type}) to: {output_tex}")

    return tex_code


if __name__ == "__main__":
    main()
