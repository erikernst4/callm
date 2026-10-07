"""
Helpers shared by the modules that produce per-sample answers with a
confidence: reading gold answers from batches, writing ``llm_outputs.csv`` /
``metrics.csv`` and computing calibration metrics.
"""

import csv
import os

import numpy as np
import torch
from ecuas import (
    ConfidenceAUCScore,
    ConfidenceBrierScore,
    ConfidenceCrossEntropy,
    ConfidenceECUAS,
    ExpectedCalibrationError,
)
from lightning.pytorch import LightningModule


def get_gold_answers(batch: dict) -> list:
    """Gold answers from ``gold_answers`` or ``label``, or ``None`` per question."""
    if "gold_answers" in batch:
        return batch["gold_answers"]
    if "label" in batch:
        return batch["label"]
    return [None] * len(batch["question"])


def format_gold_answers(gold_answers) -> str:
    """``|``-joined list (TriviaQA) or plain string (MMLU) for CSV output."""
    if isinstance(gold_answers, list):
        return "|".join(str(a) for a in gold_answers)
    return str(gold_answers or "")


def format_confidence(confidence) -> str:
    """Confidence with 6 decimals, or ``"nan"`` when missing or not numeric."""
    try:
        return f"{float(confidence):.6f}"
    except (TypeError, ValueError):
        return "nan"


def format_text(txt: str | None, limit: int | None = None) -> str:
    """Escape newlines, truncating to ``limit`` characters (plus ``...``) if given."""
    if not txt:
        return ""
    if limit is not None and len(txt) > limit:
        txt = txt[:limit] + "..."
    return txt.replace("\n", "\\n")


def write_csv(path: str, header: list, rows: list[list], description: str):
    """Write ``header`` + ``rows`` to ``path``, reporting success or failure."""
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"{description} saved to {path}")
    except Exception as e:
        print(f"Failed to save {description}: {e}")


def write_llm_outputs_csv(
    outputs: list[dict], path: str, raw_output_max_length: int | None = None
):
    """Write answers and confidences to ``llm_outputs.csv`` for the evaluator."""
    has_choices = any("choices" in out for out in outputs)
    header = ["question", "gold_answers"]
    if has_choices:
        header.append("choices")
    header += ["pred_answer", "confidence", "raw_output"]

    rows = []
    for out in outputs:
        row = [out["question"], format_gold_answers(out["gold_answers"])]
        if has_choices:
            row.append(out.get("choices", ""))
        row += [
            out["pred_answer"],
            format_confidence(out["confidence"]),
            format_text(out["raw_output"], raw_output_max_length),
        ]
        rows.append(row)
    write_csv(path, header, rows, "LLM outputs")


def write_evaluation_results_csv(
    outputs: list[dict], path: str, text_limit: int | None, description: str
):
    """Write judged outputs to ``evaluation_results.csv``."""
    rows = [
        [
            format_text(out["question"], text_limit),
            out["gold_answers"],
            format_text(out["pred_answer"], text_limit),
            format_confidence(out["confidence"]),
            "Yes" if out["correct"] else "No",
            format_text(out.get("evaluator_response", ""), text_limit),
            format_text(out["raw_output"], text_limit),
        ]
        for out in outputs
    ]
    header = [
        "Question",
        "Gold Answers",
        "Predicted Answer",
        "Confidence",
        "Correct",
        "Evaluator Response",
        "Raw Output",
    ]
    write_csv(path, header, rows, description)


def letter_match(pred_answer: str | None, gold_answers) -> bool:
    """Case-insensitive exact match of a short answer (e.g. an MMLU letter)."""
    pred = (pred_answer or "").strip().upper()
    if isinstance(gold_answers, list):
        return any(pred == g.strip().upper() for g in gold_answers)
    return pred == (gold_answers or "").strip().upper()


def answer_metrics() -> dict:
    """Calibration metrics for answers with confidences."""
    return {
        "val_ece": ExpectedCalibrationError(n_bins=10),
        "val_brier_score": ConfidenceBrierScore(),
        "val_cross_entropy": ConfidenceCrossEntropy(),
        "val_auc": ConfidenceAUCScore(),
        "val_ecuas_0": ConfidenceECUAS(n=0),
    }


def probe_metrics() -> dict:
    """Calibration metrics for probe validation."""
    return {
        "val_ece": ExpectedCalibrationError(n_bins=10),
        "val_cross_entropy": ConfidenceCrossEntropy(),
        "val_auc": ConfidenceAUCScore(),
        "val_ecuas_0": ConfidenceECUAS(n=0),
        "val_ecuas_1": ConfidenceECUAS(n=1),
        "val_ecuas_128": ConfidenceECUAS(n=128),
    }


def compute_metrics(
    confidences: torch.Tensor, correctness: torch.Tensor, metrics: dict
) -> dict[str, float]:
    results = {}
    for name, metric in metrics.items():
        metric.update(confidences, correctness)
        results[name] = float(metric.compute())
    return results


def confidences_and_correctness(outputs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Arrays of ``confidence`` (NaN when not numeric) and ``correct`` from
    ``outputs``, warning about invalid confidences.
    """
    confidences = []
    for out in outputs:
        try:
            confidences.append(float(out["confidence"]))
        except (ValueError, TypeError):
            confidences.append(float("nan"))
    confidences = np.array(confidences)
    correctness = np.array([out["correct"] for out in outputs])

    n_invalid = np.isnan(confidences).sum()
    if n_invalid > 0:
        print(
            f"Warning: {n_invalid} samples have invalid confidence (NaN). "
            "Using fallback confidence of 0.5 for these samples."
        )
    return confidences, correctness


def log_answer_metrics(
    module: LightningModule, outputs: list[dict], save_csv: bool = True
) -> dict[str, float]:
    """
    Log accuracy and calibration metrics of ``outputs`` (with ``confidence``
    and ``correct`` keys) for ``module``, saving them to ``metrics.csv`` in the
    trainer's log directory if ``save_csv``.
    """
    confidences, correctness = confidences_and_correctness(outputs)
    accuracy = float(correctness.mean()) if len(correctness) > 0 else 0.0

    # NaN confidences are handled internally by the metrics (fallback to 0.5)
    results = {"val_accuracy": accuracy} | compute_metrics(
        torch.tensor(confidences, dtype=torch.float32),
        torch.tensor(correctness, dtype=torch.float32),
        answer_metrics(),
    )
    for name, value in results.items():
        module.log(name, value, prog_bar=True, sync_dist=True)

    if save_csv:
        log_dir = module.trainer.log_dir or os.getcwd()
        write_csv(
            os.path.join(log_dir, "metrics.csv"),
            list(results.keys()),
            [list(results.values())],
            "Metrics",
        )
    return results


def read_evaluation_results(csv_path: str) -> list[dict]:
    """Read an ``evaluation_results.csv`` back into output dicts."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    outputs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                confidence = float(row["Confidence"])
            except (ValueError, KeyError):
                confidence = float("nan")
            outputs.append(
                {
                    "confidence": confidence,
                    "correct": row.get("Correct", "").strip().lower() == "yes",
                    "question": row.get("Question", ""),
                    "gold_answers": row.get("Gold Answers", ""),
                    "pred_answer": row.get("Predicted Answer", ""),
                }
            )
    print(f"Loaded {len(outputs)} results from {csv_path}")
    return outputs
