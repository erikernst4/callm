"""
Tests that the results scripts impute missing confidences with 0.5 when loading, so
every metric (including AURC, which does not impute internally) sees the same values.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import generate_mmlu_table, llm_confidence_results


def _write_csv(path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Confidence", "Correct"])
        writer.writerow(["Q1", "0.9", "Yes"])
        writer.writerow(["Q2", "", "No"])
        writer.writerow(["Q3", "nan", "Yes"])


def test_generate_mmlu_table_imputes_missing_confidences(tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path)
    confidences, correctness = generate_mmlu_table.load_csv_data(csv_path)
    np.testing.assert_array_equal(confidences, [0.9, 0.5, 0.5])
    np.testing.assert_array_equal(correctness, [1.0, 0.0, 1.0])


def test_llm_confidence_results_imputes_missing_confidences(tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path)
    confidences, correctness = llm_confidence_results.load_csv_data(csv_path)
    np.testing.assert_allclose(confidences.numpy(), [0.9, 0.5, 0.5])
    np.testing.assert_array_equal(correctness.numpy(), [1.0, 0.0, 1.0])
