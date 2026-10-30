"""
Tests for re-parsing verbalized confidences from saved (truncated) raw outputs.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.reparse_verbalized_confidences import reparse_confidence, reparse_csv


@pytest.mark.parametrize(
    "raw_output, old, expected",
    [
        # Missing confidence recovered from markdown-formatted probability
        ("**Pulsar**\\n\\nProbability: **0.95**", np.nan, 0.95),
        # Old parser skipped the bold value and took one from a later correction
        (
            "**New York**\\nProbability: **0.05**\\n\\n*(Correction: Probability: 1.0)*",
            1.0,
            0.05,
        ),
        # Probability beyond the saved 200 characters: keep the old value
        ("**Lancelot Gobbo** is actually a character from...", 0.95, 0.95),
        # Nothing to parse and nothing before: stays missing
        ("**Lancelot Gobbo** is actually a character from...", np.nan, np.nan),
    ],
)
def test_reparse_confidence(raw_output, old, expected):
    result = reparse_confidence(raw_output, old)
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "conf_col, raw_col",
    [("confidence", "raw_output"), ("Confidence", "Raw Output")],
)
def test_reparse_csv_updates_only_confidence(tmp_path, conf_col, raw_col):
    csv_path = tmp_path / "outputs.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", conf_col, raw_col])
        writer.writerow(["Q1", "nan", "A\\nProbability: **0.8**"])
        writer.writerow(["Q2", "0.3", "B\\nProbability: 0.3"])
        writer.writerow(["Q3", "", "C is..."])

    changed = reparse_csv(csv_path)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert changed == 1
    assert [r[conf_col] for r in rows] == ["0.800000", "0.3", ""]
    assert [r[raw_col] for r in rows] == [
        "A\\nProbability: **0.8**",
        "B\\nProbability: 0.3",
        "C is...",
    ]
