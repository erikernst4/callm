#!/usr/bin/env python3
"""
Standalone script to export per-sample validation predictions (confidence and correctness)
to `val_predictions.csv` using the best checkpoint from a training run.

Usage:
    uv run python scripts/export_val_predictions.py validate --config <config.yaml> --ckpt_path best
"""

import sys
from pathlib import Path

# Add project root to path so we can import cli and callm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cli import CalibrationCLI
from callm.callbacks import SaveValPredictionsCallback


class ExportValPredictionsCLI(CalibrationCLI):
    def instantiate_trainer(self, **kwargs):
        # Instantiate the trainer as usual
        trainer = super().instantiate_trainer(**kwargs)
        # Inject our custom callback to save the predictions
        trainer.callbacks.append(SaveValPredictionsCallback())
        return trainer


def main():
    ExportValPredictionsCLI(
        parser_kwargs={
            "validate": {
                "default_config_files": ["configs/config_base_validation.yaml"]
            },
            "parser_mode": "omegaconf",
        },
    )


if __name__ == "__main__":
    main()
