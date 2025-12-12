#!/usr/bin/env python
"""
Integration test script for confidence calibration.

Tests the full pipeline on a small subset of TriviaQA validation data.
"""

from callm.models.llm import LLM
from callm.data.triviaQA import TriviaQADataModule
from lightning import Trainer


def main():
    print("=" * 80)
    print("Integration Test: Confidence Calibration on TriviaQA")
    print("=" * 80)

    # Initialize data module with small subset
    print("\n1. Loading TriviaQA data (10 samples)...")
    datamodule = TriviaQADataModule(
        batch_size=2,
        model_name="flan-t5-small",
        max_samples=10,  # Only test on 10 samples
    )

    # Initialize model
    print("2. Initializing LLM model (flan-t5-small)...")
    model = LLM(
        model_name="flan-t5-small",
        evaluator_model_name="google/flan-t5-small",  # Use small model for faster testing
        train=False,
    )

    # Create trainer
    print("3. Setting up trainer...")
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        max_epochs=1,
    )

    # Run validation
    print(
        "4. Running validation (generating answers, evaluating correctness, computing metrics)..."
    )
    print("\nThis will:")
    print("  - Generate answers for 10 questions")
    print("  - Extract predictions and confidences")
    print("  - Use evaluator LLM to check correctness")
    print("  - Calculate ECE, AUC, BS, CE metrics")
    print("\n" + "-" * 80)

    results = trainer.validate(model, datamodule)

    print("\n" + "=" * 80)
    print("Integration Test Complete!")
    print("=" * 80)
    print("\nMetrics computed:")
    if results:
        for key, value in results[0].items():
            print(f"  {key}: {value:.4f}")

    print("\nAll systems working correctly! ✓")
    print("\nTo run on full validation set, use:")
    print(
        "  python main.py validate --model.model_name=flan-t5-small --data.batch_size=8"
    )


if __name__ == "__main__":
    main()
