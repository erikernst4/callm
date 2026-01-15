import os
import csv
import torch
import shutil
from unittest.mock import MagicMock
from cli import CalibrationTrainer


def test_resume_feature():
    # Setup test directories
    test_dir = "test_resume_run"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    prev_run_dir = os.path.join(test_dir, "prev_run")
    os.makedirs(prev_run_dir)

    # 1. Create dummy llm_outputs.csv (3 items)
    csv_path = os.path.join(test_dir, "llm_outputs.csv")
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "question",
                "gold_answers",
                "pred_answer",
                "confidence",
                "raw_output",
                "exact_match",
            ]
        )
        writer.writerow(["Q1", "yes", "yes", "0.9", "raw1", "True"])  # Index 0
        writer.writerow(["Q2", "no", "maybe", "0.5", "raw2", "False"])  # Index 1
        writer.writerow(["Q3", "blue", "red", "0.8", "raw3", "False"])  # Index 2

    # 2. Simulate partial run output (processed index 0)
    processed_data = [
        {
            "index": 0,
            "question": "Q1",
            "gold_answers": ["yes"],
            "pred_answer": "yes",
            "confidence": "0.9",
            "raw_output": "raw1",
            "exact_match": True,
            "output_ids": None,
            "correct": True,
        }
    ]
    torch.save(
        processed_data, os.path.join(prev_run_dir, "temp_eval_results_rank0_0.pt")
    )

    # 3. Initialize Trainer and run evaluation with resume_from
    trainer = CalibrationTrainer()
    # Mock validate to avoid actual model run

    captured_model = None
    captured_dm = None

    def mock_validate(model, datamodule, **kwargs):
        nonlocal captured_model, captured_dm
        captured_model = model
        captured_dm = datamodule

        # Mock trainer for the model
        mock_trainer = MagicMock()
        mock_trainer.global_rank = 0

        # Manually set log_dir to match what evaluation() would produce:
        # csv is at test_resume_run/llm_outputs.csv
        # eval dir should be test_resume_run_evaluation
        expected_log_dir = os.path.join(test_dir, "llm_outputs_evaluation")
        if not os.path.exists(expected_log_dir):
            os.makedirs(expected_log_dir, exist_ok=True)

        mock_trainer.log_dir = expected_log_dir

        model.trainer = mock_trainer

        # Simulate hooks
        # DataModule setup needs to be called to populate skip_indices
        datamodule.setup()

        # Model hook to load resumed files
        model.on_validation_start()

    trainer.validate = mock_validate

    print("Running evaluation with resume_from...")
    trainer.evaluation(
        llm_outputs_path=csv_path,
        resume_from=prev_run_dir,
        flush_outputs_every_n_steps=1,
        save_outputs=True,
    )

    # Verifications
    print("\nVerifying...")

    # 1. Check if DataModule skipped indices correctly
    # DM now uses resume_from to find skip_indices internally during setup()
    print(f"Skipped indices: {captured_dm.skip_indices}")
    assert 0 in captured_dm.skip_indices
    assert len(captured_dm.skip_indices) == 1

    dataset = captured_dm.dataset
    print(f"Dataset size: {len(dataset)}")
    # Should have 2 items left (indices 1 and 2)
    assert len(dataset) == 2
    # Verify indices in dataset
    indices = dataset["index"]
    print(f"Dataset indices: {indices}")
    assert 1 in indices
    assert 2 in indices
    assert 0 not in indices

    # 2. Check if EvaluatorModule loaded initial files
    # Model now finds files itself via resume_from
    assert captured_model.resume_from == prev_run_dir

    print(f"Flushed output files in model: {captured_model.flushed_output_files}")
    assert len(captured_model.flushed_output_files) == 1
    # Check if file was copied to new log dir
    copied_file = captured_model.flushed_output_files[0]
    assert "llm_outputs_evaluation" in copied_file
    assert os.path.exists(copied_file)

    # Clean up
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    print("\nSUCCESS: Verification passed!")


if __name__ == "__main__":
    test_resume_feature()
