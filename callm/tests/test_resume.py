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


def test_extract_activations_resume_feature():
    test_dir = "test_extract_resume_run"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    prev_run_dir = os.path.join(test_dir, "lightning_logs", "version_0")
    os.makedirs(prev_run_dir, exist_ok=True)

    # 1. Create dummy chunk in prev_run_dir
    chunk_0 = [
        {
            "sample_idx": 0,
            "question": "Q1",
            "gold_answers": ["A"],
            "pred_answer": "A",
            "raw_output": "A",
            "activations": {0: torch.zeros(5, 16, dtype=torch.bfloat16)},
            "seq_len": 5,
            "prompt_length": 3,
        },
        {
            "sample_idx": 1,
            "question": "Q2",
            "gold_answers": ["B"],
            "pred_answer": "B",
            "raw_output": "B",
            "activations": {0: torch.zeros(6, 16, dtype=torch.bfloat16)},
            "seq_len": 6,
            "prompt_length": 4,
        },
    ]
    torch.save(chunk_0, os.path.join(prev_run_dir, "activations_chunk_rank0_0.pt"))

    # Also test that a corrupted chunk is safely skipped
    corrupted_path = os.path.join(prev_run_dir, "activations_chunk_rank0_1.pt")
    with open(corrupted_path, "wb") as f:
        f.write(b"corrupted bytes")

    # 2. Test DataModule skip_indices
    from callm.data import MMLUDataModule

    dm = MMLUDataModule(resume_from=test_dir)
    skip_indices = dm._get_skip_indices()
    assert 0 in skip_indices
    assert 1 in skip_indices
    assert len(skip_indices) == 2

    # 3. Test ActivationExtractor resume
    from callm.models.activation_extractor import ActivationExtractor

    mock_llm = MagicMock()
    mock_llm.parameters.return_value = []
    mock_llm.model_name = "mock_model"
    mock_llm.tokenizer = MagicMock()

    new_log_dir = os.path.join(test_dir, "lightning_logs", "version_1")
    os.makedirs(new_log_dir, exist_ok=True)
    out_dir = os.path.join(test_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    extractor = ActivationExtractor(
        llm=mock_llm,
        output_dir=out_dir,
        resume_from=test_dir,
    )
    mock_trainer = MagicMock()
    mock_trainer.log_dir = new_log_dir
    mock_trainer.global_rank = 0
    extractor.trainer = mock_trainer

    extractor.on_validation_start()

    assert extractor.sample_counter == 2
    assert len(extractor.flushed_output_files) == 1
    assert os.path.exists(extractor.flushed_output_files[0])
    assert "version_1" in extractor.flushed_output_files[0]

    # Clean up
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    print("\nSUCCESS: ActivationExtractor resume verification passed!")


if __name__ == "__main__":
    test_resume_feature()
    test_extract_activations_resume_feature()
