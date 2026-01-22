import csv
from unittest.mock import MagicMock

# Add project root to sys.path if needed, though running from root should handle it.
# Assuming cli.py is in the root
from cli import CalibrationTrainer


def test_evaluate_csv_subcommand_logic(tmp_path):
    # Create a dummy CSV file
    csv_path = tmp_path / "test_eval_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Question",
                "Gold Answers",
                "Predicted Answer",
                "Confidence",
                "Correct",
                "Evaluator Response",
                "Raw Output",
            ]
        )
        writer.writerow(["Q1", "A1", "A1", "0.9", "Yes", "Reasoning", "Raw"])
        writer.writerow(["Q2", "A2", "A3", "0.2", "No", "Reasoning", "Raw"])

    # Mock EvaluatorModule
    # We don't need to mock instantiation of EvaluatorModule inside evaluate_csv anymore
    # Instead we pass a mock model instance
    mock_model = MagicMock()

    trainer = CalibrationTrainer()
    trainer.evaluate_csv(csv_path=str(csv_path), model=mock_model)

    # Verify load_evaluation_results_from_csv was called on the mock model
    mock_model.load_evaluation_results_from_csv.assert_called_once_with(str(csv_path))

    # Verify calculate_metrics was called
    mock_model.calculate_metrics.assert_called_once()


def test_evaluate_csv_missing_path(capsys):
    trainer = CalibrationTrainer()
    trainer.evaluate_csv(csv_path=None)
    captured = capsys.readouterr()
    assert "Error: Please provide a valid CSV path using --csv_path." in captured.out
