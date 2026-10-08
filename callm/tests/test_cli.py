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


def _print_config(args, capsys) -> dict:
    """Config printed by the real CLI (``main.py``) for ``args``, without running."""
    import pytest
    import yaml

    from cli import CalibrationCLI
    from main import PARSER_KWARGS

    with pytest.raises(SystemExit):
        CalibrationCLI(parser_kwargs=PARSER_KWARGS, args=[*args, "--print_config"])
    return yaml.safe_load(capsys.readouterr().out)


def test_interpolations_follow_cli_overrides_in_evaluation(capsys, tmp_path):
    outputs = str(tmp_path / "llm_outputs.csv")
    config = _print_config(
        [
            "evaluation",
            "--trainer.evaluator_model_name=Qwen/Qwen2.5-3B",
            "--trainer.evaluator_batch_size=8",
            f"--llm_outputs_path={outputs}",
            "--num_workers=2",
            f"--resume_from={tmp_path}",
        ],
        capsys,
    )
    model, data = config["model"]["init_args"], config["data"]["init_args"]
    assert model["model_name"] == data["model_name"] == "Qwen/Qwen2.5-3B"
    assert data["batch_size"] == 8
    assert data["llm_outputs_path"] == outputs
    assert data["num_workers"] == 2
    assert model["resume_from"] == data["resume_from"] == str(tmp_path)


def test_interpolations_follow_cli_overrides_in_validate(capsys):
    config = _print_config(
        ["validate", "--model.init_args.model_name=Qwen/Qwen2.5-3B"], capsys
    )
    model, data = config["model"]["init_args"], config["data"]["init_args"]
    assert data["model_name"] == "Qwen/Qwen2.5-3B"
    assert model["extractor"]["init_args"]["model_name"] == "Qwen/Qwen2.5-3B"


def test_trainer_without_default_root_dir_logs_to_cwd(tmp_path, monkeypatch):
    from lightning.pytorch.loggers import CSVLogger

    monkeypatch.chdir(tmp_path)
    trainer = CalibrationTrainer(default_root_dir=None, accelerator="cpu")
    csv_loggers = [lg for lg in trainer.loggers if isinstance(lg, CSVLogger)]
    assert [lg.save_dir for lg in csv_loggers] == [str(tmp_path)]


def test_evaluation_builds_evaluator_with_module_defaults(tmp_path):
    from unittest.mock import Mock, patch

    outputs = tmp_path / "run" / "llm_outputs.csv"
    outputs.parent.mkdir()
    outputs.write_text("question,gold_answers,pred_answer,confidence,raw_output\n")

    trainer = CalibrationTrainer(default_root_dir=str(tmp_path), accelerator="cpu")
    with (
        patch("callm.models.evaluator.initialize_model") as init_model,
        patch("callm.models.evaluator.get_tokenizer_for_model"),
        patch.object(CalibrationTrainer, "validate") as validate,
    ):
        model = Mock()
        model.parameters.return_value = iter([])
        model.config = Mock(spec=["pad_token_id", "eos_token_id"], pad_token_id=0)
        init_model.return_value = model
        trainer.evaluation(llm_outputs_path=str(outputs))

    evaluator = validate.call_args.args[0]
    assert evaluator.flush_outputs_every_n_steps == -1
    assert evaluator.save_outputs is False
    assert evaluator.resume_from is None
    evaluator._maybe_flush()  # must not fail comparing None with 0
