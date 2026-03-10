from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.trainer import Trainer
from callm.models.evaluator import EvaluatorModule
from callm.data.evaluator_data import EvaluatorDataModule
from callm.utils import get_last_llm_outputs_path
import os
from lightning.pytorch.loggers import CSVLogger

# Disable tokenizer parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CalibrationTrainer(Trainer):
    def __init__(
        self,
        *args,
        evaluate_correctness: bool = False,
        evaluator_model_name: str = "google/flan-t5-base",
        evaluator_batch_size: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.evaluate_correctness = evaluate_correctness
        self.evaluator_model_name = evaluator_model_name
        self.evaluator_batch_size = evaluator_batch_size

    def evaluation(
        self,
        llm_outputs_path: str = None,
        num_workers: int = None,
        flush_outputs_every_n_steps: int = None,
        save_outputs: bool = None,
        resume_from: str = None,
        use_existing_csv: bool = False,
        **kwargs,
    ):
        """Run correctness evaluation on LLM outputs or recalculate metrics."""
        # Check if model and datamodule were already instantiated by LightningCLI
        evaluator_model = kwargs.get("model")
        evaluator_dm = kwargs.get("datamodule")

        # Ensure they are the correct type
        if (
            evaluator_model is not None
            and not isinstance(evaluator_model, EvaluatorModule)
            or (
                evaluator_dm is not None
                and not isinstance(evaluator_dm, EvaluatorDataModule)
            )
        ):
            raise ValueError(
                "Evaluator model and datamodule must be of type EvaluatorModule and EvaluatorDataModule"
            )

        # Resolve llm_outputs_path if not provided
        if llm_outputs_path is None:
            # Look for the last run in lightning_logs
            log_dir = self.default_root_dir or os.getcwd()
            llm_outputs_path = get_last_llm_outputs_path(log_dir)

        if not llm_outputs_path or not os.path.exists(llm_outputs_path):
            print(
                "Error: LLM outputs not found. Please provide a valid path using --llm_outputs_path."
            )
            return

        output_dir = os.path.dirname(llm_outputs_path)
        eval_log_dir = f"{output_dir}_evaluation"

        if use_existing_csv:
            import glob

            csv_files = glob.glob(
                os.path.join(eval_log_dir, "version_*", "evaluation_results.csv")
            )
            if not csv_files:
                print(
                    f"Error: No existing evaluation_results.csv found in {eval_log_dir}."
                )
                return

            # Use the latest version
            csv_path = sorted(
                csv_files,
                key=lambda x: int(os.path.basename(os.path.dirname(x)).split("_")[-1]),
            )[-1]

            if evaluator_model is None:
                evaluator_model = EvaluatorModule(
                    model_name=self.evaluator_model_name,
                    flush_outputs_every_n_steps=flush_outputs_every_n_steps,
                    save_outputs=save_outputs,
                    resume_from=resume_from,
                )

            print(f"Using existing CSV for metrics computation: {csv_path}")
            evaluator_model.load_evaluation_results_from_csv(csv_path)

            logger = CSVLogger(
                save_dir=os.path.dirname(eval_log_dir),
                name=os.path.basename(eval_log_dir),
            )

            metrics_log = {}

            def log_printer(name, value, **kwargs):
                print(f"{name}: {value}")
                metrics_log[name] = value.item() if hasattr(value, "item") else value

            evaluator_model.log = log_printer
            evaluator_model.calculate_metrics()

            logger.log_metrics(metrics_log)
            logger.save()
            return

        # Use num_workers from parameter or default to 0
        if num_workers is None:
            num_workers = 0

        # Create or update evaluator components
        if evaluator_dm is None:
            evaluator_dm = EvaluatorDataModule(
                llm_outputs_path=llm_outputs_path,
                model_name=self.evaluator_model_name,
                batch_size=self.evaluator_batch_size,
                num_workers=num_workers,
                resume_from=resume_from,
            )
        else:
            evaluator_dm_attributes = evaluator_dm.__dict__
            evaluator_dm_attributes.update(
                llm_outputs_path=llm_outputs_path, resume_from=resume_from
            )
            evaluator_dm = EvaluatorDataModule(**evaluator_dm_attributes)

        if evaluator_model is None:
            evaluator_model = EvaluatorModule(
                model_name=self.evaluator_model_name,
                flush_outputs_every_n_steps=flush_outputs_every_n_steps,
                save_outputs=save_outputs,
                resume_from=resume_from,
            )

        # Configure logging to a new folder with _evaluation suffix
        logger = CSVLogger(
            save_dir=os.path.dirname(eval_log_dir), name=os.path.basename(eval_log_dir)
        )

        self.loggers = [logger]

        self.validate(evaluator_model, evaluator_dm)

    def evaluate_csv(
        self,
        csv_path: str = None,
        model: EvaluatorModule = None,
        llm_outputs_path: str = None,
        num_workers: int = None,
        resume_from: str = None,
        **kwargs,
    ):
        """
        Evaluate correctness from a CSV file.

        Args:
            csv_path: Path to the CSV file containing evaluation results.
            model: Evaluator model instance (injected by CLI).
        """
        if csv_path is None:
            print("Error: Please provide a valid CSV path using --csv_path.")
            return

        if model is None:
            # Should be enforced by CLI, but safe fallback logic or error
            print("Error: Model not provided.")
            return

        model.load_evaluation_results_from_csv(csv_path)

        # Monkeypatch log to print metrics since we are not in a loop
        def log_printer(name, value, **kwargs):
            print(f"{name}: {value}")

        model.log = log_printer
        model.calculate_metrics()


class CalibrationCLI(LightningCLI):
    """Extended LightningCLI that runs evaluation after validation."""

    def __init__(self, *args, **kwargs):
        kwargs["trainer_class"] = CalibrationTrainer
        super().__init__(*args, **kwargs)

    def after_validate(self):
        """Run correctness evaluation after LLM validation completes."""
        if not self.trainer.evaluate_correctness:
            return

        config = self.config.validate

        # Get path to LLM outputs
        log_dir = self.trainer.log_dir or os.getcwd()
        llm_outputs_path = os.path.join(log_dir, "llm_outputs.csv")

        # Get num_workers from data config if available
        num_workers = (
            getattr(config.data, "num_workers", None)
            if hasattr(config, "data")
            else None
        )

        # Run evaluation using the trainer method
        self.trainer.evaluation(
            llm_outputs_path=llm_outputs_path,
            num_workers=num_workers,
            flush_outputs_every_n_steps=config.model.init_args.flush_outputs_every_n_steps,
            save_outputs=config.model.init_args.save_outputs,
            resume_from=config.resume_from,
        )

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        """Defines the list of available subcommands and the arguments to skip."""
        return {
            "fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"},
            "validate": {"model", "dataloaders", "datamodule"},
            "test": {"model", "dataloaders", "datamodule"},
            "predict": {"model", "dataloaders", "datamodule"},
            "evaluation": {
                "model",
                "dataloaders",
                "datamodule",
                "train_dataloaders",
                "val_dataloaders",
                # llm_outputs_path is not manually added, so let it be added by signature
            },
            "evaluate_csv": {
                "model",
                "dataloaders",
                "datamodule",
                "train_dataloaders",
                "val_dataloaders",
            },
        }
