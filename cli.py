from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.trainer import Trainer
from callm.models.evaluator import EvaluatorModule
from callm.data.evaluator_data import EvaluatorDataModule
import os
from lightning.pytorch.loggers import CSVLogger


class CalibrationTrainer(Trainer):
    def evaluation(
        self,
        llm_outputs_path: str = None,
        evaluator_model_name: str = "google/flan-t5-base",
        evaluator_batch_size: int = 8,
        num_workers: int = None,
        **kwargs,
    ):
        """Run correctness evaluation on LLM outputs."""
        print(f"\n{'='*60}")
        print("Running correctness evaluation...")
        print(f"{'='*60}\n")

        # Resolve llm_outputs_path if not provided
        if llm_outputs_path is None:
            # Look for the last run in lightning_logs
            log_dir = self.default_root_dir or os.getcwd()
            lightning_logs = os.path.join(log_dir, "lightning_logs")
            if os.path.exists(lightning_logs):
                # Get all version directories sorting by creation time
                versions = sorted(
                    [
                        os.path.join(lightning_logs, d)
                        for d in os.listdir(lightning_logs)
                        if d.startswith("version_")
                    ],
                    key=os.path.getmtime,
                )
                if versions:
                    for version_dir in reversed(versions):
                        candidate_path = os.path.join(version_dir, "llm_outputs.csv")
                        if os.path.exists(candidate_path):
                            llm_outputs_path = candidate_path
                            print(f"Found latest LLM outputs at: {llm_outputs_path}")
                            break

            if llm_outputs_path is None:
                # Fallback to checking the current trainer's log dir if available
                if self.log_dir:
                    candidate_path = os.path.join(self.log_dir, "llm_outputs.csv")
                    if os.path.exists(candidate_path):
                        llm_outputs_path = candidate_path
                        print(
                            f"Found LLM outputs in current log dir: {llm_outputs_path}"
                        )

        if not llm_outputs_path or not os.path.exists(llm_outputs_path):
            print(
                "Error: LLM outputs not found. Please provide a valid path using --llm_outputs_path."
            )
            return

        # Use num_workers from parameter or default to 0
        if num_workers is None:
            num_workers = 0

        # Create evaluator components
        evaluator_dm = EvaluatorDataModule(
            llm_outputs_path=llm_outputs_path,
            model_name=evaluator_model_name,
            batch_size=evaluator_batch_size,
            num_workers=num_workers,
        )
        evaluator_model = EvaluatorModule(model_name=evaluator_model_name)

        # Configure logging to a new folder with _evaluation suffix
        output_dir = os.path.dirname(llm_outputs_path)
        eval_log_dir = f"{output_dir}_evaluation"
        logger = CSVLogger(
            save_dir=os.path.dirname(eval_log_dir), name=os.path.basename(eval_log_dir)
        )

        self.loggers = [logger]

        self.validate(evaluator_model, evaluator_dm)


class CalibrationCLI(LightningCLI):
    """Extended LightningCLI that runs evaluation after validation."""

    def __init__(self, *args, **kwargs):
        kwargs["trainer_class"] = CalibrationTrainer
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        """Add evaluator-specific arguments."""
        parser.add_argument(
            "--evaluate_correctness",
            type=bool,
            default=False,
            help="Whether to evaluate correctness after validation",
        )
        parser.add_argument(
            "--evaluator_model_name",
            default="google/flan-t5-base",
            help="Model to use for correctness evaluation",
        )
        parser.add_argument(
            "--evaluator_batch_size",
            type=int,
            default=8,
            help="Batch size for evaluator",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=None,
            help="Number of workers for DataLoader (defaults to value in data config)",
        )

    def after_validate(self):
        """Run correctness evaluation after LLM validation completes."""
        config = self.config.validate
        if not config.evaluate_correctness:
            return

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
            evaluator_model_name=config.evaluator_model_name,
            evaluator_batch_size=config.evaluator_batch_size,
            num_workers=num_workers,
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
                "evaluator_model_name",
                "evaluator_batch_size",
                # llm_outputs_path is not manually added, so let it be added by signature
            },
        }
