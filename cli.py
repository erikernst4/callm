from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.trainer import Trainer
from callm.models.evaluator import EvaluatorModule
from callm.data.evaluator_data import EvaluatorDataModule
import os


class CalibrationCLI(LightningCLI):
    """Extended LightningCLI that runs evaluation after validation."""

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

    def after_validate(self):
        """Run correctness evaluation after LLM validation completes."""
        if not self.config.get("evaluate_correctness", False):
            return
        # Get path to LLM outputs
        log_dir = self.trainer.log_dir or os.getcwd()
        llm_outputs_path = os.path.join(log_dir, "llm_outputs.csv")

        if not os.path.exists(llm_outputs_path):
            print(f"Warning: LLM outputs not found at {llm_outputs_path}")
            return

        print(f"\n{'='*60}")
        print("Running correctness evaluation...")
        print(f"{'='*60}\n")

        # Get evaluator config from CLI args
        evaluator_model_name = self.config["validate"].get(
            "evaluator_model_name", "google/flan-t5-base"
        )
        evaluator_batch_size = self.config["validate"].get("evaluator_batch_size", 8)

        # Create evaluator components
        evaluator_dm = EvaluatorDataModule(
            llm_outputs_path=llm_outputs_path,
            model_name=evaluator_model_name,
            batch_size=evaluator_batch_size,
        )
        evaluator_model = EvaluatorModule(model_name=evaluator_model_name)

        # Create a new trainer for evaluation
        # Use same accelerator/device settings as main trainer
        eval_trainer = Trainer(
            accelerator=self.trainer.accelerator.__class__.__name__.lower().replace(
                "accelerator", ""
            ),
            devices=1,  # Use single device for evaluation
            logger=self.trainer.logger,
            enable_progress_bar=True,
            inference_mode=True,
            default_root_dir=log_dir,
        )

        # Run evaluation
        eval_trainer.validate(evaluator_model, evaluator_dm)
