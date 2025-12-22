import os
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import Trainer
from callm.models.llm import LLM
from callm.models.evaluator import EvaluatorModule
from callm.data.triviaQA import TriviaQADataModule
from callm.data.evaluator_data import EvaluatorDataModule


class CalibrationCLI(LightningCLI):
    """Extended LightningCLI that runs evaluation after validation."""

    def add_arguments_to_parser(self, parser):
        """Add evaluator-specific arguments."""
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


def cli_main():
    cli = CalibrationCLI(
        LLM,
        TriviaQADataModule,
        parser_kwargs={
            "validate": {"default_config_files": ["configs/config_base.yaml"]},
            "parser_mode": "omegaconf",
        },
    )

    if hasattr(cli.model, "tokenizer") and hasattr(cli.model, "model_name"):
        if cli.datamodule.model_name != cli.model.model_name:
            raise ValueError(
                f"Data module model_name ({cli.datamodule.model_name}) doesn't match "
                f"model model_name ({cli.model.model_name})."
            )


if __name__ == "__main__":
    cli_main()
