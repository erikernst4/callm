from jsonargparse import set_parsing_settings
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.trainer import Trainer
from callm.models.evaluator import EvaluatorModule
from callm.data.triviaqa import EvaluatorDataModule
from callm.utils import get_last_llm_outputs_path
from callm.callbacks import LayerWeightsHistoryCallback, TrainingTimeCallback
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
        # Ensure CSVLogger is added by default to capture metrics
        loggers = kwargs.get("logger")
        save_dir = kwargs.get("default_root_dir") or os.getcwd()
        csv_logger = CSVLogger(save_dir=save_dir)
        if loggers is None or loggers is True:
            kwargs["logger"] = csv_logger
        elif isinstance(loggers, list):
            if not any(isinstance(logger, CSVLogger) for logger in loggers):
                loggers.append(csv_logger)
        else:
            if not isinstance(loggers, CSVLogger):
                kwargs["logger"] = [loggers, csv_logger]

        # Ensure default callbacks are added to capture and persist timing and
        # layer weight history
        callbacks = kwargs.get("callbacks")
        if callbacks is None:
            callbacks = []
        elif not isinstance(callbacks, list):
            callbacks = [callbacks]
        for callback_cls in (TrainingTimeCallback, LayerWeightsHistoryCallback):
            if not any(isinstance(cb, callback_cls) for cb in callbacks):
                callbacks.append(callback_cls())
        kwargs["callbacks"] = callbacks

        super().__init__(*args, **kwargs)
        self.evaluate_correctness = evaluate_correctness
        self.evaluator_model_name = evaluator_model_name
        self.evaluator_batch_size = evaluator_batch_size

    def _default_evaluator(
        self,
        flush_outputs_every_n_steps: int | None,
        save_outputs: bool | None,
        resume_from: str | None,
    ) -> EvaluatorModule:
        """``EvaluatorModule`` for ``evaluator_model_name``; options left as
        ``None`` keep the module's defaults."""
        options = {
            "flush_outputs_every_n_steps": flush_outputs_every_n_steps,
            "save_outputs": save_outputs,
            "resume_from": resume_from,
        }
        return EvaluatorModule(
            model_name=self.evaluator_model_name,
            **{k: v for k, v in options.items() if v is not None},
        )

    def evaluation(
        self,
        llm_outputs_path: str = None,
        num_workers: int = None,
        flush_outputs_every_n_steps: int = None,
        save_outputs: bool = None,
        resume_from: str = None,
        use_existing_csv: bool = False,
        previous_eval_csv: str = None,
        **kwargs,
    ):
        """Run correctness evaluation on LLM outputs or recalculate metrics."""
        # Check if model and datamodule were already instantiated by LightningCLI
        evaluator_model = kwargs.get("model")
        evaluator_dm = kwargs.get("datamodule")

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
                evaluator_model = self._default_evaluator(
                    flush_outputs_every_n_steps, save_outputs, resume_from
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
        elif evaluator_dm is not None:
            if llm_outputs_path and (
                not hasattr(evaluator_dm, "llm_outputs_path")
                or not evaluator_dm.llm_outputs_path
            ):
                evaluator_dm.llm_outputs_path = llm_outputs_path
            if resume_from:
                evaluator_dm.resume_from = resume_from
            if (
                previous_eval_csv
                and hasattr(evaluator_dm, "previous_eval_csv")
                and not evaluator_dm.previous_eval_csv
            ):
                evaluator_dm.previous_eval_csv = previous_eval_csv

        if evaluator_model is None:
            evaluator_model = self._default_evaluator(
                flush_outputs_every_n_steps, save_outputs, resume_from
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

    def extract_activations(
        self,
        output_dir: str = None,
        layer_indices: list[int] | None = None,
        save_dtype: str = "bfloat16",
        resume_from: str = None,
        **kwargs,
    ):
        """Extract and save LLM activations for probe training."""
        model = kwargs.get("model")
        datamodule = kwargs.get("datamodule")

        if model is None:
            raise ValueError("model is required for extract_activations")
        if datamodule is None:
            raise ValueError("datamodule is required for extract_activations")

        if output_dir is None:
            log_dir = self.default_root_dir or os.getcwd()
            output_dir = os.path.join(log_dir, "activations")

        if resume_from:
            if not hasattr(datamodule, "resume_from"):
                raise ValueError(
                    f"{type(datamodule).__name__} does not support resume_from; "
                    "already extracted samples would be processed again."
                )
            if not datamodule.resume_from:
                datamodule.resume_from = resume_from

        # Import locally to avoid circular imports
        from callm.models.activation_extractor import ActivationExtractor

        extractor = ActivationExtractor(
            llm=model,
            output_dir=output_dir,
            layer_indices=layer_indices,
            save_dtype=save_dtype,
            resume_from=resume_from,
        )

        self.validate(model=extractor, datamodule=datamodule)

    def extract_exact_answers(
        self,
        **kwargs,
    ):
        """Extract exact answer entities from LLM outputs using GCP LLM."""
        model = kwargs.get("model")
        datamodule = kwargs.get("datamodule")

        if model is None:
            raise ValueError("model is required for extract_exact_answers")
        if datamodule is None:
            raise ValueError("datamodule is required for extract_exact_answers")

        self.validate(model=model, datamodule=datamodule)


class CalibrationCLI(LightningCLI):
    """Extended LightningCLI that runs evaluation after validation."""

    def __init__(self, *args, **kwargs):
        kwargs["trainer_class"] = CalibrationTrainer
        # With parser_mode "omegaconf+", interpolations like
        # ${trainer.evaluator_model_name} are resolved after command-line
        # overrides; making them relative lets them resolve within the
        # subcommand's config (e.g. under "evaluation.").
        set_parsing_settings(omegaconf_absolute_to_relative_paths=True)
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
            "extract_activations": {
                "model",
                "dataloaders",
                "datamodule",
                "train_dataloaders",
                "val_dataloaders",
            },
            "extract_exact_answers": {
                "model",
                "dataloaders",
                "datamodule",
                "train_dataloaders",
                "val_dataloaders",
            },
        }
