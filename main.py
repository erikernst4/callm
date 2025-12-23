from callm.models.llm import LLM
from callm.data.triviaQA import TriviaQADataModule
from cli import CalibrationCLI


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
