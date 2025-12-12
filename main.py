from lightning.pytorch.cli import LightningCLI
from callm.models.llm import LLM
from callm.data.triviaQA import TriviaQADataModule


def cli_main():
    cli = LightningCLI(LLM, TriviaQADataModule)

    if hasattr(cli.model, "tokenizer") and hasattr(cli.model, "model_name"):
        if cli.datamodule.model_name != cli.model.model_name:
            raise ValueError(
                f"Data module model_name ({cli.datamodule.model_name}) doesn't match "
                f"model model_name ({cli.model.model_name})."
            )


if __name__ == "__main__":
    cli_main()
