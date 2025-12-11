from lightning.pytorch.cli import LightningCLI
from callm.models.llm import LLM
from callm.data.triviaQA import TriviaQADataModule


def cli_main():
    cli = LightningCLI(
        LLM,
        TriviaQADataModule,
        run=False,  # Don't automatically run
    )

    # Critical fix: Ensure data module uses the same tokenizer as the model
    # This prevents tokenizer mismatch issues (e.g., data tokenized with flan-t5-small
    # but model using Qwen tokenizer, which causes garbage output)
    if hasattr(cli.model, "tokenizer") and hasattr(cli.model, "model_name"):
        # Always use the model's tokenizer to ensure consistency
        if cli.datamodule.model_name != cli.model.model_name:
            print(
                f"Warning: Data module model_name ({cli.datamodule.model_name}) doesn't match "
                f"model model_name ({cli.model.model_name}). Using model's tokenizer."
            )
            cli.datamodule.tokenizer = cli.model.tokenizer
            cli.datamodule.model_name = cli.model.model_name
            # Re-setup the data module with the correct tokenizer if it was already set up
            if hasattr(cli.datamodule, "triviaQA_val"):
                cli.datamodule.setup("validate")
        else:
            # Model names match, but ensure tokenizer is the same instance
            cli.datamodule.tokenizer = cli.model.tokenizer

    # Run validation
    if cli.config.get("subcommand") == "validate" or not hasattr(
        cli.config, "subcommand"
    ):
        cli.trainer.validate(cli.model, cli.datamodule)


if __name__ == "__main__":
    cli_main()
