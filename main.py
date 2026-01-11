from cli import CalibrationCLI


def cli_main():
    CalibrationCLI(
        parser_kwargs={
            "validate": {
                "default_config_files": ["configs/config_base_validation.yaml"]
            },
            "evaluation": {
                "default_config_files": ["configs/config_base_evaluation.yaml"]
            },
            "parser_mode": "omegaconf",
        },
    )


if __name__ == "__main__":
    cli_main()
