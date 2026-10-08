from cli import CalibrationCLI

PARSER_KWARGS = {
    "validate": {"default_config_files": ["configs/config_base_validation.yaml"]},
    "evaluation": {"default_config_files": ["configs/config_base_evaluation.yaml"]},
    "evaluate_csv": {"default_config_files": ["configs/config_base_evaluation.yaml"]},
    "extract_activations": {
        "default_config_files": ["configs/config_extract_activations.yaml"]
    },
    "extract_exact_answers": {
        "default_config_files": ["configs/config_extract_exact_answers.yaml"]
    },
    # omegaconf+ resolves ${...} interpolations after command-line overrides
    "parser_mode": "omegaconf+",
}


def cli_main():
    CalibrationCLI(parser_kwargs=PARSER_KWARGS)


if __name__ == "__main__":
    cli_main()
