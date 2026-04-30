# callm — Confidence Calibration for LLMs

A framework for evaluating confidence augmented systems, built on [PyTorch Lightning](https://lightning.ai/).
Supports both local HuggingFace models and GCP Vertex AI (Gemini) backends across multiple benchmarks.

## Supported Benchmarks

| Benchmark | Task type | Semantic‑equivalence evaluation needed? |
|---|---|---|
| **TriviaQA** | Open‑ended QA | Yes — uses an evaluator LLM |
| **MMLU** | Multiple‑choice | No — exact match on answer letter |

## Calibration Metrics

| Metric | Description |
|---|---|
| **ECE** | Expected Calibration Error (L1, 10 bins) |
| **AUC** | Area Under the ROC Curve |
| **BS** | Brier Score (MSE between confidence and correctness) |
| **CE** | Binary Cross‑Entropy |
| **CCAS** | Cost of Confidence Augmented Systems |
| **ECUAS** | Expected Cost for Uncertainty-Augmented Systems (parameterised by n = 0, 1, 2, …) |
| **γ‑ECUAS** | Gamma‑ECUAS — selective prediction at operating point γ |
| **AURC** | Area Under the Risk‑Coverage curve |
| **FPR@95** | False Positive Rate at 95% recall |
| **Error Rate** | Overall prediction error rate |

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment (optional)

```bash
cp .env.example .env
```

Then edit `.env`:

```env
HF_TOKEN=your_huggingface_token_here               # needed for gated models (e.g. Llama, Mistral)
GOOGLE_APPLICATION_CREDENTIALS=path/to/creds.json   # needed for GCP / Gemini models
```

### 3. Run unit tests

```bash
uv run pytest callm/tests/ -v
```

## Usage

The CLI is built on top of `LightningCLI` and exposes three subcommands:

### `validate` — Run LLM inference and extract answers + confidences

```bash
# TriviaQA with a local HuggingFace model (default config)
uv run python main.py validate \
  --model.init_args.model_name=google/flan-t5-small \
  --data.init_args.batch_size=8

# MMLU with a local model
uv run python main.py validate \
  -c configs/config_mmlu_base_validation.yaml \
  --model.init_args.model_name=mistralai/Ministral-3-8B-Instruct-2512

# TriviaQA with a GCP Gemini model
uv run python main.py validate \
  -c configs/config_gcp_validation.yaml
```

Outputs are saved to `lightning_logs/<run>/llm_outputs.csv`.

### `evaluation` — Evaluate correctness of LLM outputs via a judge model

For benchmarks that require semantic-equivalence checking (TriviaQA):

```bash
uv run python main.py evaluation \
  --llm_outputs_path=lightning_logs/<run>/llm_outputs.csv

# Or recalculate metrics from an existing evaluation CSV:
uv run python main.py evaluation \
  --use_existing_csv \
  --llm_outputs_path=lightning_logs/<run>/llm_outputs.csv
```

### `evaluate_csv` — Compute metrics from a saved evaluation CSV

```bash
uv run python main.py evaluate_csv \
  --csv_path=lightning_logs/<run>_evaluation/version_0/evaluation_results.csv
```

## Configuration

All runs are configured via YAML. Pre-built configs live in `configs/`:

| Config | Backend | Benchmark |
|---|---|---|
| `config_base_validation.yaml` | HuggingFace | TriviaQA |
| `config_gcp_validation.yaml` | GCP (Gemini) | TriviaQA |
| `config_base_evaluation.yaml` | HuggingFace | TriviaQA (evaluator) |
| `config_gcp_evaluation.yaml` | GCP (Gemini) | TriviaQA (evaluator) |
| `config_mmlu_base_validation.yaml` | HuggingFace | MMLU |
| `config_mmlu_gcp_validation.yaml` | GCP (Gemini) | MMLU |

Any config value can be overridden from the CLI — see the [LightningCLI docs](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html).

## Project Structure

```
callm/
├── models/
│   ├── base.py              # Shared Lightning module base
│   ├── llm.py               # HuggingFace LLM (local GPU)
│   ├── gcp_llm.py           # GCP Vertex AI / Gemini LLM
│   ├── evaluator.py         # Semantic-equivalence evaluator (HF)
│   └── gcp_evaluator.py     # Semantic-equivalence evaluator (GCP)
├── data/
│   ├── triviaqa/            # TriviaQA data modules
│   ├── mmlu/                # MMLU data modules
│   ├── answers_data.py      # Shared answer-loading utilities
│   └── classification.py    # Classification data module
├── extractors/
│   ├── base.py              # Base + posterior extractors
│   ├── triviaqa.py          # TriviaQA verbalized-confidence extractor
│   └── mmlu.py              # MMLU answer/confidence extractors
├── prompts/
│   ├── base.py              # Prompt / ChatPrompt base classes
│   ├── triviaqa.py          # TriviaQA prompt templates
│   └── mmlu.py              # MMLU prompt templates
├── metrics/
│   ├── confidences.py       # Calibration metrics (ECE, AUC, BS, CE, ECUAS, …)
│   ├── classification.py    # Classification-specific metric variants
│   ├── constants.py         # Metric constants and registry
│   └── utils.py             # Metric lookup helpers
├── tests/                   # Unit & integration tests
├── config.py                # Shared config utilities
└── utils.py                 # Model loading & tokenizer helpers
configs/                     # YAML run configurations
scripts/                     # Analysis & paper-figure scripts
cli.py                       # CalibrationCLI (extends LightningCLI)
main.py                      # Entrypoint
```

## Confidence Extraction Methods

| Extractor | How confidence is obtained |
|---|---|
| **SequencePosteriorExtractor** | Product of token log‑probabilities of the generated answer |
| **IsTruePosteriorExtractor** | Log‑prob of the "True" token after an "Is this true?" follow‑up |
| **VerbalizedConfidenceExtractor** | Parsed from the model's own verbalized confidence value |

MMLU variants (`MMLUSequencePosteriorExtractor`, `MMLUVerbalizedExtractor`, etc.) adapt these strategies to multiple‑choice format.

## License

See [LICENSE](LICENSE).
