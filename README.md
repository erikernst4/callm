# Confidence Calibration for LLMs

This project reproduces results from paper 2023.emnlp-main.330, implementing confidence calibration metrics (ECE, AUC, BS, CE) on TriviaQA validation set.

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Create `.env` file (optional, only needed for Llama models):
```bash
cp .env.example .env
# Edit .env and add your HuggingFace token
```

## Running Tests

### Unit Tests
```bash
uv run pytest callm/tests/ -v
```

### Integration Test (10 samples)
```bash
uv run python test_integration.py
```

### Full Validation
```bash
uv run python main.py validate --model.model_name=flan-t5-small --data.batch_size=8
```

## Metrics

The system computes the following calibration metrics:
- **ECE** (Expected Calibration Error): Measures calibration quality
- **AUC** (Area Under Curve): Measures ranking ability
- **BS** (Brier Score): Measures prediction accuracy
- **CE** (Cross Entropy): Measures log-likelihood

## Architecture

- `callm/models/llm.py`: Main LLM model with validation logic
- `callm/extractors.py`: Answer and confidence extraction from LLM outputs
- `callm/evaluator.py`: LLM-based correctness evaluator
- `callm/metrics.py`: Calibration metrics implementation
- `callm/data/triviaQA.py`: TriviaQA dataset loading and preprocessing
- `callm/prompts.py`: Prompt templates

## Configuration

Models are cached to `~/.cache/huggingface` by default. Configure via `.env`:
```
CACHE_PATH=/your/custom/path
HF_TOKEN=your_token_here
```
