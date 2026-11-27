# callm — Confidence Calibration for LLMs

A research framework, built on [PyTorch Lightning](https://lightning.ai/), for evaluating and training
confidence-augmented LLM systems: an LLM answers benchmark questions, a confidence is attached to each
answer, and calibration metrics are computed against correctness labels.

Two lines of work share the same pipeline:

- **Confidence extraction** — confidence from the LLM itself (token posteriors, "Is this true?"
  posteriors, verbalized confidence), with local HuggingFace models or GCP Vertex AI (Gemini).
- **Probes** — small models trained on the LLM's hidden-state activations to predict whether an
  answer is correct.

## Supported benchmarks

| Benchmark | Task type | Correctness |
|---|---|---|
| **TriviaQA** | Open-ended QA | Semantic equivalence, judged by an LLM (`evaluation` subcommand) |
| **Curated TriviaQA** | Open-ended QA (curated subset) | Same as TriviaQA |
| **MMLU** | Multiple choice | Exact match on the answer letter |

## Quick start

```bash
uv sync                          # install (Python 3.13)
cp .env.example .env             # optional, see below
uv run pytest callm/tests/ -v    # run the tests
```

`.env`:

```env
HF_TOKEN=your_huggingface_token_here               # gated models (e.g. Llama, Mistral)
GOOGLE_APPLICATION_CREDENTIALS=path/to/creds.json   # GCP / Gemini models
```

`pre-commit run --all-files` runs ruff (lint + format), nbstripout, gitleaks and the full test suite;
the same hooks run on every commit.

## Usage

Everything goes through `main.py`, a `LightningCLI` (`CalibrationCLI` in `cli.py`) driven by YAML
configs in `configs/`. Any value can be overridden on the command line, e.g.
`--model.init_args.model_name=... --trainer.devices=[0]`. Configs use `${...}` interpolation (e.g. the
datamodule's `model_name: ${model.init_args.model_name}`), resolved after command-line overrides.

| Subcommand | Default config | What it does |
|---|---|---|
| `validate` | `config_base_validation.yaml` | LLM inference: answer + confidence per question → `llm_outputs.csv`; MMLU metrics directly |
| `evaluation` | `config_base_evaluation.yaml` | Judge-LLM correctness for TriviaQA → `<run>_evaluation/version_N/evaluation_results.csv` + metrics |
| `evaluate_csv` | `config_base_evaluation.yaml` | Metrics from an existing `evaluation_results.csv` |
| `extract_activations` | `config_extract_activations.yaml` | Generate answers and save `[prompt + answer]` activations for probe training |
| `extract_exact_answers` | `config_extract_exact_answers.yaml` | Locate the exact answer span in generated answers (for token-selected probes) |
| `fit` | — | Lightning training; used for probes (`config_train_probe*.yaml`) |

### Confidence extraction

```bash
# TriviaQA with a local HuggingFace model (default config)
uv run main.py validate --model.init_args.model_name=Qwen/Qwen3.5-9B --data.init_args.batch_size=8

# MMLU with a local model: correctness and metrics are computed directly
uv run main.py validate -c configs/config_mmlu_base_validation.yaml

# TriviaQA with Gemini
uv run main.py validate -c configs/config_gcp_validation.yaml

# Judge the TriviaQA answers (local judge, or -c configs/config_gcp_evaluation.yaml for Gemini)
uv run main.py evaluation --llm_outputs_path=lightning_logs/<run>/llm_outputs.csv

# Recompute metrics from an existing evaluation
uv run main.py evaluation --use_existing_csv --llm_outputs_path=lightning_logs/<run>/llm_outputs.csv
uv run main.py evaluate_csv --csv_path=lightning_logs/<run>_evaluation/version_0/evaluation_results.csv
```

With `trainer.evaluate_correctness: true`, `validate` chains into `evaluation` automatically. Long runs
can flush outputs to disk (`flush_outputs_every_n_steps`) and resume an interrupted run
(`resume_from`).

| Extractor | How confidence is obtained |
|---|---|
| `SequencePosteriorExtractor` | Product of the token probabilities of the generated answer |
| `IsTruePosteriorExtractor` | Probability of "True" after an "Is this answer true?" follow-up prompt |
| `VerbalizedConfidenceExtractor` | Confidence value stated by the model |

GCP variants (`GCPSequencePosteriorExtractor`, `GCPIsTruePosteriorExtractor`) and MMLU variants
(`MMLUSequencePosteriorExtractor`, `MMLUVerbalizedExtractor`, …) adapt these to Gemini and to the
multiple-choice format. Prompts are `Prompt` / `ChatPrompt` templates in `callm/prompts/`, referenced
from configs by import path (e.g. `prompt: callm.prompts.CHAT_LABEL_PROB_PROMPT_ZERO_SHOT`);
`disable_thinking` turns off chat-template thinking for reasoning models.

## Probes

### Pipeline

Each stage reads the previous stage's files from disk:

```bash
# 1. Generate answers and save activations (+ llm_outputs.csv)
uv run main.py extract_activations -c configs/config_extract_activations.yaml \
  --output_dir activations_cache/<name> --trainer.default_root_dir activations_cache/<name>

# 2. Correctness labels (TriviaQA; MMLU labels come from exact match)
uv run main.py evaluation -c configs/config_gcp_evaluation.yaml \
  --llm_outputs_path activations_cache/<name>/lightning_logs/version_0/llm_outputs.csv

# 3. (optional) Exact answer spans, for token-selected probes
uv run main.py extract_exact_answers -c configs/config_extract_exact_answers.yaml \
  --data.init_args.llm_outputs_path <llm_outputs.csv>

# 4. Train a probe on the saved activations
uv run main.py fit -c configs/config_train_probe.yaml \
  --data.init_args.activations_dir activations_cache/<name>/lightning_logs/version_0 \
  --data.init_args.evaluation_csv <evaluation_results.csv> \
  --trainer.default_root_dir results/<experiment>
```

Extract activations with batch size 1 (the configs and scripts do): the model runs in bfloat16, and
batching changes rounding slightly and can occasionally change a greedy answer.

### Models

All in `callm/models/confidence_llm.py`:

| Class | Use |
|---|---|
| `CachedActivationsConfidenceAugmentedLLM` | Train on pre-computed activations (`ActivationsDataModule`, `SelectedTokensActivationsDataModule`); `model_dim` required |
| `SplitConfidenceAugmentedLLM` | Cached; pools prompt and answer separately and merges them (`activations_to_use: split`) |
| `ConfidenceAugmentedLLM` | Train with the LLM in the loop: one forward pass on `[prompt + answer]` per batch (`OnTheFlyActivationsDataModule`) — no activation files needed |
| `LiveEvalConfidenceAugmentedLLM` | Evaluate a trained probe end to end (`probe_checkpoint`): generate → `[prompt + answer]` activations → probe → exact-match metrics |

A `Probe` is optional layer pooling → sequence pooling → linear layer + sigmoid, trained with BCE:

- **Layers** (`layer_idx`): a single index, a list, or `"all"`. Several layers are combined by learned
  softmax weights (`LayerWeightedAverage`), optionally after normalization
  (`GlobalZNormWeightedAverage`, `LayerwiseZNormWeightedAverage`, `LayerNormWeightedAverage`).
- **Tokens** (`activations_to_use`): `prompt`, `answer`, `both`, `split`, or exact-answer windows
  (`exact_answer`, `exact_answer_first_token`, `exact_answer_last_token`,
  `exact_answer_before_first_token`, `exact_answer_after_last_token`).
- **Sequence pooling** (`pool_strategy`): `mean` or `last` (pooled in the datamodule), or `none` to keep
  the sequence for a learned pooling module: `MeanPooling`, `AttentivePooling`,
  `QueryAttentivePooling`, `MultiHeadAttentivePooling`, `QueryMultiHeadAttentivePooling`,
  `TransformerPooling`.
- **Schedules**: `WarmupExponentialLR`, `WarmupCosineAnnealingLR`.

A training run writes `metrics.csv`, the best checkpoint (probe weights only), `training_time.json`
and, for weighted layer pooling, `layer_weights_history.json`.

### Experiments

`scripts/run_exp*.sh` are the experiment sweeps (`bash scripts/run_expN_*.sh [GPU_IDX]`). They write to
`results/<experiment>/…`, skip runs whose `metrics.csv` already exists, and then call the summarize /
plot scripts in `scripts/`.

| Script | Experiment |
|---|---|
| `run_exp1_probe_sweeps.sh` | Linear probes per layer and weighted over layers; prompt / answer / both, mean pooling |
| `run_exp2_exact_tokens_probes.sh` | Linear probes on exact-answer token windows |
| `run_exp3_attentive_pooling.sh` | Attentive sequence pooling, single-layer probes (layers 14–23) |
| `run_exp4_layer_pooling.sh` | Normalization before layer pooling |
| `run_exp5_layer_weight_init.sh` | Layer weight initialization strategies |
| `run_exp6_layer_weight_init.sh` | Exp 5 scaled to 50k samples |
| `run_exp7_attentive_pooling_50k.sh` | Exp 3 scaled to 50k samples, warmup + cosine annealing |
| `run_exp8_exact_tokens_probes_50k.sh` | Exp 2 scaled to 50k samples (exact answers + probes) |
| `run_exp9_data_scaling.sh` | Training set size vs. overfitting (attentive pooling) |
| `run_exp10_mmlu_linear_probe_mean.sh` | MMLU linear probes, mean pooling |
| `run_exp11_mmlu_attentive_pooling.sh` | MMLU attentive pooling probes |
| `run_exp12_mmlu_split_probe.sh` | MMLU probes pooling prompt and answer separately |

Helpers: `run_extract_mmlu_activations.sh` and `run_extract_curated_activations_and_tokens.sh`
(activation extraction), `run_significance_rerun.sh` (re-run an experiment with several seeds),
`precompute_labels.sh` and `compare_pipelines.sh` (on-the-fly training).

## Metrics

Metrics come from the [`ecuas`](https://pypi.org/project/ecuas/) package and are logged at the end of
each validation epoch.

| Run | Logged |
|---|---|
| Answers (`validate` on MMLU, `evaluation`, live probe evaluation) | accuracy, ECE (10 bins), Brier score, cross-entropy, AUC, 0-ECUAS |
| Probe training (`fit`) | train/val loss, ECE, cross-entropy, AUC, n-ECUAS for n = 0, 1, 128 |

ECUAS is the Expected Cost for Uncertainty-Augmented Systems. `ecuas` also provides AURC, FPR@95,
γ-ECUAS and normalized variants.

## Configs

| Config | Subcommand | Backend / data |
|---|---|---|
| `config_base_validation.yaml` | `validate` | HuggingFace, TriviaQA |
| `config_mmlu_base_validation.yaml` | `validate` | HuggingFace, MMLU |
| `config_gcp_validation.yaml` / `config_mmlu_gcp_validation.yaml` | `validate` | Gemini, TriviaQA / MMLU |
| `config_base_evaluation.yaml` / `config_gcp_evaluation.yaml` | `evaluation` | Judge: HuggingFace / Gemini |
| `config_extract_activations*.yaml` | `extract_activations` | TriviaQA, curated TriviaQA, MMLU |
| `config_extract_exact_answers.yaml` | `extract_exact_answers` | Gemini |
| `config_train_probe*.yaml` | `fit` | Probes: linear, weighted layers, attentive, MMLU, split, data scaling, on-the-fly |
| `config_evaluate_with_probe.yaml` | `validate` | Live evaluation of a trained probe |

## Project structure

```
callm/
├── models/
│   ├── base.py                  # Batched-inference base: output flushing, reloading, resume
│   ├── outputs.py               # Shared CSV writers, exact match, metric sets
│   ├── llm.py                   # HuggingFace LLM (local GPU)
│   ├── gcp_llm.py               # Gemini LLM
│   ├── evaluator.py             # Judge modules: shared base + HuggingFace judge
│   ├── gcp_evaluator.py         # Gemini judge
│   ├── activation_extractor.py  # extract_activations
│   ├── exact_answer_extractor.py# extract_exact_answers
│   ├── confidence_llm.py        # Probe training / live evaluation modules
│   ├── probe.py                 # Probe, SplitProbe
│   ├── layer_pooling.py         # Layer pooling modules
│   ├── sequence_pooling.py      # Sequence pooling modules
│   └── lr_scheduling.py         # Warmup LR schedulers
├── data/
│   ├── triviaqa/                # TriviaQA, curated TriviaQA, is-true, judge and exact-answer data
│   ├── mmlu/                    # MMLU data (tokenized and untokenized)
│   ├── answers_data.py          # Second-stage data read from a previous llm_outputs.csv
│   ├── activations_data.py      # Cached activations for probe training
│   ├── on_the_fly_data.py       # [prompt + answer] tokens for on-the-fly training
│   ├── classification.py        # Registry of classification result sets
│   └── simulation.py            # Synthetic data (scripts/simulation_results.py)
├── extractors/                  # Answer + confidence extractors (base, TriviaQA, MMLU)
├── prompts/                     # Prompt / ChatPrompt templates
├── callbacks.py                 # Timing, layer weight history, validation predictions
├── config.py                    # Paths and environment
├── utils.py                     # Model / tokenizer loading and helpers
└── tests/
configs/                         # YAML run configurations
scripts/                         # Experiment sweeps, summaries and plots
cli.py                           # CalibrationCLI and CalibrationTrainer (custom subcommands)
main.py                          # Entry point and parser settings
```

Data modules are re-exported from `callm.data`, models from `callm.models`, so configs refer to them as
`callm.data.<Name>` / `callm.models.<Name>`. `Untokenized*` data modules yield raw prompts for Gemini;
the others tokenize with the HuggingFace model's tokenizer.

## Reproducing Paper Results

Main benchmark runs (TriviaQA, MMLU) are executed using the CLI commands detailed in the [Usage](#usage) section above. The scripts in `scripts/` are used to perform post-processing, re-evaluations, classification/simulation experiments, and to generate the final paper tables and figures.

Each script can be run from the project root using `uv run python scripts/<script>.py`.

### LLM Experiments & Evaluation

| Script | Description |
|---|---|
| `reevaluate_triviaqa.py` | Re-evaluate TriviaQA results with human labels and improved exact matching (`--strict` by default) |

### Classification & OOD Experiments

| Script | Description |
|---|---|
| `image_classification.py` | Train and evaluate image classification models (ResNet, VGG, DenseNet on CIFAR) |
| `image_classification.sh` | SLURM batch script for image classification experiments |
| `ood_analysis.py` | Out-of-distribution analysis for classification models under distribution shift |
| `ood_analysis.sh` | SLURM batch script for OOD analysis |

### Tables, Figures & Analysis

| Script | Description |
|---|---|
| `llm_confidence_results.py` | Generate LaTeX tables and ECUAS/γ-ECUAS line plots for TriviaQA results |
| `generate_mmlu_table.py` | Generate LaTeX tables and ECUAS/γ-ECUAS line plots for MMLU results |
| `classification_results.py` | Generate calibration metric tables and temperature-ECUAS plots |
| `simulation_results.py` | Run simulations and generate heatmaps comparing confidence procedures |
| `metrics_scatter_plots.py` | Generate metric scatter plots across classification datasets |
| `plot_cost.py` | Plot the ECUAS cost function for different values of *n* |

## License

See [LICENSE](LICENSE).
