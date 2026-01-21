"""
Base DataModule for loading LLM outputs with answers.

Provides common functionality for data modules that need to load
LLM-generated answers from CSV files for validation or evaluation.
"""

from lightning import LightningDataModule
import csv
import os
import glob
import torch

from callm.utils import get_tokenizer_for_model, get_last_llm_outputs_path


DEFAULT_MAX_LENGTH = 512


class AnswersDataModule(LightningDataModule):
    """Base DataModule for loading LLM outputs with answers from CSV.

    This base class provides common functionality for modules that need
    to load question/answer pairs from LLM output CSV files.
    """

    def __init__(
        self,
        llm_outputs_path: str = None,
        model_name: str = "google/flan-t5-base",
        batch_size: int = 1,
        num_workers: int = 8,
        max_length: int = None,
        log_dir: str = None,
        resume_from: str = None,
        max_samples: int = None,
        seed: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.llm_outputs_path = llm_outputs_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.log_dir = log_dir
        self.resume_from = resume_from
        self.tokenizer = None
        self.skip_indices = set()
        self.max_samples = max_samples
        self.seed = seed

    def _setup_tokenizer(self):
        """Initialize tokenizer and max_length if not already set."""
        if self.tokenizer is None:
            self.tokenizer = get_tokenizer_for_model(self.model_name)

            # Determine max_length from model config if not provided
            if self.max_length is None:
                model_max_length = getattr(self.tokenizer, "model_max_length", None)
                if model_max_length and model_max_length < 1e9:
                    self.max_length = model_max_length
                else:
                    self.max_length = DEFAULT_MAX_LENGTH

    def _resolve_llm_outputs_path(self):
        """Resolve the LLM outputs path, using log_dir fallback if needed."""
        if self.llm_outputs_path is None and self.log_dir is not None:
            self.llm_outputs_path = get_last_llm_outputs_path(self.log_dir)

        if self.llm_outputs_path is None:
            raise ValueError(
                "llm_outputs_path must be provided or resolvable from log_dir"
            )

        return self.llm_outputs_path

    def _get_skip_indices(self):
        """Get indices to skip when resuming from checkpoint."""
        skip_indices = set()
        if self.resume_from:
            if not os.path.exists(self.resume_from):
                raise ValueError(f"Resume path {self.resume_from} does not exist.")
            else:
                # Look for files matching pattern
                pattern = os.path.join(self.resume_from, "temp_eval_results_rank0_*.pt")
                found_files = sorted(
                    glob.glob(pattern),
                    key=lambda x: int(x.split("_")[-1].split(".")[0]),
                )

                if found_files:
                    print(f"Found {len(found_files)} temp files to resume from.")
                    # Load all files to get indices
                    for fpath in found_files:
                        try:
                            # Load on CPU to avoid CUDA errors if not available or OOM
                            chunk = torch.load(fpath, map_location="cpu")
                            for item in chunk:
                                idx = item.get("index")
                                if idx is not None:
                                    skip_indices.add(idx)
                        except Exception as e:
                            print(f"Error loading {fpath}: {e}")

                    print(f"Found {len(skip_indices)} indices to skip.")
                else:
                    print(f"No temp files found in {self.resume_from}")
        return skip_indices

    def load_llm_outputs_from_csv(self):
        """Load LLM outputs from CSV file.

        Returns:
            List of dicts containing: question, gold_answers, pred_answer,
            confidence, raw_output, index
        """
        self._resolve_llm_outputs_path()
        self.skip_indices = self._get_skip_indices()

        rows = []
        with open(self.llm_outputs_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i in self.skip_indices:
                    continue

                required_cols = [
                    "question",
                    "gold_answers",
                    "pred_answer",
                    "confidence",
                ]
                if any(row.get(k) is None for k in required_cols):
                    raise ValueError(
                        f"Warning: Skipping malformed row {i} in {self.llm_outputs_path}. "
                    )

                rows.append(
                    {
                        "question": row["question"],
                        "gold_answers": [
                            g.strip().lower() for g in row["gold_answers"].split("|")
                        ],
                        "pred_answer": row["pred_answer"],
                        "confidence": row["confidence"],
                        "raw_output": row.get("raw_output", ""),
                        "index": i,
                    }
                )

        print(f"\nLoaded {len(rows)} rows from CSV file: {self.llm_outputs_path}")
        return rows
