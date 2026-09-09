import os
import csv
import glob
import torch
import random
from tqdm import tqdm
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader


class ActivationsDataset(Dataset):
    """Simple PyTorch Dataset for cached activations."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ActivationsDataModule(LightningDataModule):
    """
    LightningDataModule that loads pre-computed activations from disk,
    maps them to correctness labels from evaluation_results.csv, and
    splits the resulting dataset into train/val splits for probe training.
    """

    def __init__(
        self,
        activations_dir: str,
        evaluation_csv: str,
        layer_idx: int | list[int] | str = -1,
        batch_size: int = 64,
        num_workers: int = 4,
        pool_strategy: str = "mean",  # "mean", "last", "none"
        train_split: float = 0.8,
        seed: int = 42,
        activations_to_use: str = "prompt",
        max_samples: int | None = None,
        dtype: str = "float32",
        val_size: int | None = None,
    ):
        super().__init__()
        if activations_to_use not in ("prompt", "answer", "both"):
            raise ValueError(
                f"activations_to_use must be one of 'prompt', 'answer', 'both'. Got: {activations_to_use}"
            )
        self.activations_dir = activations_dir
        self.evaluation_csv = evaluation_csv
        self.layer_idx = layer_idx
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pool_strategy = pool_strategy
        self.train_split = train_split
        self.seed = seed
        self.activations_to_use = activations_to_use
        self.max_samples = max_samples
        self.dtype = dtype
        self.val_size = val_size

        self.train_dataset = None
        self.val_dataset = None
        self.layer_stats = None  # Populated in setup() for Z-norm layer pooling

    def _convert_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to the configured storage dtype."""
        if isinstance(self.dtype, torch.dtype):
            return tensor.to(self.dtype)
        dtype_str = str(self.dtype).lower()
        if dtype_str in ("bfloat16", "bf16"):
            return tensor.bfloat16()
        elif dtype_str in ("float16", "fp16", "half"):
            return tensor.half()
        elif dtype_str in ("float32", "fp32", "float"):
            return tensor.float()
        elif dtype_str in ("auto", "none", "preserve"):
            return tensor
        else:
            raise ValueError(
                f"Unsupported dtype: {self.dtype}. Choose from 'float32', 'bfloat16', 'float16', 'auto'."
            )

    def prepare_data(self):
        # Activations are already extracted on disk
        pass

    def compute_layer_statistics(self, dataset_samples: list) -> dict:
        """
        Compute per-layer and global mean/std over all multi-layer samples.

        Uses Welford's online algorithm for numerical stability.
        Only meaningful when ``is_multi_layer`` is True (multi-layer samples).

        Returns:
            dict with keys:
            - ``global_mean``: Tensor of shape (model_dim,)
            - ``global_std``:  Tensor of shape (model_dim,)
            - ``layer_means``: Tensor of shape (num_layers, model_dim)
            - ``layer_stds``:  Tensor of shape (num_layers, model_dim)
        """
        # Filter to multi-layer samples only
        multi_samples = [s for s in dataset_samples if s.get("is_multi_layer", False)]
        if not multi_samples:
            return None

        first_act = multi_samples[0]["activations"]
        # first_act shape: (num_layers, model_dim) or (num_layers, seq_len, model_dim)
        num_layers = first_act.shape[0]
        if first_act.ndim == 2:
            model_dim = first_act.shape[1]
        else:
            model_dim = first_act.shape[-1]

        # Welford's online algorithm per layer
        # For each layer, compute mean/var over all tokens across all samples
        layer_count = torch.zeros(num_layers)
        layer_mean = torch.zeros(num_layers, model_dim)
        layer_m2 = torch.zeros(num_layers, model_dim)

        for sample in multi_samples:
            act = sample["activations"]  # (num_layers, dim) or (num_layers, seq, dim)
            for layer_i in range(num_layers):
                layer_act = act[layer_i]  # (dim,) or (seq, dim)
                if layer_act.ndim == 1:
                    layer_act = layer_act.unsqueeze(0)  # (1, dim)

                for token_vec in layer_act:  # iterate over seq tokens
                    layer_count[layer_i] += 1
                    delta = token_vec - layer_mean[layer_i]
                    layer_mean[layer_i] += delta / layer_count[layer_i]
                    delta2 = token_vec - layer_mean[layer_i]
                    layer_m2[layer_i] += delta * delta2

        # Compute per-layer std
        layer_var = layer_m2 / layer_count.unsqueeze(1).clamp(min=1)
        layer_stds = torch.sqrt(layer_var)

        # Compute global stats (mean of layer means, pooled std)
        global_mean = layer_mean.mean(dim=0)
        # Pooled variance: average of per-layer variances + variance of layer means
        global_var = layer_var.mean(dim=0) + layer_mean.var(dim=0)
        global_std = torch.sqrt(global_var)

        stats = {
            "global_mean": global_mean,
            "global_std": global_std,
            "layer_means": layer_mean,
            "layer_stds": layer_stds,
        }

        print(
            f"Computed layer statistics over {int(layer_count[0].item())} vectors/layer "
            f"({len(multi_samples)} samples, {num_layers} layers, dim={model_dim})"
        )
        return stats

    def setup(self, stage: str = None):
        if not os.path.exists(self.evaluation_csv):
            raise FileNotFoundError(
                f"Evaluation CSV not found at {self.evaluation_csv}"
            )

        # 1. Load correctness labels from evaluation CSV
        correctness_map = {}
        with open(self.evaluation_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Handle capitalization of Correct/correct
            correct_col = None
            for col in reader.fieldnames or []:
                if col.lower() == "correct":
                    correct_col = col
                    break
            if correct_col is None:
                raise ValueError(
                    f"Could not find 'Correct' column in {self.evaluation_csv}"
                )

            for i, row in enumerate(reader):
                is_correct = row[correct_col].strip().lower() == "yes"
                correctness_map[i] = is_correct

        # 2. Stream load activation shards and extract target layer(s) on-the-fly to save RAM
        shard_files = sorted(
            glob.glob(os.path.join(self.activations_dir, "activations_*.pt"))
        )
        if not shard_files:
            raise FileNotFoundError(
                f"No activation shards found in {self.activations_dir}"
            )

        dataset_samples = []
        for shard_file in tqdm(
            shard_files,
            desc=f"Loading activation shards ({self.activations_to_use})",
        ):
            try:
                shard_samples = torch.load(
                    shard_file, map_location="cpu", weights_only=False
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load activation shard {shard_file}: {e}")

            for sample in shard_samples:
                idx = sample["sample_idx"]
                if idx not in correctness_map:
                    # If evaluation is incomplete or samples mismatched, skip
                    continue

                layer_keys = sorted(list(sample["activations"].keys()))
                num_layers = len(layer_keys)
                if num_layers == 0:
                    raise ValueError(f"Sample {idx} has no saved layer activations.")

                def _resolve_single_layer(l_idx: int) -> int:
                    if l_idx in sample["activations"]:
                        return l_idx
                    return layer_keys[l_idx % num_layers]

                # Support single integer, list of integers, or "all"
                if isinstance(self.layer_idx, int):
                    resolved_layer_keys = [_resolve_single_layer(self.layer_idx)]
                    is_multi_layer = False
                elif isinstance(self.layer_idx, list):
                    resolved_layer_keys = [
                        _resolve_single_layer(layer_i) for layer_i in self.layer_idx
                    ]
                    is_multi_layer = True
                elif (
                    isinstance(self.layer_idx, str) and self.layer_idx.lower() == "all"
                ):
                    resolved_layer_keys = list(layer_keys)
                    is_multi_layer = True
                else:
                    raise ValueError(
                        f"Unsupported type/value for layer_idx: {self.layer_idx}"
                    )

                # Retrieve and pool activations for the resolved layers
                layer_acts = []
                for resolved_layer_key in resolved_layer_keys:
                    act_tensor = sample["activations"][resolved_layer_key]

                    # Slice activations based on activations_to_use
                    prompt_len = sample.get("prompt_length", sample["seq_len"])
                    if self.activations_to_use == "prompt":
                        act_tensor = act_tensor[:prompt_len]
                    elif self.activations_to_use == "answer":
                        act_tensor = act_tensor[prompt_len:]
                    elif self.activations_to_use == "both":
                        pass
                    else:
                        raise ValueError(
                            f"Unknown activations_to_use: {self.activations_to_use}"
                        )

                    if act_tensor.shape[0] == 0:
                        # Safe fallback for empty slices (e.g. empty answer or old cache)
                        if self.pool_strategy in ("mean", "last"):
                            pooled_act = torch.zeros(
                                act_tensor.shape[1],
                                device=act_tensor.device,
                                dtype=act_tensor.dtype,
                            )
                        else:
                            pooled_act = act_tensor
                    else:
                        # Apply pool strategy
                        if self.pool_strategy == "mean":
                            pooled_act = act_tensor.mean(dim=0)  # (hidden_dim)
                        elif self.pool_strategy == "last":
                            pooled_act = act_tensor[-1]  # (hidden_dim)
                        elif self.pool_strategy == "none":
                            pooled_act = act_tensor  # (seq_len, hidden_dim)
                        else:
                            raise ValueError(
                                f"Unknown pool strategy: {self.pool_strategy}"
                            )

                    layer_acts.append(self._convert_dtype(pooled_act))

                if is_multi_layer:
                    # Stack along a new layer dimension (dim=0)
                    final_act = torch.stack(layer_acts, dim=0)
                else:
                    final_act = layer_acts[0]

                dataset_samples.append(
                    {
                        "activations": final_act,
                        "correct": correctness_map[idx],
                        "is_multi_layer": is_multi_layer,
                    }
                )

            del shard_samples

        if not dataset_samples:
            raise ValueError(
                "No samples could be matched between activations and evaluation CSV."
            )

        print(
            f"Loaded and mapped {len(dataset_samples)} samples from "
            f"{self.activations_dir} and {self.evaluation_csv}"
        )

        # Compute layer normalization statistics (for Z-norm layer pooling)
        self.layer_stats = self.compute_layer_statistics(dataset_samples)

        # 4. Split into train and validation
        rng = random.Random(self.seed)
        shuffled_samples = list(dataset_samples)
        rng.shuffle(shuffled_samples)

        if self.val_size is not None:
            # Fixed validation set: take last val_size samples as validation,
            # then subsample from the remainder for training.
            # This ensures the validation set is identical regardless of
            # max_samples, enabling fair comparison across training set sizes.
            if self.val_size >= len(shuffled_samples):
                raise ValueError(
                    f"val_size ({self.val_size}) must be smaller than total "
                    f"samples ({len(shuffled_samples)})"
                )
            val_samples = shuffled_samples[-self.val_size :]
            remaining = shuffled_samples[: -self.val_size]
            if self.max_samples is not None:
                train_samples = remaining[: self.max_samples]
            else:
                train_samples = remaining
        else:
            # Original behavior: ratio-based split
            split_idx = int(len(shuffled_samples) * self.train_split)
            train_samples = shuffled_samples[:split_idx]
            val_samples = shuffled_samples[split_idx:]

        self.train_dataset = ActivationsDataset(train_samples)
        self.val_dataset = ActivationsDataset(val_samples)

        print(
            f"Train split: {len(self.train_dataset)} samples, "
            f"Val split: {len(self.val_dataset)} samples."
        )

    @staticmethod
    def collate_fn(batch):
        correct = torch.tensor([item["correct"] for item in batch], dtype=torch.long)
        first_act = batch[0]["activations"]
        is_multi_layer = batch[0].get("is_multi_layer", False)
        seq_mask = None

        if not is_multi_layer:
            if first_act.ndim == 1:
                # Already pooled: stack into (batch_size, hidden_dim)
                activations = torch.stack([item["activations"] for item in batch])
            elif first_act.ndim == 2:
                # Sequence: pad into (batch_size, max_seq_len, hidden_dim)
                activations_list = [item["activations"] for item in batch]
                activations = torch.nn.utils.rnn.pad_sequence(
                    activations_list, batch_first=True, padding_value=0.0
                )
                # Generate seq_mask: True = valid, False = padding
                seq_lengths = [a.shape[0] for a in activations_list]
                max_len = activations.shape[1]
                seq_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
                for i, length in enumerate(seq_lengths):
                    seq_mask[i, :length] = True
            else:
                raise ValueError(
                    f"Unexpected activation tensor shape: {first_act.shape}"
                )
        else:
            if first_act.ndim == 2:
                # Multi-layer, pre-pooled: shape of each item is (num_layers, hidden_dim)
                # Stack into (batch_size, num_layers, hidden_dim)
                activations = torch.stack([item["activations"] for item in batch])
            elif first_act.ndim == 3:
                # Multi-layer, sequence: shape of each item is (num_layers, seq_len, hidden_dim)
                # Pad sequence dimension (dim=1) across batch.
                # To do this, permute each to (seq_len, num_layers, hidden_dim)
                permuted_list = [item["activations"].permute(1, 0, 2) for item in batch]
                padded = torch.nn.utils.rnn.pad_sequence(
                    permuted_list, batch_first=True, padding_value=0.0
                )  # (batch_size, max_seq_len, num_layers, hidden_dim)
                # Permute back to (batch_size, num_layers, max_seq_len, hidden_dim)
                activations = padded.permute(0, 2, 1, 3)
                # Generate seq_mask: True = valid, False = padding
                seq_lengths = [item["activations"].shape[1] for item in batch]
                max_len = activations.shape[2]
                seq_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
                for i, length in enumerate(seq_lengths):
                    seq_mask[i, :length] = True
            else:
                raise ValueError(
                    f"Unexpected multi-layer activation tensor shape: {first_act.shape}"
                )

        result = {
            "activations": activations,
            "correct": correct,
        }
        if seq_mask is not None:
            result["seq_mask"] = seq_mask
        return result

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


class SelectedTokensActivationsDataModule(ActivationsDataModule):
    """
    LightningDataModule that loads pre-computed activations from disk,
    aligns them with exact answer entity indices extracted by an LLM / evaluator,
    and slices the activations for specific token window strategies.

    100% Sample Retention Guarantee:
    No samples are filtered out. If an exact answer entity cannot be found or is invalid,
    it falls back to a default token slice (e.g. prompt boundary or first/last answer token).
    """

    VALID_ACTIVATIONS_TO_USE = (
        "prompt",
        "answer",
        "both",
        "exact_answer",
        "exact_answer_first_token",
        "exact_answer_last_token",
        "exact_answer_before_first_token",
        "exact_answer_after_last_token",
    )

    def __init__(
        self,
        activations_dir: str,
        evaluation_csv: str,
        layer_idx: int | list[int] | str = -1,
        batch_size: int = 64,
        num_workers: int = 4,
        pool_strategy: str = "mean",
        train_split: float = 0.8,
        seed: int = 42,
        activations_to_use: str = "prompt",
        model_name: str = "Qwen/Qwen3.5-9B",
        prompt: str = "callm.prompts.base.TRANSPARENT_PROMPT_ZERO_SHOT",
        max_samples: int | None = None,
        dtype: str = "float32",
    ):
        if activations_to_use not in self.VALID_ACTIVATIONS_TO_USE:
            raise ValueError(
                f"activations_to_use must be one of {self.VALID_ACTIVATIONS_TO_USE}. Got: {activations_to_use}"
            )

        parent_activations = (
            "both"
            if activations_to_use
            in (
                "exact_answer",
                "exact_answer_first_token",
                "exact_answer_last_token",
                "exact_answer_before_first_token",
                "exact_answer_after_last_token",
            )
            else activations_to_use
        )

        super().__init__(
            activations_dir=activations_dir,
            evaluation_csv=evaluation_csv,
            layer_idx=layer_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            pool_strategy=pool_strategy,
            train_split=train_split,
            seed=seed,
            activations_to_use=parent_activations,
            max_samples=max_samples,
            dtype=dtype,
        )
        self.activations_to_use = activations_to_use
        self.model_name = model_name
        self.prompt = prompt
        self.tokenizer = None

    def setup(self, stage: str = None):
        if not os.path.exists(self.evaluation_csv):
            raise FileNotFoundError(
                f"Evaluation CSV not found at {self.evaluation_csv}"
            )

        from callm.utils import get_tokenizer_for_model, get_indices_of_exact_answer
        import importlib

        if self.tokenizer is None:
            self.tokenizer = get_tokenizer_for_model(self.model_name)

        if isinstance(self.prompt, str):
            module_name, obj_name = self.prompt.rsplit(".", 1)
            module = importlib.import_module(module_name)
            self.prompt_obj = getattr(module, obj_name)
        else:
            self.prompt_obj = self.prompt

        # Load correctness map and exact answer information from CSV
        exact_info_map = {}
        with open(self.evaluation_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            field_map = {col.lower(): col for col in reader.fieldnames or []}

            correct_col = field_map.get("correct")
            if correct_col is None:
                raise ValueError(
                    f"Could not find 'Correct' column in {self.evaluation_csv}"
                )

            ea_col = field_map.get("exact answer") or field_map.get("exact_answer")
            vea_col = field_map.get("valid exact answer") or field_map.get(
                "valid_exact_answer"
            )
            q_col = field_map.get("question")
            r_col = field_map.get("raw output") or field_map.get("raw_output")

            for i, row in enumerate(reader):
                is_correct = row[correct_col].strip().lower() == "yes"
                exact_ans = row.get(ea_col, "") if ea_col else ""
                valid_exact = (
                    (row.get(vea_col, "").strip().lower() in ("yes", "true", "1"))
                    if vea_col
                    else False
                )
                question = row.get(q_col, "") if q_col else ""
                raw_out = row.get(r_col, "") if r_col else ""

                exact_info_map[i] = {
                    "correct": is_correct,
                    "exact_answer": exact_ans,
                    "valid_exact_answer": valid_exact,
                    "question": question,
                    "raw_output": raw_out,
                }

        shard_files = sorted(
            glob.glob(os.path.join(self.activations_dir, "activations_*.pt"))
        )
        if not shard_files:
            raise FileNotFoundError(
                f"No activation shards found in {self.activations_dir}"
            )

        dataset_samples = []
        for shard_file in tqdm(
            shard_files,
            desc=f"Loading activation shards ({self.activations_to_use})",
        ):
            try:
                shard_samples = torch.load(
                    shard_file, map_location="cpu", weights_only=False
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load activation shard {shard_file}: {e}")

            for sample in shard_samples:
                idx = sample["sample_idx"]
                if idx not in exact_info_map:
                    continue

                info = exact_info_map[idx]
                is_correct = info["correct"]

                layer_keys = sorted(list(sample["activations"].keys()))
                num_layers = len(layer_keys)
                if num_layers == 0:
                    raise ValueError(f"Sample {idx} has no saved layer activations.")

                def _resolve_single_layer(l_idx: int) -> int:
                    if l_idx in sample["activations"]:
                        return l_idx
                    return layer_keys[l_idx % num_layers]

                if isinstance(self.layer_idx, int):
                    resolved_layer_keys = [_resolve_single_layer(self.layer_idx)]
                    is_multi_layer = False
                elif isinstance(self.layer_idx, list):
                    resolved_layer_keys = [
                        _resolve_single_layer(layer_i) for layer_i in self.layer_idx
                    ]
                    is_multi_layer = True
                elif (
                    isinstance(self.layer_idx, str) and self.layer_idx.lower() == "all"
                ):
                    resolved_layer_keys = list(layer_keys)
                    is_multi_layer = True
                else:
                    raise ValueError(
                        f"Unsupported type/value for layer_idx: {self.layer_idx}"
                    )

                layer_acts = []
                for resolved_layer_key in resolved_layer_keys:
                    act_tensor = sample["activations"][resolved_layer_key]
                    prompt_len = sample.get("prompt_length", sample["seq_len"])
                    seq_len = act_tensor.shape[0]

                    if self.activations_to_use in (
                        "exact_answer",
                        "exact_answer_first_token",
                        "exact_answer_last_token",
                        "exact_answer_before_first_token",
                        "exact_answer_after_last_token",
                    ):
                        exact_range = None
                        if info["valid_exact_answer"] and info["exact_answer"]:
                            # Reconstruct prompt + answer token IDs
                            prompt_text = self.prompt_obj(question=info["question"])
                            from callm.prompts.base import ChatPrompt

                            if isinstance(self.prompt_obj, ChatPrompt):
                                chat_template_kwargs = {
                                    "tokenize": True,
                                    "enable_thinking": False,
                                }
                                has_assistant_prefill = (
                                    self.prompt_obj.assistant_template is not None
                                )
                                if has_assistant_prefill:
                                    chat_template_kwargs["continue_final_message"] = (
                                        True
                                    )
                                else:
                                    chat_template_kwargs["add_generation_prompt"] = True

                                prompt_tokens = self.tokenizer.apply_chat_template(
                                    prompt_text, **chat_template_kwargs
                                )
                                if hasattr(prompt_tokens, "input_ids"):
                                    prompt_tokens = prompt_tokens["input_ids"]
                                elif (
                                    isinstance(prompt_tokens, dict)
                                    and "input_ids" in prompt_tokens
                                ):
                                    prompt_tokens = prompt_tokens["input_ids"]

                                if torch.is_tensor(prompt_tokens):
                                    prompt_ids = prompt_tokens.squeeze().tolist()
                                    if isinstance(prompt_ids, int):
                                        prompt_ids = [prompt_ids]
                                elif isinstance(prompt_tokens, list):
                                    prompt_ids = prompt_tokens
                                else:
                                    prompt_ids = list(prompt_tokens)
                            else:
                                prompt_ids = self.tokenizer.encode(
                                    prompt_text, add_special_tokens=True
                                )
                                if torch.is_tensor(prompt_ids):
                                    prompt_ids = prompt_ids.squeeze().tolist()
                                    if isinstance(prompt_ids, int):
                                        prompt_ids = [prompt_ids]
                                elif not isinstance(prompt_ids, list):
                                    prompt_ids = list(prompt_ids)

                            answer_ids = self.tokenizer.encode(
                                info["raw_output"], add_special_tokens=False
                            )
                            if torch.is_tensor(answer_ids):
                                answer_ids = answer_ids.squeeze().tolist()
                                if isinstance(answer_ids, int):
                                    answer_ids = [answer_ids]
                            elif hasattr(answer_ids, "input_ids"):
                                answer_ids = answer_ids["input_ids"]
                            if not isinstance(answer_ids, list):
                                answer_ids = list(answer_ids)

                            full_ids = prompt_ids + answer_ids

                            exact_range = get_indices_of_exact_answer(
                                self.tokenizer,
                                full_ids,
                                info["exact_answer"],
                                len(prompt_ids),
                            )

                        if exact_range is not None:
                            lower = exact_range[0]
                            higher = exact_range[-1]
                            if self.activations_to_use == "exact_answer":
                                act_tensor = act_tensor[lower : higher + 1]
                            elif self.activations_to_use == "exact_answer_first_token":
                                act_tensor = act_tensor[lower : lower + 1]
                            elif self.activations_to_use == "exact_answer_last_token":
                                act_tensor = act_tensor[higher : higher + 1]
                            elif (
                                self.activations_to_use
                                == "exact_answer_before_first_token"
                            ):
                                start = max(0, lower - 1)
                                end = max(1, lower)
                                act_tensor = act_tensor[start:end]
                            elif (
                                self.activations_to_use
                                == "exact_answer_after_last_token"
                            ):
                                start = min(seq_len - 1, higher + 1)
                                end = min(seq_len, higher + 2)
                                if start >= end:
                                    start = max(0, seq_len - 1)
                                    end = seq_len
                                act_tensor = act_tensor[start:end]
                        else:
                            # 100% Retention Fallback when exact answer range is not found
                            if self.activations_to_use == "exact_answer":
                                act_tensor = act_tensor[prompt_len:]
                            elif self.activations_to_use == "exact_answer_first_token":
                                start = min(prompt_len, max(0, seq_len - 1))
                                act_tensor = act_tensor[start : start + 1]
                            elif self.activations_to_use == "exact_answer_last_token":
                                start = max(0, seq_len - 1)
                                act_tensor = act_tensor[start : start + 1]
                            elif (
                                self.activations_to_use
                                == "exact_answer_before_first_token"
                            ):
                                start = max(0, prompt_len - 1)
                                act_tensor = act_tensor[start : start + 1]
                            elif (
                                self.activations_to_use
                                == "exact_answer_after_last_token"
                            ):
                                start = min(prompt_len, max(0, seq_len - 1))
                                act_tensor = act_tensor[start : start + 1]

                    elif self.activations_to_use == "prompt":
                        act_tensor = act_tensor[:prompt_len]
                    elif self.activations_to_use == "answer":
                        act_tensor = act_tensor[prompt_len:]
                    elif self.activations_to_use == "both":
                        pass

                    if act_tensor.shape[0] == 0:
                        if self.pool_strategy in ("mean", "last"):
                            pooled_act = torch.zeros(
                                act_tensor.shape[1],
                                device=act_tensor.device,
                                dtype=act_tensor.dtype,
                            )
                        else:
                            pooled_act = act_tensor
                    else:
                        if self.pool_strategy == "mean":
                            pooled_act = act_tensor.mean(dim=0)
                        elif self.pool_strategy == "last":
                            pooled_act = act_tensor[-1]
                        elif self.pool_strategy == "none":
                            pooled_act = act_tensor
                        else:
                            raise ValueError(
                                f"Unknown pool strategy: {self.pool_strategy}"
                            )

                    layer_acts.append(self._convert_dtype(pooled_act))

                if is_multi_layer:
                    final_act = torch.stack(layer_acts, dim=0)
                else:
                    final_act = layer_acts[0]

                dataset_samples.append(
                    {
                        "activations": final_act,
                        "correct": is_correct,
                        "is_multi_layer": is_multi_layer,
                    }
                )

            del shard_samples

        if not dataset_samples:
            raise ValueError(
                "No samples could be matched between activations and evaluation CSV."
            )

        print(
            f"Loaded and mapped {len(dataset_samples)} samples for "
            f"SelectedTokensActivationsDataModule ({self.activations_to_use})"
        )

        rng = random.Random(self.seed)
        shuffled_samples = list(dataset_samples)
        rng.shuffle(shuffled_samples)

        split_idx = int(len(shuffled_samples) * self.train_split)
        train_samples = shuffled_samples[:split_idx]
        val_samples = shuffled_samples[split_idx:]

        self.train_dataset = ActivationsDataset(train_samples)
        self.val_dataset = ActivationsDataset(val_samples)

        print(
            f"Train split: {len(self.train_dataset)} samples, "
            f"Val split: {len(self.val_dataset)} samples."
        )
