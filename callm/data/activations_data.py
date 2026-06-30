import os
import csv
import glob
import torch
import random
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
        layer_idx: int = -1,
        batch_size: int = 64,
        num_workers: int = 4,
        pool_strategy: str = "mean",  # "mean", "last", "none"
        train_split: float = 0.8,
        seed: int = 42,
    ):
        super().__init__()
        self.activations_dir = activations_dir
        self.evaluation_csv = evaluation_csv
        self.layer_idx = layer_idx
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pool_strategy = pool_strategy
        self.train_split = train_split
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        # Activations are already extracted on disk
        pass

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

        # 2. Find and load all activation shards
        shard_files = sorted(
            glob.glob(os.path.join(self.activations_dir, "activations_*.pt"))
        )
        if not shard_files:
            raise FileNotFoundError(
                f"No activation shards found in {self.activations_dir}"
            )

        all_samples = []
        for shard_file in shard_files:
            try:
                shard_data = torch.load(
                    shard_file, map_location="cpu", weights_only=False
                )
                all_samples.extend(shard_data)
            except Exception as e:
                raise RuntimeError(f"Failed to load activation shard {shard_file}: {e}")

        # 3. Align and pool activations
        dataset_samples = []
        for sample in all_samples:
            idx = sample["sample_idx"]
            if idx not in correctness_map:
                # If evaluation is incomplete or samples mismatched, skip or warning
                continue

            # Resolve layer index
            layer_keys = sorted(list(sample["activations"].keys()))
            num_layers = len(layer_keys)
            if num_layers == 0:
                raise ValueError(f"Sample {idx} has no saved layer activations.")

            resolved_layer_key = layer_keys[self.layer_idx % num_layers]
            act_tensor = sample["activations"][resolved_layer_key]

            # Apply pool strategy
            if self.pool_strategy == "mean":
                pooled_act = act_tensor.mean(dim=0)  # (hidden_dim)
            elif self.pool_strategy == "last":
                pooled_act = act_tensor[-1]  # (hidden_dim)
            elif self.pool_strategy == "none":
                pooled_act = act_tensor  # (seq_len, hidden_dim)
            else:
                raise ValueError(f"Unknown pool strategy: {self.pool_strategy}")

            dataset_samples.append(
                {"activations": pooled_act, "correct": correctness_map[idx]}
            )

        if not dataset_samples:
            raise ValueError(
                "No samples could be matched between activations and evaluation CSV."
            )

        print(
            f"Loaded and mapped {len(dataset_samples)} samples from "
            f"{self.activations_dir} and {self.evaluation_csv}"
        )

        # 4. Split into train and validation
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

    @staticmethod
    def collate_fn(batch):
        correct = torch.tensor([item["correct"] for item in batch], dtype=torch.long)

        first_act = batch[0]["activations"]
        if first_act.ndim == 1:
            # Already pooled: stack into (batch_size, hidden_dim)
            activations = torch.stack([item["activations"] for item in batch])
        elif first_act.ndim == 2:
            # Sequence: pad into (batch_size, max_seq_len, hidden_dim)
            activations_list = [item["activations"] for item in batch]
            activations = torch.nn.utils.rnn.pad_sequence(
                activations_list, batch_first=True, padding_value=0.0
            )
        else:
            raise ValueError(f"Unexpected activation tensor shape: {first_act.shape}")

        return {
            "activations": activations,
            "correct": correct,
        }

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
