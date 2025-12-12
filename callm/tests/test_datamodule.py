#!/usr/bin/env python
"""Quick test to verify data loading works correctly."""

from callm.data.triviaQA import TriviaQADataModule

# Test data module initialization and setup
dm = TriviaQADataModule(
    batch_size=2, model_name="flan-t5-small", max_samples=5, seed=42
)

print("Preparing data...")
dm.prepare_data()

print("Setting up...")
dm.setup("validate")

print("\nDataset created successfully!")
print(f"Number of samples: {len(dm.triviaQA_val)}")

print("\nTesting dataloader...")
dataloader = dm.val_dataloader()
batch = next(iter(dataloader))

print(f"Batch keys: {batch.keys()}")
print(f"Batch size: {len(batch['label'])}")
print(f"\nFirst answer options: {batch['label'][0][:3]}")
print(f"Input IDs type: {type(batch['input_ids'])}")
print(f"Input IDs shape: {batch['input_ids'].shape}")
print(f"Attention mask shape: {batch['attention_mask'].shape}")

print("\n✓ Data module working correctly!")
print("✓ Tensors pre-stacked in collate_fn (not in validation_step)")
print("✓ Batch ready for forward pass - no processing needed!")
