#!/usr/bin/env python
"""Quick test to verify data loading works correctly."""

from callm.data.triviaQA import TriviaQADataModule
from callm.prompts import (
    CHAT_LABEL_PROB_PROMPT_ZERO_SHOT,
    VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT,
)

# 1. Test with a string prompt (should work even for models without chat templates)
print("Testing with string prompt (backward compatibility)...")
dm_string = TriviaQADataModule(
    batch_size=2,
    model_name="google/flan-t5-small",
    prompt=VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT,
    max_samples=5,
    seed=42,
)
dm_string.prepare_data()
dm_string.setup("validate")
print("✓ String prompt works correctly!")

# 2. Test with a chat prompt on a model that doesn't support it (should fail)
print("\nTesting with chat prompt on unsupported model (expecting failure)...")
dm_chat = TriviaQADataModule(
    batch_size=2,
    model_name="google/flan-t5-small",
    prompt=CHAT_LABEL_PROB_PROMPT_ZERO_SHOT,
    max_samples=5,
    seed=42,
)
try:
    dm_chat.setup("validate")
    print("X Error: Should have raised an exception for unsupported chat template!")
except (ValueError, AttributeError) as e:
    print(f"✓ Correctly raised expected error: {type(e).__name__}")

print(
    "\n✓ Integration tests reflect both supported (string) and unsupported (chat) scenarios!"
)
