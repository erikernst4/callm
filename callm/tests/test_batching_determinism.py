"""
Synthetic tests for LLM batching determinism.

Verifies that:
1. Greedy generation produces identical text outputs regardless of batch size
2. Non-generative forward pass hidden states are consistent across batch sizes
3. Attention masking correctly isolates per-sample computations in batched inference

These tests use a tiny, randomly initialised causal Qwen2 model in float32 on
CPU (with the Qwen tokenizer and its left padding), so they exercise the same
padding and masking path as the real decoder-only models. Real models in
bfloat16 are only approximately batch-invariant: rounding differs between
batch sizes and can occasionally flip a greedy decoding step.
"""

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from callm.utils import get_tokenizer_for_model

TOKENIZER_NAME = "Qwen/Qwen2.5-3B"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Build the tiny model and load the tokenizer once for all tests in this module."""
    tokenizer = get_tokenizer_for_model(TOKENIZER_NAME)
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = Qwen2ForCausalLM(config).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


# ── Test prompts with varying lengths ──────────────────────────────────
TEST_PROMPTS = [
    "What is the capital of France?",
    "Who wrote the novel '1984'?",
    "What is the boiling point of water in Celsius?",
    "Name the largest planet in our solar system.",
    "Who painted the Mona Lisa?",
    "What year did World War II end?",
]


def _hidden_states(model, inputs):
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            output_hidden_states=True,
            return_dict=True,
        )
    return outputs.hidden_states[-1]


class TestGreedyGenerationDeterminism:
    """Verify that greedy generation is identical across batch sizes."""

    def _generate_batch(self, model, tokenizer, prompts, max_new_tokens=50):
        """Generate outputs for prompts in a single batch."""
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=model.config.pad_token_id,
            )

        input_length = input_ids.shape[1]
        return [
            tokenizer.decode(output[i][input_length:], skip_special_tokens=True)
            for i in range(len(prompts))
        ]

    def _generate_single(self, model, tokenizer, prompt, max_new_tokens=50):
        """Generate output for a single prompt (batch_size=1)."""
        return self._generate_batch(model, tokenizer, [prompt], max_new_tokens)[0]

    def test_batch1_vs_batch1(self, model_and_tokenizer):
        """Same prompt, two separate batch_size=1 calls → identical output."""
        model, tokenizer = model_and_tokenizer
        prompt = TEST_PROMPTS[0]

        out1 = self._generate_single(model, tokenizer, prompt)
        out2 = self._generate_single(model, tokenizer, prompt)

        assert out1 == out2, (
            f"Same prompt produced different outputs:\n"
            f"  Run 1: {out1!r}\n"
            f"  Run 2: {out2!r}"
        )

    def test_batch1_vs_batch2(self, model_and_tokenizer):
        """Compare batch_size=1 outputs with batch_size=2 outputs."""
        model, tokenizer = model_and_tokenizer
        prompts = TEST_PROMPTS[:2]

        singles = [self._generate_single(model, tokenizer, p) for p in prompts]
        batched = self._generate_batch(model, tokenizer, prompts)

        for i, (single, batch_out) in enumerate(zip(singles, batched)):
            assert single == batch_out, (
                f"Prompt {i} ({prompts[i]!r}) produced different output:\n"
                f"  batch_size=1: {single!r}\n"
                f"  batch_size=2: {batch_out!r}"
            )

    def test_batch1_vs_batch4(self, model_and_tokenizer):
        """Compare batch_size=1 outputs with batch_size=4 outputs."""
        model, tokenizer = model_and_tokenizer
        prompts = TEST_PROMPTS[:4]

        singles = [self._generate_single(model, tokenizer, p) for p in prompts]
        batched = self._generate_batch(model, tokenizer, prompts)

        for i, (single, batch_out) in enumerate(zip(singles, batched)):
            assert single == batch_out, (
                f"Prompt {i} ({prompts[i]!r}) produced different output:\n"
                f"  batch_size=1: {single!r}\n"
                f"  batch_size=4: {batch_out!r}"
            )

    def test_different_batch_compositions(self, model_and_tokenizer):
        """Same prompt in different batch contexts → identical output."""
        model, tokenizer = model_and_tokenizer
        target_prompt = TEST_PROMPTS[0]

        # Batch 1: target + short prompt; batch 2: target + very different prompt
        batch1 = self._generate_batch(
            model, tokenizer, [target_prompt, TEST_PROMPTS[1]]
        )
        batch2 = self._generate_batch(
            model, tokenizer, [target_prompt, TEST_PROMPTS[4]]
        )

        assert batch1[0] == batch2[0], (
            f"Same prompt in different batch contexts produced different output:\n"
            f"  Context 1: {batch1[0]!r}\n"
            f"  Context 2: {batch2[0]!r}"
        )

    def test_batch_size_sweep(self, model_and_tokenizer):
        """Sweep batch sizes and verify all outputs match batch_size=1."""
        model, tokenizer = model_and_tokenizer
        prompts = TEST_PROMPTS

        reference_outputs = [
            self._generate_single(model, tokenizer, p) for p in prompts
        ]

        for batch_size in [2, 3, 6, 16, 32, 64]:
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]
                batch_outputs = self._generate_batch(model, tokenizer, batch_prompts)

                for j, (ref, out) in enumerate(
                    zip(reference_outputs[start : start + batch_size], batch_outputs)
                ):
                    assert ref == out, (
                        f"batch_size={batch_size}, prompt {start + j}: mismatch\n"
                        f"  Reference: {ref!r}\n"
                        f"  Batched:   {out!r}"
                    )


class TestForwardPassDeterminism:
    """Verify that non-generative forward pass hidden states are consistent."""

    def test_hidden_states_batch1_vs_batch2(self, model_and_tokenizer):
        """Hidden states should be identical for the same prompt regardless of batch."""
        model, tokenizer = model_and_tokenizer
        prompts = TEST_PROMPTS[:2]

        # Single forward passes, keeping only valid (non-padding) positions
        hidden_states_single = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            hs = _hidden_states(model, inputs)
            valid_mask = inputs["attention_mask"][0] == 1
            hidden_states_single.append(hs[0][valid_mask].cpu())

        # Batched forward pass
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        hs_batched = _hidden_states(model, inputs)

        for i in range(2):
            valid_mask = inputs["attention_mask"][i] == 1
            hs_b = hs_batched[i][valid_mask].cpu()
            hs_s = hidden_states_single[i]

            # For non-generative forward, hidden states should be very close
            # (identical up to floating-point precision)
            max_diff = (hs_b - hs_s).abs().max().item()
            assert max_diff < 1e-4, (
                f"Hidden states for prompt {i} differ by {max_diff:.6e} "
                f"between batch_size=1 and batch_size=2"
            )

    def test_attention_mask_isolates_samples(self, model_and_tokenizer):
        """
        Verify that changing one sample's content in a batch does NOT affect
        the other sample's hidden states (attention mask isolation).
        """
        model, tokenizer = model_and_tokenizer
        fixed_prompt = TEST_PROMPTS[0]

        # Fixed prompt with two different companions
        inputs1 = tokenizer(
            [fixed_prompt, TEST_PROMPTS[1]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs2 = tokenizer(
            [fixed_prompt, TEST_PROMPTS[4]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        hs1 = _hidden_states(model, inputs1)
        hs2 = _hidden_states(model, inputs2)

        # Extract hidden states for the fixed prompt (index 0)
        valid1 = inputs1["attention_mask"][0] == 1
        valid2 = inputs2["attention_mask"][0] == 1
        h1_fixed = hs1[0][valid1.to(model.device)].cpu()
        h2_fixed = hs2[0][valid2.to(model.device)].cpu()

        max_diff = (h1_fixed - h2_fixed).abs().max().item()
        assert max_diff < 1e-4, (
            f"Fixed prompt's hidden states changed when batch companion changed. "
            f"Max diff: {max_diff:.6e}. This means attention masking is NOT "
            f"properly isolating samples in the batch."
        )
