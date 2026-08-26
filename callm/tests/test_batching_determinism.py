"""
Synthetic tests for LLM batching determinism.

Verifies that:
1. Greedy generation produces identical text outputs regardless of batch size
2. Non-generative forward pass hidden states are consistent across batch sizes
3. Attention masking correctly isolates per-sample computations in batched inference

These tests use google/flan-t5-small for speed. The same principles apply to
larger models like Qwen3.5-9B.
"""

import pytest
import torch
from callm.utils import initialize_model, get_tokenizer_for_model


# Use small model for fast testing
TEST_MODEL_NAME = "google/flan-t5-small"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load model and tokenizer once for all tests in this module."""
    model, is_seq2seq = initialize_model(TEST_MODEL_NAME)
    tokenizer = get_tokenizer_for_model(TEST_MODEL_NAME)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer, is_seq2seq


# ── Test prompts with varying lengths ──────────────────────────────────
TEST_PROMPTS = [
    "What is the capital of France?",
    "Who wrote the novel '1984'?",
    "What is the boiling point of water in Celsius?",
    "Name the largest planet in our solar system.",
    "Who painted the Mona Lisa?",
    "What year did World War II end?",
]


class TestGreedyGenerationDeterminism:
    """Verify that greedy generation is identical across batch sizes."""

    def _generate_single(self, model, tokenizer, prompt, is_seq2seq, max_new_tokens=50):
        """Generate output for a single prompt (batch_size=1)."""
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
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

        if is_seq2seq:
            generated = output[0]
        else:
            generated = output[0][input_ids.shape[1] :]

        return tokenizer.decode(generated, skip_special_tokens=True)

    def _generate_batch(self, model, tokenizer, prompts, is_seq2seq, max_new_tokens=50):
        """Generate outputs for multiple prompts in a single batch."""
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

        results = []
        input_length = input_ids.shape[1] if not is_seq2seq else 0
        for i in range(len(prompts)):
            if is_seq2seq:
                generated = output[i]
            else:
                generated = output[i][input_length:]
            results.append(tokenizer.decode(generated, skip_special_tokens=True))

        return results

    def test_batch1_vs_batch1(self, model_and_tokenizer):
        """Same prompt, two separate batch_size=1 calls → identical output."""
        model, tokenizer, is_seq2seq = model_and_tokenizer
        prompt = TEST_PROMPTS[0]

        out1 = self._generate_single(model, tokenizer, prompt, is_seq2seq)
        out2 = self._generate_single(model, tokenizer, prompt, is_seq2seq)

        assert out1 == out2, (
            f"Same prompt produced different outputs:\n"
            f"  Run 1: {out1!r}\n"
            f"  Run 2: {out2!r}"
        )

    def test_batch1_vs_batch2(self, model_and_tokenizer):
        """Compare batch_size=1 outputs with batch_size=2 outputs."""
        model, tokenizer, is_seq2seq = model_and_tokenizer
        prompts = TEST_PROMPTS[:2]

        # Generate individually
        singles = [
            self._generate_single(model, tokenizer, p, is_seq2seq) for p in prompts
        ]

        # Generate as a batch
        batched = self._generate_batch(model, tokenizer, prompts, is_seq2seq)

        for i, (single, batch_out) in enumerate(zip(singles, batched)):
            assert single == batch_out, (
                f"Prompt {i} ({prompts[i]!r}) produced different output:\n"
                f"  batch_size=1: {single!r}\n"
                f"  batch_size=2: {batch_out!r}"
            )

    def test_batch1_vs_batch4(self, model_and_tokenizer):
        """Compare batch_size=1 outputs with batch_size=4 outputs."""
        model, tokenizer, is_seq2seq = model_and_tokenizer
        prompts = TEST_PROMPTS[:4]

        singles = [
            self._generate_single(model, tokenizer, p, is_seq2seq) for p in prompts
        ]
        batched = self._generate_batch(model, tokenizer, prompts, is_seq2seq)

        for i, (single, batch_out) in enumerate(zip(singles, batched)):
            assert single == batch_out, (
                f"Prompt {i} ({prompts[i]!r}) produced different output:\n"
                f"  batch_size=1: {single!r}\n"
                f"  batch_size=4: {batch_out!r}"
            )

    def test_different_batch_compositions(self, model_and_tokenizer):
        """Same prompt in different batch contexts → identical output."""
        model, tokenizer, is_seq2seq = model_and_tokenizer
        target_prompt = TEST_PROMPTS[0]

        # Batch 1: target + short prompt
        batch1 = self._generate_batch(
            model,
            tokenizer,
            [target_prompt, TEST_PROMPTS[1]],
            is_seq2seq,
        )

        # Batch 2: target + very different prompt
        batch2 = self._generate_batch(
            model,
            tokenizer,
            [target_prompt, TEST_PROMPTS[4]],
            is_seq2seq,
        )

        assert batch1[0] == batch2[0], (
            f"Same prompt in different batch contexts produced different output:\n"
            f"  Context 1: {batch1[0]!r}\n"
            f"  Context 2: {batch2[0]!r}"
        )

    def test_batch_size_sweep(self, model_and_tokenizer):
        """Sweep batch sizes 1..6 and verify all outputs are identical."""
        model, tokenizer, is_seq2seq = model_and_tokenizer
        prompts = TEST_PROMPTS

        # Ground truth: batch_size=1
        reference_outputs = [
            self._generate_single(model, tokenizer, p, is_seq2seq) for p in prompts
        ]

        # Test various batch sizes
        for batch_size in [2, 3, 6, 16, 32, 64]:
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]
                batch_outputs = self._generate_batch(
                    model, tokenizer, batch_prompts, is_seq2seq
                )

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
        model, tokenizer, is_seq2seq = model_and_tokenizer

        prompts = TEST_PROMPTS[:2]

        # Single forward passes
        hidden_states_single = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            # Seq2seq models require decoder_input_ids for forward pass
            extra_kwargs = {}
            if is_seq2seq:
                decoder_start_id = model.config.decoder_start_token_id or 0
                extra_kwargs["decoder_input_ids"] = torch.tensor(
                    [[decoder_start_id]]
                ).to(model.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    **extra_kwargs,
                )
            # Get last layer hidden states, only valid (non-padding) positions
            if is_seq2seq:
                hs = outputs.encoder_hidden_states[-1]
            else:
                hs = outputs.hidden_states[-1]
            valid_mask = attention_mask[0] == 1
            hidden_states_single.append(hs[0][valid_mask].cpu())

        # Batched forward pass
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        # Seq2seq models require decoder_input_ids for forward pass
        extra_kwargs = {}
        if is_seq2seq:
            decoder_start_id = model.config.decoder_start_token_id or 0
            extra_kwargs["decoder_input_ids"] = torch.tensor(
                [[decoder_start_id]] * len(prompts)
            ).to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **extra_kwargs,
            )
        if is_seq2seq:
            hs_batched = outputs.encoder_hidden_states[-1]
        else:
            hs_batched = outputs.hidden_states[-1]

        for i in range(2):
            valid_mask = attention_mask[i] == 1
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
        model, tokenizer, is_seq2seq = model_and_tokenizer

        # Fixed prompt + two different companions
        fixed_prompt = TEST_PROMPTS[0]

        # Batch 1: fixed + companion A
        inputs1 = tokenizer(
            [fixed_prompt, TEST_PROMPTS[1]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        # Seq2seq models require decoder_input_ids for forward pass
        extra_kwargs1 = {}
        extra_kwargs2 = {}
        if is_seq2seq:
            decoder_start_id = model.config.decoder_start_token_id or 0
            extra_kwargs1["decoder_input_ids"] = torch.tensor(
                [[decoder_start_id]] * 2
            ).to(model.device)
            extra_kwargs2["decoder_input_ids"] = torch.tensor(
                [[decoder_start_id]] * 2
            ).to(model.device)

        with torch.no_grad():
            out1 = model(
                input_ids=inputs1["input_ids"].to(model.device),
                attention_mask=inputs1["attention_mask"].to(model.device),
                output_hidden_states=True,
                return_dict=True,
                **extra_kwargs1,
            )

        # Batch 2: fixed + companion B (very different)
        inputs2 = tokenizer(
            [fixed_prompt, TEST_PROMPTS[4]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            out2 = model(
                input_ids=inputs2["input_ids"].to(model.device),
                attention_mask=inputs2["attention_mask"].to(model.device),
                output_hidden_states=True,
                return_dict=True,
                **extra_kwargs2,
            )

        # Extract hidden states for the fixed prompt (index 0)
        if is_seq2seq:
            hs1 = out1.encoder_hidden_states[-1]
            hs2 = out2.encoder_hidden_states[-1]
        else:
            hs1 = out1.hidden_states[-1]
            hs2 = out2.hidden_states[-1]

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
