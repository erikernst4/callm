"""Tests for continue_final_message vs add_generation_prompt logic in data modules."""

from unittest.mock import patch, Mock
import torch

from callm.prompts import (
    ChatPrompt,
    CHAT_LABEL_PROB_PROMPT_ZERO_SHOT,
    CHAT_LABEL_PROB_PROMPT_ONE_SHOT,
    CHAT_VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT,
    CHAT_IS_TRUE_PROB_PROMPT,
    GCP_CHAT_LABEL_PROB_PROMPT_ZERO_SHOT,
    GCP_CHAT_LABEL_PROB_PROMPT_ONE_SHOT,
    GCP_CHAT_IS_TRUE_PROB_PROMPT,
)


class TestChatPromptAssistantPrefill:
    """Tests that ChatPrompt correctly includes assistant prefill messages."""

    def test_zero_shot_has_assistant_prefill(self):
        prompt = CHAT_LABEL_PROB_PROMPT_ZERO_SHOT
        assert prompt.assistant_template is not None
        messages = prompt(question="What is the capital of France?")
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "Guess:"

    def test_one_shot_has_assistant_prefill(self):
        prompt = CHAT_LABEL_PROB_PROMPT_ONE_SHOT
        assert prompt.assistant_template is not None
        messages = prompt(question="What is the capital of France?")
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "Guess:"

    def test_verbalized_has_assistant_prefill(self):
        prompt = CHAT_VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT
        assert prompt.assistant_template is not None
        messages = prompt(question="What is the capital of France?")
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "Guess:"

    def test_is_true_has_no_assistant_prefill(self):
        prompt = CHAT_IS_TRUE_PROB_PROMPT
        assert prompt.assistant_template is None
        messages = prompt(question="What is 2+2?", answer="4")
        assert messages[-1]["role"] == "user"

    def test_custom_prompt_without_assistant(self):
        prompt = ChatPrompt(system="Be helpful", user="{{ question }}")
        assert prompt.assistant_template is None
        messages = prompt(question="Hello")
        assert messages[-1]["role"] == "user"

    def test_custom_prompt_with_assistant(self):
        prompt = ChatPrompt(
            system="Be helpful", user="{{ question }}", assistant="Answer:"
        )
        assert prompt.assistant_template is not None
        messages = prompt(question="Hello")
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "Answer:"


class TestTriviaQAChatTemplateKwargs:
    """Tests that TriviaQADataModule passes correct kwargs to apply_chat_template."""

    def _mock_dataset(self):
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(
            side_effect=lambda key: {
                "question": ["What is 2+2?"],
                "answer": [
                    {"aliases": ["4", "four"], "normalized_aliases": ["4", "four"]}
                ],
            }[key]
        )
        return mock_dataset

    @patch(
        "callm.data.triviaqa.triviaqa.subsample_dataset", side_effect=lambda d, *a: d
    )
    @patch("callm.data.triviaqa.triviaqa.load_dataset")
    def test_uses_continue_final_message_with_prefill(
        self, mock_load_dataset, mock_subsample
    ):
        """When prompt has assistant prefill, should use continue_final_message=True."""
        from callm.data.triviaqa import TriviaQADataModule

        mock_load_dataset.return_value = {"validation": self._mock_dataset()}

        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        dm = TriviaQADataModule(
            batch_size=1,
            model_name="test-model",
            prompt=CHAT_LABEL_PROB_PROMPT_ZERO_SHOT,
            max_samples=1,
        )
        dm.tokenizer = mock_tokenizer
        dm.setup("validate")

        for c in mock_tokenizer.apply_chat_template.call_args_list:
            kwargs = c.kwargs
            assert "continue_final_message" in kwargs
            assert kwargs["continue_final_message"] is True
            assert "add_generation_prompt" not in kwargs

    @patch(
        "callm.data.triviaqa.triviaqa.subsample_dataset", side_effect=lambda d, *a: d
    )
    @patch("callm.data.triviaqa.triviaqa.load_dataset")
    def test_uses_add_generation_prompt_without_prefill(
        self, mock_load_dataset, mock_subsample
    ):
        """When prompt has no assistant prefill, should use add_generation_prompt=True."""
        from callm.data.triviaqa import TriviaQADataModule

        mock_load_dataset.return_value = {"validation": self._mock_dataset()}

        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        prompt_no_prefill = ChatPrompt(
            system="Answer the question.", user="Question: {{ question }}"
        )
        dm = TriviaQADataModule(
            batch_size=1,
            model_name="test-model",
            prompt=prompt_no_prefill,
            max_samples=1,
        )
        dm.tokenizer = mock_tokenizer
        dm.setup("validate")

        for c in mock_tokenizer.apply_chat_template.call_args_list:
            kwargs = c.kwargs
            assert "add_generation_prompt" in kwargs
            assert kwargs["add_generation_prompt"] is True
            assert "continue_final_message" not in kwargs


class TestIsTrueDataChatTemplateKwargs:
    """Tests that IsTrueDataModule passes correct kwargs to apply_chat_template."""

    def _create_csv(self):
        """Helper to create a temporary CSV file for testing."""
        import tempfile
        import csv

        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        )
        writer = csv.writer(f)
        writer.writerow(
            ["question", "gold_answers", "pred_answer", "confidence", "raw_output"]
        )
        writer.writerow(["What is 2+2?", "4|four", "4", "0.9", "Guess: 4"])
        f.close()
        return f.name

    @patch(
        "callm.data.triviaqa.is_true_data.subsample_dataset",
        side_effect=lambda d, *a: d,
    )
    @patch("callm.data.answers_data.get_tokenizer_for_model")
    def test_uses_add_generation_prompt_for_default_prompt(
        self, mock_get_tokenizer, mock_subsample
    ):
        """CHAT_IS_TRUE_PROB_PROMPT has no assistant prefill, so should use add_generation_prompt."""
        import os
        from callm.data.triviaqa import IsTrueDataModule

        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.model_max_length = 512
        mock_get_tokenizer.return_value = mock_tokenizer

        csv_path = self._create_csv()
        try:
            dm = IsTrueDataModule(
                prompt=CHAT_IS_TRUE_PROB_PROMPT,
                batch_size=1,
                model_name="test-model",
                llm_outputs_path=csv_path,
                max_samples=1,
            )
            dm.setup("validate")

            for c in mock_tokenizer.apply_chat_template.call_args_list:
                kwargs = c.kwargs
                assert "add_generation_prompt" in kwargs
                assert kwargs["add_generation_prompt"] is True
                assert "continue_final_message" not in kwargs
        finally:
            os.unlink(csv_path)

    @patch(
        "callm.data.triviaqa.is_true_data.subsample_dataset",
        side_effect=lambda d, *a: d,
    )
    @patch("callm.data.answers_data.get_tokenizer_for_model")
    def test_uses_continue_final_message_with_prefill(
        self, mock_get_tokenizer, mock_subsample
    ):
        """When using a prompt with assistant prefill, should use continue_final_message."""
        import os
        from callm.data.triviaqa import IsTrueDataModule

        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.model_max_length = 512
        mock_get_tokenizer.return_value = mock_tokenizer

        prompt_with_prefill = ChatPrompt(
            system="Is the answer correct? Reply True or False.",
            user="Question: {{ question }}\nAnswer: {{ answer }}",
            assistant="Answer:",
        )

        csv_path = self._create_csv()
        try:
            dm = IsTrueDataModule(
                prompt=prompt_with_prefill,
                batch_size=1,
                model_name="test-model",
                llm_outputs_path=csv_path,
                max_samples=1,
            )
            dm.setup("validate")

            for c in mock_tokenizer.apply_chat_template.call_args_list:
                kwargs = c.kwargs
                assert "continue_final_message" in kwargs
                assert kwargs["continue_final_message"] is True
                assert "add_generation_prompt" not in kwargs
        finally:
            os.unlink(csv_path)


class TestGCPChatPromptFormat:
    """Tests that ChatPrompt with gcp=True produces GCP-compatible output."""

    def test_gcp_zero_shot_uses_model_role(self):
        messages = GCP_CHAT_LABEL_PROB_PROMPT_ZERO_SHOT(
            question="What is the capital of France?"
        )
        roles = [m["role"] for m in messages]
        assert "assistant" not in roles
        assert "model" in roles

    def test_gcp_zero_shot_uses_parts_key(self):
        messages = GCP_CHAT_LABEL_PROB_PROMPT_ZERO_SHOT(
            question="What is the capital of France?"
        )
        for msg in messages:
            assert "parts" in msg
            assert "content" not in msg

    def test_gcp_system_in_messages(self):
        messages = GCP_CHAT_LABEL_PROB_PROMPT_ZERO_SHOT(
            question="What is the capital of France?"
        )
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "text" in system_msgs[0]["parts"][0]

    def test_gcp_prompt_message_order(self):
        messages = GCP_CHAT_LABEL_PROB_PROMPT_ZERO_SHOT(
            question="What is the capital of France?"
        )
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "model"]

    def test_gcp_is_true_has_no_model_prefill(self):
        messages = GCP_CHAT_IS_TRUE_PROB_PROMPT(question="What is 2+2?", answer="4")
        roles = [m["role"] for m in messages]
        assert "model" not in roles
        assert roles == ["system", "user"]

    def test_gcp_custom_prompt(self):
        prompt = ChatPrompt(
            system="Be helpful", user="{{ question }}", assistant="Answer:", gcp=True
        )
        messages = prompt(question="Hello")
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "model"
        assert messages[2]["parts"][0]["text"] == "Answer:"

    def test_standard_format_unchanged(self):
        """Ensure gcp=False (default) still produces standard HuggingFace format."""
        messages = CHAT_LABEL_PROB_PROMPT_ZERO_SHOT(
            question="What is the capital of France?"
        )
        for msg in messages:
            assert "content" in msg
            assert "parts" not in msg
        roles = [m["role"] for m in messages]
        assert "assistant" in roles
        assert "model" not in roles

    def test_gcp_parts_text_content(self):
        messages = GCP_CHAT_LABEL_PROB_PROMPT_ONE_SHOT(
            question="What is the capital of France?"
        )
        user_msg = [m for m in messages if m["role"] == "user"][0]
        assert (
            user_msg["parts"][0]["text"] == "Question: What is the capital of France?"
        )
        model_msg = [m for m in messages if m["role"] == "model"][0]
        assert model_msg["parts"][0]["text"] == "Guess:"
