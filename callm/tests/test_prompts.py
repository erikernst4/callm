import pytest
from callm.prompts.base import Prompt, ChatPrompt


def test_prompt_raises_on_none_values():
    prompt = Prompt("Question: {{ question }}")

    # Should work fine with a valid string
    assert prompt(question="What is 2+2?") == "Question: What is 2+2?"

    # Should break if an important field is None
    with pytest.raises(ValueError, match="cannot be None"):
        prompt(question=None)


def test_chat_prompt_raises_on_none_values():
    chat_prompt = ChatPrompt(user="Question: {{ question }}")

    # Should work fine
    messages = chat_prompt(question="Why is the sky blue?")
    assert len(messages) == 1
    assert messages[0]["content"] == "Question: Why is the sky blue?"

    # Should break if None
    with pytest.raises(ValueError, match="cannot be None"):
        chat_prompt(question=None)


def test_prompt_raises_on_missing_fields_with_strict_undefined():
    prompt = Prompt("Question: {{ question }}\nChoices: {{ choices }}")

    # Missing 'choices' entirely
    with pytest.raises(Exception):  # May be jinja2.exceptions.UndefinedError
        prompt(question="What is 2+2?")


def test_jsonargparse_serialization_predefined_prompts():
    import jsonargparse
    import callm.prompts as prompts

    test_prompts = [
        (
            "callm.prompts.GCP_CHAT_MMLU_LABEL_PROB_PROMPT",
            prompts.GCP_CHAT_MMLU_LABEL_PROB_PROMPT,
        ),
        (
            "callm.prompts.CHAT_MMLU_LABEL_PROB_PROMPT",
            prompts.CHAT_MMLU_LABEL_PROB_PROMPT,
        ),
        (
            "callm.prompts.CHAT_MMLU_VERBALIZED_PROMPT",
            prompts.CHAT_MMLU_VERBALIZED_PROMPT,
        ),
        (
            "callm.prompts.CHAT_LABEL_PROB_PROMPT_ZERO_SHOT",
            prompts.CHAT_LABEL_PROB_PROMPT_ZERO_SHOT,
        ),
    ]

    for expected_path, prompt_obj in test_prompts:
        parser = jsonargparse.ArgumentParser()
        parser.add_subclass_arguments(Prompt, "prompt")
        cfg = parser.parse_object({"prompt": prompt_obj})
        dumped = parser.dump(cfg)
        assert "Unable to serialize" not in dumped
        assert expected_path in dumped

        reloaded = parser.parse_string(dumped)
        assert reloaded.prompt is prompt_obj


def test_jsonargparse_datamodule_default_prompt():
    import jsonargparse
    import callm.prompts as prompts
    from callm.data.mmlu.untokenized_mmlu import UntokenizedMMLUDataModule

    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(UntokenizedMMLUDataModule, "data")
    defaults = parser.get_defaults()
    dumped = parser.dump(defaults)

    assert "Unable to serialize" not in dumped
    assert "callm.prompts.GCP_CHAT_MMLU_LABEL_PROB_PROMPT" in dumped

    reloaded = parser.parse_string(dumped)
    inst = getattr(parser, "instantiate", getattr(parser, "instantiate_classes"))(
        reloaded
    )
    assert inst.data.prompt is prompts.GCP_CHAT_MMLU_LABEL_PROB_PROMPT
