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


def test_jsonargparse_serialization():
    import jsonargparse

    prompt = Prompt("Question: {{ question }}")
    parser = jsonargparse.ArgumentParser()
    parser.add_subclass_arguments(Prompt, "prompt")
    cfg = parser.parse_object({"prompt": prompt})
    dumped = parser.dump(cfg)
    assert dumped

    chat_prompt = ChatPrompt(user="Question: {{ question }}")
    parser_chat = jsonargparse.ArgumentParser()
    parser_chat.add_subclass_arguments(Prompt, "prompt")
    cfg_chat = parser_chat.parse_object({"prompt": chat_prompt})
    dumped_chat = parser_chat.dump(cfg_chat)
    assert dumped_chat
