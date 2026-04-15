"""
Prompts package — re-exports all prompts for backward compatibility.

TriviaQA prompts (and the base Prompt/ChatPrompt classes) live in
``callm.prompts.triviaqa``; MMLU prompts live in ``callm.prompts.mmlu``.
"""

from callm.prompts.base import Prompt, ChatPrompt  # noqa: F401

# Base classes + TriviaQA prompts
from callm.prompts.triviaqa import (  # noqa: F401
    LABEL_PROB_PROMPT_ZERO_SHOT,
    LABEL_PROB_PROMPT_ONE_SHOT,
    VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT,
    IS_TRUE_PROB_PROMPT,
    CHAT_LABEL_PROB_PROMPT_ZERO_SHOT,
    CHAT_LABEL_PROB_PROMPT_ONE_SHOT,
    CHAT_VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT,
    CHAT_IS_TRUE_PROB_PROMPT,
    GCP_CHAT_LABEL_PROB_PROMPT_ZERO_SHOT,
    GCP_CHAT_LABEL_PROB_PROMPT_ONE_SHOT,
    GCP_CHAT_VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT,
    GCP_CHAT_IS_TRUE_PROB_PROMPT,
)

# MMLU prompts
from callm.prompts.mmlu import (  # noqa: F401
    CHAT_MMLU_LABEL_PROB_PROMPT,
    CHAT_MMLU_VERBALIZED_PROMPT,
    CHAT_MMLU_IS_TRUE_PROMPT,
    GCP_CHAT_MMLU_LABEL_PROB_PROMPT,
    GCP_CHAT_MMLU_VERBALIZED_PROMPT,
    GCP_CHAT_MMLU_IS_TRUE_PROMPT,
    format_choices,
    answer_index_to_letter,
)
