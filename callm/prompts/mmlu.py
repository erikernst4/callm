"""
MMLU-specific prompts for confidence-augmented generation.

Prompts accept `question` (str) and `choices` (str, pre-formatted as
"A. ...\nB. ...\nC. ...\nD. ..."). For IsTrue prompts, `answer` (str)
is also required.
"""

from callm.prompts.triviaqa import ChatPrompt


# ─── Label Probability (Sequence Posterior) ───────────────────────────

CHAT_MMLU_LABEL_PROB_PROMPT = ChatPrompt(
    system=(
        "Answer the following multiple choice question. "
        "Respond with ONLY a single letter: A, B, C, or D. "
        "No other words or explanation."
    ),
    user="Question: {{ question }}\n\n{{ choices }}",
    assistant="Answer:",
)

GCP_CHAT_MMLU_LABEL_PROB_PROMPT = ChatPrompt(
    system=(
        "Answer the following multiple choice question. "
        "Respond with ONLY a single letter: A, B, C, or D. "
        "No other words or explanation."
    ),
    user="Question: {{ question }}\n\n{{ choices }}",
    assistant="Answer:",
    gcp=True,
)


# ─── Verbalized Confidence ───────────────────────────────────────────

CHAT_MMLU_VERBALIZED_PROMPT = ChatPrompt(
    system=(
        "Answer the following multiple choice question and provide the "
        "probability that your answer is correct (0.0 to 1.0). "
        "Give ONLY the answer letter and probability, no other words "
        "or explanation.\n\n"
        "Format:\n"
        "Answer: <A, B, C, or D>\n"
        "Probability: <0.0 to 1.0>"
    ),
    user="Question: {{ question }}\n\n{{ choices }}",
    assistant="Answer:",
)

GCP_CHAT_MMLU_VERBALIZED_PROMPT = ChatPrompt(
    system=(
        "Answer the following multiple choice question and provide the "
        "probability that your answer is correct (0.0 to 1.0). "
        "Give ONLY the answer letter and probability, no other words "
        "or explanation.\n\n"
        "Format:\n"
        "Answer: <A, B, C, or D>\n"
        "Probability: <0.0 to 1.0>"
    ),
    user="Question: {{ question }}\n\n{{ choices }}",
    assistant="Answer:",
    gcp=True,
)


# ─── Is True ─────────────────────────────────────────────────────────

CHAT_MMLU_IS_TRUE_PROMPT = ChatPrompt(
    system=(
        "Evaluate the correctness of the proposed answer for the given "
        "multiple choice question.\n\n"
        "Response must be only the word 'True' or 'False'. "
        "Nothing else, no explanation, no other words."
    ),
    user=(
        "Question: {{ question }}\n\n"
        "{{ choices }}\n\n"
        "Proposed Answer: {{ answer }}\n\n"
        "Is the proposed answer correct?"
    ),
)

GCP_CHAT_MMLU_IS_TRUE_PROMPT = ChatPrompt(
    system=(
        "Evaluate the correctness of the proposed answer for the given "
        "multiple choice question.\n\n"
        "Response must be only the word 'True' or 'False'. "
        "Nothing else, no explanation, no other words."
    ),
    user=(
        "Question: {{ question }}\n\n"
        "{{ choices }}\n\n"
        "Proposed Answer: {{ answer }}\n\n"
        "Is the proposed answer correct?"
    ),
    gcp=True,
)


def format_choices(choices: list[str]) -> str:
    """Format a list of choices as 'A. ...\nB. ...\nC. ...\nD. ...'."""
    letters = ["A", "B", "C", "D"]
    return "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))


def answer_index_to_letter(index: int) -> str:
    """Convert answer index (0-3) to letter (A-D)."""
    return ["A", "B", "C", "D"][index]
