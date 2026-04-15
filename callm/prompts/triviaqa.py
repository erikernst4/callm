from callm.prompts.base import Prompt, ChatPrompt

LABEL_PROB_PROMPT_ZERO_SHOT = Prompt("""Provide your best guess for the following question. Give ONLY the guess, no other words or explanation. Return the answer in the following format: 'Guess: <your guess>'.

Question: {{ question }}""")

LABEL_PROB_PROMPT_ONE_SHOT = Prompt("""Provide your best guess for the following question. Give ONLY the guess, no other words or explanation. Return the answer in the following format: 'Guess: <your guess>'.

For example:
Question: What is the capital of France?
Guess: Paris

Question: {{ question }}""")

VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT = Prompt("""Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation. For example:

Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>

Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>

The question is: {{ question }}""")

IS_TRUE_PROB_PROMPT = Prompt("""Evaluate the factual correctness of the proposed answer for the given question.

Question: {{ question }}
Proposed Answer: {{ answer }}

Is the proposed answer correct?

Response must be only the word 'True' or 'False'. Nothing else, no explanation, no other words.""")


CHAT_LABEL_PROB_PROMPT_ZERO_SHOT = ChatPrompt(
    system="Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.",
    user="Question: {{ question }}",
    assistant="Guess:",
)

CHAT_LABEL_PROB_PROMPT_ONE_SHOT = ChatPrompt(
    system="Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.\n\nFor example:\nQuestion: What is the capital of France?\nGuess: Paris",
    user="Question: {{ question }}",
    assistant="Guess:",
)

CHAT_VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT = ChatPrompt(
    system="Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation.\n\nFormat:\nGuess: <short answer>\nProbability: <0.0 to 1.0>",
    user="Question: {{ question }}",
    assistant="Guess:",
)

CHAT_IS_TRUE_PROB_PROMPT = ChatPrompt(
    system="Evaluate the factual correctness of the proposed answer for the given question.\n\nResponse must be only the word 'True' or 'False'. Nothing else, no explanation, no other words.",
    user="Question: {{ question }}\nProposed Answer: {{ answer }}\n\nIs the proposed answer correct?",
)

# GCP-compatible ChatPrompts (use "model" role, "parts" key, separate system instruction)
GCP_CHAT_LABEL_PROB_PROMPT_ZERO_SHOT = ChatPrompt(
    system="Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.",
    user="Question: {{ question }}",
    assistant="Guess:",
    gcp=True,
)

GCP_CHAT_LABEL_PROB_PROMPT_ONE_SHOT = ChatPrompt(
    system="Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.\n\nFor example:\nQuestion: What is the capital of France?\nGuess: Paris",
    user="Question: {{ question }}",
    assistant="Guess:",
    gcp=True,
)

GCP_CHAT_VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT = ChatPrompt(
    system="Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation.\n\nFormat:\nGuess: <short answer>\nProbability: <0.0 to 1.0>",
    user="Question: {{ question }}",
    assistant="Guess:",
    gcp=True,
)

GCP_CHAT_IS_TRUE_PROB_PROMPT = ChatPrompt(
    system="Evaluate the factual correctness of the proposed answer for the given question.\n\nResponse must be only the word 'True' or 'False'. Nothing else, no explanation, no other words.",
    user="Question: {{ question }}\nProposed Answer: {{ answer }}\n\nIs the proposed answer correct?",
    gcp=True,
)
