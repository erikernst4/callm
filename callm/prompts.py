from jinja2 import Template


class Prompt:
    """A prompt for a language model. This is a wrapper around a Jinja2 template in order to make it serializable for jsonargparse."""

    def __init__(self, template: str):
        self.template = Template(template)

    def __call__(self, **kwargs):
        return self.template.render(**kwargs)


class ChatPrompt(Prompt):
    """A multi-role prompt for a language model using apply_chat_template."""

    def __init__(self, system: str = None, user: str = None, assistant: str = None):
        self.system_template = Prompt(system) if system else None
        self.user_template = Prompt(user) if user else None
        self.assistant_template = Prompt(assistant) if assistant else None

    def __call__(self, **kwargs):
        messages = []
        if self.system_template:
            system_content = self.system_template(**kwargs)
            messages.append({"role": "system", "content": system_content})

        if self.user_template:
            messages.append({"role": "user", "content": self.user_template(**kwargs)})

        if self.assistant_template:
            messages.append(
                {
                    "role": "assistant",
                    "content": self.assistant_template(**kwargs),
                }
            )

        return messages


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
    system="Provide your best guess for the following question. Give ONLY the guess, no other words or explanation. Return the answer in the following format: 'Guess: <your guess>'.",
    user="Question: {{ question }}",
)

CHAT_LABEL_PROB_PROMPT_ONE_SHOT = ChatPrompt(
    system="Provide your best guess for the following question. Give ONLY the guess, no other words or explanation. Return the answer in the following format: 'Guess: <your guess>'.\n\nFor example:\nQuestion: What is the capital of France?\nGuess: Paris",
    user="Question: {{ question }}",
)

CHAT_VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT = ChatPrompt(
    system="""Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation. For example:

Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>

Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>""",
    user="The question is: {{ question }}",
)

CHAT_IS_TRUE_PROB_PROMPT = ChatPrompt(
    system="Evaluate the factual correctness of the proposed answer for the given question.\n\nResponse must be only the word 'True' or 'False'. Nothing else, no explanation, no other words.",
    user="Question: {{ question }}\nProposed Answer: {{ answer }}\n\nIs the proposed answer correct?",
)
