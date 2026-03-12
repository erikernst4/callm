from jinja2 import Template


class Prompt:
    """A prompt for a language model. This is a wrapper around a Jinja2 template in order to make it serializable for jsonargparse."""

    def __init__(self, template: str):
        self.template = Template(template)

    def __call__(self, **kwargs):
        return self.template.render(**kwargs)


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
