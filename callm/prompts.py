from jinja2 import Template

LABEL_PROB_PROMPT = Template("""Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.    

For example:

Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>

The question is: {{ question }}""")

VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT = Template("""Provide your best guess and the probability that it is correct (0.0 to 1.0) forthe following question. Give ONLY the guess and probability, no other words or explanation. For example:

Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>

Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>

The question is: {{ question }}""")

def generate_prompt(self, prompt: Template, **kwargs):
    return prompt.render(**kwargs)