from callm.prompts.base import Prompt

SEMANTIC_EQUIVALENCE_PROMPT = Prompt(
    """Is the predicted answer to my question semantically equivalent to any of the gold standard answers?

Question: {{ question }}
Predicted answer: {{ pred_answer }}
Gold Standard Answers: {{ gold_answers }}

Please answer with a single word, either 'Yes.' or 'No.'"""
)

SEMANTIC_EQUIVALENCE_PROMPT_FOR_FREE_ANSWERS = Prompt(
    """You are evaluating whether a generated answer correctly answers a TriviaQA question.

The generated answer may contain additional explanations, reasoning, or irrelevant text. Your task is to determine whether it correctly answers the question.

The list of gold standard answers contains normalized aliases of the expected answer. These aliases are not necessarily exhaustive. A generated answer should be considered correct if it unambiguously identifies the same entity, concept, value, or fact as any of the gold standard answers, even if it uses a different valid name or wording.

Ignore differences in:

- capitalization, punctuation, and formatting;
- articles (e.g., "the", "a");
- abbreviations or acronyms;
- alternative spellings or transliterations;
- valid aliases or synonymous names;
- additional explanatory text that does not change the answer.

The generated answer is incorrect if:

- it identifies a different entity or fact;
- it expresses uncertainty without providing a correct answer;
- it gives multiple conflicting answers without clearly selecting the correct one;
- the correct answer cannot be unambiguously inferred from the response.

Question:
{{ question }}

Generated Answer:
{{ pred_answer }}

Gold Standard Answers (normalized aliases):
{{ gold_answers }}

Please answer with a single word, either 'Yes.' or 'No.'"""
)
