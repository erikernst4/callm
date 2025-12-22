"""
Correctness evaluator using an LLM to check semantic equivalence.

Based on the paper's approach: using another LLM to determine if the
predicted answer is semantically equivalent to the ground truth.
"""

from typing import List
import torch
from transformers import AutoTokenizer
from jinja2 import Template
from callm.utils import initialize_model


# Semantic equivalence prompt from the paper
SEMANTIC_EQUIVALENCE_PROMPT_121 = Template("""Are the following two answers to my question Q semantically equivalent?

Q: {{ question }}
A1: {{ gold_answer }}
A2: {{ pred_answer }}

Please answer with a single word, either "Yes." or "No.", and explain your reasoning.""")

SEMANTIC_EQUIVALENCE_PROMPT_12MULTI = Template("""Is the following answer to my question Q semantically equivalent to any of the following answers?

Question: {{ question }}
Answer: {{ pred_answer }}
Answers: {{ gold_answers }}

Please answer with a single word, either "Yes." or "No.", and explain your reasoning.""")


class CorrectnessEvaluator:
    """
    LLM-based correctness evaluator.

    Uses a separate LLM instance to check if predicted answers are
    semantically equivalent to ground truth answers.
    """

    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Initialize the evaluator.

        Args:
            model_name: HuggingFace model name for the evaluator
        """
        self.model_name = model_name

        self.model, self.is_seq2seq = initialize_model(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True if model_name.startswith("Qwen/") else False,
        )

        # Set padding token for causal models
        if not self.is_seq2seq and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    def evaluate(
        self, question: str, pred_answer: str, gold_answers: List[str]
    ) -> bool:
        """
        Evaluate if predicted answer is semantically equivalent to any gold answer.

        Args:
            question: The original question
            pred_answer: The predicted answer
            gold_answers: List of acceptable ground truth answers

        Returns:
            True if the predicted answer is semantically equivalent to any gold answer
        """
        # Exact match
        if pred_answer in gold_answers:
            return True

        # Semantic equivalence
        prompt = SEMANTIC_EQUIVALENCE_PROMPT_12MULTI.render(
            question=question, gold_answers=gold_answers, pred_answer=pred_answer
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        )

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,  # Low temperature for more deterministic output
                do_sample=False,
            )

        # Decode response
        if self.is_seq2seq:
            # Seq2seq: decode the full output
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Causal: skip the prompt tokens
            input_length = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(
                outputs[0][input_length:], skip_special_tokens=True
            )

        # Parse response for Yes/No
        # Look for "Yes" at the beginning (case-insensitive)
        response_lower = response.lower().strip()
        if response_lower.startswith("yes"):
            return True
        else:
            return False
