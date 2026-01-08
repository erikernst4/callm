"""
Answer extractors for parsing LLM responses.

Extractors parse the LLM's text output to extract the answer and confidence score.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import re
import numpy as np
import torch
from lightning.pytorch import LightningModule
from callm.utils import get_tokenizer_for_model


class BaseExtractor(LightningModule, ABC):
    """Abstract base class for answer extractors."""

    @abstractmethod
    def forward(self, text: str, *args, **kwargs) -> Tuple[str, float]:
        """
        Extract answer and confidence from LLM response text.

        Args:
            text: Raw text output from the LLM

        Returns:
            Tuple of (answer: str, confidence: float)
            - answer: The extracted answer string
            - confidence: Confidence score between 0.0 and 1.0, or np.nan if invalid
        """
        pass

    def extract_answer(self, text: str):
        answer = ""

        # Extract guess/answer
        guess_match = re.search(r"Guess:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if guess_match:
            answer = guess_match.group(1).strip()
        else:
            # Fallback: try to extract first line as answer
            lines = text.strip().split("\n")
            if lines:
                answer = lines[0].strip()
        return answer


class VerbalizedConfidenceExtractor(BaseExtractor):
    """
    Extractor for VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT responses.

    Expected format:
        Guess: <answer>
        Probability: <confidence>
    """

    def forward(self, text: str, *args, **kwargs) -> Tuple[str, float]:
        """
        Extract answer and confidence from verbalized confidence format.

        Args:
            text: Raw text output from the LLM

        Returns:
            Tuple of (answer, confidence)
            - confidence is np.nan if not found or invalid
        """
        answer = self.extract_answer(text)
        confidence = np.nan  # Use NaN to indicate missing/invalid data

        # Extract probability/confidence
        prob_match = re.search(
            r"Probability:\s*(-?[0-9]*\.?[0-9]+)", text, re.IGNORECASE
        )
        if prob_match:
            try:
                prob_value = float(prob_match.group(1))
                # Only accept valid range [0, 1]
                if 0.0 <= prob_value <= 1.0:
                    confidence = prob_value
                # Out of range is invalid - keep as np.nan
            except ValueError:
                # Invalid format - keep as np.nan
                pass

        return answer, confidence


class SequencePosteriorConfidenceExtractor(BaseExtractor):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = get_tokenizer_for_model(model_name)

    def forward(
        self, text: str, logits: torch.Tensor, output_ids: torch.Tensor, *args, **kwargs
    ) -> Tuple[str, float]:
        if not text:
            return "", np.nan

        answer = self.extract_answer(text)
        # Get answer token scores
        # Get first appearance of answer in text
        answer_start = text.find(answer)
        from IPython import embed

        embed()
        if answer_start != -1:
            # Encode answer to token IDs
            answer_tokens = self.tokenizer(answer, return_tensors="pt")["input_ids"][
                0
            ].to(self.device)
            # Find token positions in output_ids
            answer_token_positions = []
            for i in range(len(output_ids) - len(answer_tokens) + 1):
                if torch.equal(output_ids[i : i + len(answer_tokens)], answer_tokens):
                    answer_token_positions.extend(range(i, i + len(answer_tokens)))
                    break  # Only first occurrence
            if not answer_token_positions:
                return answer, np.nan
            # Get logits for answer tokens
        return answer, np.nan
