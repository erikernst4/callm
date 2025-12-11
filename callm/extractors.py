"""
Answer extractors for parsing LLM responses.

Extractors parse the LLM's text output to extract the answer and confidence score.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import re
import numpy as np


class BaseExtractor(ABC):
    """Abstract base class for answer extractors."""

    @abstractmethod
    def extract(self, text: str) -> Tuple[str, float]:
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


class VerbalizedConfidenceExtractor(BaseExtractor):
    """
    Extractor for VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT responses.

    Expected format:
        Guess: <answer>
        Probability: <confidence>
    """

    def extract(self, text: str) -> Tuple[str, float]:
        """
        Extract answer and confidence from verbalized confidence format.

        Args:
            text: Raw text output from the LLM

        Returns:
            Tuple of (answer, confidence)
            - confidence is np.nan if not found or invalid
        """
        # Initialize defaults
        answer = ""
        confidence = np.nan  # Use NaN to indicate missing/invalid data

        # Extract guess/answer
        guess_match = re.search(r"Guess:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if guess_match:
            answer = guess_match.group(1).strip()
        else:
            # Fallback: try to extract first line as answer
            lines = text.strip().split("\n")
            if lines:
                answer = lines[0].strip()

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
