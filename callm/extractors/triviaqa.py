"""
TriviaQA Answer extractors for parsing LLM responses.
"""

from typing import Tuple
import re
import numpy as np

# Re-export base classes previously contained here for compatibility
from callm.extractors.base import (  # noqa: F401
    BaseExtractor,
    SequencePosteriorExtractor,
    IsTruePosteriorExtractor,
    GCPSequencePosteriorExtractor,
    GCPIsTruePosteriorExtractor,
)


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
