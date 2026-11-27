"""
TriviaQA Answer extractors for parsing LLM responses.
"""

from typing import Tuple

# Re-export base classes previously contained here for compatibility
from callm.extractors.base import (  # noqa: F401
    BaseExtractor,
    SequencePosteriorExtractor,
    IsTruePosteriorExtractor,
    GCPSequencePosteriorExtractor,
    GCPIsTruePosteriorExtractor,
    extract_verbalized_probability,
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
        return self.extract_answer(text), extract_verbalized_probability(text)
