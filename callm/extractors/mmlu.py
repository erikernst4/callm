"""
MMLU-specific answer extractors.

Extractors parse the LLM's text output to extract the answer letter (A/B/C/D)
and confidence score for MMLU multiple-choice questions.
"""

import re
import numpy as np
from typing import Tuple

from callm.extractors.base import (
    BaseExtractor,
    SequencePosteriorExtractor,
    GCPSequencePosteriorExtractor,
)


class MMLUBaseExtractor(BaseExtractor):
    """Base extractor for MMLU that extracts a letter answer (A/B/C/D)."""

    def extract_answer(self, text: str) -> str:
        """Extract the A/B/C/D choice from the text."""
        # Look for "Answer: X" pattern first
        answer_match = re.search(r"Answer:\s*([A-D])\b", text, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()

        # Look for standalone letter at start of text
        start_match = re.match(r"\s*([A-D])\b", text, re.IGNORECASE)
        if start_match:
            return start_match.group(1).upper()

        # Look for any standalone A/B/C/D
        match = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        return ""

    def forward(self, text: str, *args, **kwargs) -> Tuple[str, float]:
        raise NotImplementedError("Use a specific MMLU extractor subclass.")


class MMLUVerbalizedExtractor(MMLUBaseExtractor):
    """
    Extractor for CHAT_MMLU_VERBALIZED_PROMPT responses.

    Expected format:
        Answer: <A/B/C/D>
        Probability: <confidence>
    """

    def forward(self, text: str, *args, **kwargs) -> Tuple[str, float]:
        answer = self.extract_answer(text)
        confidence = np.nan

        # Extract probability/confidence
        prob_match = re.search(
            r"Probability:\s*(-?[0-9]*\.?[0-9]+)", text, re.IGNORECASE
        )
        if prob_match:
            try:
                prob_value = float(prob_match.group(1))
                if 0.0 <= prob_value <= 1.0:
                    confidence = prob_value
            except ValueError:
                pass

        return answer, confidence


class MMLUSequencePosteriorExtractor(SequencePosteriorExtractor):
    """Sequence posterior extractor for MMLU letter answers.

    Computes the joint log probability of the answer letter token(s)
    from the model's logits.
    """

    def extract_answer(self, text: str) -> str:
        """Extract the A/B/C/D choice from the text."""
        answer_match = re.search(r"Answer:\s*([A-D])\b", text, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()

        start_match = re.match(r"\s*([A-D])\b", text, re.IGNORECASE)
        if start_match:
            return start_match.group(1).upper()

        match = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        return ""


class GCPMMLUSequencePosteriorExtractor(GCPSequencePosteriorExtractor):
    """
    Sequence posterior confidence extractor for MMLU on GCP models.

    Returns confidence by exponentiating the sum of log probabilities
    returned by the GCP model for the answer letter.
    """

    def extract_answer(self, text: str) -> str:
        """Extract the A/B/C/D choice from the text."""
        answer_match = re.search(r"Answer:\s*([A-D])\b", text, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()

        start_match = re.match(r"\s*([A-D])\b", text, re.IGNORECASE)
        if start_match:
            return start_match.group(1).upper()

        match = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        return ""
