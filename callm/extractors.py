"""
Answer extractors for parsing LLM responses.

Extractors parse the LLM's text output to extract the answer and confidence score.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List
import re
import numpy as np
import torch
from lightning.pytorch import LightningModule
from callm.utils import get_tokenizer_for_model


class BaseExtractor(LightningModule, ABC):
    """Abstract base class for answer extractors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        # Extract guess/answer — handles both prefilled and non-prefilled outputs
        guess_match = re.search(r"Guess:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if guess_match:
            answer = guess_match.group(1).strip()
        else:
            # Fallback: use the full text (stripped), taking first line
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


class SequencePosteriorExtractor(BaseExtractor, ABC):
    """Abstract base class for sequence posterior confidence extractors.

    Provides common logit extraction and probability computation logic.
    Subclasses implement get_target_token_indices to specify which tokens
    to compute probabilities for.
    """

    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = get_tokenizer_for_model(model_name)

    def get_target_token_indices(self, text: str, encodings: dict) -> List[int]:
        """Get the token indices to compute probability for.

        Args:
            text: The generated text
            encodings: Tokenizer output with offset_mapping

        Returns:
            List of token indices to use for probability computation
        """
        choice = self.extract_answer(text)
        return self._get_token_indices_for_text(text, choice, encodings)

    def compute_probability_from_logits(
        self, logits: torch.Tensor, token_indices: List[int]
    ) -> float:
        """Compute joint probability from logits at specified token indices.

        Args:
            logits: Logits tensor from the model
            token_indices: Indices of tokens to compute probability for

        Returns:
            Joint probability as a float
        """
        if not token_indices:
            return np.nan

        answer_logits = logits[token_indices]

        # Check if logits are 1D (optimized storage) or 2D (full vocab)
        if answer_logits.ndim == 1:
            # 1D: These are already log probabilities of the tokens
            max_log_probs_per_token = answer_logits
        else:
            # 2D: [num_tokens, vocab_size] - Traditional full logits
            # Compute log probabilities
            log_probs = torch.log_softmax(answer_logits, dim=-1)
            # Get the maximum log probability for each token
            max_log_probs_per_token = log_probs.max(dim=-1).values

        joint_log_prob = max_log_probs_per_token.sum().item()
        confidence = np.exp(joint_log_prob)

        return confidence

    def _get_token_indices_for_text(
        self, text: str, query: str, encodings: dict
    ) -> List[int]:
        """Helper to find token indices for a specific substring within the text.

        Args:
            text: The full generated text
            query: The substring to find indices for
            encodings: Tokenizer output with offset_mapping

        Returns:
            List of token indices corresponding to the query
        """
        start_pos = text.find(query)
        if start_pos == -1:
            return []

        offsets = encodings.offset_mapping
        end_pos = start_pos + len(query)
        token_indices = []

        for idx, (start, end) in enumerate(offsets):
            # Check intersection between token range and query range
            overlap_start = max(start, start_pos)
            overlap_end = min(end, end_pos)
            if overlap_start < overlap_end:
                token_indices.append(idx)

        return token_indices

    def forward(
        self, text: str, logits: torch.Tensor, output_ids: torch.Tensor, *args, **kwargs
    ) -> Tuple[str, float]:
        if not text:
            return "", np.nan

        answer = self.extract_answer(text)

        if logits is None:
            return answer, np.nan

        encodings = self.tokenizer(text, return_offsets_mapping=True)
        token_indices = self.get_target_token_indices(text, encodings)

        if not token_indices:
            return answer, np.nan

        confidence = self.compute_probability_from_logits(logits, token_indices)
        return answer, confidence


class IsTruePosteriorExtractor(SequencePosteriorExtractor):
    """Extractor for IS_TRUE_PROB_PROMPT that computes P(True) from sequence posterior.

    The prompt asks if a proposed answer is:
        (A) True or
        (B) False?

    This extractor computes the probability of the original answer being True.
    If the model chooses "(A)", the confidence is the sequence posterior of "(A)".
    If the model chooses "(B)", the confidence is 1.0 - sequence_posterior of "(B)".
    """

    def extract_answer(self, text: str) -> str:
        """Extract the A/B choice from the text."""
        # Find first occurrence of A or B as a whole word
        match = re.search(r"\b(A|B)\b", text, re.IGNORECASE)
        if match:
            return match.group(0).upper()

        # Fallback to True/False if labels not found
        if "True" in text:
            return "True"
        if "False" in text:
            return "False"

        return ""


class GCPSequencePosteriorExtractor(BaseExtractor):
    """
    Sequence posterior confidence extractor for GCP models.

    Returns confidence by exponentiating the sum of log probabilities
    returned by the GCP model, ignoring any tokenizer alignment.
    """

    def __init__(self, model_name: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        text: str,
        logits: torch.Tensor = None,
        output_ids: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> Tuple[str, float]:
        if not text:
            return "", np.nan

        answer = self.extract_answer(text)

        if logits is None:
            return answer, np.nan

        # For GCP models, the returned 'logits' are already the log probabilities
        # of the generated tokens. We just sum them.
        try:
            joint_log_prob = logits.sum().item()
            confidence = np.exp(joint_log_prob)
        except Exception:
            confidence = np.nan

        return answer, confidence


class GCPIsTruePosteriorExtractor(GCPSequencePosteriorExtractor):
    """
    Extractor for IS_TRUE_PROB_PROMPT that computes joint logprob for GCP generated text.
    """

    def extract_answer(self, text: str) -> str:
        """Extract the A/B choice from the text."""
        # Find first occurrence of A or B as a whole word
        match = re.search(r"\b(A|B)\b", text, re.IGNORECASE)
        if match:
            return match.group(0).upper()

        # Fallback to True/False if labels not found
        if "True" in text:
            return "True"
        if "False" in text:
            return "False"

        return ""
