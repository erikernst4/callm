"""
Base answer extractors for parsing LLM responses.

Provides the foundational classes for extracting answers and confidence scores
from generated model text.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List
import re
import numpy as np
import torch
from lightning.pytorch import LightningModule
from callm.utils import get_tokenizer_for_model


# "Probability:" followed by markdown emphasis or brackets, e.g. "Probability: **0.95**"
_PROBABILITY_PATTERN = re.compile(
    r"Probability:[\s*_<]*(-?[0-9]*\.?[0-9]+)", re.IGNORECASE
)


def extract_verbalized_probability(text: str) -> float:
    """Extract a verbalized probability in [0, 1], or np.nan if missing or invalid."""
    prob_match = _PROBABILITY_PATTERN.search(text)
    if prob_match:
        try:
            prob_value = float(prob_match.group(1))
        except ValueError:
            return np.nan
        # Only accept valid range [0, 1]; out of range is invalid
        if 0.0 <= prob_value <= 1.0:
            return prob_value
    return np.nan


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

        # Remove markdown bold/italic formatting (e.g. **answer** or __answer__)
        answer = re.sub(r"^[\*\_]+|[\*\_]+$", "", answer).strip()

        return answer


class SequencePosteriorExtractor(BaseExtractor, ABC):
    """Abstract base class for sequence posterior confidence extractors.

    Provides common logit extraction and probability computation logic.
    Subclasses implement get_target_token_indices to specify which tokens
    to compute probabilities for.
    """

    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = get_tokenizer_for_model(model_name)

    def get_target_token_indices(self, text: str, offsets: list) -> List[int]:
        """Get the token indices to compute probability for.

        Args:
            text: The generated text
            offsets: List of (start, end) character offset tuples per token

        Returns:
            List of token indices to use for probability computation
        """
        choice = self.extract_answer(text)
        return self._get_token_indices_for_text(text, choice, offsets)

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

    def _build_offset_mapping_from_ids(self, output_ids: torch.Tensor) -> list:
        """Build character offset mapping by incremental prefix decoding.

        Decodes output_ids[0:1], output_ids[0:2], ..., output_ids[0:n] and
        computes each token's character span from the length differences.
        This is resilient to whitespace/special characters because the
        tokenizer reconstructs the full string context at each step.

        Args:
            output_ids: 1D tensor of generated token IDs

        Returns:
            List of (start, end) character offset tuples per token
        """
        offsets = []
        prev_len = 0
        ids_list = output_ids.tolist()
        for i in range(len(ids_list)):
            prefix_text = self.tokenizer.decode(
                ids_list[: i + 1], skip_special_tokens=True
            )
            curr_len = len(prefix_text)
            if curr_len > prev_len:
                offsets.append((prev_len, curr_len))
            else:
                # Special/control token that produces no text
                offsets.append((0, 0))
            prev_len = curr_len
        return offsets

    def _get_token_indices_for_text(
        self, text: str, query: str, offsets: list
    ) -> List[int]:
        """Helper to find token indices for a specific substring within the text.

        Args:
            text: The full generated text
            query: The substring to find indices for
            offsets: List of (start, end) character offset tuples per token

        Returns:
            List of token indices corresponding to the query
        """
        start_pos = text.find(query)
        if start_pos == -1:
            return []

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

        # Try native offset mapping first; fall back to manual reconstruction
        # for tokenizers that don't support it (e.g. Mistral's MistralCommonBackend)
        try:
            encodings = self.tokenizer(text, return_offsets_mapping=True)
            offsets = encodings.offset_mapping
        except (ValueError, TypeError):
            if output_ids is None:
                return answer, np.nan
            offsets = self._build_offset_mapping_from_ids(output_ids)

        token_indices = self.get_target_token_indices(text, offsets)

        if not token_indices:
            return answer, np.nan

        confidence = self.compute_probability_from_logits(logits, token_indices)
        return answer, confidence


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


class IsTrueExtractor(BaseExtractor):
    """Base class for IsTrue extractors (the prompt asks if a proposed answer is True/False).

    Combine it with a posterior extractor, which computes the raw confidence:
    the posterior of the generated answer (True/A or False/B).

    Args:
        normalize: renormalize the posterior over the True/False label pair,
            p(answer) / (p(True) + p(False)). Needs the model's top_logprobs.
        return_p_true: return P(True) instead of the posterior of the generated
            answer, i.e. 1 - confidence for False/B answers.
    """

    LABEL_PAIRS = {
        "true": ("true", "false"),
        "false": ("true", "false"),
        "a": ("a", "b"),
        "b": ("a", "b"),
    }
    TRUE_ANSWERS = {"true", "a"}
    FALSE_ANSWERS = {"false", "b"}

    def __init__(
        self, *args, normalize: bool = False, return_p_true: bool = False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.normalize = normalize
        self.return_p_true = return_p_true

    def forward(
        self, text, logits=None, output_ids=None, *args, top_logprobs=None, **kwargs
    ):
        if self.normalize:
            answer = self.extract_answer(text) if text else ""
            confidence = self.normalized_posterior(answer, top_logprobs)
        else:
            answer, confidence = super().forward(
                text, logits, output_ids, *args, **kwargs
            )
        if self.return_p_true:
            confidence = self.to_p_true([answer], [confidence])[0]
        return answer, confidence

    def extract_answer(self, text: str) -> str:
        """Extract the A/B choice, falling back to True/False."""
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

    @classmethod
    def normalized_posterior(cls, answer: str, top_logprobs) -> float:
        """Posterior of an IsTrue answer renormalized over its True/False label pair.

        Uses the first non-whitespace generation step of `top_logprobs` (a list of
        {token: logprob} dicts, one per step). Returns p(answer) / (p(True) + p(False)),
        summing case/whitespace variants of each label. Returns np.nan if the answer
        is not a label, or the answer label is not (tied for) top-1 at the first step.
        """
        answer = (answer or "").strip().lower()
        if answer not in cls.LABEL_PAIRS or not top_logprobs:
            return np.nan
        pair = cls.LABEL_PAIRS[answer]

        step = next((s for s in top_logprobs if s and max(s, key=s.get).strip()), None)
        if step is None:
            return np.nan
        # The answer label must be (tied for) top-1: bf16 logits often tie exactly,
        # and topk may then list another token first.
        answer_logprobs = [lp for t, lp in step.items() if t.strip().lower() == answer]
        if not answer_logprobs or max(answer_logprobs) < max(step.values()) - 1e-6:
            return np.nan

        mass = {label: 0.0 for label in pair}
        for token, logprob in step.items():
            token = token.strip().lower()
            if token in mass:
                mass[token] += float(np.exp(logprob))
        return mass[answer] / (mass[pair[0]] + mass[pair[1]])

    @classmethod
    def to_p_true(cls, answers, confidences) -> np.ndarray:
        """Convert IsTrue confidences (posterior of the generated answer) to P(True).

        True/A answers keep their confidence, False/B answers get 1 - confidence,
        and anything else is NaN. Scripts use it on saved llm_outputs.csv files.
        """
        result = []
        for answer, conf in zip(answers, confidences):
            # bools come from pd.read_csv parsing an all-True/False column
            if isinstance(answer, (str, bool, np.bool_)):
                answer = str(answer).strip().lower()
            else:
                answer = ""
            conf = float(conf)
            if answer in cls.TRUE_ANSWERS:
                result.append(conf)
            elif answer in cls.FALSE_ANSWERS:
                result.append(1.0 - conf)
            else:
                result.append(np.nan)
        return np.array(result, dtype=float)


class IsTruePosteriorExtractor(IsTrueExtractor, SequencePosteriorExtractor):
    """IsTrue extractor for HF models (tokenizer-aligned sequence posterior)."""

    def __init__(
        self, model_name: str, normalize: bool = False, return_p_true: bool = False
    ):
        super().__init__(model_name, normalize=normalize, return_p_true=return_p_true)


class GCPIsTruePosteriorExtractor(IsTrueExtractor, GCPSequencePosteriorExtractor):
    """IsTrue extractor for GCP models (sum of the generated tokens' logprobs)."""

    def __init__(
        self,
        model_name: str = None,
        normalize: bool = False,
        return_p_true: bool = False,
    ):
        super().__init__(model_name, normalize=normalize, return_p_true=return_p_true)
