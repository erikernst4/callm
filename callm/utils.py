import os
import re
import glob
import torch
import string
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Glm4vForConditionalGeneration,
    Mistral3ForConditionalGeneration,
    FineGrainedFP8Config,
    MistralCommonBackend,
)
from callm.config import CACHE_PATH, HF_TOKEN


def get_tokenizer_for_model(model_name: str, hf_token: str = None):
    if hf_token is None:
        hf_token = HF_TOKEN
    if model_name.startswith("zai-org"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_name.startswith("meta-llama/"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", use_auth_token=hf_token
        )
    elif model_name.startswith("Qwen/"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True,
            use_auth_token=hf_token,
        )
    elif model_name.startswith("mistralai"):
        tokenizer = MistralCommonBackend.from_pretrained(model_name)
    else:
        raise NotImplementedError(
            f"Model {model_name} not supported in get_tokenizer_for_model"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def generation_pad_token_id(model, tokenizer):
    """
    ``pad_token_id`` to pass to ``generate()``: the model config's, else the
    tokenizer's.
    """
    pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
    return pad_token_id


def initialize_model(model_name: str, hf_token: str = None):
    """Load a supported causal (decoder-only) model."""
    model = None
    if hf_token is None:
        hf_token = HF_TOKEN
    if model_name == "zai-org/GLM-4.6V-Flash":
        model = model = Glm4vForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_name,
        )
    elif (
        model_name.startswith("Qwen/")
        or model_name.startswith("meta-llama/")
        or model_name.startswith("zai-org")
    ):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=CACHE_PATH,
            trust_remote_code=True if model_name.startswith("Qwen/") else False,
            # use_auth_token=hf_token,
        )
    elif model_name.startswith("mistralai"):
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name, quantization_config=FineGrainedFP8Config(dequantize=True)
        )
    else:
        raise NotImplementedError(
            f"Model {model_name} not supported in initialize_model"
        )

    return model


def get_last_llm_outputs_path(log_dir: str):
    llm_outputs_path = None
    # Look for the last run in lightning_logs
    lightning_logs = os.path.join(log_dir, "lightning_logs")
    if os.path.exists(lightning_logs):
        # Get all version directories sorting by creation time
        versions = sorted(
            [
                os.path.join(lightning_logs, d)
                for d in os.listdir(lightning_logs)
                if d.startswith("version_")
            ],
            key=os.path.getmtime,
        )
        if versions:
            for version_dir in reversed(versions):
                candidate_path = os.path.join(version_dir, "llm_outputs.csv")
                if os.path.exists(candidate_path):
                    llm_outputs_path = candidate_path
                    print(f"Found latest LLM outputs at: {llm_outputs_path}")
                    break

    if llm_outputs_path is None:
        # Fallback to checking the current trainer's log dir if available
        if log_dir:
            candidate_path = os.path.join(log_dir, "llm_outputs.csv")
            if os.path.exists(candidate_path):
                llm_outputs_path = candidate_path
                print(f"Found LLM outputs in current log dir: {llm_outputs_path}")
    return llm_outputs_path


def find_chunk_files(resume_dir: str, prefix: str = "activations_chunk") -> list[str]:
    """Find chunk files in resume_dir or its lightning_logs subdirectories."""
    if not resume_dir:
        return []
    if not os.path.exists(resume_dir):
        raise ValueError(f"Resume path {resume_dir} does not exist.")

    # 1. Direct search in directory
    pattern = os.path.join(resume_dir, f"{prefix}_rank*.pt")
    found_files = glob.glob(pattern)

    # 2. Search in lightning_logs/version_*
    if not found_files:
        version_dirs = sorted(
            glob.glob(os.path.join(resume_dir, "lightning_logs", "version_*")),
            key=lambda x: int(os.path.basename(x).split("_")[-1])
            if os.path.basename(x).split("_")[-1].isdigit()
            else -1,
        )
        if version_dirs:
            for v_dir in reversed(version_dirs):
                files = glob.glob(os.path.join(v_dir, f"{prefix}_rank*.pt"))
                if files:
                    found_files = files
                    break

    def get_chunk_num(p: str) -> int:
        match = re.search(rf"{prefix}_rank\d+_(\d+)\.pt", os.path.basename(p))
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)\.pt$", os.path.basename(p))
        return int(match.group(1)) if match else 0

    return sorted(found_files, key=get_chunk_num)


def get_skip_indices_from_chunks(
    resume_dir: str, prefix: str = "activations_chunk"
) -> set[int]:
    """Get sample indices to skip by inspecting chunk files in resume_dir."""
    skip_indices = set()
    if not resume_dir:
        return skip_indices

    found_files = find_chunk_files(resume_dir, prefix=prefix)
    if found_files:
        print(f"Found {len(found_files)} temp chunk files to resume from.")
        for fpath in found_files:
            try:
                try:
                    chunk = torch.load(
                        fpath, map_location="cpu", weights_only=False, mmap=True
                    )
                except TypeError:
                    chunk = torch.load(fpath, map_location="cpu", weights_only=False)
                for item in chunk:
                    idx = item.get("sample_idx", item.get("index"))
                    if idx is not None:
                        skip_indices.add(idx)
                del chunk
            except Exception as e:
                print(f"Skipping corrupted or unreadable chunk {fpath}: {e}")
        print(f"Found {len(skip_indices)} indices to skip.")
    else:
        print(f"No temp chunk files found in {resume_dir}")

    return skip_indices


def subsample_dataset(dataset: Dataset, max_samples: int, seed: int = None):
    if max_samples is not None:
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(min(len(dataset), max_samples)))
    return dataset


_DELETED_CHARS = "'’‘-*"
_SPACED_CHARS = "".join(
    c for c in string.punctuation + "“”–—…" if c not in _DELETED_CHARS
)
_PUNCT_TABLE = str.maketrans(
    {**{c: "" for c in _DELETED_CHARS}, **{c: " " for c in _SPACED_CHARS}}
)


def normalize_answer(text: str) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    t = text.lower()
    t = re.sub(r"\s*\([^)]*\)", "", t)  # Remove parenthetical content
    t = t.translate(_PUNCT_TABLE)  # Drop apostrophes/hyphens/markdown, space others
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    return " ".join(t.split())


MISSING_CONFIDENCE_FALLBACK = 0.5


def impute_missing_confidences(
    confidences: np.ndarray, fallback: float = MISSING_CONFIDENCE_FALLBACK
) -> np.ndarray:
    """Replace missing (NaN) confidences, so every metric sees the same values."""
    return np.where(np.isnan(confidences), fallback, confidences)


def check_exact_match(pred_answer: str, gold_answers: list) -> bool:
    if pred_answer is None:
        return False

    norm_pred = normalize_answer(pred_answer)
    if not norm_pred:  # Empty answers never match, even aliases like "(USA)"
        return False
    norm_golds = [normalize_answer(g) for g in gold_answers]

    if norm_pred in norm_golds:
        return True

    return False


def get_indices_of_exact_answer(
    tokenizer, input_output_ids, exact_answer: str, prompt_len: int
):
    """
    Finds the range of token indices [lower, higher] in the prompt+answer sequence
    corresponding to the exact_answer substring using shrinking token search.
    """
    if not exact_answer or not isinstance(exact_answer, str):
        return None

    exact_answer_clean = exact_answer.lower().strip()
    if not exact_answer_clean or exact_answer_clean == "no answer":
        return None

    lower = prompt_len
    higher = len(input_output_ids) - 1

    if lower >= len(input_output_ids):
        return None

    full_generated_answer = tokenizer.decode(input_output_ids[lower:])
    exact_answer_index = full_generated_answer.lower().find(exact_answer_clean)

    if exact_answer_index == -1:
        return None

    true_exact_answer = full_generated_answer[
        exact_answer_index : exact_answer_index + len(exact_answer_clean)
    ]

    while higher >= lower and true_exact_answer in tokenizer.decode(
        input_output_ids[lower : higher + 1]
    ):
        higher -= 1
    higher += 1

    while lower <= higher and true_exact_answer in tokenizer.decode(
        input_output_ids[lower : higher + 1]
    ):
        lower += 1
    lower -= 1

    if lower > higher:
        return None

    return list(range(lower, higher + 1))
