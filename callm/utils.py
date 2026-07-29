from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from callm.config import CACHE_PATH, HF_TOKEN
import os
import re
from datasets import Dataset
from transformers import Glm4vForConditionalGeneration
from transformers import (
    Mistral3ForConditionalGeneration,
    FineGrainedFP8Config,
    MistralCommonBackend,
)


def get_tokenizer_for_model(model_name: str, hf_token: str = None):
    if hf_token is None:
        hf_token = HF_TOKEN
    if model_name.startswith("google/flan-t5") or model_name.startswith("zai-org"):
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


def initialize_model(model_name: str, hf_token: str = None):
    model = None
    is_seq2seq = None
    if hf_token is None:
        hf_token = HF_TOKEN
    if model_name.startswith("google/flan-t5"):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=CACHE_PATH)
        is_seq2seq = True
    elif model_name == "zai-org/GLM-4.6V-Flash":
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
        is_seq2seq = False
    elif model_name.startswith("mistralai"):
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name, quantization_config=FineGrainedFP8Config(dequantize=True)
        )
    else:
        raise NotImplementedError(
            f"Model {model_name} not supported in initialize_model"
        )

    return model, is_seq2seq


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


def subsample_dataset(dataset: Dataset, max_samples: int, seed: int = None):
    if max_samples is not None:
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(min(len(dataset), max_samples)))
    return dataset


def normalize_answer(text: str) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    t = text.lower().strip()
    t = re.sub(r"\s*\([^)]*\)", "", t)  # Remove parenthetical content
    t = t.replace("'", "").replace("-", "")
    t = t.replace("**", "")
    t = t.strip("\"'")
    t = t.strip()
    t = t.rstrip(".")
    t = t.strip()
    return t


def check_exact_match(pred_answer: str, gold_answers: list) -> bool:
    if pred_answer is None:
        return False

    norm_pred = normalize_answer(pred_answer)
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
