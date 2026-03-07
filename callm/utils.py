from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from callm.config import CACHE_PATH, HF_TOKEN
import os
from datasets import Dataset
from transformers import AutoProcessor, Glm4vForConditionalGeneration


def get_tokenizer_for_model(model_name: str, hf_token: str = None):
    if hf_token is None:
        hf_token = HF_TOKEN
    if model_name == "zai-org/GLM-4.6V-Flash":
        tokenizer = AutoProcessor.from_pretrained(model_name)
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
