from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from callm.config import CACHE_PATH, HF_TOKEN


def get_tokenizer_for_model(model_name: str, hf_token: str = None):
    if hf_token is None:
        hf_token = HF_TOKEN
    if model_name.startswith("google/flan-t5"):
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
    elif model_name.startswith("Qwen/") or model_name.startswith("meta-llama/"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=CACHE_PATH,
            trust_remote_code=True if model_name.startswith("Qwen/") else False,
            use_auth_token=hf_token,
        )
        is_seq2seq = False
    else:
        raise NotImplementedError(
            f"Model {model_name} not supported in initialize_model"
        )

    return model, is_seq2seq
