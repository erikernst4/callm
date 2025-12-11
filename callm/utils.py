from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from callm.config import CACHE_PATH


def get_tokenizer_for_model(model_name: str, HF_TOKEN: str = None):
    if model_name in [
        "flan-t5-small",
        "flan-t5-base",
        "flan-t5-large",
        "flan-t5-xl",
        "flan-t5-xxl",
    ]:
        model_load_name = f"google/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_load_name)
    elif model_name in ["Llama-2-7b-chat-hf"]:
        model_load_name = f"meta-llama/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(
            model_load_name, padding_side="left", use_auth_token=HF_TOKEN
        )
    elif model_name.startswith("Qwen/"):
        # Qwen models (e.g., Qwen/Qwen3-0.6B)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", trust_remote_code=True
        )
    else:
        raise NotImplementedError(
            f"Model {model_name} not supported in get_tokenizer_for_model"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def initialize_model(model_name: str):
    model = None
    is_seq2seq = None

    if model_name.startswith("google/flan-t5") or model_name.startswith("t5-"):
        # Seq2Seq model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=CACHE_PATH)
        is_seq2seq = True
    elif model_name.startswith("Qwen/") or model_name.startswith("meta-llama/"):
        # Causal model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=CACHE_PATH,
            trust_remote_code=True if model_name.startswith("Qwen/") else False,
        )
        is_seq2seq = False
    else:
        # Try seq2seq first, fall back to causal
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, cache_dir=CACHE_PATH
            )
            is_seq2seq = True
        except ValueError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, cache_dir=CACHE_PATH
            )
            is_seq2seq = False

    return model, is_seq2seq
