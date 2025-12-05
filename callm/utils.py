from transformers import AutoTokenizer

def get_tokenizer_for_model(model_name: str, HF_TOKEN: str = None):
    from transformers import AutoTokenizer

    if model_name in ['flan-t5-small', 'flan-t5-base', 'flan-t5-large','flan-t5-xl', 'flan-t5-xxl']:
        model_load_name = f'google/{model_name}'
        tokenizer = AutoTokenizer.from_pretrained(model_load_name)
    elif model_name in ['Llama-2-7b-chat-hf']:
        model_load_name = f'meta-llama/{model_name}'
        tokenizer = AutoTokenizer.from_pretrained(model_load_name, padding_side='left', use_auth_token=HF_TOKEN)
    elif model_name.startswith('Qwen/'):
        # Qwen models (e.g., Qwen/Qwen3-0.6B)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left',
            trust_remote_code=True
        )
    else:
        raise NotImplementedError(f"Model {model_name} not supported in get_tokenizer_for_model")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer