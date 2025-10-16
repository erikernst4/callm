from lightning.pytorch import LightningModule
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

class LLM(LightningModule):
    def __init__(
        self,
        model_name: str = "flan-t5-small",
        HF_TOKEN: str = None,
        train: bool = False,
    ):
        super().__init__()

        if model_name in ['flan-t5-small', 'flan-t5-base', 'flan-t5-large','flan-t5-xl', 'flan-t5-xxl']:
            model_load_name = f'google/{model_name}'
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_load_name, cache_dir=CACHE_PATH)
            self.tokenizer = AutoTokenizer.from_pretrained(model_load_name)
        elif model_name in ['Llama-2-7b-chat-hf']:
            model_load_name = f'meta-llama/{model_name}'
            self.model = AutoModelForCausalLM.from_pretrained(model_load_name, cache_dir=CACHE_PATH, use_auth_token=HF_TOKEN)
            self.model.config.pad_token_id = self.model.config.eos_token_id
            self.tokenizer = AutoTokenizer.from_pretrained(model_load_name, padding_side='left', use_auth_token=HF_TOKEN)
        else:
            raise NotImplementedError
        
        if not train:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, inputs, target):
        return self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=1000)

    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch["data"], batch["label"]
        output = self.model(inputs, target)
        
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)
